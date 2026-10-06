//! The backend under a scripted bus.
//!
//! No pseudo-terminals here, unlike the sibling that inspired this module: the interesting
//! failures are not serial-driver failures but *protocol* ones — a reply that arrives out of
//! order, a servo that is silent, a profile that drifted, a torque-off that did not take — and a
//! transport whose registers a test can read and write is a sharper instrument for those than a
//! pair of PTYs pretending to be fifteen servos.
//!
//! Each test's comment names the failure it exists to prevent, per `CONTRIBUTING.md`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::ErrorKind;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;

/// Byte offset of the block every device answers: the servo's own present-position register.
/// The node answers fifteen bytes there too, which is what makes one `sync_read` carry both — so
/// the node's fields and the servo's `block::*` offsets share this base.
const NODE_BASE: usize = reg::PRESENT_POSITION_L as usize;

#[derive(Default)]
struct BusState {
    /// A register file per device, in `[node, joints…]` order.
    regs: Vec<[u8; 256]>,
    /// Bytes waiting to be read by the backend.
    out: VecDeque<u8>,
    /// Every frame the backend wrote, in order. The bring-up *sequence* is an assertion.
    log: Vec<Vec<u8>>,
    /// Devices that never answer. Their slot still passes, which is the point.
    silent: HashSet<u8>,
    /// Nonzero status bytes to report, per id.
    alarms: HashMap<u8, u8>,
    /// Answer in reverse id order, which the protocol permits and arrival order would break.
    reverse: bool,
    /// A servo that keeps its torque bit set whatever is written.
    stuck_enabled: Option<u8>,
    /// A servo that refuses to energise, for the partial-enable path.
    refuses_enable: Option<u8>,
    /// EEPROM registers that accept a write and do not take it — a different firmware.
    stick_eeprom: bool,
    /// The first id, which is the node's.
    node_id: u8,
}

/// A handle a test keeps while the backend owns the transport.
#[derive(Clone)]
struct Bus(Arc<Mutex<BusState>>);

impl Bus {
    fn new(calibration: &Calibration) -> Self {
        let mut ids: Vec<u8> = Vec::with_capacity(NUM_JOINTS + 1);
        ids.push(calibration.imu_id);
        ids.extend_from_slice(&crate::model::JOINT_IDS);
        let mut regs: Vec<[u8; 256]> = Vec::with_capacity(ids.len());
        for (slot, _) in ids.iter().enumerate() {
            let mut file = [0u8; 256];
            if slot == 0 {
                // Gyro, then a half-precision quaternion whose x is 0.5. Non-zero on purpose: an
                // all-zero quaternion is what the node sends before SFLP has written its table,
                // and the decoder would hold a default forever instead of converging.
                file[NODE_BASE + 7] = 0x38;
                file[NODE_BASE + imu_block::STATUS] = wire::IMU_FLAG_SFLP_VALID;
            } else {
                let joint = slot - 1;
                let zero = calibration.joints[joint].zero_ticks as u16;
                file[NODE_BASE + block::POSITION] = zero as u8;
                file[NODE_BASE + block::POSITION + 1] = (zero >> 8) as u8;
                // A plausible standing robot rather than all-zero, so a decode that reads the
                // wrong offset has something to be wrong about.
                file[NODE_BASE + block::SPEED] = 2;
                file[NODE_BASE + block::CURRENT] = 4;
                file[NODE_BASE + block::VOLTAGE] = 74; // 7.4 V
                file[NODE_BASE + block::TEMPERATURE] = 30;
                // The travel window a command is clamped into.
                file[reg::MAX_ANGLE_LIMIT_L as usize] = 0xFF;
                file[reg::MAX_ANGLE_LIMIT_L as usize + 1] = 0x0F; // 4095
                // The factory gains in both places the real servo has them: 50/51 are the RAM
                // pair a daemon writes, 21/22 the EEPROM pair a reboot reloads them from.
                file[reg::KP as usize] = 32;
                file[reg::KD as usize] = 40;
                file[reg::EEPROM_KP as usize] = 32;
                file[reg::EEPROM_KD as usize] = 40;
                // Non-zero, which is what this servo reports: a locked servo takes an EEPROM write
                // into RAM and does not persist it, so the lock is part of the fixture.
                file[reg::LOCK as usize] = 1;
            }
            regs.push(file);
        }
        Self(Arc::new(Mutex::new(BusState {
            regs,
            node_id: calibration.imu_id,
            ..Default::default()
        })))
    }

    fn with<R>(&self, f: impl FnOnce(&mut BusState) -> R) -> R {
        f(&mut self.0.lock().unwrap())
    }

    fn get<T>(&self, f: impl FnOnce(&BusState) -> T) -> T {
        f(&self.0.lock().unwrap())
    }

    fn log(&self) -> Vec<Vec<u8>> {
        self.get(|s| s.log.clone())
    }

    fn frames_after(&self, n: usize) -> Vec<Vec<u8>> {
        self.get(|s| s.log[n.min(s.log.len())..].to_vec())
    }

    fn frame_count(&self) -> usize {
        self.get(|s| s.log.len())
    }

    /// The joint's present position, as the fake's register file has it.
    fn present_ticks(&self, joint: usize) -> u16 {
        self.get(|s| {
            let file = &s.regs[1 + joint];
            wire::le_u16(
                file[NODE_BASE + block::POSITION],
                file[NODE_BASE + block::POSITION + 1],
            )
        })
    }

    fn set_imu_counter(&self, counter: u8) {
        self.with(|s| s.regs[0][NODE_BASE + imu_block::COUNT] = counter);
    }

    fn set_imu_status(&self, status: u8) {
        self.with(|s| s.regs[0][NODE_BASE + imu_block::STATUS] = status);
    }

    fn slot_of(&self, id: u8) -> usize {
        self.get(|s| {
            if id == s.node_id {
                0
            } else {
                1 + crate::model::JOINT_IDS
                    .iter()
                    .position(|j| *j == id)
                    .expect("frame names an id we asked for")
            }
        })
    }
}

struct FakeBus {
    bus: Bus,
}

impl FakeBus {
    fn ack(id: u8, status: u8, data: &[u8]) -> Vec<u8> {
        let len = (data.len() + 2) as u8;
        let sum = wire::checksum(id, len, status, 0, data);
        let mut frame = vec![0xFF, 0xFF, id, len, status];
        frame.extend_from_slice(data);
        frame.push(sum);
        frame
    }

    fn handle(&self, frame: &[u8]) {
        let bus = &self.bus;
        let node_id = bus.get(|s| s.node_id);
        let id = frame[2];
        match frame[4] {
            wire::inst::SYNC_READ => {
                let addr = frame[5] as usize;
                let len = frame[6] as usize;
                let mut ids: Vec<u8> = frame[7..frame.len() - 1].to_vec();
                if bus.get(|s| s.reverse) {
                    ids.reverse();
                }
                for id in ids {
                    if bus.get(|s| s.silent.contains(&id)) {
                        continue;
                    }
                    let (data, status) = bus.get(|s| {
                        let slot = if id == node_id {
                            0
                        } else {
                            1 + crate::model::JOINT_IDS
                                .iter()
                                .position(|j| *j == id)
                                .unwrap()
                        };
                        let data: Vec<u8> = (0..len).map(|i| s.regs[slot][addr + i]).collect();
                        (data, s.alarms.get(&id).copied().unwrap_or(0))
                    });
                    bus.with(|s| s.out.extend(Self::ack(id, status, &data)));
                }
            }
            wire::inst::SYNC_WRITE => {
                let addr = frame[5] as usize;
                let width = frame[6] as usize;
                let mut cursor = 7;
                while cursor < frame.len() - 1 {
                    let id = frame[cursor];
                    let data = frame[cursor + 1..cursor + 1 + width].to_vec();
                    let slot = self.bus.slot_of(id);
                    let stick = self.bus.get(|s| s.stick_eeprom)
                        && (addr == reg::EEPROM_KP as usize || addr == reg::EEPROM_KD as usize);
                    if !stick {
                        self.bus.with(|s| {
                            for (i, byte) in data.iter().enumerate() {
                                s.regs[slot][addr + i] = *byte;
                            }
                        });
                    }
                    self.after_write(id, addr, &data);
                    cursor += 1 + width;
                }
            }
            wire::inst::WRITE => {
                let addr = frame[5] as usize;
                let data = frame[6..frame.len() - 1].to_vec();
                let slot = self.bus.slot_of(id);
                self.bus.with(|s| {
                    for (i, byte) in data.iter().enumerate() {
                        s.regs[slot][addr + i] = *byte;
                    }
                });
                self.after_write(id, addr, &data);
                self.bus.with(|s| s.out.extend(Self::ack(id, 0, &[])));
            }
            wire::inst::REBOOT => {
                // Measured on the real servo: no reply at all, and the EEPROM gains come back
                // over the RAM ones. Both halves matter to what the daemon must do next.
                let slot = self.bus.slot_of(id);
                self.bus.with(|s| {
                    s.regs[slot][reg::KP as usize] = 32;
                    s.regs[slot][reg::KD as usize] = 40;
                    s.regs[slot][reg::TORQUE_ENABLE as usize] = 0;
                });
            }
            other => panic!("the backend wrote instruction {other:#04x}, which no test expects"),
        }
    }

    /// The things a real servo does *after* a register write, which is where these tests differ.
    fn after_write(&self, id: u8, addr: usize, data: &[u8]) {
        let (stuck, refuses) = self
            .bus
            .get(|s| (s.stuck_enabled == Some(id), s.refuses_enable == Some(id)));
        if addr == reg::TORQUE_ENABLE as usize {
            if stuck {
                let slot = self.bus.slot_of(id);
                self.bus
                    .with(|s| s.regs[slot][reg::TORQUE_ENABLE as usize] = 1);
            }
            if refuses && data.first() == Some(&1) {
                let slot = self.bus.slot_of(id);
                self.bus
                    .with(|s| s.regs[slot][reg::TORQUE_ENABLE as usize] = 0);
            }
        }
    }
}

impl Transport for FakeBus {
    fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.bus.with(|s| s.log.push(bytes.to_vec()));
        self.handle(bytes);
        Ok(())
    }

    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.bus.with(|s| {
            if s.out.is_empty() {
                return Err(std::io::Error::new(ErrorKind::TimedOut, "no reply"));
            }
            let n = buf.len().min(s.out.len());
            for byte in buf.iter_mut().take(n) {
                *byte = s.out.pop_front().unwrap();
            }
            Ok(n)
        })
    }

    fn set_timeout(&mut self, _timeout: Duration) -> std::io::Result<()> {
        Ok(())
    }
}

/// A backend and a handle to the bus it writes to.
fn bus() -> (ScsIo, Bus) {
    bus_with(Calibration::default())
}

fn bus_with(calibration: Calibration) -> (ScsIo, Bus) {
    let handle = Bus::new(&calibration);
    let io = ScsIo::with_transport(
        Box::new(FakeBus {
            bus: handle.clone(),
        }),
        calibration,
    );
    (io, handle)
}

/// The address every frame wrote, for finding the goal write among the others.
fn address(frame: &[u8]) -> u8 {
    frame[5]
}

/// Byte offset of a joint's two-byte field inside a `sync_write` frame.
///
/// `FF FF FE LEN 83 ADDR WIDTH` is seven bytes, then each joint occupies `id + data`.
fn goal_field(frame: &[u8], joint: usize) -> u16 {
    let at = 7 + joint * 3 + 1;
    wire::le_u16(frame[at], frame[at + 1])
}

#[test]
fn bring_up_writes_the_pose_before_it_energises() {
    // The failure this prevents, measured on this hardware: energising servos whose goal register
    // still holds a stale target drives every joint at once, and the inrush tripped the SBC's
    // undervoltage lockout four times in eight minutes. The order is the fix, so the order is
    // what is asserted.
    let (mut io, bus) = bus();
    io.set_gain(200).unwrap();
    let before = bus.frame_count();

    io.set_torque(true).unwrap();
    let log = bus.frames_after(before);

    let goal = log
        .iter()
        .position(|f| f[4] == wire::inst::SYNC_WRITE && address(f) == reg::GOAL_POSITION_L)
        .expect("bring-up must write the goal block");
    let torque = log
        .iter()
        .position(|f| f[4] == wire::inst::WRITE && address(f) == reg::TORQUE_ENABLE)
        .expect("bring-up must energise, with an acknowledged write — see `write_torque`");
    assert!(
        goal < torque,
        "the goal write has to come before torque, or the servos drive to wherever the goal \
         register was left"
    );
    // And what it writes is the pose the robot is already in, so the error term is zero.
    for joint in [0usize, 7, NUM_JOINTS - 1] {
        assert_eq!(
            goal_field(&log[goal], joint),
            bus.present_ticks(joint),
            "bring-up must adopt the present pose, not command one"
        );
    }
}

#[test]
fn a_partial_enable_is_refused_and_leaves_torque_off() {
    // A subset of servos energised means the robot is held up by some joints and limp in others,
    // and the control loop cannot tell which. Refusing is the only safe reading.
    let (mut io, bus) = bus();
    bus.with(|s| s.refuses_enable = Some(crate::model::JOINT_IDS[3]));

    let err = io.set_torque(true).unwrap_err().to_string();
    assert!(err.contains("partial enable"), "got: {err}");
    assert!(
        err.contains(JOINT_NAMES[3]),
        "the refusal must name the joint: {err}"
    );
}

/// A released servo must not be commanded, or the command re-arms it and the release is undone.
///
/// Measured: writing a goal to a servo whose torque was just cut turns the torque back on,
/// intermittently, because the goal is stale by the time it lands. The control loop writes goals
/// every tick, so without this the robot cannot be left limp at all.
#[test]
fn a_released_servo_is_not_told_where_to_go() {
    let (mut io, bus) = bus();
    let targets = JointTargets::new([0.0; NUM_JOINTS]);

    // Energised: the goal frame goes out.
    io.set_torque(true).unwrap();
    let before = bus.frame_count();
    io.write(&targets).unwrap();
    assert!(
        bus.frames_after(before)
            .iter()
            .any(|f| f[4] == wire::inst::SYNC_WRITE && address(f) == reg::GOAL_POSITION_L),
        "an energised robot must be commanded"
    );

    // Released: nothing goes out, however many times the loop asks.
    io.set_torque(false).unwrap();
    let before = bus.frame_count();
    for _ in 0..20 {
        io.write(&targets).unwrap();
    }
    assert_eq!(
        bus.frame_count(),
        before,
        "twenty ticks of a released robot must not put one goal frame on the bus"
    );

    // And the bring-up path may still write to a released servo, because that is the order that
    // keeps energising from driving every joint at once.
    let before = bus.frame_count();
    io.set_torque(true).unwrap();
    let log = bus.frames_after(before);
    let goals = log
        .iter()
        .position(|f| f[4] == wire::inst::SYNC_WRITE && address(f) == reg::GOAL_POSITION_L);
    let torque = log
        .iter()
        .position(|f| f[4] == wire::inst::WRITE && address(f) == reg::TORQUE_ENABLE);
    assert!(
        matches!((goals, torque), (Some(g), Some(t)) if g < t),
        "the pre-energise goal write must survive the guard, and still come before torque"
    );
}

/// **Register 40 is written the slow way, and this is the test that keeps it that way.**
///
/// A `sync_write` to register 40 does not stick on this hardware: the servo reads back 0
/// immediately afterwards and is holding again seconds later, with nothing on this side writing 1.
/// A direct `WRITE` releases the joint and stays released. So the shape here is load-bearing, not a
/// style choice — and an unacknowledged frame is exactly the one whose silent failure leaves a
/// robot stiff.
#[test]
fn torque_is_written_with_an_acknowledged_write_per_servo() {
    let (mut io, bus) = bus();
    let before = bus.frame_count();
    io.set_torque(true).unwrap();

    let log = bus.frames_after(before);
    let torque: Vec<&Vec<u8>> = log
        .iter()
        .filter(|f| address(f) == reg::TORQUE_ENABLE && f[4] != wire::inst::SYNC_READ)
        .collect();
    assert_eq!(
        torque.len(),
        crate::model::JOINT_IDS.len(),
        "one write per servo, not one broadcast"
    );
    assert!(
        torque.iter().all(|f| f[4] == wire::inst::WRITE),
        "a sync_write to register 40 is the failure this guards: {:?}",
        torque.iter().map(|f| f[4]).collect::<Vec<_>>()
    );
    // Each frame names exactly one id, and between them every joint is covered.
    let named: Vec<u8> = torque.iter().map(|f| f[2]).collect();
    assert_eq!(named, crate::model::JOINT_IDS.to_vec());
    // And byte for byte, because "an acknowledged write" is only true if the frame says so.
    assert_eq!(
        torque[0],
        &vec![
            0xFF,
            0xFF,
            0x14,
            0x04,
            wire::inst::WRITE,
            reg::TORQUE_ENABLE,
            0x01,
            0xBB
        ],
        "id 20, torque on: len 4, checksum over id+len+inst+addr+data"
    );
}

#[test]
fn a_servo_that_will_not_release_is_an_error() {
    // `relax` that silently did not happen is worse than one that failed: the robot is
    // energised, the operator believes it is not, and the next thing that happens is a hand in a
    // joint.
    let (mut io, bus) = bus();
    bus.with(|s| s.stuck_enabled = Some(crate::model::JOINT_IDS[7]));
    io.set_torque(true).unwrap();

    let err = io.set_torque(false).unwrap_err().to_string();
    assert!(err.contains("still energised"), "got: {err}");
    assert!(
        err.contains(JOINT_NAMES[7]),
        "the refusal must name the joint: {err}"
    );
}

#[test]
fn reboot_invalidates_the_profile_so_the_next_enable_rewrites_it() {
    // Measured on this hardware: `0x08` brings the EEPROM gains back over the RAM ones. A robot
    // that reboots a servo and then energises without rewriting is walking on 32/40 while its
    // policy was trained against 6/20 — the mismatch this invalidation exists to make impossible.
    let (mut io, bus) = bus();
    io.set_gain(200).unwrap();
    io.set_torque(true).unwrap();
    let before = bus.frame_count();

    io.reboot(crate::model::JOINT_IDS[2]).unwrap();
    io.set_torque(true).unwrap();
    let log = bus.frames_after(before);

    assert!(
        log.iter()
            .any(|f| f[4] == wire::inst::SYNC_WRITE && address(f) == reg::KP),
        "the enable after a reboot must rewrite the volatile P/D"
    );
    // And the profile it wrote is the profile, not the EEPROM values the reboot restored.
    let write = log
        .iter()
        .find(|f| f[4] == wire::inst::SYNC_WRITE && address(f) == reg::KP)
        .unwrap();
    assert_eq!(write[8], 6, "Kp for joint 0 comes from the profile");
    assert_eq!(write[9], 20, "Kd for joint 0 comes from the profile");
}

#[test]
fn a_stray_ack_left_by_a_reboot_does_not_corrupt_the_next_read() {
    // `reboot` writes torque off and then the reboot instruction, which is never answered — so the
    // torque write's own ack is still in the port when the next tick reads. It has the same id as
    // the block the tick wants, and a shorter frame. A reader that trusted the length byte alone
    // would either wait for bytes that never come or splice this ack into a servo's block, which
    // shows up as one joint at a plausible, wrong angle.
    let (mut io, bus) = bus();
    let victim = crate::model::JOINT_IDS[4];
    io.reboot(victim).unwrap();
    assert!(
        bus.get(|s| !s.out.is_empty()),
        "the fixture must actually leave an unread ack, or this test proves nothing"
    );

    let sensors = io.read().unwrap();
    // The joint the stray ack named still reads its own position, not a byte of that ack.
    let expected = 0.0;
    assert!(
        (sensors.positions[4] - expected).abs() < 1e-9,
        "the stray ack leaked into the block: got {}",
        sensors.positions[4]
    );
    assert_eq!(
        io.health().missing_replies,
        0,
        "and it was not mistaken for a missing servo"
    );
}

#[test]
fn slow_sensors_rewrites_a_drifted_profile_and_says_so() {
    // One servo resetting itself out from under the other fourteen is invisible in every
    // observation the policy sees, and is the case a once-per-second check exists for.
    let (mut io, bus) = bus();
    io.set_gain(200).unwrap();
    assert_eq!(io.health().gain_repairs, 0);

    // A reboot is the protocol path that produces this on real hardware.
    io.reboot(crate::model::JOINT_IDS[5]).unwrap();

    let slow = io.slow_sensors().unwrap();
    assert_eq!(io.health().gain_repairs, 1, "the drift should be repaired");
    assert!(
        (slow.volts - 7.4).abs() < 1e-9,
        "volts came from the same read"
    );
    assert_eq!(slow.temps_c, [30.0; NUM_JOINTS]);

    // A second read finds nothing left to repair, so the repair is not a per-second bus write.
    io.slow_sensors().unwrap();
    assert_eq!(io.health().gain_repairs, 1);
    let _ = bus;
}

#[test]
fn an_unchanged_gain_costs_no_bus_traffic() {
    // `Safety::apply` passes the gain on *every* tick. If an unchanged gain rewrote the profile,
    // that is a RAM write plus a readback per tick — a fifth of the 20 ms period spent restating
    // what is already true, on a bus whose tick already costs ≈4.9 ms.
    let (mut io, bus) = bus();
    io.set_gain(200).unwrap();
    io.set_torque(true).unwrap();
    let before = bus.frame_count();

    for _ in 0..50 {
        io.set_gain(200).unwrap();
    }
    assert_eq!(
        bus.frame_count(),
        before,
        "an unchanged gain touched the bus"
    );
}

#[test]
fn a_gain_change_under_load_is_applied_in_both_directions() {
    // The control loop tapers the gain while the robot is energised — the standing policy runs
    // softer than the walking one, limp-fall softer still — so both directions have to work.
    // Refusing the stiffening would leave the robot stuck at whatever it last went limp to.
    let (mut io, bus) = bus();
    io.set_gain(200).unwrap();
    io.set_torque(true).unwrap();
    let before = bus.frame_count();

    io.set_gain(50).unwrap();
    assert_eq!(io.calibration().pd.registers(0, 50)[0], 2, "limp is soft");

    io.set_gain(200).unwrap();
    let log = bus.frames_after(before);
    let writes: Vec<&Vec<u8>> = log
        .iter()
        .filter(|f| f[4] == wire::inst::SYNC_WRITE && address(f) == reg::KP)
        .collect();
    assert_eq!(
        writes.len(),
        2,
        "one write per changed gain, and none for an unchanged one"
    );
    assert_eq!(writes[0][8], 2, "the limp gain landed");
    assert_eq!(writes[1][8], 6, "and back to the running stiffness");
}

#[test]
fn replies_out_of_order_still_land_on_the_right_joint() {
    // The protocol has devices answer in their listed slots at ≈295 µs each, and devices do miss
    // them, so arrival order is a coincidence. Trusting it silently swaps two joints' angles —
    // and because the neighbours are on the same leg, the robot reads as merely badly tuned.
    let (mut io, bus) = bus();
    bus.with(|s| s.reverse = true);
    for (joint, id) in crate::model::JOINT_IDS.iter().enumerate() {
        let slot = bus.slot_of(*id);
        let ticks = (2048 + 100 * joint as i32) as u16;
        bus.with(|s| {
            s.regs[slot][NODE_BASE + block::POSITION] = ticks as u8;
            s.regs[slot][NODE_BASE + block::POSITION + 1] = (ticks >> 8) as u8;
        });
    }

    let sensors = io.read().unwrap();
    for joint in 0..NUM_JOINTS {
        let expected = 100.0 * joint as f64 * wire::RAD_PER_COUNT;
        assert!(
            (sensors.positions[joint] - expected).abs() < 1e-9,
            "joint {joint} got {} not {expected}",
            sensors.positions[joint]
        );
    }
}

#[test]
fn a_silent_servo_is_named_and_the_rest_are_still_read() {
    // A missing device costs its own slot, not the transaction. Failing the whole tick instead
    // would turn one bad servo into a robot that cannot see any of its joints.
    let (mut io, bus) = bus();
    let dead = crate::model::JOINT_IDS[9];
    bus.with(|s| {
        s.silent.insert(dead);
    });

    let err = io.read().unwrap_err().to_string();
    assert!(
        err.contains(&dead.to_string()),
        "the error must name the id: {err}"
    );
    assert!(
        err.contains(JOINT_NAMES[9]),
        "and the joint, so a person can find it: {err}"
    );
    assert_eq!(io.health().missing_replies, 1);
}

#[test]
fn an_alarm_is_reported_without_failing_the_tick() {
    // One joint overheating must not blind the loop to the other fourteen, but a masked fault is
    // how a burnt servo is discovered late. Both halves are asserted.
    let (mut io, bus) = bus();
    bus.with(|s| {
        s.alarms.insert(crate::model::JOINT_IDS[1], 0x04);
    });

    let sensors = io.read().expect("an alarm is not a bus failure");
    assert_eq!(sensors.positions.len(), NUM_JOINTS);
    assert_eq!(io.health().servo_alarms, 1);
}

#[test]
fn currents_are_a_magnitude_in_milliamps() {
    // The offset and the scale are both easy to be wrong about, and a wrong current is invisible
    // until something is about to overheat.
    let (mut io, bus) = bus();
    let slot = bus.slot_of(crate::model::JOINT_IDS[0]);
    // A load word of 0x8004 is four counts in the negative direction, which is 26 mA of load.
    bus.with(|s| {
        s.regs[slot][NODE_BASE + block::CURRENT] = 0x04;
        s.regs[slot][NODE_BASE + block::CURRENT + 1] = 0x80;
    });

    let sensors = io.read().unwrap();
    assert!((sensors.currents_ma[0] - 4.0 * wire::MA_PER_CURRENT_COUNT).abs() < 1e-9);
    assert!(sensors.currents_ma[0] > 0.0, "load is a magnitude");
}

#[test]
fn a_repeated_sample_counter_marks_the_imu_stale() {
    // The node runs at 120 Hz and the loop at 50, so two ticks inside one node refresh is
    // ordinary — the counter distinguishes that from a node that has stopped. A byte comparison
    // cannot, which is why the Dynamixel backend can only count the weaker signal.
    let (mut io, bus) = bus();
    bus.set_imu_counter(1);
    io.read().unwrap();
    assert_eq!(io.imu_stale().run, 0);
    bus.set_imu_counter(2);
    io.read().unwrap();
    assert_eq!(
        io.imu_stale().run,
        0,
        "an advancing counter is a fresh sample"
    );

    // The node stops producing while still answering: the counter repeats.
    bus.set_imu_counter(5);
    io.read().unwrap();
    io.read().unwrap();
    io.read().unwrap();
    assert!(
        io.imu_stale().run >= 2,
        "a repeated counter is a stale read, got {}",
        io.imu_stale().run
    );
}

#[test]
fn imu_is_not_ready_until_the_node_says_sflp_is_valid() {
    // Two independent facts: the decoder has seen enough non-zero quaternions to have a value at
    // all, and the chip's own fusion is running. Either alone reports an orientation that is
    // unknown.
    let (mut io, bus) = bus();
    bus.set_imu_status(0);
    for _ in 0..(STALE_RUN_WARN + 30) {
        io.read().unwrap();
    }
    assert!(
        !io.imu_ready(),
        "without the node's SFLP_VALID bit the orientation is a default, not a measurement"
    );

    bus.set_imu_status(wire::IMU_FLAG_SFLP_VALID);
    io.read().unwrap();
    assert!(io.imu_ready());
}

#[test]
fn position_mapping_is_its_own_inverse_and_clamps_to_the_servo_window() {
    // Calibration is one measurement because the map inverts. If it did not, the number stored
    // would be a pair that can disagree, and the robot would drift a little on every enable.
    let mut calibration = Calibration::default();
    calibration.joints[0].zero_ticks = 2236;
    calibration.joints[0].direction = -1.0;
    calibration.validate().unwrap();
    let (io, _) = bus_with(calibration.clone());

    // A narrowed window first, so the clamp below is exercised on a real limit rather than on
    // 0..4095, which nothing inside a joint's travel would ever reach.
    for rad in [-0.4, -0.1, 0.0, 0.1, 0.4] {
        let ticks = io.ticks_for(0, rad);
        let back = io.radians(0, ticks);
        assert!((back - rad).abs() <= wire::RAD_PER_COUNT, "{rad} -> {back}");
    }

    let mut narrow = calibration.clone();
    narrow.joints[0].limits = (2200, 2300);
    narrow.validate().unwrap();
    let (io, _) = bus_with(narrow);
    // Zero is inside the window and must survive; a long way past it must not, or a bad
    // calibration (or a wound encoder) drives the joint into its housing.
    assert_eq!(
        wire::position_raw_to_counts(io.ticks_for(0, 0.0)),
        2236,
        "zero must map to the calibrated tick"
    );
    // Which edge a big angle hits depends on the direction sign: this joint is `-1`, so a
    // positive angle moves towards lower ticks. Getting that backwards is the same mistake that
    // makes a joint read forward on the way out and backward on the way back.
    for (rad, edge) in [(10.0, 2200), (-10.0, 2300)] {
        assert_eq!(
            wire::position_raw_to_counts(io.ticks_for(0, rad)),
            edge,
            "{rad} rad should clamp to the servo's window edge"
        );
    }
}

#[test]
fn the_tick_is_one_transaction_for_the_node_and_every_servo() {
    // The whole reason the node's FeeTech block was made fifteen bytes wide: one request per
    // tick, not two. A second transaction per tick is a fifth of a 20 ms period.
    let (mut io, bus) = bus();
    let before = bus.frame_count();
    io.read().unwrap();
    let log = bus.frames_after(before);

    assert_eq!(log.len(), 1, "one request, one write");
    assert_eq!(log[0][4], wire::inst::SYNC_READ);
    assert_eq!(log[0][5], reg::PRESENT_POSITION_L);
    assert_eq!(log[0][6], block::LEN as u8);
    // The node is listed first, so its reply is not queued behind fifteen servo slots.
    assert_eq!(log[0][7], 200);
}

#[test]
fn volts_and_temperature_come_from_the_gain_transaction() {
    // One extra transaction per second carries gains, voltage and temperature, because 50..63 is
    // contiguous. Reading them separately would be a second transaction for the same numbers.
    let (mut io, bus) = bus();
    io.set_gain(200).unwrap();
    let before = bus.frame_count();
    let slow = io.slow_sensors().unwrap();
    let log = bus.frames_after(before);

    assert_eq!(log.len(), 1, "slow sensors must be one transaction");
    assert_eq!(address(&log[0]), reg::KP);
    assert!((slow.volts - 7.4).abs() < 1e-9);
    assert_eq!(slow.temps_c, [30.0; NUM_JOINTS]);
}

#[test]
fn persisting_gains_unlocks_writes_reads_back_and_relocks() {
    // Four steps that are load-bearing together, and the third is the one that is easy to skip:
    // a *locked* servo accepts a write into RAM and silently does not persist it, so without the
    // readback a call that did nothing at all reports success — and the robot comes back at the
    // factory stiffness after its next power cycle with nothing to point at.
    let (mut io, bus) = bus();
    io.set_gain(200).unwrap();
    let before = bus.frame_count();
    let backup = io.persist_gains().unwrap();

    assert_eq!(
        backup.kp, [32; NUM_JOINTS],
        "the factory gains, read before writing"
    );
    assert_eq!(backup.kd, [40; NUM_JOINTS]);

    let log = bus.frames_after(before);
    let writes: Vec<(u8, u8)> = log
        .iter()
        .filter(|f| f[4] == wire::inst::SYNC_WRITE)
        .map(|f| (address(f), f[8]))
        .collect();
    assert!(
        writes.contains(&(reg::LOCK, 0)),
        "the lock has to be cleared before an EEPROM write means anything: {writes:?}"
    );
    assert!(
        writes.contains(&(reg::EEPROM_KP, 6)),
        "the profile is what gets written"
    );
    assert!(
        writes
            .iter()
            .rposition(|(addr, _)| *addr == reg::LOCK)
            .unwrap()
            > writes
                .iter()
                .position(|(addr, _)| *addr == reg::EEPROM_KP)
                .unwrap(),
        "and the lock goes back on afterwards: {writes:?}"
    );

    // What a later reader sees is the profile, not the factory pair — including the mouth, which
    // is the one joint with a P of its own because it is a lighter mechanism.
    let now = io.eeprom_gains().unwrap();
    let want_kp: [u8; NUM_JOINTS] =
        std::array::from_fn(|joint| if joint == MOUTH_INDEX { 10 } else { 6 });
    assert_eq!(now.kp, want_kp);
    assert_eq!(now.kd, [20; NUM_JOINTS]);
    assert_eq!(io.calibration().pd.kp, 6);
}

#[test]
fn persisting_gains_is_refused_while_the_servos_are_energised() {
    // It is a flash write, and the servo is not being asked to hold a pose through it.
    let (mut io, _) = bus();
    io.set_gain(200).unwrap();
    io.set_torque(true).unwrap();
    let err = io.persist_gains().unwrap_err().to_string();
    assert!(err.contains("energised"), "{err}");
}

#[test]
fn a_persist_that_does_not_read_back_says_so_and_still_relocks() {
    // The servo may be a different firmware. Reporting success on an unverified flash write is how
    // a robot ends up with a stiffness nobody can account for, so this fails loudly — and it must
    // still put the lock back, because a servo left permanently writable is the worse state.
    let (mut io, bus) = bus();
    io.set_gain(200).unwrap();
    // Make the EEPROM registers refuse to take the value.
    bus.with(|s| s.stick_eeprom = true);
    let err = io.persist_gains().unwrap_err().to_string();
    assert!(err.contains("did not read back"), "{err}");
    let locks: Vec<u8> = bus
        .log()
        .iter()
        .filter(|f| f[4] == wire::inst::SYNC_WRITE && address(f) == reg::LOCK)
        .map(|f| f[8])
        .collect();
    assert_eq!(
        locks.last(),
        Some(&1),
        "the lock is restored even on the failure path, got {locks:?}"
    );
}

#[test]
fn surveying_the_bus_writes_nothing_at_all() {
    // Commissioning happens with a robot that may be held up by hand, and the one form of a
    // commissioning tool that is safe around a person is the one that *cannot* write. Asserted on
    // the wire rather than trusted to a caller: every frame the survey sent must be a read.
    let (mut io, bus) = bus();
    let slot = bus.slot_of(crate::model::JOINT_IDS[0]);
    bus.with(|s| {
        s.regs[slot][reg::MODEL_L as usize] = 0x0A;
        s.regs[slot][reg::MODEL_H as usize] = 0x1F; // the 0x1f0a this servo reports
        s.regs[slot][0] = 3; // firmware 3.46
        s.regs[slot][1] = 46;
        s.regs[slot][reg::MODE as usize] = 4;
        s.regs[slot][reg::BAUD_RATE as usize] = 0;
        s.regs[slot][reg::LOCK as usize] = 1;
    });

    let survey = io.survey().unwrap();

    assert!(
        bus.log().iter().all(|f| f[4] == wire::inst::SYNC_READ),
        "a survey sent something other than a read: {:?}",
        bus.log()
            .iter()
            .map(|f| format!("{:#04x}@{:#04x}", f[4], f[5]))
            .collect::<Vec<_>>()
    );
    let first = survey.servos[0];
    assert_eq!(first.model, 0x1F0A);
    assert_eq!(first.firmware, (3, 46));
    assert_eq!(
        first.mode, 4,
        "a servo in the wrong mode is a servo that will not follow"
    );
    assert_eq!(first.baud_code, 0, "code 0 is 1 Mbps");
    assert_eq!(
        first.lock, 1,
        "a locked servo writes to RAM and silently does not persist"
    );
    assert_eq!(survey.raw_ticks[0], 2048);
    assert_eq!(
        survey.kp[0], 32,
        "the factory gains, so a profile that has not landed shows"
    );
    assert!((survey.volts[0] - 7.4).abs() < 1e-9);
    assert_eq!(survey.imu_status, wire::IMU_FLAG_SFLP_VALID);
}

#[test]
fn the_servos_own_travel_window_is_read_and_then_honoured() {
    // The clamp is the last thing between a bad calibration and a joint in its housing, and the
    // servo knows its own window. A daemon that never asks is a daemon clamping to a guess.
    let (mut io, bus) = bus();
    let slot = bus.slot_of(crate::model::JOINT_IDS[0]);
    bus.with(|s| {
        // A narrow window *around* the calibrated zero, which is the realistic shape: someone
        // configured this servo's travel. 2000 = 0x07d0, 2100 = 0x0834.
        s.regs[slot][reg::MIN_ANGLE_LIMIT_L as usize] = 0xD0;
        s.regs[slot][reg::MIN_ANGLE_LIMIT_L as usize + 1] = 0x07;
        s.regs[slot][reg::MAX_ANGLE_LIMIT_L as usize] = 0x34;
        s.regs[slot][reg::MAX_ANGLE_LIMIT_L as usize + 1] = 0x08;
    });

    io.read_travel_limits().unwrap();
    assert_eq!(io.calibration().joints[0].limits, (2000, 2100));
    // Past the window, it stops — in the direction the joint's sign says that way is.
    assert_eq!(wire::position_raw_to_counts(io.ticks_for(0, 10.0)), 2100);
    assert_eq!(wire::position_raw_to_counts(io.ticks_for(0, -10.0)), 2000);
}

#[test]
fn a_calibrated_zero_outside_the_servos_own_window_stops_the_bus_opening() {
    // Found while reading the clamp back: because `ticks_for` clamps into the window, a zero the
    // window does not contain makes the joint's own zero unreachable. Every command lands on the
    // edge, the robot stands permanently offset by however far out the number was, and nothing in
    // the observation explains it. Refused while the numbers are still in hand.
    let mut calibration = Calibration::default();
    calibration.joints[0].zero_ticks = 2048;
    let (mut io, bus) = bus_with(calibration);
    let slot = bus.slot_of(crate::model::JOINT_IDS[0]);
    bus.with(|s| {
        s.regs[slot][reg::MIN_ANGLE_LIMIT_L as usize] = 0x00;
        s.regs[slot][reg::MIN_ANGLE_LIMIT_L as usize + 1] = 0x0E; // 3584
        s.regs[slot][reg::MAX_ANGLE_LIMIT_L as usize] = 0xFF;
        s.regs[slot][reg::MAX_ANGLE_LIMIT_L as usize + 1] = 0x0F; // 4095
    });

    let err = io.read_travel_limits().unwrap_err().to_string();
    assert!(err.contains(JOINT_NAMES[0]), "{err}");
    assert!(err.contains("2048") && err.contains("3584"), "{err}");
}

#[test]
fn a_servo_whose_own_limits_make_no_sense_stops_the_bus_opening() {
    // 0..0 is a servo that cannot move anywhere. Opening the bus on it would mean every command
    // silently becoming one position.
    let (mut io, bus) = bus();
    let slot = bus.slot_of(crate::model::JOINT_IDS[2]);
    bus.with(|s| {
        s.regs[slot][reg::MAX_ANGLE_LIMIT_L as usize] = 0;
        s.regs[slot][reg::MAX_ANGLE_LIMIT_L as usize + 1] = 0;
    });
    let err = io.read_travel_limits().unwrap_err().to_string();
    assert!(err.contains(JOINT_NAMES[2]), "{err}");
}

#[test]
fn a_calibration_that_cannot_describe_a_robot_is_refused() {
    // A direction of 0 turns two joints into mirrors that happen to agree; a zero outside the
    // single turn is another joint's zero. Both are caught here rather than as a robot that
    // walks sideways.
    let mut calibration = Calibration::default();
    calibration.joints[4].direction = 0.0;
    assert!(calibration.validate().is_err());

    let mut calibration = Calibration::default();
    calibration.joints[4].zero_ticks = 5000;
    assert!(calibration.validate().is_err());

    let mut calibration = Calibration::default();
    calibration.joints[4].limits = (3000, 2000);
    assert!(calibration.validate().is_err());
}

#[test]
fn the_gain_scale_keeps_a_limp_gain_limp() {
    // Passing the policy's 200 through to a FeeTech servo would ask for a P twenty times the
    // factory value. Scaling keeps the meaning: bigger is stiffer, and 50 is genuinely soft.
    let pd = PdProfile::default();
    assert_eq!(pd.registers(0, 200), [6, 20]);
    assert_eq!(pd.registers(MOUTH_INDEX, 200), [10, 20]);
    assert_eq!(pd.registers(0, 50), [2, 20]);
    assert_eq!(pd.registers(0, 0), [1, 20], "a P of zero is an open loop");
    // Monotone in the policy gain, which is what makes a gain change mean "stiffer" at all.
    assert!(pd.registers(0, 200)[0] > pd.registers(0, 150)[0]);
    assert!(pd.registers(0, 150)[0] > pd.registers(0, 50)[0]);
}
