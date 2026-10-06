//! The FeeTech SCS/HLS bus: the 1910 servos and the `imu_to_dxl` node on one UART.
//!
//! [`wire`] is the bytes; this module is the port, the clock and the [`RobotIo`] the control
//! loop drives.
//!
//! # Why this exists beside [`crate::bus`]
//!
//! [`crate::bus`] speaks Dynamixel protocol 2.0 to XL330s. This robot has HD-1910-C001 servos
//! instead, which speak the SCS/HLS family protocol: a different header, a one's-complement
//! checksum rather than a CRC, per-device status packets rather than one appended to a
//! broadcast, and no `0x8A` fast sync read. It is not a register-table difference — the
//! framing is different, so it is a different backend rather than a different table.
//!
//! What is *not* different is the seam: this implements [`RobotIo`] unchanged, so the control
//! loop, policy, safety and observations above it do not know which bus they are on.
//!
//! # Why the P/D profile is a runtime concern
//!
//! A servo's position gain lives in two places. Registers 21/22 are EEPROM and survive power
//! loss; 50/51 are RAM and are **reset from 21/22 by a reboot** — measured on this hardware:
//! writing 50/51 = 99/88 and issuing `0x08` brings 32/40 back, which is what 21/22 holds.
//!
//! So a robot whose P/D matters has to write them *every* bring-up, and has to notice when a
//! single servo has reset itself out from under the other fourteen. That is what
//! [`ScsIo::ensure_gains`] and the check in [`RobotIo::slow_sensors`] are for. Writing EEPROM
//! is deliberately not automatic: a wrong persistent gain is a robot that lurches on every
//! future boot, and EEPROM writes are the one thing here that can brick a servo.
//!
//! # The two bus budgets
//!
//! One tick is one `sync_read` of sixteen devices at address 56 and length 15 — measured at
//! ≈4.9 ms of wire time, with ≈295 µs per device slot because each answers separately. A
//! device that is *absent* still consumes its slot, so a missing servo costs a slot, not a
//! timeout. The burst is therefore ended by silence ([`BURST_IDLE`]) rather than by waiting
//! out a serial timeout, and capped ([`BURST_LIMIT`]) so a dead bus costs 12 ms rather than
//! 30 — most of a 20 ms period.

pub mod wire;

#[cfg(test)]
mod tests;

use std::io::Write as _;
use std::time::{Duration, Instant};

use crate::imu::{IMU_BLOCK_LEN, SflpDecoder};
use crate::io::{ImuStale, IoError, JointTargets, Result, RobotIo, Sensors, SlowSensors};
use crate::model::{JOINT_NAMES, MOUTH_INDEX, NUM_JOINTS};
use wire::{ExpectedReply, FrameReader, block, gains_block, imu_block, reg};

/// The bus rate every device is configured for. Not a preference: the servos' baud code and the
/// node's are both set to 1 Mbps on this robot, and a mismatch is silence, not a slower bus.
pub const BAUD_RATE: u32 = 1_000_000;

/// Silence that ends a reply burst.
///
/// Measured on this hardware: the longest gap between two adjacent servo replies is 1.94 ms
/// over 135,000 samples, so 2 ms leaves about 3% margin and is one scheduler hiccup away from
/// truncating a burst — which shows up as the *last* few ids reported missing, the failure this
/// constant exists to prevent.
pub const BURST_IDLE: Duration = Duration::from_millis(4);

/// Hard ceiling on one burst, so a bus that is present but dead costs 12 ms rather than the
/// serial timeout's 30 — which is most of a 20 ms control period, and enough to make the loop
/// miss ticks rather than merely read badly.
pub const BURST_LIMIT: Duration = Duration::from_millis(12);

/// Register read that covers the next tick's block. Sixteen devices at 15 bytes.
const TICK_DATA_LEN: usize = block::LEN;

/// Where a joint's mechanical zero sits, and which way it turns.
///
/// The map is its own inverse — `q = 0` reads as `zero_ticks` and commands `zero_ticks` — which
/// is what makes calibration a single measurement rather than a pair that can disagree:
/// command the joint to its zero, read the position back, store that number. It also means the
/// table is invalidated by a change to the servo's own offset register (31), because that moves
/// the raw encoder window the number was measured in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointCalibration {
    pub zero_ticks: i32,
    /// `+1` or `-1`: the sign that takes a raw count delta into the joint's own positive
    /// direction. Config carries it as an integer because a value between the two is a
    /// typo, not a gain.
    pub direction: f64,
    /// Raw counts the servo itself will accept, read from registers 9..12 at open. Commands are
    /// clamped into this, so a calibration mistake cannot drive a joint into its housing.
    pub limits: (i32, i32),
}

impl Default for JointCalibration {
    fn default() -> Self {
        Self {
            zero_ticks: 2048,
            direction: 1.0,
            limits: (0, 4095),
        }
    }
}

/// Volatile position P/D written on every bring-up.
///
/// The values are **not** measured on this robot. 6/20 is where two independent forks of this
/// codebase that both walk landed (`kp_fw` 5–6 in their training, P=6 D=20 in the servo's RAM),
/// against a factory 32/40 and a BAM actuator default of 32. That is evidence, not proof; the
/// matching simulator constant is `kp_fw`, and the two must be changed together or the policy is
/// trained against a stiffness the robot does not have.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PdProfile {
    pub kp: u8,
    pub kd: u8,
    /// The mouth is a lighter mechanism than a leg and both forks give it its own P.
    pub mouth_kp: u8,
}

impl Default for PdProfile {
    fn default() -> Self {
        Self {
            kp: 6,
            kd: 20,
            mouth_kp: 10,
        }
    }
}

impl PdProfile {
    /// The register pair for one joint at a policy gain, in the policy's own units.
    ///
    /// The policy's `gain` is an XL330 figure — 200 nominal, 50 limp — and passing it through
    /// unchanged would ask a FeeTech servo for a P twenty times its factory value. Scaling it
    /// onto the profile keeps the *meaning* of a gain change (bigger is stiffer, and limp is
    /// genuinely soft) without pretending the two families share a unit.
    pub fn registers(&self, joint: usize, policy_gain: u16) -> [u8; 2] {
        let nominal = u32::from(self.kp.max(1));
        let mouth = u32::from(self.mouth_kp.max(1));
        let base = if joint == MOUTH_INDEX { mouth } else { nominal };
        // Rounded to nearest, floored at 1: a P of 0 is not a soft joint, it is an open loop.
        let scaled = (base * u32::from(policy_gain) + 100) / 200;
        [scaled.clamp(1, 255) as u8, self.kd]
    }

    /// Whether `to` would make any joint stiffer than `from`, which is the one change that must
    /// not happen under load. Compared on the *scaled* value rather than on the policy gain, so
    /// it stays honest if the profile ever gains a joint whose gain does not scale.
    pub fn stiffens(&self, from: u16, to: u16) -> bool {
        (0..NUM_JOINTS).any(|joint| self.registers(joint, to)[0] > self.registers(joint, from)[0])
    }
}

/// Everything about this particular robot that the servos cannot be asked.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    /// Bus id of the `imu_to_dxl` node. 200 by default, which is what the node ships as and what
    /// the Dynamixel side already calls it.
    pub imu_id: u8,
    pub joints: [JointCalibration; NUM_JOINTS],
    pub pd: PdProfile,
    /// Sensor→trunk mounting rotation for the node's quaternion, scalar-first.
    pub imu_mount: [f64; 4],
    pub baud_rate: u32,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            imu_id: 200,
            joints: [JointCalibration::default(); NUM_JOINTS],
            pd: PdProfile::default(),
            imu_mount: SflpDecoder::DEFAULT_MOUNT,
            baud_rate: BAUD_RATE,
        }
    }
}

impl Calibration {
    /// Reject a table that cannot describe a robot, rather than discovering it mid-stride.
    ///
    /// Checked here as well as in the config schema because this type is also built by tests and
    /// by `robotctl`, and a direction of 0 turns two joints into mirrors that happen to agree.
    pub fn validate(&self) -> std::result::Result<(), String> {
        for (joint, cal) in self.joints.iter().enumerate() {
            if !matches!(cal.direction, -1.0 | 1.0) {
                return Err(format!(
                    "{}: direction is {}, and only +1 or -1 is a robot",
                    JOINT_NAMES[joint], cal.direction
                ));
            }
            if !(0..4096).contains(&cal.zero_ticks) {
                return Err(format!(
                    "{}: zero_ticks is {}, outside the single turn 0..4095",
                    JOINT_NAMES[joint], cal.zero_ticks
                ));
            }
            if cal.limits.0 >= cal.limits.1 || cal.limits.1 > 4095 {
                return Err(format!(
                    "{}: travel limits {:?} are not a window inside 0..4095",
                    JOINT_NAMES[joint], cal.limits
                ));
            }
        }
        Ok(())
    }
}

/// What a backend needs from a serial port, so that everything above it is testable without one.
pub trait Transport: Send {
    fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()>;
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize>;
    fn set_timeout(&mut self, timeout: Duration) -> std::io::Result<()>;
}

struct SerialTransport(Box<dyn serialport::SerialPort>);

impl Transport for SerialTransport {
    fn write_all(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        // Deliberately no `flush`/`tcdrain`. On this board that call costs a fixed ~12 ms per
        // transaction regardless of frame length, which is two thirds of a 20 ms period; with
        // it the loop measured 36.3 Hz and 48% missed ticks, without it 50.0 Hz and none.
        // `write_all` has already handed the bytes to the driver.
        self.0.write_all(bytes)
    }

    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        std::io::Read::read(&mut self.0, buf)
    }

    fn set_timeout(&mut self, timeout: Duration) -> std::io::Result<()> {
        self.0.set_timeout(timeout).map_err(std::io::Error::other)
    }
}

/// IMU staleness, from the node's own sample counter where it exists.
///
/// [`crate::bus`] can only compare bytes, because its 12-byte read at address 124 stops eight
/// bytes short of the counter at offset 18 of the node's 20-byte diagnostic block. Reading 15
/// bytes at address 56 puts the counter at offset 12, so this backend asks the better question:
/// *did the node produce a new frame*, rather than *are these bytes the same*. Two consecutive
/// reads landing inside one 120 Hz node refresh is ordinary and the counter says so; identical
/// bytes are then not evidence of anything.
#[derive(Debug, Default)]
struct StaleTracker {
    last_block: Option<[u8; IMU_BLOCK_LEN]>,
    last_count: Option<u8>,
    stale: ImuStale,
}

impl StaleTracker {
    /// Records one block and returns the run length it belongs to, 0 when the block is fresh.
    fn observe(&mut self, block: &[u8; IMU_BLOCK_LEN], count: Option<u8>) -> u64 {
        let repeated = match (count, self.last_count) {
            (Some(now), Some(before)) => now == before,
            // No counter (an older node, or the DXL personality): fall back to comparing bytes,
            // which is what [`crate::bus`] does and is a weaker signal for exactly that reason.
            _ => self.last_block == Some(*block),
        };
        if repeated && self.last_block.is_some() {
            self.stale.total = self.stale.total.saturating_add(1);
            self.stale.run = self.stale.run.saturating_add(1);
        } else {
            self.stale.run = 0;
        }
        self.last_block = Some(*block);
        self.last_count = count;
        self.stale.run
    }
}

/// Every joint's EEPROM P/D, as it was before somebody changed it.
///
/// Returned by [`ScsIo::persist_gains`] so the values that were there can be written down: a
/// persistent gain is the one setting on this bus that outlives the robot's own configuration, and
/// "we changed it and did not record what it was" is not recoverable from the robot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GainBackup {
    pub kp: [u8; NUM_JOINTS],
    pub kd: [u8; NUM_JOINTS],
}

/// What one servo says about itself.
///
/// Read once, at commissioning, and never on the tick: a robot's firmware revision is not a thing
/// that changes while it walks, and this is four registers the control loop has no use for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServoIdentity {
    pub id: u8,
    pub model: u16,
    /// Major and minor, as the servo reports them. Not the release version — see the node's own
    /// identity window for that.
    pub firmware: (u8, u8),
    /// 4 is the pure position/PD mode this robot runs.
    pub mode: u8,
    pub baud_code: u8,
    /// Register 55. Non-zero means a write lands in RAM but is *not* persisted, which is a
    /// silent difference rather than an error — the reason this is read out loud.
    pub lock: u8,
    pub limits: (i32, i32),
}

/// A read-only survey of the bus, for commissioning a robot rather than driving one.
///
/// Everything here is something a person needs *before* there is a calibration to open with: the
/// raw encoder count each joint reads right now, so a mechanical zero can be measured and written
/// down; the firmware and mode each servo is running, so a wrong one is visible before the robot
/// moves; and the gains, so the profile can be seen to have landed rather than believed to have.
#[derive(Debug, Clone, PartialEq)]
pub struct Survey {
    pub servos: [ServoIdentity; NUM_JOINTS],
    /// Raw encoder counts, uncalibrated — this is the number that becomes `zero_ticks`.
    pub raw_ticks: [u16; NUM_JOINTS],
    pub kp: [u8; NUM_JOINTS],
    pub kd: [u8; NUM_JOINTS],
    pub volts: [f64; NUM_JOINTS],
    pub temps_c: [f64; NUM_JOINTS],
    /// Torque-enable as the servo reports it (register 40). Here because it is the one question a
    /// read-only survey can answer that nothing else can: `robot.relax` says "torque off", and the
    /// only way to see whether the *servo* agrees is to read the register. Free with the identity
    /// block — register 40 is one byte past `mode`, so this costs one byte and no transaction.
    pub torque: [u8; NUM_JOINTS],
    /// The same register read *the way the driver reads it for its own verification* — a
    /// one-byte `sync_read` at address 40 rather than a byte sliced out of the long block.
    ///
    /// Two ways of asking one question, kept side by side because they disagreed on hardware: the
    /// long read said torque was **on** while the driver's own read-back said **off**, and the
    /// driver believed itself and reported a successful relax. A survey tool that can only see one
    /// of them cannot show which question the servo answered differently.
    pub torque_direct: [u8; NUM_JOINTS],
    /// The node's status byte and sample counter, so an unbuilt or unresponsive node is visible
    /// here rather than as an orientation that never becomes ready.
    pub imu_status: u8,
    pub imu_counter: u8,
}

/// Counters a health report can quote, and a person can act on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScsHealth {
    /// Bursts that ended with one or more asked ids silent.
    pub missing_replies: u64,
    /// Bytes dropped while resynchronising, summed over every transaction.
    pub discarded_bytes: u64,
    /// Servos that answered with a non-zero status byte, by occurrence rather than by joint.
    pub servo_alarms: u64,
    /// Times the volatile P/D had drifted and was rewritten.
    pub gain_repairs: u64,
}

pub struct ScsIo {
    transport: Box<dyn Transport>,
    calibration: Calibration,
    /// The node first, then the joints in [`JOINT_NAMES`] order. Slot order is also answer order,
    /// so putting the node first is what keeps its reply off the end of the servo burst.
    ids: Vec<u8>,
    decoder: SflpDecoder,
    stale: StaleTracker,
    health: ScsHealth,
    /// The policy gain last written, in the policy's units.
    gain: u16,
    /// Whether every joint demonstrably holds the profile in RAM right now. Cleared by a reboot,
    /// by a detected drift, and by a failed verification — never assumed.
    gains_verified: bool,
    /// Last torque state we set, and whether every servo confirmed it.
    enabled: bool,
    /// The node's status byte from the last block: SFLP validity is a fact about the chip, not
    /// about the decoder's history, and `imu_ready` wants both.
    imu_status: u8,
    io_timeout: Duration,
}

impl ScsIo {
    pub fn open(port: &str, calibration: Calibration) -> Result<Self> {
        calibration.validate().map_err(IoError::Bus)?;
        let serial = serialport::new(port, calibration.baud_rate)
            .timeout(BURST_LIMIT)
            .open()
            .map_err(|e| IoError::Port {
                path: port.to_owned(),
                source: std::io::Error::other(e),
            })?;
        let mut io = Self::with_transport(Box::new(SerialTransport(serial)), calibration);
        io.read_travel_limits()?;
        Ok(io)
    }

    /// Read each servo's own travel window, so a command is clamped to it.
    ///
    /// One transaction at open, and worth it: the clamp in [`Self::ticks_for`] is the last thing
    /// between a wrong calibration and a joint driven into its housing, and the servo already
    /// knows its own window — the only question is whether anyone asked. The values live in
    /// EEPROM, so they survive a power cycle and are part of what this robot *is*; a servo whose
    /// window is not a window is treated as a reason not to open the bus at all.
    ///
    /// Not called by [`Self::with_transport`], so a test can state its own window.
    pub fn read_travel_limits(&mut self) -> Result<()> {
        let blocks = self.read_joint_blocks(reg::MIN_ANGLE_LIMIT_L, 4)?;
        for (joint, block) in blocks.iter().enumerate() {
            let low = wire::position_raw_to_counts(wire::le_u16(block[0], block[1]));
            let high = wire::position_raw_to_counts(wire::le_u16(block[2], block[3]));
            if low >= high || high > 4095 {
                return Err(IoError::Bus(format!(
                    "{}: firmware travel limits {low}..{high} are not a window inside one turn; \
                     refusing to drive a joint whose own limits make no sense",
                    JOINT_NAMES[joint]
                )));
            }
            // The calibration has to be reachable. `ticks_for` clamps into the window, so a zero
            // outside it means the joint can never be commanded to its own zero: every command
            // lands on the window edge, the robot stands permanently offset, and nothing in the
            // observation says why. Cheaper to refuse here, while the numbers are in hand.
            let zero = self.calibration.joints[joint].zero_ticks;
            if zero < low || zero > high {
                return Err(IoError::Bus(format!(
                    "{}: calibrated zero_ticks {zero} is outside the servo's own {low}..{high}; \
                     the calibration and this servo disagree about where the joint is",
                    JOINT_NAMES[joint]
                )));
            }
            self.calibration.joints[joint].limits = (low, high);
        }
        Ok(())
    }

    /// The whole backend over an arbitrary transport. Tests drive this; `open` drives the port.
    pub fn with_transport(transport: Box<dyn Transport>, calibration: Calibration) -> Self {
        let mut ids = Vec::with_capacity(NUM_JOINTS + 1);
        ids.push(calibration.imu_id);
        ids.extend_from_slice(&crate::model::JOINT_IDS);
        let decoder = SflpDecoder::new(calibration.imu_mount);
        Self {
            transport,
            calibration,
            ids,
            decoder,
            stale: StaleTracker::default(),
            health: ScsHealth::default(),
            // Not 200: nothing has been written yet, so any first `set_gain` is a stiffening
            // and is refused until torque is off. Starting at the nominal value would make the
            // first softening look like a no-op and skip the write.
            gain: 0,
            gains_verified: false,
            enabled: false,
            imu_status: 0,
            io_timeout: BURST_LIMIT,
        }
    }

    pub fn health(&self) -> ScsHealth {
        self.health
    }

    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    /// One request, every reply, in the order the ids were listed.
    ///
    /// Ordered by *request*, not by arrival: the protocol has the devices answer in their listed
    /// slots, but a slot is ≈295 µs and devices do miss them, so arrival order is a coincidence
    /// this must not depend on. A missing id is reported with the ones that did answer, in one
    /// error, because "which servo is silent" is the question.
    fn request(&mut self, frame: &[u8], expected: &ExpectedReply) -> Result<Vec<Vec<u8>>> {
        self.transport
            .write_all(frame)
            .map_err(|source| IoError::Bus(format!("SCS write: {source}")))?;

        // The ids of *this* request, not every device on the bus: a gains read names only the
        // servos, and demanding a reply from the node would fail a transaction the node was
        // never asked to join.
        let ids: Vec<u8> = expected.ids().to_vec();
        let mut got: Vec<Option<Vec<u8>>> = vec![None; ids.len()];
        let mut reader = FrameReader::new();
        let started = Instant::now();
        // The first read may legitimately wait for the whole slot chain, so it gets the burst
        // budget; every read after a byte arrives only has to outlast the gap between replies.
        self.transport
            .set_timeout(self.io_timeout)
            .map_err(|source| IoError::Bus(format!("SCS timeout: {source}")))?;

        let mut buf = [0u8; 512];
        loop {
            if got.iter().all(Option::is_some) {
                break;
            }
            match self.transport.read(&mut buf) {
                Ok(0) => {}
                Ok(n) => {
                    reader.push(&buf[..n]);
                    // Any byte means the burst is underway, so the wait shrinks to the gaps.
                    self.transport
                        .set_timeout(BURST_IDLE)
                        .map_err(|source| IoError::Bus(format!("SCS timeout: {source}")))?;
                    while let Some(frame) = reader.next_frame(expected) {
                        // A frame that parses is one of the ids we asked for; the reader only
                        // accepts those, so this cannot index out of range.
                        let ack = wire::parse_ack(&frame).expect("reader returns parsed frames");
                        if let Some(slot) = ids.iter().position(|id| *id == ack.id) {
                            got[slot] = Some(frame);
                        }
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => break,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(source) => return Err(IoError::Bus(format!("SCS read: {source}"))),
            }
            if started.elapsed() >= BURST_LIMIT {
                break;
            }
        }

        self.health.discarded_bytes += reader.take_discarded();

        let missing: Vec<u8> = ids
            .iter()
            .zip(&got)
            .filter(|(_, block)| block.is_none())
            .map(|(id, _)| *id)
            .collect();
        if !missing.is_empty() {
            self.health.missing_replies += 1;
            let named: Vec<String> = missing
                .iter()
                .map(|id| match duck_ipc_proto::joint_of(*id) {
                    Some(name) => format!("{id} ({name})"),
                    None => format!("{id} (imu node)"),
                })
                .collect();
            return Err(IoError::Bus(format!(
                "SCS sync_read: no reply from {}",
                named.join(", ")
            )));
        }

        let mut blocks = Vec::with_capacity(got.len());
        for frame in got.into_iter().flatten() {
            let ack = wire::parse_ack(&frame).expect("reader returns parsed frames");
            if ack.status != 0 {
                self.health.servo_alarms += 1;
                // The status byte is the servo's own alarm register made visible. It does not
                // fail the tick — one joint overheating must not blind the loop to the other
                // fourteen — but it is named, because a masked fault is how a burnt servo is
                // discovered late.
                tracing::warn!(
                    id = ack.id,
                    joint = duck_ipc_proto::joint_of(ack.id).unwrap_or("imu node"),
                    status = ack.status,
                    "SCS device reports a nonzero status byte"
                );
            }
            if ack.data.len() != expected.reply_data_len() {
                return Err(IoError::ShortRead {
                    what: "SCS reply block",
                    expected: expected.reply_data_len(),
                    got: ack.data.len(),
                });
            }
            blocks.push(ack.data);
        }
        Ok(blocks)
    }

    /// Read one block from every joint (never the IMU node).
    fn read_joint_blocks(&mut self, addr: u8, len: usize) -> Result<Vec<Vec<u8>>> {
        let frame = wire::sync_read(&crate::model::JOINT_IDS, addr, len as u8);
        let expected = ExpectedReply::new(&crate::model::JOINT_IDS, len);
        self.request(&frame, &expected)
    }

    /// Radians for one joint, from its calibrated zero.
    fn radians(&self, joint: usize, raw: u16) -> f64 {
        let cal = &self.calibration.joints[joint];
        (wire::position_raw_to_counts(raw) as f64 - cal.zero_ticks as f64)
            * wire::RAD_PER_COUNT
            * cal.direction
    }

    /// The goal word for one joint, clamped to the servo's own travel window.
    ///
    /// Clamped here rather than left to the servo's limit registers because the failure is
    /// asymmetric: the servo will refuse or fold a target past its limits, and on this hardware
    /// a wound joint has already been seen reading past one turn (raw 6289) — a number a
    /// calibration table then interprets as several revolutions of error.
    fn ticks_for(&self, joint: usize, radians: f64) -> u16 {
        let cal = &self.calibration.joints[joint];
        let counts =
            cal.zero_ticks + (radians * cal.direction / wire::RAD_PER_COUNT).round() as i32;
        let clamped = counts.clamp(cal.limits.0, cal.limits.1);
        wire::position_counts_to_raw(clamped)
    }

    /// Write the volatile P/D profile, and refuse to believe it until it reads back.
    ///
    /// Idempotent and cheap enough to call on every bring-up: one `sync_write` and one read,
    /// together about a millisecond, against the alternative of trusting a register that a
    /// reboot silently resets.
    fn ensure_gains(&mut self) -> Result<()> {
        if self.gains_verified {
            return Ok(());
        }
        self.write_gains()
    }

    fn write_gains(&mut self) -> Result<()> {
        let rows: Vec<[u8; 2]> = (0..NUM_JOINTS)
            .map(|joint| self.calibration.pd.registers(joint, self.gain))
            .collect();
        let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        let frame = wire::sync_write(&crate::model::JOINT_IDS, reg::KP, &refs)
            .map_err(|e| IoError::Bus(format!("SCS gain frame: {e}")))?;
        self.transport
            .write_all(&frame)
            .map_err(|source| IoError::Bus(format!("SCS gain write: {source}")))?;
        self.gains_verified = self.verify_gains(&rows)?;
        if !self.gains_verified {
            return Err(IoError::Bus(
                "SCS servo P/D did not read back after being written; refusing to enable"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Whether every joint's RAM P/D currently equals `want`.
    fn verify_gains(&mut self, want: &[[u8; 2]]) -> Result<bool> {
        let blocks = self.read_joint_blocks(reg::KP, gains_block::LEN)?;
        Ok(blocks.iter().zip(want).all(|(block, want)| {
            block[gains_block::KP] == want[0] && block[gains_block::KD] == want[1]
        }))
    }

    /// The profile this robot should be holding right now.
    fn want_gains(&self) -> Vec<[u8; 2]> {
        (0..NUM_JOINTS)
            .map(|joint| self.calibration.pd.registers(joint, self.gain))
            .collect()
    }

    /// Read every joint's torque-enable bit in one transaction.
    fn torque_enabled(&mut self) -> Result<Vec<bool>> {
        let blocks = self.read_joint_blocks(reg::TORQUE_ENABLE, 1)?;
        Ok(blocks.iter().map(|b| b[0] != 0).collect())
    }

    /// Torque enable, one servo at a time, with an acknowledgement each — and deliberately **not**
    /// a `sync_write`.
    ///
    /// **Measured on this hardware, and it cost a robot that would not let go.** A `sync_write` to
    /// register 50 (P/D) lands, and one to register 42 (goal position) lands, so the frame shape is
    /// right. The same frame at register 40 does not stick: the servo answers 0 to a read
    /// immediately afterwards — which is what made the old verification say the release had
    /// happened — and it is holding again seconds later with nothing on this side writing 1. A
    /// direct `WRITE` to 40 from a bench tool releases the joint and stays released. Register 40 is
    /// the torque switch; treat the others' evidence as not covering it.
    ///
    /// It is also the write where a silent failure is worst. An unacknowledged frame that a servo
    /// drops leaves the robot stiff, and the only witness is a read-back that can catch the
    /// register mid-update. A `WRITE` is acknowledged, so a refusal arrives as a status byte rather
    /// than as an inference.
    ///
    /// Cost: fifteen round trips instead of one — about 5 ms at the measured slot time, on bring-up,
    /// `init` and `relax` only. Not on the tick.
    fn write_torque(&mut self, on: bool) -> Result<()> {
        let value = [u8::from(on)];
        for &id in crate::model::JOINT_IDS.iter() {
            let frame = wire::write(id, reg::TORQUE_ENABLE, &value);
            let expected = ExpectedReply::new(&[id], 0);
            self.request(&frame, &expected).map_err(|e| {
                IoError::Bus(format!(
                    "SCS torque write to id {id} was not acknowledged: {e}"
                ))
            })?;
        }
        Ok(())
    }

    /// One servo's registers, read with a plain `READ`.
    ///
    /// For the bench tool rather than the control path: the tick reads all sixteen devices in one
    /// transaction, and this exists so a diagnosis can ask one servo one question without the
    /// whole bus. `wire::read` rather than a one-id `sync_read` because this is a single servo and
    /// the acknowledged instruction is the one whose reply is checked against the length asked for.
    pub fn read_register(&mut self, id: u8, addr: u8, len: usize) -> Result<Vec<u8>> {
        let frame = wire::read(id, addr, len as u8);
        let expected = ExpectedReply::new(&[id], len);
        let mut blocks = self.request(&frame, &expected)?;
        blocks
            .pop()
            .ok_or_else(|| IoError::Bus(format!("servo {id} answered nothing")))
    }

    /// One servo's registers, written with an acknowledgement.
    ///
    /// `WRITE`, not `sync_write`: a servo that refuses an unacknowledged frame is indistinguishable
    /// from one that took it, which is the whole difficulty this pair of helpers was added to
    /// resolve.
    pub fn write_register(&mut self, id: u8, addr: u8, data: &[u8]) -> Result<()> {
        let frame = wire::write(id, addr, data);
        let expected = ExpectedReply::new(&[id], 0);
        self.request(&frame, &expected).map(|_| ())
    }

    /// Every joint's EEPROM P/D, reading only.
    pub fn eeprom_gains(&mut self) -> Result<GainBackup> {
        let blocks = self.read_joint_blocks(reg::EEPROM_KP, 2)?;
        let mut backup = GainBackup {
            kp: [0; NUM_JOINTS],
            kd: [0; NUM_JOINTS],
        };
        for (joint, block) in blocks.iter().enumerate() {
            backup.kp[joint] = block[0];
            backup.kd[joint] = block[1];
        }
        Ok(backup)
    }

    /// Burn the volatile profile into EEPROM, so a servo that resets comes back holding it.
    ///
    /// **Never called automatically, and never from the control loop.** Two reasons, and the second
    /// is the one that matters. A wrong persistent gain is a robot that lurches on every boot from
    /// then on, with nothing in the daemon's own configuration to point at. And an interrupted
    /// EEPROM write is the one operation on this bus that can leave a servo unusable, which is not
    /// a risk to take on a path that runs unattended.
    ///
    /// The four steps are load-bearing together. Registers 21/22 are only writable while the lock
    /// register is clear, but a locked servo *accepts* the write into RAM and silently does not
    /// persist it — so a forgotten unlock is not an error, it is a change that evaporates at the
    /// next power cycle. Hence: read the locks, clear them, write, read back, restore each lock to
    /// what it was. Returns the gains that were there.
    ///
    /// Refused while energised: this is a flash write, and the servo is not being asked to hold a
    /// pose through it.
    pub fn persist_gains(&mut self) -> Result<GainBackup> {
        if self.enabled {
            return Err(IoError::Bus(
                "refusing to write EEPROM gains while the servos are energised; relax first"
                    .to_owned(),
            ));
        }
        let backup = self.eeprom_gains()?;
        let locks = self.read_joint_blocks(reg::LOCK, 1)?;
        let locks: Vec<u8> = locks.iter().map(|block| block[0]).collect();

        // The lock is one byte per servo; the gains are two. Both are one broadcast frame.
        let unlock = vec![[0u8]; NUM_JOINTS];
        let unlock_refs: Vec<&[u8]> = unlock.iter().map(|row| row.as_slice()).collect();
        self.sync_write_raw(reg::LOCK, &unlock_refs)?;
        let rows = self.want_gains();
        let refs: Vec<&[u8]> = rows.iter().map(|row| row.as_slice()).collect();
        self.sync_write_raw(reg::EEPROM_KP, &refs)
            .map_err(|e| IoError::Bus(format!("EEPROM gain write: {e}")))?;

        let written = self.eeprom_gains()?;
        // Restore the locks before reporting, so a failed verification still leaves the servos as
        // they were found rather than permanently writable.
        let lock_rows: Vec<[u8; 1]> = locks.iter().map(|lock| [*lock]).collect();
        let lock_refs: Vec<&[u8]> = lock_rows.iter().map(|row| row.as_slice()).collect();
        self.sync_write_raw(reg::LOCK, &lock_refs)?;

        let want_kp: [u8; NUM_JOINTS] = std::array::from_fn(|joint| rows[joint][0]);
        let want_kd: [u8; NUM_JOINTS] = std::array::from_fn(|joint| rows[joint][1]);
        if written.kp != want_kp || written.kd != want_kd {
            return Err(IoError::Bus(
                "EEPROM gains did not read back after being written; the servo may be a different \
                 firmware and its profile is now uncertain"
                    .to_owned(),
            ));
        }
        // The volatile pair is the same profile, so there is nothing left to repair.
        self.gains_verified = true;
        Ok(backup)
    }

    /// One broadcast write, nothing to wait for.
    fn sync_write_raw(&mut self, addr: u8, refs: &[&[u8]]) -> Result<()> {
        let frame = wire::sync_write(&crate::model::JOINT_IDS, addr, refs)
            .map_err(|e| IoError::Bus(format!("SCS write frame: {e}")))?;
        self.transport
            .write_all(&frame)
            .map_err(|source| IoError::Bus(format!("SCS write: {source}")))
    }

    /// Survey the bus. **Reads only** — no torque, no goal, no gain, no EEPROM.
    ///
    /// Three `sync_read`s and nothing else, which is a property worth stating and worth testing:
    /// commissioning happens with a robot possibly held up by hand, and a tool that energised a
    /// joint "to measure it" would be measuring a different pose. The one form of this that is
    /// safe to run on a robot someone is touching is the one that cannot write.
    pub fn survey(&mut self) -> Result<Survey> {
        let identity = self.read_joint_blocks(0, reg::TORQUE_ENABLE as usize + 1)?;
        let gains = self.read_joint_blocks(reg::KP, gains_block::LEN)?;
        let tick = self.read_full_block()?;

        let mut servos = [ServoIdentity {
            id: 0,
            model: 0,
            firmware: (0, 0),
            mode: 0,
            baud_code: 0,
            lock: 0,
            limits: (0, 0),
        }; NUM_JOINTS];
        let mut raw_ticks = [0u16; NUM_JOINTS];
        let mut kp = [0u8; NUM_JOINTS];
        let mut kd = [0u8; NUM_JOINTS];
        let mut volts = [0.0f64; NUM_JOINTS];
        let mut temps_c = [0.0f64; NUM_JOINTS];
        let mut torque = [0u8; NUM_JOINTS];
        let mut torque_direct = [0u8; NUM_JOINTS];
        let direct = self.read_joint_blocks(reg::TORQUE_ENABLE, 1)?;

        for (joint, block) in identity.iter().enumerate() {
            servos[joint] = ServoIdentity {
                id: block[reg::ID as usize],
                model: wire::le_u16(block[reg::MODEL_L as usize], block[reg::MODEL_H as usize]),
                firmware: (block[0], block[1]),
                mode: block[reg::MODE as usize],
                baud_code: block[reg::BAUD_RATE as usize],
                lock: gains[joint][gains_block::LOCK],
                limits: (
                    wire::position_raw_to_counts(wire::le_u16(
                        block[reg::MIN_ANGLE_LIMIT_L as usize],
                        block[reg::MIN_ANGLE_LIMIT_L as usize + 1],
                    )),
                    wire::position_raw_to_counts(wire::le_u16(
                        block[reg::MAX_ANGLE_LIMIT_L as usize],
                        block[reg::MAX_ANGLE_LIMIT_L as usize + 1],
                    )),
                ),
            };
            kp[joint] = gains[joint][gains_block::KP];
            kd[joint] = gains[joint][gains_block::KD];
            torque[joint] = block[reg::TORQUE_ENABLE as usize];
            torque_direct[joint] = direct[joint][0];
            volts[joint] = f64::from(gains[joint][gains_block::VOLTAGE]) * wire::VOLTS_PER_COUNT;
            temps_c[joint] = f64::from(gains[joint][gains_block::TEMPERATURE]);
            raw_ticks[joint] = wire::le_u16(
                tick[1 + joint][block::POSITION],
                tick[1 + joint][block::POSITION + 1],
            );
        }

        Ok(Survey {
            servos,
            raw_ticks,
            kp,
            kd,
            volts,
            temps_c,
            torque,
            torque_direct,
            imu_status: tick[0][imu_block::STATUS],
            imu_counter: tick[0][imu_block::COUNT],
        })
    }

    /// The tick's own transaction, returned as raw blocks, for a caller that wants the encoder
    /// counts rather than the calibrated radians [`RobotIo::read`] produces.
    fn read_full_block(&mut self) -> Result<Vec<Vec<u8>>> {
        let frame = wire::sync_read(&self.ids, reg::PRESENT_POSITION_L, block::LEN as u8);
        let expected = ExpectedReply::new(&self.ids, TICK_DATA_LEN);
        self.request(&frame, &expected)
    }

    /// The pose the robot is in right now, for a caller that has to start a ramp from it.
    pub fn present_positions(&mut self) -> Result<[f64; NUM_JOINTS]> {
        Ok(self.read()?.positions)
    }

    /// Ramp every joint from where it is now to `target`, linearly.
    ///
    /// Only ever called by an explicit `init`, matching [`crate::bus::DynamixelIo::interpolate_to`]
    /// — the control loop must never move the robot on its own, because that would make an update
    /// restart a fall risk. Blocking, and deliberately so: nothing else should be talking to the
    /// bus while this runs.
    ///
    /// The caller owns energising first, and on this backend that is what makes the ramp safe:
    /// [`RobotIo::set_torque`] writes the pose the robot is already in before anything is
    /// energised, so the ramp starts from a zero error term rather than from whatever the goal
    /// registers held when the board last lost power.
    pub fn interpolate_to(
        &mut self,
        target: &[f64; NUM_JOINTS],
        duration: Duration,
        step: Duration,
    ) -> Result<()> {
        let start = self.present_positions()?;
        let steps = (duration.as_secs_f64() / step.as_secs_f64())
            .ceil()
            .max(1.0) as u32;
        for i in 1..=steps {
            let t = i as f64 / steps as f64;
            let mut next = [0.0; NUM_JOINTS];
            for (joint, value) in next.iter_mut().enumerate() {
                *value = start[joint] + (target[joint] - start[joint]) * t;
            }
            self.write(&JointTargets::new(next))?;
            std::thread::sleep(step);
        }
        Ok(())
    }

    /// Write the goal block and verify it landed, for bring-up.
    ///
    /// [`RobotIo::write`] deliberately does not verify: it runs every tick, and a broadcast goal
    /// write has no acknowledgement to check. Bring-up is the one place where "the servos are
    /// holding what we last sent" has to be established before torque arrives, so it pays for
    /// the read.
    /// The goal frame itself, with no opinion about whether it should be sent.
    ///
    /// Split out because one caller legitimately writes goals to servos that are still released:
    /// [`Self::set_torque`] writes the pose the robot is already in *before* it energises, which is
    /// the measured bring-up order and the reason energising does not drive every joint at once.
    fn send_goals(&mut self, targets: &JointTargets) -> Result<()> {
        // One frame for the whole robot: fifteen bus turnarounds saved, and nothing to wait
        // for, because a broadcast write is not acknowledged by any device.
        let rows: Vec<[u8; 2]> = (0..NUM_JOINTS)
            .map(|joint| {
                self.ticks_for(joint, targets.positions[joint])
                    .to_le_bytes()
            })
            .collect();
        let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
        let frame = wire::sync_write(&crate::model::JOINT_IDS, reg::GOAL_POSITION_L, &refs)
            .map_err(|e| IoError::Bus(format!("SCS goal frame: {e}")))?;
        self.transport
            .write_all(&frame)
            .map_err(|source| IoError::Bus(format!("SCS goal write: {source}")))
    }

    fn write_and_verify_goals(&mut self, targets: &JointTargets) -> Result<()> {
        // `send_goals`, not `write`: this runs *before* torque goes on, so `enabled` is still
        // false and the guard in `write` would (correctly, for its own caller) skip it.
        self.send_goals(targets)?;
        let want: Vec<u16> = (0..NUM_JOINTS)
            .map(|joint| self.ticks_for(joint, targets.positions[joint]))
            .collect();
        let blocks = self.read_joint_blocks(reg::GOAL_POSITION_L, 2)?;
        for (joint, block) in blocks.iter().enumerate() {
            let got = wire::le_u16(block[0], block[1]);
            if got != want[joint] {
                return Err(IoError::Bus(format!(
                    "{}: goal read back {got}, wrote {} — refusing to energise a joint that is \
                     not where it was told to be",
                    JOINT_NAMES[joint], want[joint]
                )));
            }
        }
        Ok(())
    }
}

impl RobotIo for ScsIo {
    fn read(&mut self) -> Result<Sensors> {
        // One transaction carries the node and all fifteen servos, because the node's FeeTech
        // block is the same fifteen bytes wide — see `wire::imu_block`.
        let frame = wire::sync_read(&self.ids, reg::PRESENT_POSITION_L, block::LEN as u8);
        let expected = ExpectedReply::new(&self.ids, TICK_DATA_LEN);
        let blocks = self.request(&frame, &expected)?;

        let mut sensors = Sensors::default();

        let Some(node) = blocks.first() else {
            return Err(IoError::Bus("SCS sync_read returned no blocks".to_owned()));
        };
        if node.len() < imu_block::CTRL_LEN {
            return Err(IoError::ShortRead {
                what: "imu node block",
                expected: imu_block::CTRL_LEN,
                got: node.len(),
            });
        }
        let mut raw = [0u8; IMU_BLOCK_LEN];
        raw.copy_from_slice(&node[..IMU_BLOCK_LEN]);

        let status = node[imu_block::STATUS];
        let count = node[imu_block::COUNT];
        let run = self.stale.observe(&raw, Some(count));
        if run == STALE_RUN_WARN || (run > STALE_RUN_WARN && run.is_multiple_of(500)) {
            tracing::warn!(
                consecutive = run,
                total = self.stale.stale.total,
                counter = count,
                "imu node has not produced a new sample for {run} reads — orientation is frozen"
            );
        }
        sensors.imu = self.decoder.decode(&raw);
        self.imu_status = status;

        for (joint, block) in blocks[1..].iter().enumerate() {
            let position = wire::le_u16(
                block[wire::block::POSITION],
                block[wire::block::POSITION + 1],
            );
            let speed = wire::le_u16(block[wire::block::SPEED], block[wire::block::SPEED + 1]);
            let current =
                wire::le_u16(block[wire::block::CURRENT], block[wire::block::CURRENT + 1]);
            let cal = &self.calibration.joints[joint];
            sensors.positions[joint] = self.radians(joint, position);
            // Velocity carries the joint's sign for the same reason position does: one
            // direction table, or a joint reads forward on the way out and backward on the way
            // back.
            sensors.velocities[joint] = wire::signed_magnitude(speed) as f64
                * wire::RAD_PER_SEC_PER_SPEED_COUNT
                * cal.direction;
            // Magnitude, as `Sensors` documents: what consumers want is load, and a joint
            // holding a squat is near-zero velocity at non-zero current. Six and a half
            // milliamps per count, from the vendor's own worked example.
            sensors.currents_ma[joint] =
                wire::signed_magnitude(current).unsigned_abs() as f64 * wire::MA_PER_CURRENT_COUNT;
        }
        Ok(sensors)
    }

    fn write(&mut self, targets: &JointTargets) -> Result<()> {
        // **A released servo is not told where to go.** Measured on this hardware with
        // `scs_commission --probe-goal-refresh`: a servo whose torque has just been cut re-arms
        // itself when a goal arrives, so the goal the control loop writes every tick silently
        // undoes a `relax`. It is intermittent rather than per-servo — the same joints answered
        // both ways across runs — which fits the race it is: the goal is read from the servo, then
        // the servo is released, and by the time the goal lands the joint has sagged, so the goal
        // is no longer where the servo is and the servo re-engages to correct it. That is also why
        // the FeeTech port this replaces released "most of the time and occasionally not".
        //
        // Gain writes do **not** do this (probed the same way, four joints, no re-arm), so the
        // drift guard needs no such guard.
        if !self.enabled {
            return Ok(());
        }
        self.send_goals(targets)
    }

    fn set_gain(&mut self, kp: u16) -> Result<()> {
        // Called on **every tick** — `Safety::apply` passes the gain its caller decided, and the
        // caller tapers it deliberately: the standing policy runs softer than the walking one and
        // limp-fall softer still. So two things follow, and both are load-bearing.
        //
        // An unchanged gain must cost nothing. The profile is a RAM write plus a readback, and
        // paying that per tick would spend a fifth of the 20 ms period re-stating what is already
        // true — the bus has other work.
        if kp == self.gain && self.gains_verified {
            return Ok(());
        }
        // And a *changed* gain is applied, in both directions, rather than refused. A stiffening
        // under load does grow every error term at once, which is why `set_torque` adopts the pose
        // before it energises anything; but refusing the change here would override a decision
        // that is the control loop's, and stand→walk→limp are exactly the transitions this
        // backend would then be unable to make.
        let previous = self.gain;
        self.gain = kp;
        self.gains_verified = false;
        if let Err(e) = self.ensure_gains() {
            // Put the old gain back, so a failed write is not reported as the new one being in
            // force: the drift guard compares the servos against `self.gain`, and a value that
            // never landed would make it rewrite the profile on every `slow_sensors` forever.
            self.gain = previous;
            self.gains_verified = false;
            return Err(e);
        }
        Ok(())
    }

    fn set_torque(&mut self, on: bool) -> Result<()> {
        if !on {
            self.write_torque(false)?;
            let states = self.torque_enabled()?;
            self.enabled = false;
            // A servo that will not release is the one failure here worth an error: the robot
            // cannot be handled, and `relax` would silently not have happened.
            let stuck: Vec<String> = states
                .iter()
                .enumerate()
                .filter(|(_, on)| **on)
                .map(|(joint, _)| {
                    format!(
                        "{} ({})",
                        crate::model::JOINT_IDS[joint],
                        JOINT_NAMES[joint]
                    )
                })
                .collect();
            if !stuck.is_empty() {
                return Err(IoError::Bus(format!(
                    "SCS servos still energised after torque off: {}",
                    stuck.join(", ")
                )));
            }
            return Ok(());
        }

        // Measured on this hardware, and the reason this order is not negotiable: energising a
        // servo whose goal register still holds a stale target drives every joint at once, at
        // whatever current the pack will give — 15 servos × up to 26° of error inrush tripped the
        // board's undervoltage lockout and reset the SBC, four times in eight minutes. Writing
        // the pose the robot is already in makes the error zero, so there is nothing to rush to.
        let present = self.read()?;
        self.ensure_gains()?;
        self.write_and_verify_goals(&JointTargets::new(present.positions))?;

        self.write_torque(true)?;
        let states = self.torque_enabled()?;
        let off: Vec<String> = states
            .iter()
            .enumerate()
            .filter(|(_, on)| !**on)
            .map(|(joint, _)| {
                format!(
                    "{} ({})",
                    crate::model::JOINT_IDS[joint],
                    JOINT_NAMES[joint]
                )
            })
            .collect();
        if !off.is_empty() {
            // All or nothing, because a partial enable means some joints are holding a pose
            // while others are limp: the robot is being held by a subset of its servos and the
            // control loop does not know which.
            self.enabled = false;
            return Err(IoError::Bus(format!(
                "SCS partial enable: {} did not energise; relax and retry",
                off.join(", ")
            )));
        }
        self.enabled = true;
        Ok(())
    }

    fn reboot(&mut self, id: u8) -> Result<()> {
        if !crate::model::JOINT_IDS.contains(&id) {
            return Err(IoError::Bus(format!("SCS reboot: {id} is not a joint")));
        }
        // Torque off first: the servo comes back with its EEPROM gains but keeps whatever it was
        // driving towards, and a reboot while energised is a joint with no loop holding it.
        self.transport
            .write_all(&wire::write(id, reg::TORQUE_ENABLE, &[0]))
            .map_err(|source| IoError::Bus(format!("SCS reboot torque off: {source}")))?;
        self.transport
            .write_all(&wire::reboot(id))
            .map_err(|source| IoError::Bus(format!("SCS reboot: {source}")))?;
        // It does not answer — measured: no reply, back after ≈823 ms. Waiting would stall the
        // loop for most of a second, so the caller owns the wait and the re-enable.
        //
        // The RAM gains are gone with it: a reboot reloads 21/22 over 50/51. Marking the profile
        // unverified is what makes the *next* bring-up rewrite it, rather than the robot walking
        // on a stiffness nobody chose.
        self.gains_verified = false;
        self.enabled = false;
        Ok(())
    }

    fn slow_sensors(&mut self) -> Result<SlowSensors> {
        let blocks = self.read_joint_blocks(reg::KP, gains_block::LEN)?;

        // The same transaction that carries voltage and temperature also carries the volatile
        // gains, which is the only periodic opportunity to notice a servo that has reset itself
        // out from under the other fourteen. Repairing is cheap; not noticing means walking on a
        // stiffness only that joint has.
        let want = self.want_gains();
        let drifted: Vec<usize> = blocks
            .iter()
            .enumerate()
            .filter(|(joint, block)| {
                block[gains_block::KP] != want[*joint][0]
                    || block[gains_block::KD] != want[*joint][1]
            })
            .map(|(joint, _)| joint)
            .collect();
        if !drifted.is_empty() {
            let named: Vec<String> = drifted
                .iter()
                .map(|joint| {
                    format!(
                        "{} ({})",
                        crate::model::JOINT_IDS[*joint],
                        JOINT_NAMES[*joint]
                    )
                })
                .collect();
            tracing::warn!(
                joints = named.join(", "),
                "SCS volatile P/D drifted from the profile; rewriting"
            );
            self.health.gain_repairs += 1;
            self.gains_verified = false;
            self.write_gains()?;
        } else {
            self.gains_verified = true;
        }

        let mut volts = 0.0;
        let mut temps_c = [0.0f64; NUM_JOINTS];
        for (joint, block) in blocks.iter().enumerate() {
            volts += f64::from(block[gains_block::VOLTAGE]) * wire::VOLTS_PER_COUNT;
            temps_c[joint] = f64::from(block[gains_block::TEMPERATURE]);
        }
        Ok(SlowSensors {
            volts: volts / NUM_JOINTS as f64,
            temps_c,
        })
    }

    fn imu_stale(&self) -> ImuStale {
        self.stale.stale
    }

    fn imu_ready(&self) -> bool {
        // Two independent facts, and both are needed. The decoder has seen enough non-zero
        // quaternions to have a value at all; the node's status bit says the chip's own fusion is
        // running. Either alone reports a robot as oriented when its orientation is unknown.
        self.decoder.ready() && self.imu_status & wire::IMU_FLAG_SFLP_VALID != 0
    }
}

/// Same threshold [`crate::bus`] warns at, so one robot's journal reads the same either way.
const STALE_RUN_WARN: u64 = 25;
