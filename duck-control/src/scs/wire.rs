//! FeeTech SCS/HLS framing, registers and conversions — the bytes, and nothing else.
//!
//! Split from [`super`] on purpose: this module has no port, no clock and no `RobotIo`, so
//! every vector below is checked against the reference implementations without hardware. The
//! same bytes are asserted three ways elsewhere — `soft_imu_to_dxl/v1/host/bus.py` and the
//! node's own `host/tools/test_protocols.c` — and the duplicated vectors are the point: three
//! independent encoders that agree byte for byte is what makes a checksum disagreement a bug
//! here rather than a mystery on the bus.
//!
//! ```text
//! instruction: FF FF ID LEN INST [ADDR] [DATA...] ~SUM
//!              LEN = 1 (INST) + ADDR? + DATA + 1 (SUM)
//! ack:         FF FF ID LEN STATUS [DATA...] ~SUM
//!              LEN = 1 (STATUS) + DATA + 1 (SUM)
//! SUM          = ID + LEN + INST + ADDR + ΣDATA    (ADDR counts as 0 when absent)
//! ```
//!
//! Sources: `FTServo_Linux-main/src/SCS.cpp` (`writeBuf`, `Ack`, `Read`, `syncReadPacketTx`,
//! `syncWrite`), `INST.h`, `HLSCL.h`, and the measurements in
//! `soft_imu_to_dxl/v1/docs/bus_timing_borrow_plan.md` §1 against HD-1910-C001 firmware 3.46.

/// Sum that goes into the checksum: every byte from ID up to (but excluding) the checksum
/// itself. `ADDR` is included only when the frame carries one, and the reference library adds
/// it as 0 for the instructions without it.
fn checksum_sum(id: u8, len: u8, inst: u8, addr: u8, data: &[u8]) -> u8 {
    let mut sum = id.wrapping_add(len).wrapping_add(inst).wrapping_add(addr);
    for byte in data {
        sum = sum.wrapping_add(*byte);
    }
    sum
}

/// The byte actually sent: bitwise complement of [`checksum_sum`].
pub fn checksum(id: u8, len: u8, inst: u8, addr: u8, data: &[u8]) -> u8 {
    !checksum_sum(id, len, inst, addr, data)
}

/// Instruction set (`INST.h`), plus the reboot the vendor table omits.
pub mod inst {
    pub const PING: u8 = 0x01;
    pub const READ: u8 = 0x02;
    pub const WRITE: u8 = 0x03;
    pub const REG_WRITE: u8 = 0x04;
    pub const REG_ACTION: u8 = 0x05;
    pub const RECOVERY: u8 = 0x06;
    /// Reboot one servo. Absent from the vendor instruction table we have, and present on
    /// HD-1910-C001 firmware 3.46 anyway: measured 2026-09-24, it does not answer, the device
    /// is back after ~823 ms, EEPROM settings survive, and the RAM gains return to their EEPROM
    /// values. That last clause is why [`super::ScsIo::reboot`] re-arms the P/D write.
    pub const REBOOT: u8 = 0x08;
    pub const RESET: u8 = 0x0A;
    pub const CAL: u8 = 0x0B;
    pub const SYNC_READ: u8 = 0x82;
    pub const SYNC_WRITE: u8 = 0x83;
}

/// HLS memory table addresses (`HLSCL.h`, cross-checked against the servo itself).
pub mod reg {
    pub const MODEL_L: u8 = 3;
    pub const MODEL_H: u8 = 4;
    pub const ID: u8 = 5;
    pub const BAUD_RATE: u8 = 6;
    pub const SECOND_ID: u8 = 7;
    pub const MIN_ANGLE_LIMIT_L: u8 = 9;
    pub const MAX_ANGLE_LIMIT_L: u8 = 11;
    pub const CW_DEAD: u8 = 26;
    pub const CCW_DEAD: u8 = 27;
    pub const OFS_L: u8 = 31;
    pub const MODE: u8 = 33;
    pub const TORQUE_ENABLE: u8 = 40;
    pub const ACC: u8 = 41;
    pub const GOAL_POSITION_L: u8 = 42;
    pub const GOAL_TORQUE_L: u8 = 44;
    pub const GOAL_SPEED_L: u8 = 46;
    pub const TORQUE_LIMIT_L: u8 = 48;
    /// EEPROM position P and D. Unlike 50/51 these survive power loss, and a reboot reloads
    /// 50/51 *from* them — measured on this hardware, which is why the volatile pair is what a
    /// running robot writes and this pair is what it may choose to persist.
    pub const EEPROM_KP: u8 = 21;
    pub const EEPROM_KD: u8 = 22;
    /// Volatile position P. `None` of the documented tables list 50/51 — they are a firmware
    /// feature, and the only P/D a running robot can change without touching EEPROM.
    pub const KP: u8 = 50;
    pub const KD: u8 = 51;
    pub const LOCK: u8 = 55;
    pub const PRESENT_POSITION_L: u8 = 56;
    pub const PRESENT_SPEED_L: u8 = 58;
    pub const PRESENT_LOAD_L: u8 = 60;
    pub const PRESENT_VOLTAGE: u8 = 62;
    pub const PRESENT_TEMPERATURE: u8 = 63;
    /// The servo's own alarm/status byte. Non-zero is a fault, and the one place a latched
    /// overload or over-temperature is visible from the bus.
    pub const PRESENT_STATUS: u8 = 65;
    pub const MOVING: u8 = 66;
    pub const PRESENT_CURRENT_L: u8 = 69;
}

/// Byte offsets inside the 15-byte block at [`reg::PRESENT_POSITION_L`].
///
/// Written out rather than derived so the widths are visible: several are two bytes wide, and
/// an off-by-one here reads a plausible number out of the wrong register. Every offset is
/// `register − 56`.
pub mod block {
    pub const POSITION: usize = 0;
    pub const SPEED: usize = 2;
    pub const LOAD: usize = 4;
    pub const VOLTAGE: usize = 6;
    pub const TEMPERATURE: usize = 7;
    pub const STATUS: usize = 9;
    pub const CURRENT: usize = 13;
    /// Bytes in the block. Fifteen is not a preference: `sync_read` gives every device one
    /// length, the servo documents 56..70, and the `imu_to_dxl` node answers the same 15 so a
    /// single transaction can carry both. Reading 20 works on this firmware and was rejected
    /// anyway — it pushes a sixteen-device transaction from ≈4.9 ms to ≈5.7 ms against a 6 ms
    /// budget. See `bus_timing_borrow_plan.md` §2.
    pub const LEN: usize = 15;
}

/// Byte offsets inside the `imu_to_dxl` node's answer at the same address.
///
/// The node packs its FeeTech personality as the DXL control block's first twelve bytes
/// followed by the counter and flags it has room for here and not at address 124. Byte-for-byte
/// equal to what [`crate::imu::SflpDecoder`] already decodes, which is what lets one read feed
/// both.
pub mod imu_block {
    /// Gyro xyz then the SFLP quaternion xyz, exactly `crate::imu::IMU_BLOCK_LEN`.
    pub const CTRL_LEN: usize = 12;
    /// Frames the node has produced, wrapping. A repeat means it did not refresh — a fact
    /// upstream structurally cannot see, because its 12-byte DXL read stops short of the
    /// counter at offset 18 of the node's 20-byte diagnostic block.
    pub const COUNT: usize = 12;
    pub const STATUS: usize = 13;
}

/// `TELEM_FLAG_SFLP_VALID` in [`imu_block::STATUS`]: the chip is producing fused output.
pub const IMU_FLAG_SFLP_VALID: u8 = 0x01;

/// Volatile-gain block fetched by [`super::RobotIo::slow_sensors`], starting at [`reg::KP`].
///
/// One transaction covers the gains *and* the pack voltage and case temperatures, because
/// 50..63 is contiguous and 14 bytes: `Kp Kd Ki · · lock present_position(2) speed(2) load(2)
/// voltage temperature`. Fifteen servos' replies cost about a millisecond, once a second.
pub mod gains_block {
    pub const KP: usize = 0;
    pub const KD: usize = 1;
    pub const LOCK: usize = 5;
    pub const VOLTAGE: usize = 12;
    pub const TEMPERATURE: usize = 13;
    pub const LEN: usize = 14;
}

pub const BROADCAST_ID: u8 = 0xFE;
/// Largest frame the protocol can produce (`LEN` is a `u8`).
pub const MAX_FRAME: usize = 4 + 255;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A frame arrived but its header, length or checksum did not hold together.
    BadFrame(&'static str),
    /// A reply named a device nobody asked.
    UnexpectedId(u8),
    /// The device answered with a non-zero status byte. Named rather than folded into a
    /// generic bus error because it is the difference between a lost frame and a servo
    /// reporting an overload.
    DeviceStatus { id: u8, status: u8 },
    /// A reply was shorter than the request demanded.
    ShortData { got: usize, want: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::BadFrame(why) => write!(f, "bad SCS frame: {why}"),
            Error::UnexpectedId(id) => write!(f, "SCS reply from id {id}, which was not asked"),
            Error::DeviceStatus { id, status } => {
                write!(f, "SCS servo {id} reports status {status:#04x}")
            }
            Error::ShortData { got, want } => {
                write!(f, "SCS reply carried {got} bytes, expected {want}")
            }
        }
    }
}

impl std::error::Error for Error {}

// ── packet construction ─────────────────────────────────────────────────────

/// Instruction with no ADDR (PING, REG_ACTION, REBOOT, RESET, CAL): `LEN = 2`, and the
/// checksum still counts ADDR as zero because the reference library does.
pub fn instruction_no_addr(id: u8, instruction: u8) -> Vec<u8> {
    let len = 2u8;
    let sum = checksum(id, len, instruction, 0, &[]);
    vec![0xFF, 0xFF, id, len, instruction, sum]
}

/// Instruction with ADDR + DATA: `LEN = DATA + 3`.
pub fn instruction(id: u8, instruction: u8, addr: u8, data: &[u8]) -> Vec<u8> {
    let len = (data.len() as u8).wrapping_add(3);
    let sum = checksum(id, len, instruction, addr, data);
    let mut frame = Vec::with_capacity(6 + data.len());
    frame.extend_from_slice(&[0xFF, 0xFF, id, len, instruction, addr]);
    frame.extend_from_slice(data);
    frame.push(sum);
    frame
}

pub fn ping(id: u8) -> Vec<u8> {
    instruction_no_addr(id, inst::PING)
}

/// READ: the one data byte is the length, so `LEN = 4`.
pub fn read(id: u8, addr: u8, len: u8) -> Vec<u8> {
    instruction(id, inst::READ, addr, &[len])
}

pub fn write(id: u8, addr: u8, data: &[u8]) -> Vec<u8> {
    instruction(id, inst::WRITE, addr, data)
}

pub fn reboot(id: u8) -> Vec<u8> {
    instruction_no_addr(id, inst::REBOOT)
}

pub fn reset(id: u8) -> Vec<u8> {
    instruction_no_addr(id, inst::RESET)
}

/// SYNC_READ: `FF FF FE (IDN+4) 82 ADDR nLen ID... ~SUM`.
///
/// The reply is one status packet *per listed id*, and the device answers in the slot its
/// position in the list gives it — a device that is absent still consumes its slot, measured
/// linear at ≈295 µs. So the collector must gather by id rather than assume arrival order, and
/// this is also why the IMU node is listed first: it then answers before the servo burst
/// instead of after fourteen of them.
pub fn sync_read(ids: &[u8], addr: u8, len: u8) -> Vec<u8> {
    let frame_len = (ids.len() as u8).wrapping_add(4);
    let mut sum = BROADCAST_ID
        .wrapping_add(frame_len)
        .wrapping_add(inst::SYNC_READ)
        .wrapping_add(addr)
        .wrapping_add(len);
    let mut frame = Vec::with_capacity(7 + ids.len());
    frame.extend_from_slice(&[
        0xFF,
        0xFF,
        BROADCAST_ID,
        frame_len,
        inst::SYNC_READ,
        addr,
        len,
    ]);
    for id in ids {
        frame.push(*id);
        sum = sum.wrapping_add(*id);
    }
    frame.push(!sum);
    frame
}

/// SYNC_WRITE: `FF FF FE ((nLen+1)*IDN+4) 83 ADDR nLen ID DATA... ~SUM`.
///
/// One packet sets every goal, which is the difference between one bus turnaround and fifteen
/// — and the reason the frame length is checked against `MAX_FRAME` rather than truncated.
pub fn sync_write(ids: &[u8], addr: u8, payloads: &[&[u8]]) -> Result<Vec<u8>, Error> {
    if ids.is_empty() || ids.len() != payloads.len() {
        return Err(Error::BadFrame("sync_write ids/payloads differ in length"));
    }
    let width = payloads[0].len();
    if payloads.iter().any(|p| p.len() != width) {
        return Err(Error::BadFrame("sync_write payloads differ in width"));
    }
    let width_u8 =
        u8::try_from(width).map_err(|_| Error::BadFrame("sync_write payload too wide"))?;
    let frame_len = ((width + 1) * ids.len() + 4) as u8;
    let mut sum = BROADCAST_ID
        .wrapping_add(frame_len)
        .wrapping_add(inst::SYNC_WRITE)
        .wrapping_add(addr)
        .wrapping_add(width_u8);
    let mut frame = vec![
        0xFF,
        0xFF,
        BROADCAST_ID,
        frame_len,
        inst::SYNC_WRITE,
        addr,
        width_u8,
    ];
    for (id, payload) in ids.iter().zip(payloads) {
        frame.push(*id);
        sum = sum.wrapping_add(*id);
        for byte in *payload {
            frame.push(*byte);
            sum = sum.wrapping_add(*byte);
        }
    }
    frame.push(!sum);
    Ok(frame)
}

// ── packet parsing ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ack {
    pub id: u8,
    pub status: u8,
    pub data: Vec<u8>,
}

/// Parse one complete status frame. `frame` must be exactly one frame: bytes are not scanned
/// for a header here, because a caller that has to resynchronise wants [`FrameReader`].
pub fn parse_ack(frame: &[u8]) -> Result<Ack, Error> {
    if frame.len() < 6 {
        return Err(Error::BadFrame("shorter than a ping ack"));
    }
    if frame[0] != 0xFF || frame[1] != 0xFF {
        return Err(Error::BadFrame("missing header"));
    }
    let len = frame[3] as usize;
    if len < 2 {
        return Err(Error::BadFrame("LEN < 2"));
    }
    if frame.len() != len + 4 {
        return Err(Error::BadFrame("LEN does not match frame length"));
    }
    let body = &frame[2..frame.len() - 1];
    let expect = !body.iter().fold(0u8, |acc, b| acc.wrapping_add(*b));
    if expect != frame[frame.len() - 1] {
        return Err(Error::BadFrame("checksum"));
    }
    Ok(Ack {
        id: frame[2],
        status: frame[4],
        data: frame[5..frame.len() - 1].to_vec(),
    })
}

/// Total frame length if `buf` starts with a complete frame, else `None`.
pub fn frame_len(buf: &[u8]) -> Option<usize> {
    if buf.len() < 4 {
        return None;
    }
    if buf[0] != 0xFF || buf[1] != 0xFF || buf[3] < 2 {
        return None;
    }
    let total = buf[3] as usize + 4;
    (buf.len() >= total).then_some(total)
}

/// What the reader is willing to accept, and how long it should be.
///
/// A reader that trusted the length byte alone cannot tell an answer from its own echo: our
/// own single-device `WRITE` to a servo is `FF FF <id> <len> 03 …`, which has a valid header
/// and a plausible length, so a reader hunting only for `FF FF` will sit waiting for a
/// several-hundred-byte frame that never comes and swallow the real answers behind it. Knowing
/// the id *and* the exact reply length turns that into an immediate resynchronise. This is the
/// difference between "the bus is fine but a reply is occasionally missing" and a loop that
/// stalls.
pub trait Expects {
    /// Total bytes of this device's reply, or `None` if it was not asked.
    fn reply_len(&self, id: u8) -> Option<usize>;
}

/// The ids a request named, and the reply length they share.
#[derive(Debug, Clone)]
pub struct ExpectedReply {
    ids: Vec<u8>,
    data_len: usize,
    /// Total frame length: `LEN + 4`, with `LEN = data + 2`.
    total: usize,
}

impl ExpectedReply {
    pub fn new(ids: &[u8], data_len: usize) -> Self {
        Self {
            ids: ids.to_vec(),
            data_len,
            total: data_len + 6,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The ids this request listed, in the order the caller listed them.
    pub fn ids(&self) -> &[u8] {
        &self.ids
    }

    /// Bytes of payload each reply should carry, so a caller can check the body as well as the
    /// frame — a device that answers the wrong question with the right framing is otherwise
    /// indistinguishable from one that answered correctly.
    pub fn reply_data_len(&self) -> usize {
        self.data_len
    }
}

impl Expects for ExpectedReply {
    fn reply_len(&self, id: u8) -> Option<usize> {
        self.ids.contains(&id).then_some(self.total)
    }
}

/// Reassembles status frames out of a byte stream.
///
/// Tolerates what a half-duplex bus actually delivers: the transceiver can echo what we wrote,
/// a read can be cut short, and a servo can answer late enough that its frame lands after the
/// next request. Anything that is not the reply we are waiting for is dropped one byte at a
/// time, so a false `FF FF` inside a payload costs one resynchronisation rather than the rest
/// of the run.
#[derive(Debug, Default)]
pub struct FrameReader {
    buf: Vec<u8>,
    /// Bytes thrown away while hunting for a header, for the health counters.
    discarded: u64,
}

impl FrameReader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Bytes discarded while resynchronising since the last call.
    pub fn take_discarded(&mut self) -> u64 {
        std::mem::take(&mut self.discarded)
    }

    pub fn buffered(&self) -> usize {
        self.buf.len()
    }

    /// The next complete, checksum-valid reply the caller is waiting for.
    ///
    /// `None` means "more bytes needed", which is also what a frame that failed its checksum
    /// ends as: on a bus where an echo and a real answer share the wire, a corrupted frame is a
    /// dropped byte, not a fault to hand upstairs.
    pub fn next_frame(&mut self, expected: &impl Expects) -> Option<Vec<u8>> {
        loop {
            let Some(start) = self.buf.windows(2).position(|w| w == [0xFF, 0xFF]) else {
                let drop = self.buf.len().saturating_sub(1);
                self.discarded += drop as u64;
                self.buf.drain(..drop);
                return None;
            };
            if start > 0 {
                self.discarded += start as u64;
                self.buf.drain(..start);
            }
            if self.buf.len() < 4 {
                return None;
            }
            // The length byte is only a hint until the id says we asked for this device.
            let Some(total) = expected.reply_len(self.buf[2]) else {
                self.discarded += 1;
                self.buf.remove(0);
                continue;
            };
            if self.buf[3] as usize + 4 != total || total > MAX_FRAME {
                self.discarded += 1;
                self.buf.remove(0);
                continue;
            }
            if self.buf.len() < total {
                return None;
            }
            let frame: Vec<u8> = self.buf.drain(..total).collect();
            match parse_ack(&frame) {
                Ok(_) => return Some(frame),
                Err(_) => self.discarded += total as u64,
            }
        }
    }
}

// ── conversions ─────────────────────────────────────────────────────────────

/// Position is a 15-bit signed count: bit 15 is the direction flag and bits 14..0 the
/// magnitude (`HLSCL::ReadPos`). Not two's complement — a difference that is invisible until a
/// joint passes the encoder's mid-point and then reads as a huge positive angle.
pub fn position_raw_to_counts(raw: u16) -> i32 {
    let magnitude = (raw & 0x7FFF) as i32;
    if raw & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

pub fn position_counts_to_raw(counts: i32) -> u16 {
    let magnitude = counts.unsigned_abs().min(0x7FFF) as u16;
    if counts < 0 {
        magnitude | 0x8000
    } else {
        magnitude
    }
}

/// Counts per revolution.
///
/// The HLS memory table gives the unit of both goal and present position as **0.087°**, i.e.
/// 4096 counts per 360°, and the 16-bit field as `-32767..32767` with bit 15 the direction bit
/// — so the field is multi-turn (about ±8 revolutions), which is why a joint wound past one
/// turn reads above 4095 instead of wrapping.
pub const POSITION_COUNTS_PER_REV: f64 = 4096.0;

pub const RAD_PER_COUNT: f64 = std::f64::consts::TAU / POSITION_COUNTS_PER_REV;

/// RPM per speed count (`60 * 0.732 = 43.92 rpm` in the reference example).
pub const RPM_PER_SPEED_COUNT: f64 = 0.732;

pub const RAD_PER_SEC_PER_SPEED_COUNT: f64 = RPM_PER_SPEED_COUNT * std::f64::consts::TAU / 60.0;

/// Milliamps per current count (`Torque = 300 * 6.5 = 1950 mA` in the example).
pub const MA_PER_CURRENT_COUNT: f64 = 6.5;

/// Volts per count of `PRESENT_VOLTAGE`.
pub const VOLTS_PER_COUNT: f64 = 0.1;

pub fn le_u16(low: u8, high: u8) -> u16 {
    u16::from_le_bytes([low, high])
}

/// Speed and current share a 16-bit word whose top bit is a direction flag, like position.
pub fn signed_magnitude(word: u16) -> i32 {
    position_raw_to_counts(word)
}

pub fn position_rad(low: u8, high: u8) -> f64 {
    position_raw_to_counts(le_u16(low, high)) as f64 * RAD_PER_COUNT
}

pub fn position_from_rad(rad: f64) -> [u8; 2] {
    let counts = (rad / RAD_PER_COUNT).round() as i32;
    position_counts_to_raw(counts).to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A status frame for `id` carrying `data`, checksum included.
    ///
    /// Hand-written frames are how a test comes to assert the right shape with the wrong
    /// checksum — the id is inside the sum — so every reply in this module is built here.
    fn ack_frame(id: u8, status: u8, data: &[u8]) -> Vec<u8> {
        let len = (data.len() + 2) as u8;
        let sum = checksum(id, len, status, 0, data);
        let mut frame = vec![0xFF, 0xFF, id, len, status];
        frame.extend_from_slice(data);
        frame.push(sum);
        frame
    }

    /// The vector the node's own C tests and `host/bus.py` both assert. A checksum that
    /// disagrees with the reference is a bus that never answers, which is the failure this
    /// prevents.
    #[test]
    fn ping_matches_the_reference_vector() {
        assert_eq!(ping(200), vec![0xFF, 0xFF, 0xC8, 0x02, 0x01, 0x34]);
    }

    #[test]
    fn read_matches_the_reference_vector() {
        assert_eq!(
            read(200, reg::PRESENT_POSITION_L, 12),
            vec![0xFF, 0xFF, 0xC8, 0x04, 0x02, 0x38, 0x0C, 0xED]
        );
    }

    /// The real robot's runtime layout: the node first, then all fifteen servos, at the block
    /// address and length the servos document. Byte-exact against
    /// `bus_timing_borrow_plan.md` §11(c), which is measured from `host/bus.py`.
    #[test]
    fn sixteen_device_sync_read_matches_the_measured_layout() {
        let ids = [
            200, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34,
        ];
        assert_eq!(
            sync_read(&ids, reg::PRESENT_POSITION_L, block::LEN as u8),
            vec![
                0xFF, 0xFF, 0xFE, 0x14, 0x82, 0x38, 0x0F, 0xC8, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x14,
                0x15, 0x16, 0x17, 0x18, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x12
            ]
        );
    }

    /// One goal write for all fifteen, which is the whole reason `sync_write` exists here.
    /// Length is `(2+1)*15 + 4 = 49`, at the protocol's limit for a two-byte field.
    #[test]
    fn sync_write_goals_is_one_frame_for_the_whole_robot() {
        let ids: Vec<u8> = (1..=15).collect();
        let payloads: Vec<[u8; 2]> = (0..15).map(|i| (1000 + i as u16).to_le_bytes()).collect();
        let refs: Vec<&[u8]> = payloads.iter().map(|p| p.as_slice()).collect();
        let frame = sync_write(&ids, reg::GOAL_POSITION_L, &refs).unwrap();
        assert_eq!(frame[3] as usize, 3 * 15 + 4);
        assert_eq!(frame.len(), frame[3] as usize + 4);
    }

    /// A broadcast write is *shape*-compatible with a status frame — same header, a length that
    /// satisfies `frame.len() == LEN + 4`, and a checksum computed over the same byte range — so
    /// `parse_ack` accepts our own goal write and calls the instruction byte a status. This is
    /// not a parser bug to fix but the reason a collector must filter replies by the ids it
    /// asked for; without that filter a robot reads its own write back as a reply from id 254.
    #[test]
    fn a_broadcast_write_parses_as_an_ack_from_the_broadcast_id() {
        let frame = sync_write(&[1, 2], reg::GOAL_POSITION_L, &[&[0, 0], &[0, 0]]).unwrap();
        let ack = parse_ack(&frame).unwrap();
        assert_eq!(ack.id, BROADCAST_ID);
        assert_eq!(ack.status, inst::SYNC_WRITE);

        // And the reader refuses it, because 254 was never an id we asked to hear from.
        let mut reader = FrameReader::new();
        reader.push(&frame);
        assert_eq!(reader.next_frame(&ExpectedReply::new(&[1, 2], 15)), None);
        assert!(reader.take_discarded() >= 1);
    }

    /// A `sync_write` whose ids and payloads disagree must be refused, not truncated: a short
    /// frame here would move some servos and not others, which looks like a calibration fault.
    #[test]
    fn sync_write_refuses_ragged_input() {
        assert!(sync_write(&[1, 2], reg::GOAL_POSITION_L, &[&[0, 0]]).is_err());
        assert!(sync_write(&[1, 2], reg::GOAL_POSITION_L, &[&[0, 0], &[0, 0, 0]]).is_err());
        assert!(sync_write(&[], reg::GOAL_POSITION_L, &[]).is_err());
    }

    #[test]
    fn ack_round_trips_and_the_checksum_is_load_bearing() {
        // FF FF C8 11 00 <15 bytes 00..0E> BD, from the protocol write-up.
        let mut frame = vec![0xFF, 0xFF, 200, 0x11, 0x00];
        frame.extend(0u8..=0x0E);
        frame.push(0xBD);
        let ack = parse_ack(&frame).unwrap();
        assert_eq!(ack.id, 200);
        assert_eq!(ack.status, 0);
        assert_eq!(ack.data.len(), 15);

        let mut bad = frame.clone();
        *bad.last_mut().unwrap() ^= 0x01;
        assert_eq!(parse_ack(&bad), Err(Error::BadFrame("checksum")));
    }

    /// The reader must survive the things a half-duplex bus does to it. `host/bus.py` and the
    /// node's C test drive the same shapes.
    #[test]
    fn the_reader_resynchronises_through_noise_and_a_short_frame() {
        let expected = ExpectedReply::new(&[200], 15);

        // The measured vector, asserted byte for byte above: data 00..0E, checksum 0xBD.
        let good = ack_frame(200, 0x00, &(0u8..=0x0E).collect::<Vec<_>>());
        assert_eq!(*good.last().unwrap(), 0xBD, "the reference vector changed");

        let mut reader = FrameReader::new();
        reader.push(b"debug text\r\n");
        reader.push(&[0xFF, 0xFF, 0xC8]);
        assert_eq!(
            reader.next_frame(&expected),
            None,
            "a partial frame is not a frame"
        );

        reader.push(&good);
        assert_eq!(
            reader.next_frame(&expected).as_deref(),
            Some(good.as_slice())
        );
        assert!(
            reader.take_discarded() >= 14,
            "the debug text was discarded"
        );

        // A frame whose first byte is a plausible header but which is not one.
        reader.push(&[0xFF, 0xFF, 0x01, 0x02, 0x03, 0xFF]);
        reader.push(&good);
        assert_eq!(
            reader.next_frame(&expected).as_deref(),
            Some(good.as_slice())
        );

        // A frame that passes the length check but not the checksum is dropped, not returned.
        let mut corrupt = good.clone();
        *corrupt.last_mut().unwrap() ^= 0x01;
        reader.push(&corrupt);
        assert_eq!(reader.next_frame(&expected), None);
        reader.push(&good);
        assert_eq!(
            reader.next_frame(&expected).as_deref(),
            Some(good.as_slice())
        );

        // A device that answers with more bytes than it was asked for is a wrong frame, not a
        // longer one: the length is checked against the request, not trusted.
        let long = ack_frame(200, 0x00, &[0u8; 20]);
        reader.push(&long);
        assert_eq!(reader.next_frame(&expected), None);
        reader.push(&good);
        assert_eq!(
            reader.next_frame(&expected).as_deref(),
            Some(good.as_slice())
        );
    }

    /// The case a length-blind reader cannot survive: our own frame sits in front of the answer,
    /// and the answer must still come out. Without id filtering the reader waits for the
    /// written frame's own length to arrive and swallows the reply behind it.
    #[test]
    fn the_reader_ignores_our_own_instruction_frame() {
        let expected = ExpectedReply::new(&[10], 15);
        let good = ack_frame(10, 0x00, &(0u8..=0x0E).collect::<Vec<_>>());

        let mut reader = FrameReader::new();
        // A torque-enable write to id 10: FF FF 0A 04 03 28 01 C5. Same id as the answer and a
        // length that is plausible on its own.
        reader.push(&write(10, reg::TORQUE_ENABLE, &[1]));
        reader.push(&good);
        assert_eq!(
            reader.next_frame(&expected).as_deref(),
            Some(good.as_slice())
        );
    }

    /// Sign-magnitude is not two's complement, and the two disagree exactly past the
    /// mid-point — 0x8001 is -1 here and -32767 there. Getting it wrong turns a joint moving
    /// one count past zero into a joint reporting a full negative revolution.
    #[test]
    fn position_is_sign_magnitude_not_twos_complement() {
        assert_eq!(position_raw_to_counts(0x0000), 0);
        assert_eq!(position_raw_to_counts(0x0001), 1);
        assert_eq!(position_raw_to_counts(0x8001), -1);
        assert_eq!(position_raw_to_counts(0x7FFF), 32767);
        assert_eq!(position_raw_to_counts(0xFFFF), -32767);
        // A joint wound past one turn reads above 4095 rather than wrapping, which is what the
        // real head_yaw did at raw 6289.
        assert_eq!(position_raw_to_counts(6289), 6289);
    }

    #[test]
    fn position_round_trips_through_radians() {
        for counts in [-32767i32, -4096, -1, 0, 1, 2048, 4096, 32767] {
            let raw = position_counts_to_raw(counts);
            let bytes = raw.to_le_bytes();
            let rad = position_rad(bytes[0], bytes[1]);
            assert_eq!(
                position_raw_to_counts(le_u16(
                    position_from_rad(rad)[0],
                    position_from_rad(rad)[1]
                )),
                counts,
                "counts {counts} did not survive the round trip"
            );
        }
    }

    /// One revolution is 4096 counts and therefore 2π, which fixes every other constant here.
    #[test]
    fn one_revolution_is_two_pi() {
        assert!((RAD_PER_COUNT * POSITION_COUNTS_PER_REV - std::f64::consts::TAU).abs() < 1e-12);
        // The speed unit the reference example gives: 60 * 0.732 rpm = 43.92 rpm.
        assert!((RPM_PER_SPEED_COUNT * 60.0 - 43.92).abs() < 1e-9);
    }

    /// The gain read is one transaction covering two blocks, so its arithmetic is worth
    /// pinning: an off-by-one in `gains_block` reads voltage out of the lock register.
    #[test]
    fn the_gain_block_covers_gains_and_slow_sensors_in_one_span() {
        let base = reg::KP;
        assert_eq!(base + gains_block::KP as u8, reg::KP);
        assert_eq!(base + gains_block::KD as u8, reg::KD);
        assert_eq!(base + gains_block::LOCK as u8, reg::LOCK);
        assert_eq!(base + gains_block::VOLTAGE as u8, reg::PRESENT_VOLTAGE);
        assert_eq!(
            base + gains_block::TEMPERATURE as u8,
            reg::PRESENT_TEMPERATURE
        );
        assert_eq!(gains_block::LEN as u8, reg::PRESENT_TEMPERATURE - base + 1);
    }

    /// Same for the tick's block, where the two-byte fields make an off-by-one plausible.
    #[test]
    fn the_tick_block_offsets_match_their_registers() {
        let base = reg::PRESENT_POSITION_L;
        assert_eq!(base + block::POSITION as u8, reg::PRESENT_POSITION_L);
        assert_eq!(base + block::SPEED as u8, reg::PRESENT_SPEED_L);
        assert_eq!(base + block::LOAD as u8, reg::PRESENT_LOAD_L);
        assert_eq!(base + block::VOLTAGE as u8, reg::PRESENT_VOLTAGE);
        assert_eq!(base + block::TEMPERATURE as u8, reg::PRESENT_TEMPERATURE);
        assert_eq!(base + block::STATUS as u8, reg::PRESENT_STATUS);
        assert_eq!(base + block::CURRENT as u8, reg::PRESENT_CURRENT_L);
        // Fifteen bytes exactly: the last field ends on the last byte of the block.
        assert_eq!(block::LEN, 15);
        assert_eq!(block::CURRENT + 2, block::LEN);
    }

    /// The node's contract block is the DXL control block plus the two bytes DXL has no room
    /// for at 124. If these ever stop being 12 and 13, one `sync_read` can no longer carry
    /// both devices and the whole tick layout changes.
    #[test]
    fn the_imu_nodes_block_is_the_control_block_plus_counter_and_flags() {
        assert_eq!(imu_block::CTRL_LEN, crate::imu::IMU_BLOCK_LEN);
        assert_eq!(imu_block::COUNT, 12);
        assert_eq!(imu_block::STATUS, 13);
        assert_eq!(IMU_FLAG_SFLP_VALID, 0x01);
    }

    #[test]
    fn reboot_is_an_instruction_with_no_address() {
        assert_eq!(reboot(1), vec![0xFF, 0xFF, 0x01, 0x02, 0x08, 0xF4]);
    }
}
