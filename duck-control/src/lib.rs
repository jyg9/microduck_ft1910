//! The robot control core: everything between reading the bus and writing it.
//!
//! Deliberately not a daemon. There is no tokio here, no socket, no systemd — `robotd`
//! owns all of that. The boundary is enforced by the compiler rather than by discipline,
//! which is what stops process concerns leaking into the code that drives motors.
//!
//! The control path it holds — model, bus, [`io::RobotIo`], observations, policy, safety — is
//! designed in `docs/design/robotd-design.md` §2.

pub mod bus;
pub mod fall;
pub mod imu;
pub mod io;
pub mod model;
pub mod obs;
pub mod pickup;
pub mod policy;
pub mod safety;
/// A FeeTech SCS/HLS bus: the 1910 servos and the `imu_to_dxl` node on one UART, instead of
/// the Dynamixel bus [`bus`] speaks. See [`scs::ScsIo`] for why a robot picks one or the other.
pub mod scs;
/// A robot in MuJoCo, over TCP — the backend `robotd-design.md` §9 deferred.
pub mod sim;

pub use imu::ImuData;
pub use io::{FakeIo, IoError, JointTargets, RobotIo, Sensors, SlowSensors};
pub use model::{
    BATTERY_EMPTY_V, BATTERY_FULL_V, DEFAULT_POSITION, JOINT_IDS, JOINT_NAMES, NUM_JOINTS,
    REST_POSITION, battery_percent,
};
pub use obs::{ACTION_LEN, Command, OBS_LEN, Observation};
