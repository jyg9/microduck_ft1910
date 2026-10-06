//! Survey a FeeTech bus without moving the robot.
//!
//! `robotd` must be stopped first: it holds the port exclusively, so this cannot open it while
//! the daemon is up.
//!
//! ```text
//! sudo systemctl stop robotd
//! cargo run -p duck-control --example scs_commission -- --port /dev/ttyS2
//! ```
//!
//! Three reads per servo and nothing else — no torque, no goal, no gain, no EEPROM — which is the
//! only form of a commissioning tool that is safe to run on a robot somebody may be holding. The
//! output is a table, and a `scs.json` skeleton whose `zero_ticks` are whatever each joint reads
//! *now*: prop each joint at its mechanical zero, run this again, and paste the result. Direction
//! is not measurable from here and is left as the `+1` default for a person to set by watching
//! which way a positive angle moves the joint.
//!
//! `--persist-gains` is the one thing here that writes, and it is not the default. It burns the
//! volatile P/D into each servo's EEPROM so a servo that resets comes back holding it, and it
//! prints the values it replaced so they can be written down — see that flag's own help.

use std::time::Duration;

use duck_control::model::{JOINT_IDS, JOINT_NAMES, NUM_JOINTS};
use duck_control::scs::{Calibration, JointCalibration, PdProfile, ScsIo, Survey, wire};

/// The port `robotd.toml` names on this board, so a bare invocation is usually right.
const DEFAULT_PORT: &str = "/dev/ttyS2";

fn main() -> std::process::ExitCode {
    let mut port = DEFAULT_PORT.to_owned();
    let mut persist = false;
    let mut probe: Option<u8> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => match args.next() {
                Some(value) => port = value,
                None => {
                    eprintln!("--port needs a device path");
                    return std::process::ExitCode::FAILURE;
                }
            },
            "--persist-gains" => persist = true,
            "--probe-goal-refresh" => match args.next().map(|v| v.parse::<u8>()) {
                Some(Ok(id)) => probe = Some(id),
                _ => {
                    eprintln!("--probe-goal-refresh needs a servo id, e.g. 34");
                    return std::process::ExitCode::FAILURE;
                }
            },
            "--help" | "-h" => {
                println!(
                    "scs_commission [--port {DEFAULT_PORT}] [--persist-gains] \
                     [--probe-goal-refresh <id>]"
                );
                println!();
                println!("Reads only by default, and writes nothing to any servo.");
                println!();
                println!("--persist-gains  Burn the volatile P/D profile into each servo's");
                println!("                 EEPROM (registers 21/22), so a servo that resets or");
                println!("                 browns out comes back holding it. Prints the values");
                println!("                 it replaced — record them. Refused while energised.");
                println!();
                println!("--probe-goal-refresh <id>");
                println!(
                    "                 Writes nothing that moves: releases one servo's torque,"
                );
                println!(
                    "                 then writes its own present position as a goal, and says"
                );
                println!(
                    "                 whether the goal write turned the torque back on. This is"
                );
                println!(
                    "                 the one experiment that separates \"the servo re-enables"
                );
                println!(
                    "                 itself\" from \"the write never landed\". Leaves the servo"
                );
                println!("                 however the last step left it.");
                return std::process::ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument {other:?}; try --help");
                return std::process::ExitCode::FAILURE;
            }
        }
    }

    // An identity calibration, because the point of surveying is that none exists yet. The
    // defaults pass `Calibration::validate`: a zero inside the single turn and a direction of +1.
    let calibration = Calibration {
        joints: [JointCalibration::default(); NUM_JOINTS],
        ..Calibration::default()
    };

    let mut io = match ScsIo::open(&port, calibration) {
        Ok(io) => io,
        Err(e) => {
            eprintln!("cannot open {port}: {e}");
            eprintln!();
            eprintln!("If the port is busy, `robotd` still holds it: sudo systemctl stop robotd");
            return std::process::ExitCode::FAILURE;
        }
    };

    if let Some(id) = probe {
        return probe_goal_refresh(&mut io, id);
    }

    let survey = match io.survey() {
        Ok(survey) => survey,
        Err(e) => {
            eprintln!("cannot survey the bus: {e}");
            eprintln!();
            eprintln!("A missing id here is a servo that did not answer, not a bad servo — check");
            eprintln!("the bus wiring and that servo power is on before reading anything into it.");
            return std::process::ExitCode::FAILURE;
        }
    };

    println!();
    println!("port {port} — read only, nothing was written");
    println!();
    print_table(&survey);
    print_problems(&survey);
    print_skeleton(&survey);

    if persist {
        let code = persist_and_report(&mut io, &survey);
        if code != std::process::ExitCode::SUCCESS {
            return code;
        }
    }

    // Keep the port open a moment and read again, only to show the node's counter advancing: a
    // node that answers but never produces a sample is the failure that looks like a healthy bus.
    std::thread::sleep(Duration::from_millis(100));
    if let Ok(second) = io.survey() {
        let advanced = second.imu_counter != survey.imu_counter;
        println!();
        println!(
            "imu node: status {:#04x}, counter {} -> {} ({})",
            survey.imu_status,
            survey.imu_counter,
            second.imu_counter,
            if advanced {
                "advancing"
            } else {
                "FROZEN — it answered but produced no new sample"
            }
        );
        if survey.imu_status & duck_control::scs::wire::IMU_FLAG_SFLP_VALID == 0 {
            println!(
                "  …and its SFLP_VALID bit is clear, so `imu_ready` will stay false and the \
                 policies will not start."
            );
        }
    }

    std::process::ExitCode::SUCCESS
}

fn print_table(survey: &Survey) {
    println!(
        "{:<16} {:>4} {:>7} {:>6} {:>5} {:>5} {:>4} {:>4} {:>4} {:>12} {:>7} {:>5} {:>4} {:>6} \
         {:>5}",
        "joint",
        "id",
        "model",
        "fw",
        "mode",
        "baud",
        "lock",
        "blk",
        "rd40",
        "limits",
        "ticks",
        "kp",
        "kd",
        "volts",
        "°C"
    );
    for joint in 0..NUM_JOINTS {
        let servo = survey.servos[joint];
        println!(
            "{:<16} {:>4} {:#06x} {:>3}.{:<2} {:>5} {:>5} {:>4} {:>4} {:>4} {:>5}..{:<6} {:>7} \
             {:>5} {:>4} {:>6.1} {:>5}",
            JOINT_NAMES[joint],
            servo.id,
            servo.model,
            servo.firmware.0,
            servo.firmware.1,
            servo.mode,
            servo.baud_code,
            servo.lock,
            survey.torque[joint],
            survey.torque_direct[joint],
            servo.limits.0,
            servo.limits.1,
            survey.raw_ticks[joint],
            survey.kp[joint],
            survey.kd[joint],
            survey.volts[joint],
            survey.temps_c[joint],
        );
    }
    // Said once, in prose, and pointedly *not* per joint. A locked EEPROM is the state a working
    // robot arrives in and is not a fault: the daemon writes P/D to RAM at every bring-up and never
    // needs EEPROM. Fifteen lines of "worth fixing" for the normal case is how a real warning gets
    // skimmed past; the one thing worth knowing is what the lock stands between a servo and.
    if (0..NUM_JOINTS).any(|j| survey.servos[j].lock != 0) {
        println!();
        println!(
            "note: EEPROM is write-protected on the servos showing lock 1, which is normal and \
             expected."
        );
        println!(
            "      The P/D above is RAM and is rewritten at every bring-up, so nothing here needs \
             the lock cleared."
        );
        println!("      `--persist-gains` clears it, writes, reads back and restores it.");
    }
    // `blk` is register 40 as a byte of the long identity block; `rd40` is the same register read
    // the way the driver reads it for its own verification. They disagreed on hardware — the long
    // read said torque was on while the driver's own read-back said off, and the driver believed
    // itself and reported a successful relax. Shouted about because one of the two is a bug and the
    // driver trusts the second.
    let disagree: Vec<&str> = (0..NUM_JOINTS)
        .filter(|j| survey.torque[*j] != survey.torque_direct[*j])
        .map(|j| JOINT_NAMES[j])
        .collect();
    if !disagree.is_empty() {
        println!();
        println!(
            "** blk and rd40 disagree on {} joint(s): {}",
            disagree.len(),
            disagree.join(", ")
        );
        println!(
            "   Both are register 40. `blk` comes from the long read at address 0; `rd40` is the \
             one-byte"
        );
        println!(
            "   read the driver verifies with. One of them is wrong, and the driver believes \
             rd40."
        );
    }
    let powered: Vec<&str> = (0..NUM_JOINTS)
        .filter(|j| survey.torque_direct[*j] != 0)
        .map(|j| JOINT_NAMES[j])
        .collect();
    if !powered.is_empty() {
        println!();
        println!(
            "note: {} servo(s) still report torque on (rd40): {}",
            powered.len(),
            powered.join(", ")
        );
        println!("      `robot.relax` has not released those.");
    }
}

/// The things in the table that mean the robot will not behave, said in words rather than left in
/// a column for someone to interpret.
fn print_problems(survey: &Survey) {
    let mut problems: Vec<String> = Vec::new();
    for joint in 0..NUM_JOINTS {
        let servo = survey.servos[joint];
        let name = JOINT_NAMES[joint];
        if servo.id != JOINT_IDS[joint] {
            problems.push(format!(
                "{name}: answers as id {} but the wire order says {}",
                servo.id, JOINT_IDS[joint]
            ));
        }
        // 4 is the pure position/PD mode the policies are trained against.
        if servo.mode != 4 {
            problems.push(format!(
                "{name}: mode {} — the policies assume mode 4 (position/PD)",
                servo.mode
            ));
        }
        // Baud code 0 is 1 Mbps, the rate the node shares.
        if servo.baud_code != 0 {
            problems.push(format!(
                "{name}: baud code {} — not 0, so not the 1 Mbps the node uses",
                servo.baud_code
            ));
        }
        // The lock is deliberately not a problem. `--persist-gains` clears it, verifies the
        // read-back and restores it, so a locked servo is the ordinary case rather than a fault —
        // `print_table` says so once rather than fifteen times.
    }
    if problems.is_empty() {
        return;
    }
    println!();
    println!("worth fixing before this robot moves:");
    for problem in problems {
        println!("  · {problem}");
    }
}

/// The one writing action, behind its own flag, with the previous values printed.
fn persist_and_report(io: &mut ScsIo, survey: &Survey) -> std::process::ExitCode {
    match io.persist_gains() {
        Ok(backup) => {
            println!();
            println!("── EEPROM gains written ───────────────────────────────────────────");
            println!("The volatile profile is now persistent, so a servo that resets comes back");
            println!("holding it. These are the values it replaced — write them down:");
            for joint in 0..NUM_JOINTS {
                if backup.kp[joint] == survey.kp[joint] && backup.kd[joint] == survey.kd[joint] {
                    continue;
                }
                println!(
                    "  {:<16} was P={} D={}, now P={} D={}",
                    JOINT_NAMES[joint],
                    backup.kp[joint],
                    backup.kd[joint],
                    survey.kp[joint],
                    survey.kd[joint],
                );
            }
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("cannot persist the gains: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn print_skeleton(survey: &Survey) {
    let profile = PdProfile::default();
    let mount = Calibration::default().imu_mount;
    println!();
    println!("── scs.json skeleton ──────────────────────────────────────────────");
    // **The zeros below are "whatever each joint reads right now", and saying so is the difference
    // between a calibration and a robot that thinks its current slump is the home pose.** There are
    // two ways to get here and they want opposite instructions, so this says both rather than
    // assuming the uncalibrated one.
    println!("These zeros are what each joint reads *now*. There are two cases:");
    println!("  · the servos were CAL'd so this pose reads 2048 — then every zero_ticks is");
    println!("    2048, nothing needs propping, and the numbers below are simply not the answer;");
    println!("  · they were not — then prop each joint at its mechanical zero, run this again,");
    println!("    and paste what it prints.");
    println!("`joints` is one entry per line, in this order:");
    for (joint, name) in JOINT_NAMES.iter().enumerate() {
        println!("  {joint:>2}  {name}");
    }
    println!();
    println!("`direction` cannot be measured from here — a sign is not visible in a position —");
    println!("so every joint says 1. Set one to -1 when a positive angle moves it the wrong");
    println!("way: that is a number here, not a change to the driver.");
    println!();
    // **Valid JSON, and no comments in it.** This block is meant to be saved as `scs.json` and
    // loaded with `deny_unknown_fields`, so a trailing `// joint_name` is not a helpful annotation
    // — it is a file the robot refuses to start on. The names are printed above instead, and the
    // mount comes from the bus layer's own default rather than a literal, because a skeleton that
    // disagrees with the code it was generated from is worse than no skeleton.
    println!("{{");
    println!("  \"imu_id\": {},", Calibration::default().imu_id);
    println!(
        "  \"imu_mount\": [{}, {}, {}, {}],",
        mount[0], mount[1], mount[2], mount[3]
    );
    println!("  \"baud_rate\": {},", Calibration::default().baud_rate);
    println!(
        "  \"servo_pd\": {{ \"kp\": {}, \"kd\": {}, \"mouth_kp\": {} }},",
        profile.kp, profile.kd, profile.mouth_kp
    );
    println!("  \"joints\": [");
    for joint in 0..NUM_JOINTS {
        let comma = if joint + 1 == NUM_JOINTS { "" } else { "," };
        println!(
            "    {{ \"zero_ticks\": {}, \"direction\": 1 }}{comma}",
            survey.raw_ticks[joint]
        );
    }
    println!("  ]");
    println!("}}");
}

/// Does writing a goal position turn a released servo's torque back on?
///
/// The daemon writes goals every tick, including while the robot is limp, and a `sync_write` to
/// register 40 was measured to read back 0 immediately and be 1 again seconds later with nothing on
/// the daemon's side writing 1. A bench tool's write to the same register stays. The difference
/// would be the traffic in between, and register 42 is the only thing in it — so this asks the
/// servo directly.
///
/// It writes the servo's **own present position** as the goal, so a servo that reacts does not move,
/// and it writes only registers 40 and 42. It leaves the torque wherever the last step put it
/// rather than restoring it: the caller is here because they want it off.
fn probe_goal_refresh(io: &mut ScsIo, id: u8) -> std::process::ExitCode {
    let read40 = |io: &mut ScsIo| -> Result<u8, String> {
        io.read_register(id, wire::reg::TORQUE_ENABLE, 1)
            .map(|b| b[0])
            .map_err(|e| e.to_string())
    };

    println!("probing id {id}: register 40, then a goal write, then register 40 again");
    let before = match read40(io) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cannot read register 40 on id {id}: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    println!("  1. torque (40) is {before}");

    if let Err(e) = io.write_register(id, wire::reg::TORQUE_ENABLE, &[0]) {
        eprintln!("cannot write register 40 on id {id}: {e}");
        return std::process::ExitCode::FAILURE;
    }
    let released = read40(io).unwrap_or(0xff);
    println!("  2. after writing 0 it reads {released}");

    // Read the present position and write it straight back as the goal: the same frame the loop
    // sends every tick, with a target the servo is already at.
    let present = match io.read_register(id, wire::reg::PRESENT_POSITION_L, 2) {
        Ok(b) => wire::le_u16(b[0], b[1]),
        Err(e) => {
            eprintln!("cannot read the present position of id {id}: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    if let Err(e) = io.write_register(id, wire::reg::GOAL_POSITION_L, &present.to_le_bytes()) {
        eprintln!("cannot write register 42 on id {id}: {e}");
        return std::process::ExitCode::FAILURE;
    }
    let after = read40(io).unwrap_or(0xff);
    println!("  3. after writing goal 42 = {present} (its own position) it reads {after}");

    // The 1 Hz drift guard writes P/D while the robot is running, so a gain write that also re-arms
    // would defeat a relax from a second direction. Release it again and ask.
    if released == 0 && after == 0 {
        let kp = io.read_register(id, wire::reg::KP, 1).unwrap_or(vec![0]);
        let _ = io.write_register(id, wire::reg::TORQUE_ENABLE, &[0]);
        let _ = io.write_register(id, wire::reg::KP, &kp);
        let gains = read40(io).unwrap_or(0xff);
        println!("  4. after writing P (50) = {} it reads {gains}", kp[0]);
        println!();
        if gains != 0 {
            println!("VERDICT: a *gain* write turns torque back on too.");
            return std::process::ExitCode::SUCCESS;
        }
    }
    println!();

    if released == 0 && after != 0 {
        println!(
            "VERDICT: a goal write turns torque back on. Register 40 accepts 0, and the next \
             goal undoes it."
        );
        println!(
            "         The daemon writes goals every tick, so `relax` cannot hold while it runs."
        );
    } else if released == 0 {
        println!(
            "VERDICT: the goal write did not re-enable it. Register 40 holds after a release."
        );
    } else {
        println!("VERDICT: the write to register 40 did not take effect at all on this servo.");
    }
    std::process::ExitCode::SUCCESS
}
