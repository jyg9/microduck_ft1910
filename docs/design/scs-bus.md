# The FeeTech SCS/HLS bus

`robotd` drives one serial bus. Which protocol that bus speaks is a property of the robot, and
this page owns the FeeTech one. Everything above the bus — the control loop, the model,
observations, policy and safety — is [`robotd-design.md`](robotd-design.md)'s and is unchanged by
this: the backend implements the same `RobotIo` trait, so nothing above it knows which bus it is
on.

## What a robot declares

Two keys in `robotd.toml`, both off by default, so a shipped robot is a Dynamixel one and adding a
second backend changed nothing about it:

```toml
[bus]
scs = true
scs_config = "/etc/robot/scs.json"
```

`scs_config` is a separate file on purpose. It holds fifteen **measured** encoder zeros, each the
reading of a joint standing at its own mechanical zero — a calibration, not a preference, written
by a procedure with a robot hanging in it. `robotd.toml` is what `robotctl configure` edits key by
key, and an editor that offers to retype one of those fifteen numbers is offering to break the
robot. `Params::validate` loads and checks the file, so both the editor and the daemon refuse a bad
one, and it refuses the two keys disagreeing: `scs = true` with no file, or a file with the switch
off (which would leave the Dynamixel backend reading the same port and ignoring it).

The file, `deny_unknown_fields`, every field defaulted:

```jsonc
{
  "imu_id": 200,                    // bus id of the imu_to_dxl node
  "imu_mount": [1, 0, 0, 0],        // sensor→trunk rotation, scalar-first
  "baud_rate": 1000000,
  "servo_pd": { "kp": 6, "kd": 20, "mouth_kp": 10 },
  "joints": [ { "zero_ticks": 2236, "direction": -1 }, /* … 15, in wire order */ ]
}
```

## Why it is a backend and not a register table

The header, the checksum and the reply shape all differ from Dynamixel protocol 2.0: two `0xFF`
bytes rather than `FF FF FD 00`, a one's-complement sum rather than a CRC-16, and a status packet
per device rather than one appended to a broadcast. There is also no fast sync read (`0x8A`), which
is the instruction the Dynamixel backend's whole tick budget is built around.

`duck-control/src/scs/` is therefore split in two. `wire.rs` is the bytes — framing, registers,
conversions — with no port and no clock, which is what lets every vector be checked against the
reference implementations without hardware. `mod.rs` is the port, the clock and the `RobotIo`.

## One transaction per tick, and the node's block is why

Every device answers **fifteen bytes at address 56**. That is not a preference either: `sync_read`
gives every device on the bus one address and one length, the servo documents 56..70, and the
`imu_to_dxl` node answers the same fifteen so that a single transaction can carry both. The node's
first twelve bytes *are* the block `imu.rs` already decodes at Dynamixel address 124 — its FeeTech
personality is that block plus the sample counter and status flags DXL has no room for.

The node is listed **first**, because the protocol has each device answer in the slot its position
in the list gives it, at ≈295 µs each, and a node listed last would answer after fifteen of them.

Reading fifteen bytes also buys something the Dynamixel backend structurally cannot have: its
12-byte read stops eight bytes short of the sample counter, so it can only compare blocks for
equality. Here the counter is at offset 12 and the answer is *did the node produce a new frame*
rather than *are these bytes the same* — two ticks landing inside one 120 Hz node refresh is
ordinary, and the counter says so.

Two budgets follow from the same measurement. Replies are collected **by id**, not in arrival order,
because a device that misses its slot is a device that answers out of order. The burst ends after
`BURST_IDLE` = 4 ms of silence — the longest gap measured between adjacent replies is 1.94 ms, so
2 ms is one scheduler hiccup from truncating a burst, which shows up as the last few ids reported
missing. `BURST_LIMIT` = 12 ms caps a bus that is present but dead, rather than spending the serial
timeout's 30 ms, which is most of a 20 ms control period.

## Energising is ordered, and that order is the point

Measured on this hardware: energising servos whose goal register still holds a stale target drives
every joint at once, and fifteen servos × up to 26° of error inrush tripped the board's undervoltage
lockout — four SBC resets in eight minutes. So `set_torque(true)` writes the pose the robot is
already in, verifies it read back, and only then energises, all in one call rather than as a
convention every caller has to remember. Torque-off is one broadcast write and a readback, and a
servo that will not release is an error rather than a silent `relax`.

`reboot` does not wait: the servo answers nothing and is back after ≈823 ms, and waiting would stall
the loop for most of a second. It does mark the gain profile unverified, which is the next section.

## The volatile P/D, and why it is a runtime concern

A servo's position gains live in two places. Registers 21/22 are EEPROM and survive power loss;
registers 50/51 are RAM and are **reloaded from 21/22 by a reboot** — measured: writing 50/51 =
99/88 and issuing `0x08` brings 32/40 back, which is what 21/22 holds.

So a robot whose P/D matters has to write them on every bring-up, and has to notice a single servo
that has reset itself out from under the other fourteen. Three things do that:

- `set_torque(true)` writes the profile before it energises, and refuses to enable if it does not
  read back.
- `slow_sensors` — one transaction a second, at registers 50..63, which covers the gains *and* pack
  voltage *and* case temperatures between them — compares all fifteen against the profile and
  rewrites any that drifted, loudly.
- `reboot` clears the verified flag, so the next bring-up rewrites rather than assuming.

Writing **EEPROM is never automatic.** A wrong persistent gain is a robot that lurches on every
boot from then on, with nothing in the daemon's own configuration to point at, and an interrupted
EEPROM write is the one operation on this bus that can leave a servo unusable — neither belongs on
a path that runs unattended. The periodic guard above makes that choice cheap: a reset servo is
repaired within a second instead of walking on the wrong stiffness.

It is available, explicitly, as `scs_commission --persist-gains` (`ScsIo::persist_gains`). Four
steps that only work together: read the locks, clear them, write 21/22, read back, restore each
lock. The readback is not ceremony — a *locked* servo accepts the write into RAM and silently does
not persist it, so without it a call that did nothing at all reports success. It refuses while the
servos are energised, and it returns the values it replaced, because a setting that outlives the
robot's configuration is not recoverable from the robot.

The policy's `gain` (200 nominal, 50 limp) is an XL330 figure and is **scaled** onto this profile
rather than written through: asking a FeeTech servo for a P of 200 is asking for twenty times its
factory value.

`Safety::apply` passes that gain on **every tick**, and the control loop tapers it deliberately —
the standing policy runs softer than the walking one, limp-fall softer still. So `set_gain` applies
a changed gain in either direction and costs nothing when the gain is unchanged. Both halves matter:
refusing a stiffening would leave the robot stuck at whatever it last went limp to, overriding a
decision that is the control loop's; and rewriting the profile per tick would spend a fifth of the
20 ms period restating what is already true. What keeps a stiffening safe is not a refusal but the
bring-up order above — `set_torque` adopts the pose it finds, so the error term starts at zero.

**`servo_pd` and the simulator's `kp_fw` must be changed together.** 6/20 is where two independent
forks of this codebase that both walk landed, against a factory 32/40 and a BAM actuator default of
32. That is evidence and not proof: it has not been measured on this robot.

## What is verified and what is not

Verified without hardware — 33 tests over the two files, including byte-exact vectors shared with
`soft_imu_to_dxl/v1/host/bus.py` and the node's own C tests:

- the framing, the checksum, sign-magnitude position, the unit constants, the block offsets;
- the reader surviving an echoed instruction frame, noise, a short frame, a wrong length, a corrupt
  checksum;
- replies out of order; a silent servo named while the rest are read; an alarm reported without
  failing the tick; a repeated sample counter read as stale;
- the whole bring-up order; a partial enable refused; a servo that will not release; a reboot
  forcing the profile to be rewritten; a drifted profile repaired once and once only; a gain
  change applied in both directions and an unchanged one costing no traffic; the servo's own travel
  window read and then honoured.

**First light on the robot, 2026-10-06**: `0.16.0-dev.local.1791301647.gc2b0a21` on the zero3
(RK3566) board, `/dev/ttyS2` at 1 Mbaud. The bus opened on the first attempt — all fifteen servos
*and* the node at id 200 answered the first transaction, with no missing replies, no alarms and no
stale reads — and the loop then held `50.0 of 50.0 Hz`, `0 missed` of 436 ticks, `bus ok`,
`imu ready`, 8.22 V and 33 °C hottest servo. That is the framing, the checksum, the 21-byte block
at 56, the gains block and the node's shared 15 bytes all working against this firmware at once,
which is what those byte-exact unit vectors exist to make possible without a bus.

What first light does **not** prove, and what is still owed:

- **The numbers are still firmware 3.46's.** The ±295 µs slot, the 1.94 ms worst adjacent-reply gap,
  the 823 ms reboot and the EEPROM reload are why the idle/limit budgets and the stale window are
  what they are. Nothing here re-measured them; the 50 Hz with zero misses is consistent with them
  and is not a measurement of them.
- **The zeros are borrowed until somebody props each joint up.** They come from the CAL done by
  hand before this backend existed, and `scs_commission` is how they get re-read.
- **`direction` is inherited for eleven joints.** See below.
- **Nothing has been commanded to move.** Bring-up, holding, the gain profile and the guards have
  all run; a policy has not driven this bus yet.

`scs_commission` on the same bus then confirmed the parts first light cannot see:

- **The profile is really in RAM on all fifteen**: `kp` 6, `kd` 20, and 10 on the mouth — the
  per-joint scaling is not a comment, and the mouth's own P is the one joint that differs.
- **2048 is the zero, cross-checked.** The tool reads the raw register and the daemon reads the
  converted joint, so the two can be compared: every joint agreed to about 1 %, with the sign
  flipped, which is `direction = -1` and `zero_ticks = 2048` confirmed by two independent paths.
  (The 1 % is real and expected — `systemctl stop robotd` leaves the joints carrying their own
  weight with nothing updating the loop, so they sag slightly between the two readings.)
- **The travel windows are `0..4095` on every servo**, so the clamp into "the servo's own window"
  never bites on this robot. It is not wrong and it costs nothing, but it is also not the
  protection it was written to be: someone had never set MIN/MAX angle limits here. The guards that
  do work are the calibration, the zero-inside-the-window refusal and upstream's own ±π travel.
- **EEPROM is locked (register 55 = 1) on all fifteen**, which is the expected state and not a
  fault: the profile is RAM and is rewritten at every bring-up. `--persist-gains` is what needs the
  lock, and it clears and restores it.
- **The node's counter advances** (44 → 56, then 221 → 233 across runs) with status `0x81`, so the
  shared block carries fresh samples and the `SFLP_VALID` bit `imu_ready` waits on is set.

### `relax` could not hold, and why

`robot.relax` reported success four times and the robot stayed stiff. Three separate mistakes were
involved, and the last one is a property of the servos rather than of this code:

1. **`sync_write` to register 40 does not do what a `sync_write` to 50 does.** It reads back 0
   immediately — which is what made the old verification report a release — and the servo is holding
   again seconds later. Register 40 is written per servo with an acknowledged `WRITE` now, which also
   means a refusal arrives as a status byte instead of as an inference. Fifteen round trips, on
   bring-up and `relax` only.
2. **A released servo re-arms when it is sent a goal.** `scs_commission --probe-goal-refresh` asks
   this directly: release the servo, write its own present position back as the goal, read the torque
   register. It answers *on* — intermittently, and on different joints across runs, which is the
   shape of the race it is: the goal is read from the servo, then the servo is released, and by the
   time the goal lands the joint has sagged, so the goal is no longer where the servo is and the
   servo re-engages to close the gap. **Gain writes do not do this** (probed the same way, four
   joints, no re-arm), so the drift guard needs no equivalent. The control loop writes goals every
   tick, so this alone made a release impossible to keep; `ScsIo::write` now returns without sending
   anything while the bus is not energised, and the bring-up path has its own entry point because it
   legitimately writes the pose the robot is already in *before* torque goes on.
3. **The lock was never involved**, though it was suspected first. Firmware 3.46's own memory table
   says register 55 decides only whether an EEPROM write *survives power-down* — a locked servo
   accepts the write either way. That is also why the read-back in [`persist_gains`] is weaker
   evidence than it looks: a locked servo reads the new value back out of RAM and loses it later.
   Clearing the lock is what makes it persist, which is what the code does.

The lesson worth keeping is about the shape of the bug, not the register: **an unacknowledged write
plus a read-back that can catch a servo mid-update will report a success that did not happen.** The
read-back was correct and so was the write; what was missing was any evidence about the *next* few
milliseconds.

## Commissioning

`duck-control/examples/scs_commission.rs` is the bench procedure, and it is deliberately the only
one: it issues three reads per servo and **cannot write**, which is the one property that makes a
commissioning tool safe to run on a robot somebody is holding. It prints each servo's firmware,
mode, baud code, lock register and travel window; the raw encoder count each joint reads right now;
the volatile gains, so the profile can be seen to have landed rather than believed to have; and a
`scs.json` skeleton with those counts as candidate `zero_ticks`. Prop each joint at its mechanical
zero, run it again, paste.

```bash
sudo systemctl stop robotd          # it holds the port exclusively
cargo run -p duck-control --example scs_commission -- --port /dev/ttyS2
```

`direction` is the one field it cannot measure — a sign is not visible in a position — so it is
left at `+1` for a person to set by watching which way a positive angle moves the joint.

`deploy/scs.example.json` is this robot's, from the hand measurements recorded in the FeeTech
port's `docs/robot/joint-direction-check.md` (2026-10-03): `zero_ticks` is 2048 everywhere because
every servo was CAL'd until it read 2048 at its rest pose, and `direction` is a single `-1` because
three joints — `left_hip_pitch`, `left_ankle`, `right_ankle` — were turned by hand and the two
ankles passed the mirror check that says one global sign is enough. **Eleven joints were never
watched**, and that page flags `left_knee` first, its model axis being −Y where the hip and ankle
are +Y. The config takes a per-joint sign, so a joint that turns out inverted is one number here
rather than a constant in the driver.
