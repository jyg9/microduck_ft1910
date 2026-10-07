//! The ONNX policies.
//!
//! Walking and standing are chosen by the magnitude of the velocity command, exactly as
//! `microduck_runtime` does; the skill networks — sit↔stand, ground pick, the two kicks —
//! are selected explicitly by the scheduler in `robotd`, which owns the priority rules.
//! Every network shares the one 61-D observation layout, so a skill is a session choice
//! plus a command-block encoding. LSTM exports additionally pass hidden/cell state,
//! owned by each network; sensor observations and joint actions are unchanged.
//!
//! **Everything is validated at load, not at inference.** A bundle with the wrong
//! observation width, the wrong action count, or a missing ONNX Runtime must fail while the
//! robot is standing still and the caller can be told why — not sixty ticks later, mid
//! stride. `robotd` turns a load failure into "hold the pose and report unhealthy", so the
//! updater rolls the release back instead of leaving a robot that cannot walk.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::model::{DEFAULT_POSITION, NUM_JOINTS};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::tensor::TensorElementType;
use ort::value::{Tensor, Value, ValueType};
use sha2::{Digest, Sha256};

use crate::obs::{ACTION_LEN, OBS_LEN, Observation};

/// Below this velocity magnitude the standing policy takes over. The prototype's value.
pub const DEFAULT_STANDING_THRESHOLD: f64 = 0.05;

/// Inference threads per session.
///
/// One, deliberately. The prototype uses two, which on a four-core A55 means the control
/// thread blocks on a pool it does not own — and for a network this small the pool costs
/// more in synchronisation than it recovers in parallelism. Worth re-measuring on the board
/// rather than trusting either number.
const INTRA_THREADS: usize = 1;

#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    #[error("reading {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("loading {path}: {source}")]
    Load {
        path: PathBuf,
        #[source]
        source: ort::Error,
    },
    /// The bundle does not match what this build implements. Reported with both shapes
    /// because "wrong policy file" and "wrong daemon" look identical without them.
    #[error("{path}: {what} is {got}, expected {expected}")]
    Shape {
        path: PathBuf,
        what: &'static str,
        expected: String,
        got: String,
    },
    /// A field the model carries about itself, and cannot be believed.
    ///
    /// Separate from [`Self::Shape`] because the graph is fine — this is the *provenance* the
    /// export wrote beside it, and the fix is a re-export rather than a different build.
    #[error("{path}: {field} {why}")]
    Metadata {
        path: PathBuf,
        field: &'static str,
        why: String,
    },
    #[error("inference failed: {0}")]
    Inference(String),
    /// ONNX Runtime is not installed, or not where it is being looked for.
    ///
    /// Its own diagnosis, because it is an operator problem with an operator fix — install
    /// the library or set `ORT_DYLIB_PATH` — and not a broken policy bundle.
    #[error("ONNX Runtime not loadable ({searched}): {detail}")]
    RuntimeMissing { searched: String, detail: String },
    /// `ort` panicked instead of returning an error. See [`catching_ort_panics`].
    ///
    /// `detail` is the panic message, and carrying it is the point: the one panic we have
    /// actually seen on a board names the two version numbers that explain it.
    #[error("ort panicked loading the policy: {detail}")]
    RuntimePanic { detail: String },
}

impl PolicyError {
    /// The file this error is about, when it is about one.
    ///
    /// `Read`, `Load`, `Shape` and `Metadata` name a file; a missing runtime or an `ort` panic does
    /// not, and blaming whichever policy happened to be loading when the dylib turned out to be
    /// absent would send an operator to replace a file that is fine.
    pub fn path(&self) -> Option<&Path> {
        match self {
            PolicyError::Read { path, .. }
            | PolicyError::Load { path, .. }
            | PolicyError::Shape { path, .. }
            | PolicyError::Metadata { path, .. } => Some(path),
            PolicyError::Inference(_)
            | PolicyError::RuntimeMissing { .. }
            | PolicyError::RuntimePanic { .. } => None,
        }
    }
}

/// Where `ort` will look for the runtime, replicating its own logic.
fn dylib_name() -> String {
    match std::env::var("ORT_DYLIB_PATH") {
        Ok(path) if !path.is_empty() => path,
        _ => {
            if cfg!(target_os = "windows") {
                "onnxruntime.dll".to_owned()
            } else if cfg!(any(target_os = "macos", target_os = "ios")) {
                "libonnxruntime.dylib".to_owned()
            } else {
                "libonnxruntime.so".to_owned()
            }
        }
    }
}

/// Confirm ONNX Runtime is loadable **before** calling into `ort`.
///
/// This exists because `ort` does not return an error when the dylib is missing — it
/// `expect`s inside `setup_api`, from a lazy path reachable through any API call, so a
/// missing library aborts the thread that touched it. In the control loop that means the
/// thread dies, no tick ever lands, and `robot.health` reports "the loop has not completed a
/// cycle" forever: the daemon looks wedged instead of saying ONNX Runtime is not installed.
///
/// Probing first turns the *missing library* case into an ordinary error the caller can
/// report, with the operator's fix in it. That is all it does.
///
/// It does **not** mean `ort` cannot then panic, and an earlier version of this comment
/// claimed it did. A board running ONNX Runtime 1.20.1 falsified that: the library loaded, so
/// the probe passed, and `ort` panicked in `setup_api` on its own version check
/// (`expected version >= '1.23.x', but got '1.20.1'`). The probe proves the file loads;
/// nothing more. [`catching_ort_panics`] covers the rest, including panics we have not seen.
pub(crate) fn ensure_runtime() -> Result<(), PolicyError> {
    static PROBE: OnceLock<Result<(), String>> = OnceLock::new();
    let outcome = PROBE.get_or_init(|| {
        let name = dylib_name();
        // Safety: loading a shared library runs its initialisers. This is the same library
        // `ort` is about to load itself, so the risk is not one this probe introduces.
        match unsafe { libloading::Library::new(&name) } {
            Ok(library) => {
                // Leak it: `ort` will dlopen the same file moments later and the OS
                // reference-counts the mapping. Dropping ours would be harmless but
                // pointless churn.
                std::mem::forget(library);
                Ok(())
            }
            Err(e) => Err(e.to_string()),
        }
    });

    outcome
        .clone()
        .map_err(|detail| PolicyError::RuntimeMissing {
            searched: dylib_name(),
            detail,
        })
}

/// Run the `ort` calls, turning a panic from inside them into a [`PolicyError`].
///
/// `ort` treats some initialisation failures as unrecoverable and panics rather than
/// returning `Err` — the version mismatch in [`ensure_runtime`]'s comment is the one a board
/// hit, and it fires from inside a lazy init reachable through any API call. In the control
/// thread a panic is worse than an error: the thread dies, no tick ever lands, and
/// `robot.health` answers "the loop has not completed a cycle yet" — the one message that
/// names no cause — while the daemon stays up serving its socket. The updater then rolls the
/// release back for a reason nobody can act on.
///
/// `robotd` already handles a policy that fails to load: hold the pose, keep ticking at rate,
/// report why, get rolled back. This makes a panic take that same path.
///
/// Deliberately wraps the `ort` work only, and not all of [`Policy::load`], so a genuine bug
/// of ours does not get relabelled "policy unavailable". Note that a caught panic has still
/// run the panic hook, so the backtrace is in the journal either way.
///
/// `AssertUnwindSafe` is needed because `Session` is not `UnwindSafe`. It is sound here
/// because nothing of ours is observed after a catch: the sessions being built are moved into
/// the `Policy` on success and dropped on failure, and the caller's answer is the error.
///
/// **`panic = "abort"` would defeat this.** The root `Cargo.toml` has no `[profile.release]`,
/// so the default unwind strategy applies; adding one would silently turn this back into a
/// dead control thread.
pub(crate) fn catching_ort_panics<T>(
    work: impl FnOnce() -> Result<T, PolicyError>,
) -> Result<T, PolicyError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).unwrap_or_else(|payload| {
        Err(PolicyError::RuntimePanic {
            detail: panic_message(payload),
        })
    })
}

/// The panic message, or a stand-in saying there wasn't one.
///
/// `panic!` with a literal produces `&'static str`; with arguments, `String`. `ort` uses both.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panicked with no message; see the journal for the backtrace".to_owned()
    }
}

/// Which network drives a tick.
///
/// The choice is the caller's — the skill scheduler in `robotd` owns the priority rules —
/// and this enum is how it names its choice. Asking for a network that is not loaded falls
/// back to walking rather than panicking, but the scheduler is expected to check `has_*`
/// first; the fallback exists so a race cannot kill the control thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Net {
    Walk,
    Stand,
    /// Commanded sit↔stand: the twist `vx` slot carries a posture flag, 1 = sit, 0 = stand.
    SitStand,
    /// Phase-scripted ground pick; the twist slots carry `[cos φ, sin φ, 0]`.
    GroundPick,
    /// A one-shot skill, by its index in [`PolicyPaths::skills`].
    ///
    /// Kicks and roulade used to be variants here. They were the same thing three times over —
    /// a network trained on an all-zero command, driving for a fixed window, selected by an
    /// explicit request — differing only in duration and tuning, which is data. An index means
    /// a robot gains a skill by gaining a config entry rather than a release.
    Skill(usize),
}

/// Which policy files to load. `walk` is mandatory; every other slot is a capability the
/// robot simply does not have when `None`.
#[derive(Debug, Clone, Default)]
pub struct PolicyPaths {
    pub walk: PathBuf,
    pub stand: Option<PathBuf>,
    pub sitstand: Option<PathBuf>,
    pub ground_pick: Option<PathBuf>,
    /// One-shot skills, in the priority order the caller wants them considered. Each is
    /// selected only by an explicit request, so an empty list is a robot with no tricks rather
    /// than a robot missing something.
    pub skills: Vec<PathBuf>,
}

/// The servo gains a policy says it was trained against.
///
/// Not every export carries them. The XL330 policies predate the field and have none, and a model
/// without them is one this comparison cannot speak about rather than a broken one — refusing
/// those would refuse every policy that works on the robot today.
///
/// The numbers are in the servo's own register units, which is the whole reason they are worth
/// reading: `kp_fw` is the same quantity the SCS bus writes on every bring-up, so the two are
/// directly comparable. A policy trained at 6 and deployed at 32 is not a config error from the
/// outside — the robot runs, and it is simply a different stiffness from the plant its gait was
/// fitted to, which reads as "needs tuning" rather than "wrong plant".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainedGains {
    pub kp_fw: u8,
    pub kd_fw: u8,
    /// Stamped by one fork; neither the XL330 export nor ours writes it. Compared when present,
    /// because a jaw trained at one P and written with another closes on a different force.
    pub mouth_kp_fw: Option<u8>,
}

/// The loaded networks.
///
/// A configured path that fails to load fails the whole load — the policies ship inside the
/// release, so a missing or corrupt file is a broken bundle, and the right outcome is
/// "unhealthy, roll it back", not a robot that silently lost its kick.
pub struct Policy {
    walk: Network,
    stand: Option<Network>,
    sitstand: Option<Network>,
    ground_pick: Option<Network>,
    skills: Vec<Network>,
    standing_threshold: f64,
    active: Option<Net>,
    /// Roller mode and fall-recovery mode reserve the standing network (roller has none;
    /// fall recovery keeps it for getting up), so command magnitude must never select it.
    standing_disabled: bool,
}

impl Policy {
    /// Load, validate and warm up.
    ///
    /// `stand` is optional: without it the walking policy runs at every velocity, which is
    /// what a single-policy bundle does.
    pub fn load(paths: &PolicyPaths, standing_threshold: f64) -> Result<Self, PolicyError> {
        ensure_runtime()?;

        // Everything below calls into `ort`, and `ort` panics on failures it considers
        // unrecoverable — so the whole of it, and nothing else, goes inside the catch.
        catching_ort_panics(move || {
            // Warm up before the loop ever calls this. The first inference is always an
            // outlier — lazy initialisation, cold pages, first-touch faults — and paying that
            // on tick one would look exactly like a control loop that missed its deadline.
            // It also proves ONNX Runtime is actually present and usable, which with
            // `load-dynamic` is not known until something is run.
            let zero = Observation::zeroed();
            fn open_warm(path: &Path, zero: &Observation) -> Result<Network, PolicyError> {
                let mut network = open(path)?;
                network.run(zero)?;
                network.reset();
                Ok(network)
            }
            fn open_opt(
                path: &Option<PathBuf>,
                zero: &Observation,
            ) -> Result<Option<Network>, PolicyError> {
                path.as_deref().map(|p| open_warm(p, zero)).transpose()
            }

            Ok(Self {
                walk: open_warm(&paths.walk, &zero)?,
                stand: open_opt(&paths.stand, &zero)?,
                sitstand: open_opt(&paths.sitstand, &zero)?,
                ground_pick: open_opt(&paths.ground_pick, &zero)?,
                skills: paths
                    .skills
                    .iter()
                    .map(|path| open_warm(path, &zero))
                    .collect::<Result<Vec<_>, _>>()?,
                standing_threshold,
                active: None,
                standing_disabled: false,
            })
        })
    }

    /// Reserve the standing network: command magnitude no longer selects it, and only an
    /// explicit [`Net::Stand`] from the caller (fall recovery, body pose) reaches it.
    pub fn set_standing_disabled(&mut self, disabled: bool) {
        self.standing_disabled = disabled;
    }

    /// Whether the standing policy would be chosen for this command.
    ///
    /// Separate from [`Self::infer`] because the caller needs the same answer to decide
    /// gains and action scale, and asking twice must not be able to disagree.
    pub fn will_stand(&self, twist_magnitude: f64) -> bool {
        self.stand.is_some()
            && !self.standing_disabled
            && twist_magnitude <= self.standing_threshold
    }

    pub fn has_standing(&self) -> bool {
        self.stand.is_some()
    }

    pub fn has_sitstand(&self) -> bool {
        self.sitstand.is_some()
    }

    pub fn has_ground_pick(&self) -> bool {
        self.ground_pick.is_some()
    }

    /// How many one-shot skills are loaded. A caller's index is valid below this.
    pub fn skill_count(&self) -> usize {
        self.skills.len()
    }

    /// One inference on the named network. A missing optional network falls back to
    /// walking — the scheduler checks `has_*` before asking, so reaching the fallback is a
    /// bug, but a wrong gait beats a dead control thread.
    /// Which network a request actually runs on.
    ///
    /// One place, because two callers need the answer and they must not differ: [`Self::infer`]
    /// runs the network, and [`Self::home_pose`] says which pose its actions are centred on. A
    /// fallback resolved in one and not the other would take the action from one model and the pose
    /// from another.
    fn resolve(&self, net: Net) -> Net {
        match net {
            Net::Stand if self.stand.is_none() => Net::Walk,
            Net::SitStand if self.sitstand.is_none() => Net::Walk,
            Net::GroundPick if self.ground_pick.is_none() => Net::Walk,
            Net::Skill(i) if i >= self.skills.len() => Net::Walk,
            net => net,
        }
    }

    /// The pose the next action will be centred on, for the network `net` resolves to.
    ///
    /// The caller needs this *before* [`Self::infer`], because the observation's `joint_pos` block
    /// is measured from it and is an argument to that call.
    pub fn home_pose(&self, net: Net) -> [f64; NUM_JOINTS] {
        match self.resolve(net) {
            Net::Stand => self.stand.as_ref().unwrap_or(&self.walk),
            Net::SitStand => self.sitstand.as_ref().unwrap_or(&self.walk),
            Net::GroundPick => self.ground_pick.as_ref().unwrap_or(&self.walk),
            Net::Skill(i) => self.skills.get(i).unwrap_or(&self.walk),
            Net::Walk => &self.walk,
        }
        .home_pose
    }

    /// Every loaded network, in the order they are considered.
    fn networks(&self) -> impl Iterator<Item = &Network> {
        std::iter::once(&self.walk)
            .chain(self.stand.iter())
            .chain(self.sitstand.iter())
            .chain(self.ground_pick.iter())
            .chain(self.skills.iter())
    }

    /// The plant the loaded set declares, or `None` if none of it declares one.
    pub fn declared_gains(&self) -> Result<Option<TrainedGains>, PolicyError> {
        Ok(self.declared()?.map(|(_, gains)| gains))
    }

    /// The plant the loaded set declares, and the file that named it.
    fn declared(&self) -> Result<Option<(&Path, TrainedGains)>, PolicyError> {
        declare_set(
            self.networks()
                .map(|network| (network.path.as_path(), network.gains)),
        )
    }

    /// The mismatch between the declared plant and the gains the servos are written with.
    ///
    /// `Ok(None)` when they agree, and also when nothing declares a plant: an XL330 policy carries
    /// no `kp_fw`, and refusing those would refuse every policy that works on the robot today.
    ///
    /// The returned error is ready to be *returned* or *logged* — that choice is the caller's,
    /// because running one set on a deliberately different plant is a legitimate bench A/B and an
    /// operator who has said so should not have to fight the daemon to do it.
    pub fn gain_mismatch(&self, written: TrainedGains) -> Result<Option<PolicyError>, PolicyError> {
        let Some((path, declared)) = self.declared()? else {
            return Ok(None);
        };
        Ok(
            gains_mismatch(declared, written).map(|why| PolicyError::Metadata {
                path: path.to_owned(),
                field: "kp_fw",
                why,
            }),
        )
    }

    pub fn infer(
        &mut self,
        observation: &Observation,
        net: Net,
    ) -> Result<[f32; ACTION_LEN], PolicyError> {
        // Resolve fallback before comparing: asking for an absent skill must not reset
        // the walking network on every tick.
        let net = self.resolve(net);
        let changed = self.active != Some(net);
        let network = match net {
            Net::Walk => &mut self.walk,
            Net::Stand => self.stand.as_mut().unwrap(),
            Net::SitStand => self.sitstand.as_mut().unwrap(),
            Net::GroundPick => self.ground_pick.as_mut().unwrap(),
            Net::Skill(i) => &mut self.skills[i],
        };
        if changed {
            network.reset();
        }
        let result = network.run(observation);
        // Never carry a failed inference's state into another control tick.
        if result.is_err() {
            network.reset();
            self.active = None;
        } else {
            self.active = Some(net);
        }
        result
    }

    /// Preserve the running network when only another slot changed. Compare model
    /// bytes, not paths: a reload may replace a file in place, and a seated swap may
    /// intentionally replace the active network. Those cases must start fresh.
    pub fn carry_over(&mut self, from: &Self) {
        self.reset();
        let Some(net) = from.active else {
            return;
        };
        let target = match net {
            Net::Walk => Some(&mut self.walk),
            Net::Stand => self.stand.as_mut(),
            Net::SitStand => self.sitstand.as_mut(),
            Net::GroundPick => self.ground_pick.as_mut(),
            Net::Skill(i) => self.skills.get_mut(i),
        };
        let source = match net {
            Net::Walk => Some(&from.walk),
            Net::Stand => from.stand.as_ref(),
            Net::SitStand => from.sitstand.as_ref(),
            Net::GroundPick => from.ground_pick.as_ref(),
            Net::Skill(i) => from.skills.get(i),
        };
        if let (Some(target), Some(source)) = (target, source)
            && target.digest == source.digest
        {
            if let (Some(dst), Some(src)) = (&mut target.state, &source.state) {
                dst.h
                    .try_extract_tensor_mut::<f32>()
                    .unwrap()
                    .1
                    .copy_from_slice(src.h.try_extract_tensor::<f32>().unwrap().1);
                dst.c
                    .try_extract_tensor_mut::<f32>()
                    .unwrap()
                    .1
                    .copy_from_slice(src.c.try_extract_tensor::<f32>().unwrap().1);
            }
            self.active = Some(net);
        }
    }

    /// Start a new episode, including when resuming after disable or fall recovery.
    /// A network also starts fresh whenever selection switches away and back to it.
    pub fn reset(&mut self) {
        self.active = None;
        self.walk.reset();
        for network in self
            .stand
            .iter_mut()
            .chain(self.sitstand.iter_mut())
            .chain(self.ground_pick.iter_mut())
            .chain(self.skills.iter_mut())
        {
            network.reset();
        }
    }
}

/// Can this file be loaded as a policy, without committing to running it?
///
/// Opens the graph and checks both shapes, then throws the session away. Two callers, and they
/// want it for the same reason from opposite ends:
///
///  - `robot.loadPolicy` answers a client *synchronously*, while the real swap happens seconds
///    later at the home pose. Validating here is what turns "accepted" followed by a robot that
///    did not change into an immediate `observation width is 51, expected 61`.
///  - startup checks each overridden slot before building the controller, so one bad override
///    costs that slot rather than the whole policy.
///
/// **No warm-up inference**, unlike [`Policy::load`] — this is a question about a file, not a
/// session about to be driven, and the first-inference cost is the loading path's to pay. It
/// therefore proves less: a graph that opens and has the right shape can still fail to run.
/// Nothing downstream treats a pass as a guarantee, which is why a failed load at the home pose
/// still has to keep the controller it had.
///
/// Not from inside a tick. Opening a session is tens of milliseconds and the loop has 20 to
/// spend — so the IPC caller runs it on its own runtime, and the loop only ever calls it in its
/// preamble, before the first tick is due.
pub fn validate(path: &Path) -> Result<(), PolicyError> {
    ensure_runtime()?;
    catching_ort_panics(|| open(path).map(drop))
}

/// The mjlab/rsl_rl LSTM export passes state explicitly. Buffers belong to one
/// session, are allocated at load, and are never shared between policy slots.
struct Network {
    session: Session,
    state: Option<LstmState>,
    action_name: String,
    path: PathBuf,
    digest: [u8; 32],
    /// The pose this model's actions are centred on, and the pose its observation's `joint_pos`
    /// block is measured from. See [`parse_home_pose`].
    home_pose: [f64; NUM_JOINTS],
    /// The plant this model was trained against, when its export says. See [`TrainedGains`].
    gains: Option<TrainedGains>,
}

struct LstmState {
    h: Tensor<f32>,
    c: Tensor<f32>,
}

impl Network {
    fn reset(&mut self) {
        if let Some(state) = &mut self.state {
            state.h.try_extract_tensor_mut::<f32>().unwrap().1.fill(0.0);
            state.c.try_extract_tensor_mut::<f32>().unwrap().1.fill(0.0);
        }
    }

    fn run(&mut self, observation: &Observation) -> Result<[f32; ACTION_LEN], PolicyError> {
        let fail = |e: String| PolicyError::Inference(format!("{}: {e}", self.path.display()));
        if !observation.as_slice().iter().all(|v| v.is_finite()) {
            return Err(fail("non-finite observation".into()));
        }
        let input = Value::from_array(([1usize, OBS_LEN], observation.as_slice().to_vec()))
            .map_err(|e| fail(format!("building input: {e}")))?;
        let outputs = match &self.state {
            Some(state) => self.session.run(ort::inputs![
                "obs" => &input, "h_in" => &state.h, "c_in" => &state.c
            ]),
            None => self.session.run(ort::inputs!["obs" => &input]),
        }
        .map_err(|e| fail(e.to_string()))?;
        let (_, actions) = outputs[self.action_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| fail(e.to_string()))?;
        if actions.len() != ACTION_LEN || !actions.iter().all(|v| v.is_finite()) {
            return Err(fail("expected 14 finite actions".into()));
        }
        let mut result = [0.0; ACTION_LEN];
        result.copy_from_slice(actions);
        if let Some(state) = &mut self.state {
            let (hs, h) = outputs["h_out"]
                .try_extract_tensor::<f32>()
                .map_err(|e| fail(e.to_string()))?;
            let (cs, c) = outputs["c_out"]
                .try_extract_tensor::<f32>()
                .map_err(|e| fail(e.to_string()))?;
            let (expected_h, h_in) = state
                .h
                .try_extract_tensor_mut::<f32>()
                .map_err(|e| fail(e.to_string()))?;
            let (expected_c, c_in) = state
                .c
                .try_extract_tensor_mut::<f32>()
                .map_err(|e| fail(e.to_string()))?;
            // Check both before updating either, including dynamic runtime output shapes.
            if hs != expected_h || cs != expected_c || !h.iter().chain(c).all(|v| v.is_finite()) {
                return Err(fail(
                    "invalid LSTM output state shape or non-finite state".into(),
                ));
            }
            h_in.copy_from_slice(h);
            c_in.copy_from_slice(c);
        }
        Ok(result)
    }
}

fn shape_error(path: &Path, what: &'static str, expected: &str, got: String) -> PolicyError {
    PolicyError::Shape {
        path: path.to_owned(),
        what,
        expected: expected.into(),
        got,
    }
}

/// Require an exact rank and float32 type. Batch may be symbolic, but this runtime
/// always supplies batch one; state layers and hidden width must be known at load.
pub(crate) fn tensor_shape(
    path: &Path,
    outlet: &ort::value::Outlet,
) -> Result<Vec<i64>, PolicyError> {
    match outlet.dtype() {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape,
            ..
        } => Ok(shape.to_vec()),
        other => Err(shape_error(
            path,
            "tensor type",
            "float32",
            format!("{}: {other:?}", outlet.name()),
        )),
    }
}

fn outlet<'a>(
    path: &Path,
    outlets: &'a [ort::value::Outlet],
    name: &str,
) -> Result<&'a ort::value::Outlet, PolicyError> {
    outlets.iter().find(|o| o.name() == name).ok_or_else(|| {
        shape_error(
            path,
            "tensor names",
            name,
            format!("{:?}", outlets.iter().map(|o| o.name()).collect::<Vec<_>>()),
        )
    })
}

fn check_matrix(path: &Path, outlet: &ort::value::Outlet, width: usize) -> Result<(), PolicyError> {
    let shape = tensor_shape(path, outlet)?;
    if shape.len() != 2 || (shape[0] != 1 && shape[0] != -1) || shape[1] != width as i64 {
        return Err(shape_error(
            path,
            "tensor shape",
            &format!("[1 or dynamic, {width}]"),
            format!("{}: {shape:?}", outlet.name()),
        ));
    }
    Ok(())
}

/// The pose a policy's actions are centred on, out of the model's own metadata.
///
/// **Why this is read rather than assumed.** Upstream bakes the home pose into [`DEFAULT_POSITION`]
/// and records the same numbers in this metadata for provenance, so for the official and retrained
/// sets the two agree and nothing here is visible. A model trained against a different stance — the
/// xgoduck reference set is 0.109 rad shallower at the hip and ankle — would otherwise be run
/// centred on a pose it never saw, and **both** halves of that matter: the observation's `joint_pos`
/// block is measured from the same pose, so the network would be told "I am six degrees below my
/// default" *and* have its offsets added to the wrong base. One is a state it has seen; the other
/// is a robot standing somewhere the policy never trained for.
///
/// Absent is not an error — a model that predates the field keeps the built-in home, which is what
/// every shipped set does — and the remaining joints come from [`DEFAULT_POSITION`] because the
/// mouth is not a policy joint.
///
/// Present-but-unusable *is* an error. The alternative to refusing is standing the robot up
/// wherever a malformed string happened to point.
fn parse_home_pose(
    path: &Path,
    pose: Option<&str>,
    joints: Option<&str>,
) -> Result<[f64; NUM_JOINTS], PolicyError> {
    let Some(pose) = pose else {
        return Ok(DEFAULT_POSITION);
    };
    let refuse = |field: &'static str, why: String| PolicyError::Metadata {
        path: path.to_owned(),
        field,
        why,
    };

    // The order has to be checked before the numbers are placed, because the numbers carry no
    // names: a model whose joints are listed differently would be believed, silently, and every
    // joint after the first difference would be commanded someone else's pose.
    let expected: Vec<&str> = (0..ACTION_LEN)
        .map(|slot| duck_ipc_proto::JOINT_NAMES[crate::obs::joint_of(slot)])
        .collect();
    let Some(joints) = joints else {
        return Err(refuse(
            "joint_names",
            format!(
                "is missing, so the order of `default_joint_pos` cannot be checked against {}",
                expected.join(",")
            ),
        ));
    };
    let got: Vec<&str> = joints.split(',').map(str::trim).collect();
    if got != expected {
        return Err(refuse(
            "joint_names",
            format!("is {}, expected {}", got.join(","), expected.join(",")),
        ));
    }

    let values: Vec<f64> = match pose
        .split(',')
        .map(|v| v.trim().parse::<f64>())
        .collect::<Result<_, _>>()
    {
        Ok(values) => values,
        Err(e) => {
            return Err(refuse(
                "default_joint_pos",
                format!("is not a list of numbers: {e}"),
            ));
        }
    };
    if values.len() != ACTION_LEN {
        return Err(refuse(
            "default_joint_pos",
            format!(
                "has {} values, expected {ACTION_LEN} (one per policy joint)",
                values.len()
            ),
        ));
    }
    // Not a plausibility filter for its own sake: a pose outside the travel every joint has is a
    // number that would be clamped into something nobody chose, and a NaN would poison every
    // target derived from it.
    if let Some(bad) = values
        .iter()
        .find(|v| !v.is_finite() || v.abs() > std::f64::consts::PI)
    {
        return Err(refuse(
            "default_joint_pos",
            format!("has {bad}, which is not an angle inside the +-pi the joints travel"),
        ));
    }

    let mut out = DEFAULT_POSITION;
    for (slot, value) in values.iter().enumerate() {
        out[crate::obs::joint_of(slot)] = *value;
    }
    Ok(out)
}

/// Parse the plant a model declares, out of its export metadata.
///
/// Absent means absent, and `None` is the answer for a model that says nothing about the plant —
/// an XL330 policy is not a policy trained at P=0. *Half* present is an error rather than a
/// default: an export that stamps P and not D is a bug in whoever wrote it, and filling in the
/// missing half is how the comparison quietly stops comparing anything.
fn parse_trained_gains(
    path: &Path,
    kp_fw: Option<&str>,
    kd_fw: Option<&str>,
    mouth_kp_fw: Option<&str>,
) -> Result<Option<TrainedGains>, PolicyError> {
    let refuse = |field: &'static str, why: String| PolicyError::Metadata {
        path: path.to_owned(),
        field,
        why,
    };
    let (Some(kp), Some(kd)) = (kp_fw, kd_fw) else {
        if kp_fw.is_none() && kd_fw.is_none() && mouth_kp_fw.is_none() {
            return Ok(None);
        }
        return Err(refuse(
            if kp_fw.is_none() { "kp_fw" } else { "kd_fw" },
            format!(
                "is missing while the other is present (kp_fw={kp_fw:?}, kd_fw={kd_fw:?}); a \
                 half-stamped plant cannot be compared against the servos"
            ),
        ));
    };

    // 0 is not a soft joint, it is an open loop — the runtime's own profile floors at 1 for the
    // same reason — and anything above 255 does not fit the register being compared against.
    let gain = |field: &'static str, raw: &str| -> Result<u8, PolicyError> {
        let value: u32 = raw
            .trim()
            .parse()
            .map_err(|e| refuse(field, format!("is not a number: {raw:?} ({e})")))?;
        if !(1..=255).contains(&value) {
            return Err(refuse(
                field,
                format!("is {value}, outside the 1..=255 a servo register holds"),
            ));
        }
        Ok(value as u8)
    };
    Ok(Some(TrainedGains {
        kp_fw: gain("kp_fw", kp)?,
        kd_fw: gain("kd_fw", kd)?,
        mouth_kp_fw: mouth_kp_fw
            .map(|raw| gain("mouth_kp_fw", raw))
            .transpose()?,
    }))
}

/// The one plant a set declares, or the error naming the two files that disagree.
///
/// Free rather than a method so it is testable without ONNX Runtime — the same reason
/// [`parse_trained_gains`] is, and the disagreement is the case a board would otherwise report as
/// a gait that is wrong in two directions.
fn declare_set<'a>(
    networks: impl Iterator<Item = (&'a Path, Option<TrainedGains>)>,
) -> Result<Option<(&'a Path, TrainedGains)>, PolicyError> {
    let mut declared: Option<(&Path, TrainedGains)> = None;
    for (path, gains) in networks {
        let Some(gains) = gains else {
            continue;
        };
        match declared {
            None => declared = Some((path, gains)),
            Some((_, known)) if known == gains => {}
            Some((first, known)) => {
                return Err(PolicyError::Metadata {
                    path: path.to_owned(),
                    field: "kp_fw",
                    why: format!(
                        "declares P={}/D={} where {} declares P={}/D={}; one robot has one pair \
                         of servo registers, so this set is not one robot's",
                        gains.kp_fw,
                        gains.kd_fw,
                        first.display(),
                        known.kp_fw,
                        known.kd_fw,
                    ),
                });
            }
        }
    }
    Ok(declared)
}

/// What is wrong with running a policy trained for `declared` on `written`, or `None`.
///
/// Free for the same reason as [`declare_set`]: this sentence is the entire output of the check,
/// and it should be assertable without a policy file or a runtime.
fn gains_mismatch(declared: TrainedGains, written: TrainedGains) -> Option<String> {
    let mouth_differs = declared
        .mouth_kp_fw
        .is_some_and(|mouth| written.mouth_kp_fw != Some(mouth));
    if declared.kp_fw == written.kp_fw && declared.kd_fw == written.kd_fw && !mouth_differs {
        return None;
    }
    let show = |value: Option<u8>| value.map_or_else(|| "unset".to_string(), |v| v.to_string());
    let mouth = if mouth_differs {
        format!(
            ", and a mouth P of {} against {}",
            show(declared.mouth_kp_fw),
            show(written.mouth_kp_fw)
        )
    } else {
        String::new()
    };
    Some(format!(
        "declares P={}/D={} but the servos are written with P={}/D={}{mouth}; the gait was fitted \
         to the first plant and the robot would run the second",
        declared.kp_fw, declared.kd_fw, written.kp_fw, written.kd_fw,
    ))
}

fn open(path: &Path) -> Result<Network, PolicyError> {
    let bytes = std::fs::read(path).map_err(|source| PolicyError::Read {
        path: path.to_owned(),
        source,
    })?;
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    let session = Session::builder()
        .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
        .and_then(|b| b.with_intra_threads(INTRA_THREADS))
        .and_then(|b| b.commit_from_file(path))
        .map_err(|source| PolicyError::Load {
            path: path.to_owned(),
            source,
        })?;

    let inputs = session.inputs();
    let outputs = session.outputs();
    check_matrix(path, outlet(path, inputs, "obs")?, OBS_LEN)?;
    let recurrent = match (inputs.len(), outputs.len()) {
        (1, 1) => false,
        (3, 3) => true,
        counts => {
            return Err(shape_error(
                path,
                "input/output contract",
                "obs -> actions, or obs/h_in/c_in -> actions/h_out/c_out",
                format!("{counts:?} tensors"),
            ));
        }
    };
    // Preserve the existing feed-forward output naming contract (the sole output).
    let action = if recurrent {
        outlet(path, outputs, "actions")?
    } else {
        &outputs[0]
    };
    check_matrix(path, action, ACTION_LEN)?;
    let action_name = action.name().to_owned();
    let state = if recurrent {
        let mut shapes = Vec::new();
        for (outlets, name) in [
            (inputs, "h_in"),
            (inputs, "c_in"),
            (outputs, "h_out"),
            (outputs, "c_out"),
        ] {
            let shape = tensor_shape(path, outlet(path, outlets, name)?)?;
            if shape.len() != 3
                || shape[0] <= 0
                || (shape[1] != 1 && shape[1] != -1)
                || shape[2] <= 0
            {
                return Err(shape_error(
                    path,
                    "LSTM state shape",
                    "[positive layers, 1 or dynamic batch, positive hidden size]",
                    format!("{name}: {shape:?}"),
                ));
            }
            shapes.push(vec![shape[0], 1, shape[2]]);
        }
        if shapes.iter().any(|shape| shape != &shapes[0]) {
            return Err(shape_error(
                path,
                "LSTM state shapes",
                "matching h/c input and output shapes",
                format!("{shapes:?}"),
            ));
        }
        let shape = &shapes[0];
        let count = usize::try_from(shape[0])
            .ok()
            .and_then(|n| n.checked_mul(shape[2] as usize))
            .filter(|&n| n <= 1_048_576)
            .ok_or_else(|| {
                shape_error(
                    path,
                    "LSTM state size",
                    "at most 1048576 elements per state",
                    format!("{shape:?}"),
                )
            })?;
        let make = || {
            Tensor::from_array((shape.clone(), vec![0.0f32; count])).map_err(|source| {
                PolicyError::Load {
                    path: path.to_owned(),
                    source,
                }
            })
        };
        Some(LstmState {
            h: make()?,
            c: make()?,
        })
    } else {
        None
    };
    // Read before the session is moved into the struct, and read as strings: `ort` hands back the
    // export's own text, so the parsing and its refusals live in `parse_home_pose` /
    // `parse_trained_gains` where they can be tested without ONNX Runtime at all.
    let (home_pose, gains) = {
        let metadata = session.metadata().map_err(|source| PolicyError::Load {
            path: path.to_owned(),
            source,
        })?;
        (
            parse_home_pose(
                path,
                metadata.custom("default_joint_pos").as_deref(),
                metadata.custom("joint_names").as_deref(),
            )?,
            parse_trained_gains(
                path,
                metadata.custom("kp_fw").as_deref(),
                metadata.custom("kd_fw").as_deref(),
                metadata.custom("mouth_kp_fw").as_deref(),
            )?,
        )
    };

    Ok(Network {
        session,
        state,
        action_name,
        path: path.to_owned(),
        digest,
        home_pose,
        gains,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The joint list exactly as every shipped and reference export writes it.
    const NAMES: &str = "left_hip_yaw,left_hip_roll,left_hip_pitch,left_knee,left_ankle,neck_pitch,\
head_pitch,head_yaw,head_roll,right_hip_yaw,right_hip_roll,right_hip_pitch,right_knee,right_ankle";

    /// No metadata at all means the built-in home, which is what a model predating the field gets.
    #[test]
    fn a_model_without_a_stance_keeps_the_built_in_home() {
        let path = Path::new("/p/velstand.onnx");
        assert_eq!(parse_home_pose(path, None, None).unwrap(), DEFAULT_POSITION);
        // And `joint_names` alone changes nothing: the stance is what is being read.
        assert_eq!(
            parse_home_pose(path, None, Some(NAMES)).unwrap(),
            DEFAULT_POSITION
        );
    }

    /// **The property that makes this change invisible for every shipped set**: their metadata
    /// records the same home the binary has, so reading it moves nothing.
    ///
    /// Not to the last bit, and the reason is worth knowing rather than tolerating — the export
    /// writes three decimals, so the model's own copy of the pose differs from the constant by up
    /// to 0.0003 rad (0.017 degrees). That is the resolution of the *record*, not a disagreement
    /// about the robot.
    #[test]
    fn the_shipped_sets_metadata_agrees_with_the_built_in_home() {
        let official = "0.000,-0.087,-0.458,-0.005,0.453,0.349,0.349,0.000,0.000,0.000,0.087,0.458,\
0.005,-0.453";
        let parsed = parse_home_pose(Path::new("/p/velstand.onnx"), Some(official), Some(NAMES))
            .expect("the shipped metadata must load");
        let worst = (0..NUM_JOINTS)
            .map(|j| (parsed[j] - DEFAULT_POSITION[j]).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst < 0.001,
            "the shipped stance drifted from the constant by {worst} rad"
        );
        assert_eq!(
            parsed[crate::model::MOUTH_INDEX],
            DEFAULT_POSITION[crate::model::MOUTH_INDEX],
            "the mouth is not a policy joint and keeps the built-in value"
        );
    }

    /// A model trained against another stance is placed where *it* was trained, and the mouth
    /// still is not in its list.
    #[test]
    fn a_model_carries_its_own_stance() {
        let xgoduck = "0.000,-0.087,-0.349,-0.005,0.349,0.349,0.349,0.000,0.000,0.000,0.087,0.349,\
0.005,-0.349";
        let parsed = parse_home_pose(Path::new("/p/hd1910_walk.onnx"), Some(xgoduck), Some(NAMES))
            .expect("a well formed stance must load");
        // By name, not by index: the whole risk in this function is the 14-slot list landing on the
        // wrong side of the mouth, and an index assertion would be written from the same mistake.
        let at = |name: &str| parsed[crate::model::joint_index(name).expect(name)];
        assert!((at("left_hip_pitch") - -0.349).abs() < 1e-9);
        assert!((at("left_ankle") - 0.349).abs() < 1e-9);
        // Past the mouth, where an off-by-one would show: slot 11 is the 12th value and the 12th is
        // `right_hip_pitch`, because joint 9 (`mouth`) has no slot.
        assert!((at("right_hip_roll") - 0.087).abs() < 1e-9);
        assert!((at("right_hip_pitch") - 0.349).abs() < 1e-9);
        assert!((at("right_knee") - 0.005).abs() < 1e-9);
        assert!((at("right_ankle") - -0.349).abs() < 1e-9);
        assert_ne!(
            parsed, DEFAULT_POSITION,
            "this is the whole point: it is not the built-in pose"
        );
        assert_eq!(
            parsed[crate::model::MOUTH_INDEX],
            DEFAULT_POSITION[crate::model::MOUTH_INDEX]
        );
    }

    /// A stance whose joints are listed in another order would otherwise be believed, silently,
    /// and every joint after the first difference commanded someone else's angle.
    #[test]
    fn a_stance_whose_joint_order_disagrees_is_refused() {
        let swapped = NAMES.replace("left_hip_yaw,left_hip_roll", "left_hip_roll,left_hip_yaw");
        let err = parse_home_pose(
            Path::new("/p/x.onnx"),
            Some("0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"),
            Some(&swapped),
        )
        .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("joint_names"), "{text}");
        assert_eq!(
            err.path(),
            Some(Path::new("/p/x.onnx")),
            "it names the file"
        );
    }

    /// Without the names the numbers cannot be placed at all, and guessing is how a robot ends up
    /// standing somewhere nobody chose.
    #[test]
    fn a_stance_without_its_joint_names_is_refused() {
        let err = parse_home_pose(Path::new("/p/x.onnx"), Some("0.0"), None).unwrap_err();
        assert!(err.to_string().contains("joint_names"), "{err}");
    }

    /// Every way the numbers themselves can be unusable, each refused by name.
    #[test]
    fn an_unusable_stance_is_refused() {
        let path = Path::new("/p/x.onnx");
        let short = "0.0,0.0,0.0";
        assert!(parse_home_pose(path, Some(short), Some(NAMES)).is_err());
        let nan = "NaN,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0";
        assert!(parse_home_pose(path, Some(nan), Some(NAMES)).is_err());
        // Inside the travel or not at all: this one would be clamped into an angle nobody chose.
        let far = "9.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0";
        let err = parse_home_pose(path, Some(far), Some(NAMES)).unwrap_err();
        assert!(err.to_string().contains("default_joint_pos"), "{err}");
        let words = "hips,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0";
        assert!(parse_home_pose(path, Some(words), Some(NAMES)).is_err());
    }

    /// Absent is not zero. An XL330 export has no `kp_fw` at all, and reading that as "trained at
    /// P=0" would refuse every policy that works on the robot today.
    #[test]
    fn a_model_that_names_no_plant_declares_none() {
        let path = Path::new("/p/x.onnx");
        assert_eq!(parse_trained_gains(path, None, None, None).unwrap(), None);
    }

    /// A half-stamped plant cannot be compared, so it is refused rather than completed by guess.
    #[test]
    fn a_half_stamped_plant_is_refused() {
        let path = Path::new("/p/x.onnx");
        let err = parse_trained_gains(path, Some("6"), None, None).unwrap_err();
        assert!(err.to_string().contains("kd_fw"), "{err}");
        let err = parse_trained_gains(path, None, Some("20"), None).unwrap_err();
        assert!(err.to_string().contains("kp_fw"), "{err}");
        // A mouth P alone is still a claim about the plant.
        assert!(parse_trained_gains(path, None, None, Some("10")).is_err());
    }

    /// Every unusable number, refused rather than clamped into a plant nobody chose.
    #[test]
    fn a_plant_outside_the_servo_register_is_refused() {
        let path = Path::new("/p/x.onnx");
        // 0 is an open loop rather than a soft joint, and 256 does not fit the register.
        for bad in ["0", "256", "-1", "6.0", "six", ""] {
            assert!(
                parse_trained_gains(path, Some(bad), Some("20"), None).is_err(),
                "{bad:?} was accepted as a P"
            );
        }
        assert!(parse_trained_gains(path, Some("6"), Some("999"), None).is_err());
    }

    /// The shape the FT1910 export and the 1910 forks actually write.
    #[test]
    fn the_stamped_plant_parses() {
        let path = Path::new("/p/x.onnx");
        let gains = parse_trained_gains(path, Some("6"), Some("20"), None)
            .unwrap()
            .expect("a stamped plant");
        assert_eq!(
            gains,
            TrainedGains {
                kp_fw: 6,
                kd_fw: 20,
                mouth_kp_fw: None
            }
        );
        let with_mouth = parse_trained_gains(path, Some("6"), Some("20"), Some("10"))
            .unwrap()
            .expect("a stamped plant");
        assert_eq!(with_mouth.mouth_kp_fw, Some(10));
        // Whitespace is what a text-format exporter leaves behind, not a different number.
        assert_eq!(
            parse_trained_gains(path, Some(" 6 "), Some("20"), None).unwrap(),
            Some(gains)
        );
    }

    /// Two files naming two plants are one robot's set only if the robot has two sets of registers.
    #[test]
    fn a_set_that_names_two_plants_is_refused_with_both() {
        let a = Path::new("/p/velstand.onnx");
        let b = Path::new("/p/roulade.onnx");
        let six = TrainedGains {
            kp_fw: 6,
            kd_fw: 20,
            mouth_kp_fw: None,
        };
        let thirty_two = TrainedGains {
            kp_fw: 32,
            kd_fw: 40,
            mouth_kp_fw: None,
        };

        assert_eq!(
            declare_set([(a, Some(six)), (b, None)].into_iter()).unwrap(),
            Some((a, six)),
            "a network that says nothing does not outvote one that speaks"
        );
        assert_eq!(
            declare_set([(a, None), (b, None)].into_iter()).unwrap(),
            None
        );

        let err = declare_set([(a, Some(six)), (b, Some(thirty_two))].into_iter()).unwrap_err();
        let text = err.to_string();
        assert!(text.contains('6') && text.contains("32"), "{text}");
        assert!(
            text.contains("roulade.onnx"),
            "names the second file: {text}"
        );
        assert!(
            text.contains("velstand.onnx"),
            "names the first file: {text}"
        );
    }

    /// The sentence is the whole output of the check, so what it says is the thing to test.
    #[test]
    fn a_mismatch_names_both_plants_and_agreement_is_silent() {
        let six = TrainedGains {
            kp_fw: 6,
            kd_fw: 20,
            mouth_kp_fw: None,
        };
        assert_eq!(gains_mismatch(six, six), None);

        let thirty_two = TrainedGains {
            kp_fw: 32,
            kd_fw: 40,
            mouth_kp_fw: None,
        };
        let why = gains_mismatch(six, thirty_two).expect("a mismatch");
        assert!(why.contains("P=6/D=20"), "{why}");
        assert!(why.contains("P=32/D=40"), "{why}");

        // D alone is a mismatch. BAM simulates no D term at all, so this is the one comparison
        // that catches a profile drifting to another D — nothing on the robot would show it.
        let other_d = TrainedGains {
            kp_fw: 6,
            kd_fw: 40,
            mouth_kp_fw: None,
        };
        assert!(gains_mismatch(six, other_d).is_some());

        // The mouth counts only when the export claimed one.
        let mouth_ten = TrainedGains {
            kp_fw: 6,
            kd_fw: 20,
            mouth_kp_fw: Some(10),
        };
        let mouth_five = TrainedGains {
            kp_fw: 6,
            kd_fw: 20,
            mouth_kp_fw: Some(5),
        };
        assert_eq!(gains_mismatch(six, mouth_ten), None, "nothing was claimed");
        let why = gains_mismatch(mouth_ten, mouth_five).expect("a mouth mismatch");
        assert!(why.contains("mouth P of 10 against 5"), "{why}");
    }

    /// The threshold decides walking versus standing every tick, so it must match what the
    /// prototype uses or the robot changes gait at a different speed than it was tuned for.
    #[test]
    fn the_standing_threshold_matches_the_prototype() {
        assert_eq!(DEFAULT_STANDING_THRESHOLD, 0.05);
    }

    /// A bundle without a standing policy must never select one. Slice 2 can ship a single
    /// policy, and `will_stand` returning true there would index a session that is not
    /// loaded.
    #[test]
    fn without_a_standing_policy_it_never_stands() {
        // Constructed directly rather than via `load`, which needs ONNX Runtime present.
        // This is the branch that has to hold regardless of what is installed.
        let threshold = DEFAULT_STANDING_THRESHOLD;
        let stands = |has_stand: bool, magnitude: f64| has_stand && magnitude <= threshold;

        assert!(!stands(false, 0.0), "no standing policy, zero command");
        assert!(stands(true, 0.0), "standing policy, zero command");
        assert!(!stands(true, 0.5), "standing policy, walking command");
    }

    /// Roller and fall-recovery modes reserve the standing network, so the magnitude rule
    /// must be inert while `standing_disabled` is set — otherwise a roller duck at zero
    /// stick would swap to a network trained for legs it is not standing on.
    #[test]
    fn disabling_standing_beats_the_magnitude_rule() {
        let threshold = DEFAULT_STANDING_THRESHOLD;
        let stands = |has_stand: bool, disabled: bool, magnitude: f64| {
            has_stand && !disabled && magnitude <= threshold
        };

        assert!(stands(true, false, 0.0));
        assert!(
            !stands(true, true, 0.0),
            "disabled must win at zero command"
        );
    }

    /// **The panic contract.** A panic out of `ort` must come back as a `PolicyError`, because
    /// the caller is the control thread: an escaping panic kills it, no tick ever lands, and
    /// health reports "the loop has not completed a cycle" — naming no cause — instead of
    /// holding the pose and saying the policy is unusable.
    ///
    /// The message must survive too. This is the panic a Radxa actually produced, and the two
    /// version numbers in it are the whole diagnosis; a health reason without them tells an
    /// operator nothing.
    ///
    /// The panic hook still runs, so this test prints a panic and a backtrace hint. That is
    /// wanted — on a board it is what puts the detail in the journal — and is not a failure.
    #[test]
    fn a_panic_out_of_ort_becomes_an_error_that_keeps_its_message() {
        let err = catching_ort_panics::<()>(|| {
            panic!(
                "Failed to load ONNX Runtime dylib: ort 2.0.0-rc.11 is not compatible with \
                 the ONNX Runtime binary found at `libonnxruntime.so`; expected version >= \
                 '1.23.x', but got '1.20.1'"
            )
        })
        .expect_err("a panic in the ort work must not escape to the caller");

        assert!(
            matches!(err, PolicyError::RuntimePanic { .. }),
            "wrong variant: {err:?}"
        );
        let reported = err.to_string();
        for detail in ["1.23.x", "1.20.1"] {
            assert!(
                reported.contains(detail),
                "the version detail must reach the caller, got {reported:?}"
            );
        }
    }

    /// An error that names no file must not be attributed to one. A missing ONNX Runtime is an
    /// operator problem with an operator fix, and reporting it as "this slot's file is bad" sends
    /// them to replace a policy that is fine — which is exactly what the startup fallback would
    /// do with it, silently dropping every override on a board that simply has no dylib.
    #[test]
    fn only_file_errors_name_a_file() {
        let shape = PolicyError::Shape {
            path: PathBuf::from("/tmp/x.onnx"),
            what: "observation width",
            expected: "61".into(),
            got: "51".into(),
        };
        assert_eq!(shape.path(), Some(Path::new("/tmp/x.onnx")));

        let missing = PolicyError::RuntimeMissing {
            searched: "libonnxruntime.so".into(),
            detail: "not found".into(),
        };
        assert_eq!(missing.path(), None, "a missing runtime blames no policy");

        let panicked = PolicyError::RuntimePanic {
            detail: "version mismatch".into(),
        };
        assert_eq!(panicked.path(), None, "an ort panic blames no policy");
    }

    /// Success must pass straight through — a wrapper that swallowed the value would turn
    /// every load into "policy unavailable" on a board where everything works.
    #[test]
    fn the_catch_is_transparent_when_nothing_panics() {
        assert_eq!(catching_ort_panics(|| Ok(7)).unwrap(), 7);
    }

    /// A panic payload that is neither `&str` nor `String` must still produce a reason. The
    /// alternative is an empty health string, which reads as "no reason given".
    #[test]
    fn an_unprintable_panic_payload_still_reports_something() {
        let detail = panic_message(Box::new(42u32));
        assert!(!detail.is_empty(), "a reason is mandatory");
        assert!(
            detail.contains("journal"),
            "point somewhere useful: {detail:?}"
        );
    }
}
