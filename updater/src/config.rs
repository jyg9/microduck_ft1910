//! Per-robot configuration.
//!
//! The engine is generic; everything robot-specific lives here. Adapting to a
//! different robot should mean a new config file, new signing keys, and possibly
//! a new health probe — not engine changes. See `docs/design/updater-design.md` §10.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Root of `/etc/robot/updater.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The file this was read from, when it was read from one.
    ///
    /// Not configuration — it is how the engine hands the *same* config to the release's own
    /// `updaterd --self-test` before committing to it. That check exists to catch a new binary
    /// which rejects the board's `updater.toml`, and it can only do that if it is pointed at the
    /// file in use: `--config` has a default, so a self-test run without one silently validates
    /// `/etc/robot/updater.toml` however the running daemon was started. It read as the release
    /// being broken.
    ///
    /// `None` when the config came from text rather than a path — every test, and nothing on a
    /// board.
    #[serde(skip)]
    pub loaded_from: Option<PathBuf>,

    /// Directory of trusted minisign public keys. A signature is valid if it
    /// verifies against *any* key in here.
    ///
    /// A set rather than one key so a lost or compromised key is survivable —
    /// see `docs/design/updater-design.md` §5.4.
    pub trusted_keys_dir: PathBuf,

    /// The hardware revision a release's `min_hw_rev` is checked against, for a robot whose
    /// `robotd.toml` declares no board.
    ///
    /// The board is the answer when there is one ([`Config::hw_rev`]). This is what every robot
    /// installed before the board was a setting carries — `hw_rev = 1`, which is what its board
    /// is — and what a board provisioned against a release that still shipped it carries too,
    /// which is why it cannot win: a beta would be revision 1 for good, since the installer never
    /// rewrites this file. One integer and not a capability matrix
    /// (`docs/design/updater-design.md` §5.6).
    #[serde(default)]
    pub hw_rev: Option<u32>,

    /// Engine-owned state: lock file, update log, boot counter. Must NOT be
    /// inside any component's `install_dir` — it has to survive every swap and
    /// rollback (`docs/design/updater-design.md` §5.7).
    pub state_dir: PathBuf,

    /// Where fetched policies are kept: one directory per `org/name/revision`, outside every
    /// release directory so a policy somebody chose survives an update and a rollback
    /// (`docs/design/updater-design.md` §5.7).
    ///
    /// **Configurable because it was a constant, and a constant is right exactly once.**
    /// `/var/lib/robot/policies` is correct on a board and unwritable on the laptop the twin runs
    /// on, so every `policy.fetch` against a simulated duck refused with `Permission denied (os
    /// error 13)` naming a path no simulator should ever have been given. The default is still
    /// the board's, so nothing in a release changes; `scripts/duck-sim` sets it per duck.
    #[serde(default = "default_policy_library")]
    pub policy_library: PathBuf,

    /// Where `robotd` listens. Used by every `health = { probe = "socket" }` component
    /// and by the pre-restart `safeToRestart` query.
    ///
    /// Absent or silent is a normal state, not an error (`docs/design/architecture.md` §1.1):
    /// `robotd` may legitimately be stopped, crashed, or not yet installed.
    #[serde(default = "default_robot_socket")]
    pub robot_socket: PathBuf,

    /// Accept artifacts signed with a key marked dev-only. Off in production.
    /// See `docs/design/updater-design.md` §15.
    #[serde(default)]
    pub allow_dev_keys: bool,

    /// Permit `--inject-fault`. Off in production; a client robot must not be able
    /// to be told to fail on purpose. See [`crate::faults`].
    #[serde(default)]
    pub allow_fault_injection: bool,

    /// Largest uncompressed size an artifact may expand to, in bytes.
    ///
    /// Configurable because a model bundle of several ONNX policies is legitimately
    /// far larger than a daemon binary; a global ceiling would reject a real release.
    #[serde(default)]
    pub max_uncompressed_bytes: Option<u64>,

    /// Largest number of entries an artifact archive may contain.
    #[serde(default)]
    pub max_archive_entries: Option<usize>,

    /// How often to check each component's source for a new release.
    ///
    /// `None` disables periodic checking entirely, which also disables the only path
    /// that makes `min_supported` effective (§8.1) — a robot nobody taps update on
    /// never learns a floor exists.
    #[serde(default, with = "humantime_serde::option")]
    pub check_interval: Option<Duration>,

    /// Which updates the periodic check may apply with **no client attached**.
    ///
    /// Inert without [`Self::check_interval`]: the scheduler is the only thing that
    /// applies unattended, so a policy with no timer to run it does nothing.
    #[serde(default)]
    pub auto_apply: AutoApply,

    /// Extra uids permitted to perform **mutating** operations over the IPC socket.
    ///
    /// The uid `updaterd` itself runs as is always permitted — it could replace the
    /// daemon anyway, so denying it would be theatre. Everyone else needs listing
    /// here (or in [`Self::allow_gids`]) to apply, roll back, select or pin.
    ///
    /// Read-only requests (`status`, `log`, `listInstalled`, `check`, `subscribe`) are
    /// **not** gated by this: reaching the socket at all already requires its group
    /// (mode 0660), and support needs to be able to look at a robot without being
    /// authorised to change it.
    #[serde(default)]
    pub allow_uids: Vec<u32>,

    /// Groups permitted to perform mutating operations, by gid.
    #[serde(default)]
    pub allow_gids: Vec<u32>,

    /// Users permitted to perform mutating operations, **by name**.
    ///
    /// Preferred over [`Self::allow_uids`] for anything shipped, because
    /// `systemd-sysusers` allocates uids dynamically: a number that is correct on the
    /// board a config was written for is wrong on the next one. `btd` is listed here so
    /// the app can trigger an update, which is what M6 exists to deliver.
    ///
    /// A name that does not resolve is a warning, not a fatal error. A robot missing an
    /// optional service must still serve status and logs.
    #[serde(default)]
    pub allow_users: Vec<String>,

    /// Groups permitted to perform mutating operations, by name.
    ///
    /// Deliberately empty in the shipped config, and a test enforces that `robot` never
    /// appears here: membership of `robot` is what gets a process as far as *talking* to
    /// updaterd, and listing it would collapse that layer into the change-authority one.
    #[serde(default)]
    pub allow_groups: Vec<String>,

    /// Defaulted so a config with no components reaches [`Config::validate`] and
    /// gets a clear message, rather than a bare serde "missing field".
    #[serde(rename = "component", default)]
    pub components: BTreeMap<String, ComponentConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentConfig {
    pub source: SourceConfig,

    /// Root under which releases are staged and linked.
    pub install_dir: PathBuf,

    pub on_apply: ApplyAction,

    #[serde(default)]
    pub health: HealthCheck,

    /// Retained previous releases, for rollback. The golden release is kept
    /// independently of this count.
    #[serde(default = "default_keep_previous")]
    pub keep_previous: usize,

    /// Never-pruned known-good release (`docs/design/updater-design.md` §8.2).
    #[serde(default)]
    pub golden: Option<semver::Version>,

    /// Refuse anything but this version. Set by `robotctl pin`.
    #[serde(default)]
    pub pinned: Option<semver::Version>,

    /// Files required in the extracted artifact before hooks run or the release becomes live.
    #[serde(default)]
    pub required_files: Vec<PathBuf>,

    /// Maximum compressed artifact size. When set, the signed manifest must declare a size.
    #[serde(default)]
    pub max_artifact_bytes: Option<u64>,
}

fn default_keep_previous() -> usize {
    1
}

/// Which updates the periodic check may apply with no client attached.
///
/// One ordered setting rather than a boolean per urgency. Two booleans would let a config
/// say "apply ordinary updates automatically but not mandatory ones" — auto-updating
/// everything *except* the releases published specifically to rescue a broken fleet. That
/// combination has no legitimate use and is not worth being able to write down.
///
/// Whatever the policy, an unattended apply is still an ordinary apply: it runs the same
/// preflight, so a robot that is walking or streaming refuses and retries at the next
/// interval rather than restarting under someone's hands (`preflight.rs`), and the same
/// health gate, so a release that does not come up is rolled back.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutoApply {
    /// Never apply without a client. Availability is still logged, and a mandatory
    /// release is logged loudly — an ignored one is worth shouting about.
    Off,

    /// Only a release whose manifest declares the running version below `min_supported`.
    ///
    /// The default, and the remediation path for "we shipped a bad release": robots pull
    /// themselves forward instead of waiting for someone to open the app (§8.1). Ordinary
    /// releases still wait for a client, because when a robot restarts is the owner's
    /// decision.
    #[default]
    Mandatory,

    /// Any available release.
    ///
    /// For canary and bench robots — §16.2's Tier 2 wants lab robots that track
    /// `staging` and update on every candidate. On a client robot this takes the
    /// "when does my robot restart" decision away from its owner, which is the decision
    /// the whole app-driven update flow exists to give them.
    All,
}

impl AutoApply {
    /// May the scheduler apply a candidate of this urgency unattended?
    pub fn permits(self, mandatory: bool) -> bool {
        match self {
            AutoApply::Off => false,
            AutoApply::Mandatory => mandatory,
            AutoApply::All => true,
        }
    }
}

/// Where artifacts come from. Per-component because the daemon lives on GitHub
/// Releases and models on HF Hub (`docs/design/updater-design.md` §5.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SourceConfig {
    GithubReleases {
        /// `ORG/REPO`.
        repo: String,
        /// Tag prefix identifying the channel, e.g. `daemon-v`.
        tag_prefix: String,
        /// Release asset holding the signed manifest.
        #[serde(default = "default_manifest_asset")]
        manifest_asset: String,
        /// Tag prefix for per-branch dev builds, so `--ref my-branch` resolves to
        /// `daemon-dev-my-branch`.
        ///
        /// Separate from `tag_prefix` because the two streams must not be confusable: a
        /// dev tag *moves* (it points at whatever that branch built last) while a release
        /// tag is immutable, and `newest_version` must never consider a dev tag when
        /// resolving `latest` for the fleet.
        #[serde(default = "default_ref_tag_prefix")]
        ref_tag_prefix: String,
        /// Tag prefix for release candidates, so `--staging` resolves to
        /// `daemon-staging-v<version>`.
        ///
        /// A third prefix rather than a flag on `tag_prefix`, for the reason the second one
        /// exists: the streams must not be confusable. A candidate is flagged as a prerelease
        /// on GitHub precisely so `newest_version` cannot reach it, and giving that scan an
        /// "unless…" would put the fleet one config typo away from tracking candidates.
        #[serde(default = "default_staging_tag_prefix")]
        staging_tag_prefix: String,
        /// The GitHub API host, without a trailing slash.
        ///
        /// Defaulted so that naming nothing keeps the public service, byte for byte: a robot
        /// that must not reach it — one on a private network, or one whose owner does not want
        /// its address in someone else's access log — points this at its own mirror with config
        /// alone rather than by building a fork of this crate.
        #[serde(default = "default_github_api_base")]
        api_base: String,
        /// The host a release's own `download/...` URLs name, without a trailing slash.
        ///
        /// Separate from [`Self::api_base`] because it answers a different question. The API
        /// host is where *this daemon* sends its requests; this one is what the *manifest*
        /// wrote down, and it is what `source::github::GithubReleases` matches to recognise its
        /// own artifacts and re-resolve them through the API — which is the only way a private
        /// repository's `releases/download/...` URL is fetchable at all. It must therefore be
        /// the host the publishing CI actually used; a mirror that serves the bytes under a
        /// different name is not something this field can paper over.
        #[serde(default = "default_github_download_base")]
        download_base: String,
    },
    HfHub {
        /// `ORG/MODEL`.
        repo: String,
        /// Branch, tag, or commit. A moving branch means "latest".
        revision: String,
        #[serde(default = "default_manifest_asset")]
        manifest_file: String,
        /// The Hub host, without a trailing slash. Defaulted, for [`Self::GithubReleases`]'s
        /// `api_base` reason.
        #[serde(default = "default_hf_endpoint")]
        endpoint: String,
    },
    /// A local directory. Not a production source — this is what makes the
    /// engine testable against the real code path with no network, and backs
    /// the dev sideload flow (`docs/design/updater-design.md` §16.1).
    LocalDir { path: PathBuf },
}

/// Default prefix for dev-build tags.
///
/// Names `daemon` because that is this robot's only source-backed component today; a robot
/// with a differently-named channel sets it explicitly, the same as `tag_prefix`.
fn default_ref_tag_prefix() -> String {
    "daemon-dev-".to_owned()
}

/// Default prefix for release-candidate tags, matching what `release.yml` pushes.
fn default_staging_tag_prefix() -> String {
    "daemon-staging-v".to_owned()
}

fn default_manifest_asset() -> String {
    "manifest.json".to_owned()
}

/// The public GitHub API.
///
/// Stated here rather than in `source/github.rs` because it is a property of the *schema*: it
/// is what an absent `api_base` means, and the only reader that can leave it absent is serde.
/// A test in this module pins it against the literal the source layer's URL tests use, so the
/// two cannot drift into agreeing on a wrong value.
fn default_github_api_base() -> String {
    "https://api.github.com".to_owned()
}

/// The public GitHub download host — where `releases/download/...` URLs point.
fn default_github_download_base() -> String {
    "https://github.com".to_owned()
}

/// The public Hugging Face Hub.
fn default_hf_endpoint() -> String {
    "https://huggingface.co".to_owned()
}

/// What to do once the new release is linked.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ApplyAction {
    /// Nothing to do (the consumer picks it up on its own).
    None,
    /// Full restart. Drops motor control briefly — only for the daemon channel.
    Restart { units: Vec<String> },
    /// Signal in place, no restart. Used for models so a weights swap doesn't
    /// interrupt motor control (`docs/design/updater-design.md` §5.5).
    Reload { unit: String, signal: String },
}

/// How to decide whether the new release is good.
///
/// This gate is what makes auto-rollback meaningful, so a weak probe here
/// undermines the whole design — see `docs/design/updater-design.md` §8.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "probe", rename_all = "snake_case")]
pub enum HealthCheck {
    /// No gate: commit as soon as apply returns. Only acceptable for components
    /// that cannot break the robot.
    #[default]
    None,
    /// Ask `robotd` over its unix socket and wait for it to report healthy.
    ///
    /// No path here on purpose. The socket is one robot-wide fact, not a per-component
    /// one, so it lives in [`Config::robot_socket`].
    Socket {
        #[serde(with = "humantime_serde")]
        timeout: Duration,
    },
    /// Run a command; exit status 0 means healthy. Escape hatch for probes that
    /// don't fit the socket model.
    Command {
        program: PathBuf,
        #[serde(default)]
        args: Vec<String>,
        #[serde(with = "humantime_serde")]
        timeout: Duration,
    },
}

impl HealthCheck {
    pub fn timeout(&self) -> Option<Duration> {
        match self {
            HealthCheck::None => None,
            HealthCheck::Socket { timeout, .. } | HealthCheck::Command { timeout, .. } => {
                Some(*timeout)
            }
        }
    }
}

fn default_robot_socket() -> PathBuf {
    PathBuf::from("/run/robotd.sock")
}

fn default_policy_library() -> PathBuf {
    PathBuf::from(crate::policy::LIBRARY_ROOT)
}

impl Config {
    /// Parse from TOML text. Always validated — an invalid config must not be
    /// constructible.
    pub fn from_toml(text: &str) -> Result<Self, crate::Error> {
        let config: Self = toml::from_str(text).map_err(|e| crate::Error::Config(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn load(path: &std::path::Path) -> Result<Self, crate::Error> {
        let text = std::fs::read_to_string(path).map_err(|e| crate::Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let mut config = Self::from_toml(&text)?;
        // Absolute, because the self-test runs as a fresh process whose working directory is not
        // this one's.
        config.loaded_from = Some(path.canonicalize().unwrap_or_else(|_| path.to_path_buf()));
        Ok(config)
    }

    /// The hardware revision this robot is: the board `robotd.toml` declares, else this file's
    /// `hw_rev`, else `zero3`'s.
    ///
    /// Read from the file every time rather than once at startup, so `robotctl configure`
    /// applies without restarting `updaterd`.
    pub fn hw_rev(&self) -> u32 {
        self.hw_rev_given(robotd_params::board::Board::declared(std::path::Path::new(
            robotd_params::DEFAULT_PATH,
        )))
    }

    fn hw_rev_given(&self, declared: Option<robotd_params::board::Board>) -> u32 {
        declared
            .map(robotd_params::board::Board::hw_rev)
            .or(self.hw_rev)
            .unwrap_or_else(|| robotd_params::board::Board::default().hw_rev())
    }

    /// Bounds applied when extracting an artifact.
    pub fn archive_limits(&self) -> crate::verify::ArchiveLimits {
        let defaults = crate::verify::ArchiveLimits::default();
        crate::verify::ArchiveLimits {
            max_uncompressed_bytes: self
                .max_uncompressed_bytes
                .unwrap_or(defaults.max_uncompressed_bytes),
            max_entries: self.max_archive_entries.unwrap_or(defaults.max_entries),
        }
    }

    pub fn component(&self, id: &str) -> Result<&ComponentConfig, crate::Error> {
        self.components
            .get(id)
            .ok_or_else(|| crate::Error::UnknownComponent(id.to_owned()))
    }

    /// Reject self-inconsistent configs at load time rather than mid-update.
    ///
    /// Each of these would otherwise surface as data loss or a mysterious failure
    /// during an update, which is the worst time to discover them.
    pub fn validate(&self) -> Result<(), crate::Error> {
        let bad = |msg: String| Err(crate::Error::Config(msg));

        if self.components.is_empty() {
            return bad("no components configured; nothing could ever be updated".into());
        }

        for (name, component) in &self.components {
            for required in &component.required_files {
                if required.as_os_str().is_empty()
                    || !required
                        .components()
                        .all(|part| matches!(part, std::path::Component::Normal(_)))
                {
                    return bad(format!(
                        "component {name}: required_files entry {} must be a nonempty relative file path without ..",
                        required.display()
                    ));
                }
            }
            if component.max_artifact_bytes == Some(0) {
                return bad(format!(
                    "component {name}: max_artifact_bytes must be positive"
                ));
            }

            // A relative install_dir would resolve against the daemon's cwd,
            // which systemd does not guarantee.
            if !component.install_dir.is_absolute() {
                return bad(format!(
                    "component {name}: install_dir must be absolute, got {}",
                    component.install_dir.display()
                ));
            }

            // The decisive one: engine state inside a release tree would be
            // destroyed by the very swap or rollback it exists to record.
            if self.state_dir.starts_with(&component.install_dir) {
                return bad(format!(
                    "state_dir {} is inside component {name}'s install_dir {} — a swap or \
                     rollback would destroy the update log and boot counter",
                    self.state_dir.display(),
                    component.install_dir.display()
                ));
            }

            // Two components sharing a tree would prune each other's releases.
            for (other_name, other) in &self.components {
                if other_name != name && component.install_dir == other.install_dir {
                    return bad(format!(
                        "components {name} and {other_name} share install_dir {} — they would \
                         prune each other's releases",
                        component.install_dir.display()
                    ));
                }
            }

            // Golden exists to be a rollback target that is never pruned; a
            // keep_previous of 0 leaves nothing else to fall back to.
            if component.keep_previous == 0 && component.golden.is_none() {
                return bad(format!(
                    "component {name}: keep_previous = 0 with no golden release leaves no \
                     rollback target"
                ));
            }
        }

        if !self.state_dir.is_absolute() {
            return bad(format!(
                "state_dir must be absolute, got {}",
                self.state_dir.display()
            ));
        }

        // Same rule as `state_dir`, for the same reason: this is resolved by a daemon whose
        // working directory is not anybody's, and a relative path there is a directory nobody
        // meant to write to.
        if !self.policy_library.is_absolute() {
            return bad(format!(
                "policy_library must be absolute, got {}",
                self.policy_library.display()
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The example config shipped in the repo must actually parse. Catches drift
    /// between docs and code.
    #[test]
    fn example_config_parses() {
        let text = include_str!("../updater.example.toml");
        let config = Config::from_toml(text).expect("example config must be valid");

        let daemon = config.component("daemon").unwrap();

        // The bootstrap state is over: robotd exists, so the example config gates for
        // real. These assertions run in the other direction now — they catch a
        // *regression* to the inert values, which would silently disable auto-rollback
        // and look like nothing at all in a diff.
        let ApplyAction::Restart { units } = &daemon.on_apply else {
            panic!(
                "daemon on_apply must restart robotd, not {:?}",
                daemon.on_apply
            );
        };
        assert!(
            units.contains(&"robotd".to_string()),
            "must restart robotd: {units:?}"
        );
        // updaterd must never restart itself (it would die mid-swap) and must not
        // restart btd (it would drop the app's progress connection).
        assert!(
            !units.iter().any(|u| u == "updaterd" || u == "btd"),
            "updaterd must not restart itself or btd: {units:?}"
        );
        assert!(
            matches!(daemon.health, HealthCheck::Socket { .. }),
            "daemon must have a real health gate, got {:?}",
            daemon.health
        );

        // One component per model, each independently versioned (§5.5).
        assert!(config.component("model-walk").is_ok());
        assert!(config.component("model-jump").is_ok());
    }

    /// **Everything `on_apply` restarts must actually ship.**
    ///
    /// "What the daemon artifact contains" is stated in three places — this config, each
    /// unit's `ExecStart`, and the release workflows' copy lists — and nothing else compares
    /// them. An artifact missing a binary installs cleanly and then fails its own restart
    /// step, which rolls the release back on every robot, with the cause three files away
    /// from the symptom.
    ///
    /// The workflows are checked by string search rather than by parsing YAML: one assertion
    /// does not justify a YAML dependency, and the strings searched for are exactly the ones
    /// that must not go missing.
    #[test]
    fn every_unit_on_apply_restarts_is_actually_shipped() {
        let text = include_str!("../updater.example.toml");
        let config = Config::from_toml(text).unwrap();
        let daemon = config.component("daemon").unwrap();

        let ApplyAction::Restart { units } = &daemon.on_apply else {
            panic!("expected a restart action; the other assertions cover that");
        };

        // updater/ -> repo root
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("updater/ has a parent");
        // Both workflows that build an artifact. A dev build missing `robotd` fails on the
        // board in exactly the same way as a release missing it — `systemctl restart robotd`
        // with no such unit — and a teammate hitting that would have no reason to suspect the
        // packaging rather than their own branch.
        //
        // `_build-release.yml`, not `release.yml`: the packaging recipe lives in the reusable
        // workflow that both the staging and the stable path call, and `release.yml` is now only the
        // entry point choosing between them. The assertion below fails loudly on a file it cannot
        // parse, which is what caught this rename rather than silently passing.
        let workflows = [
            ".github/workflows/_build-release.yml",
            ".github/workflows/dev.yml",
        ]
        .map(|w| {
            (
                w,
                std::fs::read_to_string(repo.join(w)).unwrap_or_else(|_| panic!("{w} must exist")),
            )
        });
        let workspace = std::fs::read_to_string(repo.join("Cargo.toml")).unwrap();

        for unit in units {
            // A crate of that name must exist in the workspace, or there is no binary to
            // restart.
            assert!(
                workspace.contains(&format!("\"{unit}\"")),
                "on_apply restarts `{unit}` but no such workspace member exists"
            );

            // A unit file must exist in the repo...
            let unit_file = repo
                .join(unit)
                .join("systemd")
                .join(format!("{unit}.service"));
            assert!(
                unit_file.exists(),
                "on_apply restarts `{unit}` but {} does not exist",
                unit_file.display()
            );

            // ...and every workflow that builds an artifact must ship both the binary and
            // that unit file. Without these two lines the release installs successfully and
            // then fails its own restart step.
            for (name, workflow) in &workflows {
                assert!(
                    workflow.contains(&format!("release/{unit} staged/")),
                    "{name} does not copy the `{unit}` binary into the artifact"
                );
                assert!(
                    workflow.contains(&format!("{unit}/systemd/{unit}.service=")),
                    "{name} does not include `{unit}.service` in the artifact"
                );
            }
        }
    }

    /// `deploy/updater.toml` is what a client robot actually runs, so its safety-relevant
    /// values are asserted rather than reviewed.
    ///
    /// Every one of these is a single word or `true`/`false` away from being wrong in a way
    /// no diff makes obvious: a robot that trusts dev keys, one that can be told to fail on
    /// purpose, one that reaches a public host unattended, or one whose update gate does not
    /// gate. All four look fine and behave fine right up to the moment they matter.
    ///
    /// The egress one is the reason this file differs from `updater.example.toml` at all: the
    /// example describes the mechanism a fleet robot wants, and this file is a robot that is
    /// not on that fleet.
    #[test]
    fn shipped_config_is_safe_for_a_client_robot() {
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("updater/ has a parent");
        let text = std::fs::read_to_string(repo.join("deploy/updater.toml"))
            .expect("deploy/updater.toml must exist — scripts/install.sh installs it");
        let config = Config::from_toml(&text).expect("the shipped config must be valid");

        assert!(
            !config.allow_dev_keys,
            "a client robot must not trust dev keys: it would install anything a teammate builds"
        );
        assert!(
            !config.allow_fault_injection,
            "a client robot must not accept --inject-fault"
        );
        // **Absent, and this is the load-bearing line of the whole file.** The periodic check
        // is the only thing this robot does without being asked, and it is an outbound request
        // to a host we do not run. Losing it costs `min_supported`: a withdrawn release is
        // noticed when somebody asks rather than on its own. That is the trade a standalone
        // robot makes, and it is the opposite of the trade a fleet robot makes — which is why
        // the example keeps a `check_interval` and this file does not.
        assert!(
            config.check_interval.is_none(),
            "the shipped config must not poll: the periodic check is the one request this \
             robot makes with nobody watching"
        );
        // Numeric ids never belong in a shipped config: sysusers allocates dynamically, so a
        // number that is right on one board is wrong on the next. Names are resolved at startup.
        assert!(
            config.allow_uids.is_empty() && config.allow_gids.is_empty(),
            "the shipped config must grant by name, not by number: {:?} / {:?}",
            config.allow_uids,
            config.allow_gids
        );

        // The layering this whole design rests on. Naming a specific *service* is a narrow
        // claim — "btd may relay an update request from the app". Naming the `robot` group is
        // not: membership of it is what gets a process as far as *talking* to updaterd, so
        // listing it here would collapse the socket-access and change-authority layers into
        // one, and anything that could read status could replace the firmware.
        assert!(
            !config.allow_groups.iter().any(|g| g == "robot"),
            "the robot group must not have change authority: {:?}",
            config.allow_groups
        );

        // btd is expected: an update the owner starts from their phone is M6's headline, and
        // without this it returns PERMISSION_DENIED.
        assert!(
            config.allow_users.iter().any(|u| u == "btd"),
            "btd must be able to relay an update request from the app: {:?}",
            config.allow_users
        );
        // And mediad, for the mutating calls its route table permits: `account.login`, and
        // `policy.install`/`policy.fetch`, which were routed to WebRTC before this line existed
        // and so answered PERMISSION_DENIED. Asserted rather than assumed because the failure is
        // silent in the worst way — a button that reads as a broken feature rather than as a
        // missing line in a config file.
        assert!(
            config.allow_users.iter().any(|u| u == "mediad"),
            "mediad must be able to sign the robot in to an account: {:?}",
            config.allow_users
        );
        // Neither entry may become a *group*: the two above are services, and the layering
        // argument above is about exactly that distinction.
        assert!(
            config.allow_groups.is_empty(),
            "change authority is granted to named services, not to groups: {:?}",
            config.allow_groups
        );
        // Narrower than "not `all`": nothing here may install a release without a person
        // asking. There is no channel this robot trusts enough to take one from unattended, and
        // with no `check_interval` above there is no timer that could run it anyway — asserted
        // both ways so that restoring one without the other is a test failure rather than a
        // robot that restarts on its own.
        assert_eq!(
            config.auto_apply,
            AutoApply::Off,
            "the shipped config must apply nothing unattended"
        );

        let daemon = config
            .component("daemon")
            .expect("the daemon component must exist");
        let ApplyAction::Restart { units } = &daemon.on_apply else {
            panic!(
                "daemon on_apply must restart robotd, not {:?}",
                daemon.on_apply
            );
        };
        assert!(
            units.contains(&"robotd".to_string()),
            "must restart robotd: {units:?}"
        );
        assert!(
            !units.iter().any(|u| u == "updaterd" || u == "btd"),
            "updaterd must not restart itself or btd: {units:?}"
        );
        assert!(
            matches!(daemon.health, HealthCheck::Socket { .. }),
            "the shipped config must have a real health gate, got {:?}",
            daemon.health
        );

        // **Locally-sideloaded signed releases, and no host at all.** A `github_releases` source
        // here would be a standing egress to a public host on a robot that is not on the
        // manufacturer's fleet — and every mechanism that matters is indifferent to which source
        // produced the bytes: a sideloaded artifact is verified exactly like a downloaded one
        // (signature, not a skipped check), and the health gate, golden release and rollback all
        // run on the result. This is the assertion that keeps a later edit from quietly
        // restoring the public channel.
        //
        // There is no `tag_prefix`/`staging_tag_prefix` to assert any more, and that is the
        // intended change rather than an omission: a directory has no channels, so `--staging`
        // against it refuses by name (`source/local.rs`) instead of resolving a candidate.
        let SourceConfig::LocalDir { path } = &daemon.source else {
            panic!(
                "the shipped daemon source must be a local_dir so this robot makes no \
                 outbound request, got {:?}",
                daemon.source
            );
        };
        assert!(
            path.is_absolute(),
            "a relative drop directory would resolve against updaterd's working directory, \
             which systemd does not guarantee: {}",
            path.display()
        );

        // One component, because a second would be one nobody has shipped — and a source that
        // cannot answer makes every `check` report a failure, which teaches whoever reads robot
        // status to ignore failures.
        //
        // The daemon's directory may be empty on a robot nobody has sideloaded to, and a check
        // against it then fails with "no manifests in <dir>". That is the honest answer — there
        // is nothing to install — and it is not noise, because nothing polls: it is only ever
        // said to somebody who asked.
        assert_eq!(
            config.components.keys().collect::<Vec<_>>(),
            vec!["daemon"],
            "the shipped config should carry only the component this robot actually runs"
        );
    }

    /// A config that names none of the defaulted source fields, so what they come back as is the
    /// schema's answer rather than a file's.
    ///
    /// Shared by the two tests below rather than written twice, because "which fields are
    /// optional" is one fact and a second copy would be a second place for it to be wrong.
    const NO_OPTIONAL_SOURCE_FIELDS: &str = r#"
trusted_keys_dir = "/etc/robot/trusted_keys"
state_dir = "/var/lib/robot/updater"

[component.daemon]
install_dir = "/opt/robot/daemon"

[component.daemon.source]
type       = "github_releases"
repo       = "ORG/duck-daemon"
tag_prefix = "daemon-v"

[component.daemon.on_apply]
action = "restart"
units  = ["robotd"]

[component.models]
install_dir = "/opt/robot/model/walk"

[component.models.source]
type     = "hf_hub"
repo     = "ORG/gait-model"
revision = "main"

[component.models.on_apply]
action = "none"
"#;

    /// **Naming no host keeps the public one, which is what makes the fields additive.**
    ///
    /// The three values are asserted against literals that also appear in the source layer's URL
    /// tests, and deliberately not read from a shared constant: one constant would satisfy both
    /// sides of the contract while being wrong in the one place that matters, which is the request
    /// a robot that configured nothing actually sends.
    ///
    /// The failure this prevents is silent in both directions — a default that drifts sends a
    /// robot to a host nobody configured, and a field that stopped being defaulted turns every
    /// existing `updater.toml` into a parse error on the next boot.
    #[test]
    fn a_source_with_no_host_named_keeps_the_public_one() {
        let config =
            Config::from_toml(NO_OPTIONAL_SOURCE_FIELDS).expect("naming no host must still load");

        let SourceConfig::GithubReleases {
            api_base,
            download_base,
            ..
        } = &config.component("daemon").unwrap().source
        else {
            panic!("the daemon source here is a github_releases source");
        };
        assert_eq!(api_base, "https://api.github.com");
        assert_eq!(download_base, "https://github.com");

        let SourceConfig::HfHub { endpoint, .. } = &config.component("models").unwrap().source
        else {
            panic!("the model source here is an hf_hub source");
        };
        assert_eq!(endpoint, "https://huggingface.co");
    }

    /// **The channel prefixes default to what the workflows actually push.**
    ///
    /// A `github_releases` source that names no prefix depends on these three strings matching
    /// `.github/workflows/`: `daemon-v` from `_build-release.yml` and `_promote-release.yml`,
    /// `daemon-staging-v` from the candidate path, `daemon-dev-` from `dev.yml`. A wrong default
    /// does not fail loudly — it reports "no releases with tag prefix", which reads as "there is
    /// no candidate" rather than as "this board is looking in the wrong place", so the two are
    /// pinned together here where a change to either is one diff.
    ///
    /// No longer asserted against `deploy/updater.toml`, because that file's source is a local
    /// directory now and a directory has no channels. The contract is with the workflows, so this
    /// is where it belongs rather than in the shipped-config test it used to live in.
    #[test]
    fn the_channel_prefixes_default_to_what_the_workflows_push() {
        let config =
            Config::from_toml(NO_OPTIONAL_SOURCE_FIELDS).expect("naming no prefix must still load");
        let SourceConfig::GithubReleases {
            ref_tag_prefix,
            staging_tag_prefix,
            ..
        } = &config.component("daemon").unwrap().source
        else {
            panic!("the daemon source here is a github_releases source");
        };
        assert_eq!(ref_tag_prefix, "daemon-dev-");
        assert_eq!(
            staging_tag_prefix, "daemon-staging-v",
            "the default candidate prefix must match the tag release.yml pushes"
        );
    }

    /// Wherever a placeholder repository is named, `scripts/install.sh`'s guard must recognise it.
    ///
    /// `scripts/install.sh` refuses to install a config still containing `ORG/`, so a robot
    /// cannot be provisioned pointing at a repository that does not exist — which would install
    /// fine and then never find another update. The shipped config names no repository today (its
    /// source is a local directory), so the branch that matters here is the other one: the guard
    /// must survive in the script for boards provisioned before that, and for anyone who puts a
    /// GitHub source back. Where a placeholder does appear it must be the `repo` value and not
    /// stray text — say, the literal left behind in a comment, which the script's `grep` would
    /// match and its `sed` would not fix.
    #[test]
    fn installer_guard_and_shipped_config_agree_about_the_placeholder() {
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap();
        let config = std::fs::read_to_string(repo.join("deploy/updater.toml")).unwrap();
        let script = std::fs::read_to_string(repo.join("scripts/install.sh"))
            .expect("scripts/install.sh must exist");

        let placeholder_present = config.contains("ORG/");
        assert!(
            script.contains("ORG/"),
            "install.sh must keep checking for the placeholder while one can still appear"
        );
        if placeholder_present {
            assert!(
                config.contains("repo           = \"ORG/"),
                "the placeholder should be the `repo` value, not stray text the guard would \
                 also match"
            );
        }
    }

    /// The truth table, stated once. Every unattended-apply decision routes through this.
    #[test]
    fn auto_apply_permits_the_right_urgencies() {
        assert!(
            !AutoApply::Off.permits(true),
            "off must ignore even mandatory"
        );
        assert!(!AutoApply::Off.permits(false));

        assert!(AutoApply::Mandatory.permits(true));
        assert!(
            !AutoApply::Mandatory.permits(false),
            "the default must not install ordinary releases behind the owner's back"
        );

        assert!(AutoApply::All.permits(true), "all must include mandatory");
        assert!(AutoApply::All.permits(false));
    }

    /// A declared board is the answer, whatever `hw_rev` says: a beta provisioned against a
    /// release whose `updater.toml` still carried `hw_rev = 1` must not stay revision 1. Without
    /// a board, `hw_rev` is the answer, and without either it is the zero3.
    #[test]
    fn the_declared_board_wins_over_hw_rev() {
        use robotd_params::board::Board;
        let base = r#"
            trusted_keys_dir = "/etc/robot/keys"
            state_dir = "/var/lib/robot/updater"
            [component.daemon]
            install_dir = "/opt/robot/daemon"
            source = { type = "local_dir", path = "/var/tmp/rel" }
            on_apply = { action = "none" }
        "#;
        let bare = Config::from_toml(base).unwrap();
        assert_eq!(bare.hw_rev_given(None), Board::Zero3.hw_rev());
        assert_eq!(bare.hw_rev_given(Some(Board::Beta)), Board::Beta.hw_rev());

        let legacy = Config::from_toml(&format!("hw_rev = 1\n{base}")).unwrap();
        assert_eq!(legacy.hw_rev_given(None), 1);
        assert_eq!(legacy.hw_rev_given(Some(Board::Beta)), Board::Beta.hw_rev());
    }

    /// A board's config says nothing about the policy library and must keep getting the board's
    /// path; a twin's says where it can actually write, and must be believed. Both halves matter:
    /// the default is what every release depends on, and the override is the whole reason this
    /// stopped being a constant — `/var/lib/robot/policies` is unwritable on the laptop the
    /// simulator runs on, and a `policy.fetch` there refused with `Permission denied`.
    #[test]
    fn policy_library_defaults_to_the_board_and_can_be_moved() {
        let base = r#"
            trusted_keys_dir = "/etc/robot/keys"
            hw_rev = 1
            state_dir = "/var/lib/robot/updater"
            [component.daemon]
            install_dir = "/opt/robot/daemon"
            source = { type = "local_dir", path = "/var/tmp/rel" }
            on_apply = { action = "none" }
        "#;
        assert_eq!(
            Config::from_toml(base).unwrap().policy_library,
            PathBuf::from(crate::policy::LIBRARY_ROOT),
            "a config that does not mention it must still get the board's library"
        );

        let moved = format!("policy_library = \"/tmp/ducks/duck-a/policies\"\n{base}");
        assert_eq!(
            Config::from_toml(&moved).unwrap().policy_library,
            PathBuf::from("/tmp/ducks/duck-a/policies")
        );

        let relative = format!("policy_library = \"policies\"\n{base}");
        let why = Config::from_toml(&relative).unwrap_err().to_string();
        assert!(
            why.contains("policy_library must be absolute"),
            "a relative library is a directory nobody meant to write to: {why}"
        );
    }

    /// The default has to be `mandatory`: a config that omits the field still needs to
    /// remediate a withdrawn release, and `off` would leave the fleet stuck on it.
    #[test]
    fn auto_apply_defaults_to_mandatory_and_parses_each_variant() {
        let base = r#"
            trusted_keys_dir = "/etc/robot/keys"
            hw_rev = 1
            state_dir = "/var/lib/robot/updater"
            [component.daemon]
            install_dir = "/opt/robot/daemon"
            source = { type = "local_dir", path = "/var/tmp/rel" }
            on_apply = { action = "none" }
        "#;
        assert_eq!(
            Config::from_toml(base).unwrap().auto_apply,
            AutoApply::Mandatory
        );

        for (text, expected) in [
            ("off", AutoApply::Off),
            ("mandatory", AutoApply::Mandatory),
            ("all", AutoApply::All),
        ] {
            let config = Config::from_toml(&format!("auto_apply = \"{text}\"\n{base}")).unwrap();
            assert_eq!(config.auto_apply, expected, "parsing {text:?}");
        }

        // A typo must be refused rather than silently defaulted. `auto_apply = "always"`
        // reading as `mandatory` would look like the setting took effect and quietly not.
        assert!(
            Config::from_toml(&format!("auto_apply = \"always\"\n{base}")).is_err(),
            "an unrecognised policy must be a config error"
        );

        // The old boolean must not linger anywhere: `deny_unknown_fields` means a config
        // still carrying it fails loudly instead of being read as "off".
        assert!(
            Config::from_toml(&format!("auto_apply_mandatory = true\n{base}")).is_err(),
            "the superseded field must be rejected, not ignored"
        );
    }

    #[test]
    fn health_timeout_accepts_humantime() {
        let config = Config::from_toml(
            r#"
            trusted_keys_dir = "/etc/robot/keys"
            hw_rev = 1
            state_dir = "/var/lib/robot/updater"
            [component.daemon]
            install_dir = "/opt/robot/daemon"
            source = { type = "local_dir", path = "/var/tmp/rel" }
            on_apply = { action = "none" }
            health = { probe = "socket", timeout = "45s" }
            "#,
        )
        .unwrap();

        assert_eq!(
            config.component("daemon").unwrap().health.timeout(),
            Some(Duration::from_secs(45))
        );
    }

    fn config_with(extra_component: &str) -> Result<Config, crate::Error> {
        Config::from_toml(&format!(
            r#"
            trusted_keys_dir = "/etc/robot/keys"
            hw_rev = 1
            state_dir = "/var/lib/robot/updater"
            {extra_component}
            "#
        ))
    }

    #[test]
    fn rejects_state_dir_inside_install_dir() {
        let err = Config::from_toml(
            r#"
            trusted_keys_dir = "/etc/robot/keys"
            hw_rev = 1
            state_dir = "/opt/robot/daemon/state"
            [component.daemon]
            install_dir = "/opt/robot/daemon"
            source = { type = "local_dir", path = "/var/tmp/rel" }
            on_apply = { action = "none" }
            "#,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("would destroy the update log"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_shared_install_dir() {
        let err = config_with(
            r#"
            [component.daemon]
            install_dir = "/opt/robot/shared"
            source = { type = "local_dir", path = "/var/tmp/rel" }
            on_apply = { action = "none" }
            [component.model]
            install_dir = "/opt/robot/shared"
            source = { type = "local_dir", path = "/var/tmp/rel" }
            on_apply = { action = "none" }
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("prune each other"), "got: {err}");
    }

    #[test]
    fn rejects_relative_install_dir() {
        let err = config_with(
            r#"
            [component.daemon]
            install_dir = "opt/robot/daemon"
            source = { type = "local_dir", path = "/var/tmp/rel" }
            on_apply = { action = "none" }
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("must be absolute"), "got: {err}");
    }

    #[test]
    fn rejects_no_rollback_target() {
        let err = config_with(
            r#"
            [component.daemon]
            install_dir = "/opt/robot/daemon"
            source = { type = "local_dir", path = "/var/tmp/rel" }
            keep_previous = 0
            on_apply = { action = "none" }
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("no rollback target"), "got: {err}");
    }

    #[test]
    fn component_guards_reject_invalid_paths_and_zero_budget() {
        for setting in [
            "required_files = ['']",
            "required_files = ['.']",
            "required_files = ['/bin/worker']",
            "required_files = ['../worker']",
            "required_files = ['bin/../../worker']",
            "max_artifact_bytes = 0",
        ] {
            let err = config_with(&format!(
                r#"
                [component.daemon]
                install_dir = "/opt/robot/daemon"
                source = {{ type = "local_dir", path = "/var/tmp/rel" }}
                on_apply = {{ action = "none" }}
                {setting}
                "#
            ))
            .unwrap_err();
            assert!(matches!(err, crate::Error::Config(_)), "{setting}: {err}");
        }
    }

    #[test]
    fn rejects_empty_components() {
        let err = Config::from_toml(
            r#"
            trusted_keys_dir = "/etc/robot/keys"
            hw_rev = 1
            state_dir = "/var/lib/robot/updater"
            "#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("no components"), "got: {err}");
    }

    /// A typo in a key name must fail loudly rather than be silently ignored —
    /// a misspelled `keep_previous` would quietly prune the rollback target.
    #[test]
    fn rejects_unknown_fields() {
        let err = config_with(
            r#"
            [component.daemon]
            install_dir = "/opt/robot/daemon"
            source = { type = "local_dir", path = "/var/tmp/rel" }
            on_apply = { action = "none" }
            keep_previouss = 3
            "#,
        )
        .unwrap_err();
        assert!(matches!(err, crate::Error::Config(_)), "got: {err:?}");
    }
}
