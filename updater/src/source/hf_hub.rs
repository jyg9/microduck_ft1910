//! Hugging Face Hub source — the model channel.
//!
//! Files resolve at `https://huggingface.co/{repo}/resolve/{revision}/{file}`.
//! **HF signs nothing for us**, so we publish our own minisign signature alongside
//! every artifact and verify that. See `docs/design/updater-design.md` §5.1.
//!
//! Two things differ from GitHub Releases and shape the code:
//!
//!  - There is no "releases" concept, only git revisions. `revision` in config is
//!    typically a moving branch, so "latest" means "whatever that branch points at
//!    right now" — and the *manifest's own* `version` field is authoritative for
//!    what we actually installed, never the branch name.
//!  - Exact versions map to tags by convention (`v1.2.3`), so an exact fetch is the
//!    same code path with a different revision.

use std::path::Path;

use serde::Deserialize;

use crate::Error;
use crate::manifest::Manifest;
use crate::source::{FetchedArtifact, ProgressSink, SignedBytes, Source, http, trim_base};

const SIG_SUFFIX: &str = ".minisig";

pub struct HfHub {
    repo: String,
    revision: String,
    manifest_file: String,
    /// The Hub host, without a trailing slash.
    ///
    /// Configurable for the same reason GitHub's hosts are: a robot that must not reach the
    /// public service points this at its own mirror. The default lives in `crate::config`,
    /// so a robot that names nothing sends exactly the URLs it sent before the field
    /// existed. It covers both the resolve and the refs endpoint because they are one
    /// service — a mirror that served one and not the other would answer "no manifest" for
    /// every version, which reads as a withdrawn release rather than as a misconfigured host.
    endpoint: String,
    client: reqwest::Client,
}

/// Subset of the HF repo API used to enumerate tags for an exact-version lookup.
#[derive(Debug, Deserialize)]
struct RepoRefs {
    #[serde(default)]
    tags: Vec<GitRef>,
}

#[derive(Debug, Deserialize)]
struct GitRef {
    name: String,
}

impl HfHub {
    pub fn new(repo: String, revision: String, manifest_file: String, endpoint: String) -> Self {
        Self {
            repo,
            revision,
            manifest_file,
            endpoint: trim_base(&endpoint),
            client: http::client().unwrap_or_default(),
        }
    }

    fn resolve_url(&self, revision: &str, file: &str) -> String {
        format!("{}/{}/resolve/{revision}/{file}", self.endpoint, self.repo)
    }

    /// The endpoint that answers which tags exist, used to turn a bare 404 into a list.
    fn refs_url(&self) -> String {
        format!("{}/api/models/{}/refs", self.endpoint, self.repo)
    }

    /// Tag naming convention for an exact version.
    fn tag_for(version: &semver::Version) -> String {
        format!("v{version}")
    }

    async fn signed_manifest(&self, revision: &str) -> Result<SignedBytes<Manifest>, Error> {
        let manifest_url = self.resolve_url(revision, &self.manifest_file);
        let sig_url = format!("{manifest_url}{SIG_SUFFIX}");

        let bytes = http::get_bytes(&self.client, &manifest_url, None).await?;
        let signature = http::get_bytes(&self.client, &sig_url, None).await?;

        let parsed: Manifest = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Corrupt(format!("manifest at {manifest_url}: {e}")))?;

        Ok(SignedBytes {
            bytes,
            signature,
            parsed,
        })
    }

    /// Does the repo have a tag for this version?
    ///
    /// Checked so a missing version fails with "no tag v1.2.3 (available: …)" rather
    /// than a bare 404 from the resolve endpoint.
    async fn tag_exists(&self, tag: &str) -> Result<bool, Error> {
        let url = self.refs_url();
        let bytes = match http::get_bytes(&self.client, &url, None).await {
            Ok(bytes) => bytes,
            // The refs API is a convenience; if it's unavailable, fall through and let
            // the resolve attempt produce the error.
            Err(e) => {
                tracing::debug!(error = %e, "could not list refs; skipping the tag check");
                return Ok(true);
            }
        };
        let refs: RepoRefs = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Network(format!("parsing refs for {}: {e}", self.repo)))?;
        Ok(refs.tags.iter().any(|t| t.name == tag))
    }
}

#[async_trait::async_trait]
impl Source for HfHub {
    async fn latest_manifest(&self) -> Result<SignedBytes<Manifest>, Error> {
        // `revision` is usually a moving branch, so this is "whatever it points at
        // now". The manifest's `version` is what we record as installed.
        self.signed_manifest(&self.revision).await
    }

    async fn manifest_for(
        &self,
        version: &semver::Version,
    ) -> Result<SignedBytes<Manifest>, Error> {
        let tag = Self::tag_for(version);
        if !self.tag_exists(&tag).await? {
            return Err(Error::Network(format!(
                "{} has no tag {tag} for version {version}",
                self.repo
            )));
        }
        let signed = self.signed_manifest(&tag).await?;

        // A tag whose manifest disagrees with it means the publisher tagged the wrong
        // commit. Refusing beats installing something other than what was asked for.
        if signed.parsed.version != *version {
            return Err(Error::Corrupt(format!(
                "tag {tag} contains a manifest for version {} — the tag and manifest disagree",
                signed.parsed.version
            )));
        }
        Ok(signed)
    }

    async fn fetch_artifact(
        &self,
        manifest: &Manifest,
        dest_dir: &Path,
        progress: ProgressSink,
    ) -> Result<FetchedArtifact, Error> {
        tokio::fs::create_dir_all(dest_dir)
            .await
            .map_err(|e| Error::Io {
                path: dest_dir.to_path_buf(),
                source: e,
            })?;

        // A model manifest's `url` is a path *within the repo*, unlike GitHub's
        // absolute asset URLs — so resolve it against the same revision the manifest
        // came from, keeping the artifact and its manifest consistent.
        let (artifact_url, sig_url) = if manifest.url.starts_with("https://") {
            (manifest.url.clone(), manifest.sig_url.clone())
        } else {
            (
                self.resolve_url(&self.revision, &manifest.url),
                self.resolve_url(&self.revision, &manifest.sig_url),
            )
        };

        // Untrusted until the signature is checked, so refuse anything that isn't a
        // bare filename.
        let artifact_name = super::github::safe_file_name(&manifest.url)?;
        let artifact = dest_dir.join(&artifact_name);
        let signature = dest_dir.join(format!("{artifact_name}{SIG_SUFFIX}"));

        let bytes =
            http::download_to(&self.client, &artifact_url, &artifact, None, &progress).await?;

        let sig_bytes = http::get_bytes(&self.client, &sig_url, None).await?;
        tokio::fs::write(&signature, &sig_bytes)
            .await
            .map_err(|e| Error::Io {
                path: signature.clone(),
                source: e,
            })?;

        Ok(FetchedArtifact {
            artifact,
            signature,
            bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The public Hub, as a literal rather than read from `crate::config`'s default — the
    /// literal is what a robot that configures no endpoint sends, and taking it from the same
    /// constant would let a typo satisfy both sides.
    fn source() -> HfHub {
        HfHub::new(
            "ORG/gait-model".into(),
            "main".into(),
            "manifest.json".into(),
            "https://huggingface.co".into(),
        )
    }

    #[test]
    fn resolve_url_matches_the_hub_layout() {
        assert_eq!(
            source().resolve_url("main", "manifest.json"),
            "https://huggingface.co/ORG/gait-model/resolve/main/manifest.json"
        );
    }

    /// **A configured Hub host is where both requests go.**
    ///
    /// Two endpoints, one field: a mirror that were used for the resolve but not for the refs
    /// lookup would answer "has no tag v1.2.3 (available: …)" for a version that is there —
    /// which reads as a withdrawn release and sends somebody to the wrong repository.
    #[test]
    fn a_configured_endpoint_replaces_the_public_one_in_both_urls() {
        let hub = HfHub::new(
            "ORG/gait-model".into(),
            "main".into(),
            "manifest.json".into(),
            "https://hub.internal".into(),
        );
        assert_eq!(
            hub.resolve_url("main", "manifest.json"),
            "https://hub.internal/ORG/gait-model/resolve/main/manifest.json"
        );
        assert_eq!(
            hub.refs_url(),
            "https://hub.internal/api/models/ORG/gait-model/refs"
        );
    }

    /// The trailing-slash case, for the reason `source::trim_base` gives: a doubled slash is
    /// a different path, and the only symptom is "no manifest for version …".
    #[test]
    fn a_trailing_slash_on_the_endpoint_is_not_doubled() {
        let hub = HfHub::new(
            "ORG/gait-model".into(),
            "main".into(),
            "manifest.json".into(),
            "https://hub.internal/".into(),
        );
        assert_eq!(
            hub.resolve_url("main", "m.json"),
            "https://hub.internal/ORG/gait-model/resolve/main/m.json"
        );
        assert_eq!(
            hub.refs_url(),
            "https://hub.internal/api/models/ORG/gait-model/refs"
        );
    }

    #[test]
    fn exact_versions_map_to_v_prefixed_tags() {
        assert_eq!(HfHub::tag_for(&semver::Version::new(3, 1, 0)), "v3.1.0");
    }

    #[test]
    fn refs_parsing_tolerates_extra_fields() {
        let json = serde_json::json!({
            "branches": [{ "name": "main", "targetCommit": "abc" }],
            "tags": [{ "name": "v1.0.0", "targetCommit": "def" }],
            "converts": []
        });
        let refs: RepoRefs = serde_json::from_value(json).unwrap();
        assert_eq!(refs.tags.len(), 1);
        assert_eq!(refs.tags[0].name, "v1.0.0");
    }

    #[test]
    fn missing_tags_field_is_not_an_error() {
        let refs: RepoRefs = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(refs.tags.is_empty());
    }
}
