//! GitHub Releases source — the daemon channel.
//!
//! "Latest" is resolved by listing releases and taking the highest **semver** among
//! tags matching `tag_prefix`, not by using `/releases/latest`: that endpoint is
//! repo-wide, so it breaks the moment a second channel shares the repo, and it
//! answers "most recently published" rather than "highest version" — which differ as
//! soon as you publish a patch to an older line.
//! See `docs/design/updater-design.md` §6.
//!
//! Nothing here is trusted. Tags, release names and asset URLs are all attacker- or
//! mistake-influenced; the only thing that makes an artifact acceptable is the
//! minisign signature the caller checks afterwards.

use std::path::Path;

use serde::Deserialize;

use crate::Error;
use crate::manifest::Manifest;
use crate::source::{FetchedArtifact, ProgressSink, SignedBytes, Source, http, trim_base};

/// Releases fetched per page when scanning for the newest tag. One page is plenty
/// for any real channel; further pages are fetched only if a page comes back full.
const PER_PAGE: usize = 100;
const MAX_PAGES: usize = 5;

/// The signature that accompanies every signed file.
const SIG_SUFFIX: &str = ".minisig";

/// What the release-asset API needs to return bytes rather than JSON metadata.
const OCTET_STREAM: &str = "application/octet-stream";

/// Where a GitHub Releases source talks to.
///
/// Two hosts rather than one base because they are two services: `api` answers metadata and
/// serves a private release's assets, while `download` is the host a *manifest's own* `url`
/// names. A mirror can reasonably serve one and proxy the other, and keeping them apart is
/// what lets `split_release_url` recognise our own download URLs on a host that is not
/// github.com — the recognition that makes a private repository's artifacts fetchable at all.
///
/// Configured so that a robot which must not reach the public service can be pointed at a
/// private one with config alone. Defaults live in `crate::config`, where every other
/// schema default lives, so a robot that names neither field sends exactly the URLs it
/// sent before these existed.
#[derive(Debug, Clone)]
pub struct Hosts {
    api: String,
    download: String,
}

impl Hosts {
    pub fn new(api: impl Into<String>, download: impl Into<String>) -> Self {
        Self {
            api: trim_base(&api.into()),
            download: trim_base(&download.into()),
        }
    }
}

pub struct GithubReleases {
    repo: String,
    tag_prefix: String,
    manifest_asset: String,
    ref_tag_prefix: String,
    staging_tag_prefix: String,
    hosts: Hosts,
    client: reqwest::Client,
}

/// Only the fields we use. GitHub adds fields freely, so this is deliberately not
/// exhaustive.
#[derive(Debug, Deserialize)]
struct Release {
    /// What [`GithubReleases::release_for_tag`] re-asks by when the tag answer lists nothing.
    id: u64,
    tag_name: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    assets: Vec<Asset>,
}

#[derive(Debug, Deserialize)]
struct Asset {
    name: String,
    /// The API endpoint for this asset, `/repos/{owner}/{repo}/releases/assets/{id}`.
    ///
    /// Used in preference to `browser_download_url` because that one **404s on a private
    /// repository**, with or without a token — verified against this repo. The API endpoint
    /// serves the bytes with a token and `Accept: application/octet-stream`, and works for
    /// public repos too, so there is one path rather than two.
    url: String,
}

impl GithubReleases {
    pub fn new(
        repo: String,
        tag_prefix: String,
        manifest_asset: String,
        ref_tag_prefix: String,
        staging_tag_prefix: String,
        hosts: Hosts,
    ) -> Self {
        Self {
            repo,
            tag_prefix,
            ref_tag_prefix,
            staging_tag_prefix,
            manifest_asset,
            hosts,
            // A failure here means a broken TLS setup, which is fatal for every
            // request anyway; fall back to a default client so construction stays
            // infallible and the error surfaces on first use.
            client: http::client().unwrap_or_default(),
        }
    }

    /// The API URL answering "the release published under this tag".
    fn release_by_tag_url(&self, tag: &str) -> String {
        format!("{}/repos/{}/releases/tags/{tag}", self.hosts.api, self.repo)
    }

    /// The API URL answering "this release", identified by its numeric id.
    fn release_by_id_url(&self, id: u64) -> String {
        format!("{}/repos/{}/releases/{id}", self.hosts.api, self.repo)
    }

    /// One page of the release listing.
    fn releases_page_url(&self, page: usize) -> String {
        format!(
            "{}/repos/{}/releases?per_page={PER_PAGE}&page={page}",
            self.hosts.api, self.repo
        )
    }

    /// The prefix one of *our* release-download URLs carries.
    ///
    /// Mirrors [`Self::split_release_url`], which is the only reader, so the two cannot
    /// drift: a prefix built one way and matched another is a private artifact that 404s.
    fn download_prefix(&self) -> String {
        format!("{}/{}/releases/download/", self.hosts.download, self.repo)
    }

    fn tag_for(&self, version: &semver::Version) -> String {
        format!("{}{}", self.tag_prefix, version)
    }

    /// The tag a named ref resolves to: `daemon-dev-` + the branch name.
    ///
    /// The ref is appended verbatim. Branch names are already valid git refs, so slashes in
    /// `feature/foo` need no handling — and rewriting them would resolve to a tag that does
    /// not exist, failing with "release not found" instead of anything informative.
    fn ref_tag_for(&self, git_ref: &str) -> String {
        format!("{}{}", self.ref_tag_prefix, git_ref)
    }

    /// The tag a release candidate lives under: `daemon-staging-v` + the version.
    fn staging_tag_for(&self, version: &semver::Version) -> String {
        format!("{}{}", self.staging_tag_prefix, version)
    }

    /// The release behind `tag`, with its assets.
    ///
    /// **An empty asset list from the tag lookup is asked again by id.** From 2026-09-23 the
    /// unauthenticated `releases/tags/{tag}` answer — and the listing — carry no assets for any
    /// release published since, while `releases/{id}` for the same release lists every one. A
    /// board asks without a token, so it saw a finished build as an empty one and refused it,
    /// every time. The second request costs nothing on a release that really is empty, and it
    /// is what makes [`Error::ReleaseNotReady`] mean what it says.
    async fn release_for_tag(&self, tag: &str) -> Result<Release, Error> {
        let url = self.release_by_tag_url(tag);
        let release = self.release_at(&url, tag).await?;
        if !release.assets.is_empty() {
            return Ok(release);
        }

        let url = self.release_by_id_url(release.id);
        tracing::debug!(%tag, id = release.id, "tag lookup listed no assets; asking by id");
        self.release_at(&url, tag).await
    }

    async fn release_at(&self, url: &str, tag: &str) -> Result<Release, Error> {
        let bytes = http::get_bytes(&self.client, url, Some("application/vnd.github+json")).await?;
        serde_json::from_slice(&bytes)
            .map_err(|e| Error::Network(format!("parsing release {tag}: {e}")))
    }

    /// Highest **stable** semver among matching tags.
    ///
    /// Drafts and prereleases are skipped: a draft isn't published, and a prerelease is
    /// by definition not what a client robot should install. Reach one with an explicit
    /// `--version` or `--ref`.
    ///
    /// Two independent reasons a build is skipped — GitHub's `prerelease` flag *and* a
    /// semver prerelease component — because dev builds (`0.2.0-dev.5.abc1234`) must
    /// never become `latest` for the fleet, and relying on someone remembering a
    /// checkbox is not a safeguard.
    async fn newest_version(&self) -> Result<semver::Version, Error> {
        self.newest_under(&self.tag_prefix, false).await
    }

    /// The newest release candidate, which is the same scan with the prerelease flag allowed.
    ///
    /// Only the *GitHub* flag is allowed, never a semver prerelease: candidates carry the plain
    /// version they will be promoted under (`0.3.0`), while dev builds carry `0.3.0-dev.17.abc`
    /// and live under a third prefix. So the two exclusions in [`Self::newest_under`] are not
    /// redundant here — dropping one still excludes branch builds, which is the point.
    async fn newest_staging_version(&self) -> Result<semver::Version, Error> {
        self.newest_under(&self.staging_tag_prefix, true).await
    }

    /// Highest version among tags carrying `prefix`.
    ///
    /// `allow_flagged_prerelease` is the whole difference between the stable channel and the
    /// staging one, and it is a parameter rather than a field so that the call site — one of
    /// exactly two — states which channel it means.
    async fn newest_under(
        &self,
        prefix: &str,
        allow_flagged_prerelease: bool,
    ) -> Result<semver::Version, Error> {
        let mut best: Option<semver::Version> = None;

        for page in 1..=MAX_PAGES {
            let url = self.releases_page_url(page);
            let bytes =
                http::get_bytes(&self.client, &url, Some("application/vnd.github+json")).await?;
            let releases: Vec<Release> = serde_json::from_slice(&bytes)
                .map_err(|e| Error::Network(format!("parsing releases: {e}")))?;

            let count = releases.len();
            for release in releases {
                if release.draft || (release.prerelease && !allow_flagged_prerelease) {
                    continue;
                }
                if let Some(version) = version_under(prefix, &release.tag_name)
                    // A semver prerelease is a dev build, whatever the release was flagged as.
                    && version.pre.is_empty()
                    && best.as_ref().is_none_or(|b| version > *b)
                {
                    best = Some(version);
                }
            }

            // A short page is the last page.
            if count < PER_PAGE {
                break;
            }
        }

        best.ok_or_else(|| {
            Error::Network(format!(
                "no releases in {} with tag prefix {prefix:?}",
                self.repo
            ))
        })
    }

    /// The API download URL for a named asset. Pair it with [`OCTET_STREAM`].
    ///
    /// A release with *no* assets at all is reported separately, because it is not the same
    /// situation as a missing one. Assets are uploaded at the end of a release build, so an
    /// empty release is every release for the few minutes before that finishes — a state to
    /// wait out, not a fault to investigate. A release that has assets but not this one is the
    /// fault case, and there the list of what *is* there is the whole diagnostic.
    fn asset_url(&self, release: &Release, name: &str) -> Result<String, Error> {
        if release.assets.is_empty() {
            return Err(Error::ReleaseNotReady {
                repo: self.repo.clone(),
                tag: release.tag_name.clone(),
            });
        }

        release
            .assets
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.url.clone())
            .ok_or_else(|| {
                let available: Vec<_> = release.assets.iter().map(|a| a.name.as_str()).collect();
                Error::Network(format!(
                    "release {} has no asset named {name:?} (has: {})",
                    release.tag_name,
                    available.join(", ")
                ))
            })
    }

    /// Split one of our own release-download URLs into `(tag, asset name)`.
    ///
    /// `None` for anything else, including another repository's release URL — which matters
    /// because the manifest this comes from is unverified at that point, so a URL naming a
    /// foreign repo must not become an authenticated API request against it.
    fn split_release_url(&self, url: &str) -> Option<(String, String)> {
        let prefix = self.download_prefix();
        let (tag, name) = url.strip_prefix(&prefix)?.split_once('/')?;
        Some((tag.to_owned(), name.to_owned()))
    }

    /// Where to actually fetch a URL from a signed manifest, and with which `Accept`.
    ///
    /// A private repo's `releases/download/...` URL 404s even with a token, so one of ours is
    /// re-resolved through the release API. Anything else is fetched verbatim — a manifest
    /// pointing at a CDN keeps working, and the bytes are hash- and signature-checked either
    /// way.
    async fn resolve_download(&self, url: &str) -> Result<(String, Option<&'static str>), Error> {
        let Some((tag, name)) = self.split_release_url(url) else {
            return Ok((url.to_owned(), None));
        };

        let release = self.release_for_tag(&tag).await?;
        let api_url = self.asset_url(&release, &name)?;
        tracing::debug!(%tag, %name, "resolved asset through the release API");
        Ok((api_url, Some(OCTET_STREAM)))
    }

    async fn signed_manifest(&self, tag: &str) -> Result<SignedBytes<Manifest>, Error> {
        let release = self.release_for_tag(tag).await?;

        let manifest_url = self.asset_url(&release, &self.manifest_asset)?;
        let sig_name = format!("{}{SIG_SUFFIX}", self.manifest_asset);
        let sig_url = self.asset_url(&release, &sig_name)?;

        let bytes = http::get_bytes(&self.client, &manifest_url, Some(OCTET_STREAM)).await?;
        let signature = http::get_bytes(&self.client, &sig_url, Some(OCTET_STREAM)).await?;

        // Parsed for convenience only; the *bytes* are what the caller verifies,
        // since the signature covers exactly what was received.
        let parsed: Manifest = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Corrupt(format!("manifest at {manifest_url}: {e}")))?;

        Ok(SignedBytes {
            bytes,
            signature,
            parsed,
        })
    }
}

#[async_trait::async_trait]
impl Source for GithubReleases {
    async fn latest_manifest(&self) -> Result<SignedBytes<Manifest>, Error> {
        let version = self.newest_version().await?;
        let tag = self.tag_for(&version);
        tracing::debug!(repo = %self.repo, %tag, "resolved latest");
        self.signed_manifest(&tag).await
    }

    async fn manifest_for(
        &self,
        version: &semver::Version,
    ) -> Result<SignedBytes<Manifest>, Error> {
        self.signed_manifest(&self.tag_for(version)).await
    }

    async fn manifest_at_ref(&self, git_ref: &str) -> Result<SignedBytes<Manifest>, Error> {
        let tag = self.ref_tag_for(git_ref);
        tracing::debug!(repo = %self.repo, %tag, %git_ref, "resolving ref");
        self.signed_manifest(&tag).await
    }

    async fn staging_manifest(&self) -> Result<SignedBytes<Manifest>, Error> {
        let version = self.newest_staging_version().await?;
        let tag = self.staging_tag_for(&version);
        tracing::debug!(repo = %self.repo, %tag, "resolved newest candidate");
        self.signed_manifest(&tag).await
    }

    async fn staging_manifest_for(
        &self,
        version: &semver::Version,
    ) -> Result<SignedBytes<Manifest>, Error> {
        self.signed_manifest(&self.staging_tag_for(version)).await
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

        // The filename comes from a signed manifest, but the signature is only
        // checked *after* download — so treat it as untrusted here and refuse
        // anything that isn't a bare name.
        let artifact_name = safe_file_name(&manifest.url)?;
        let artifact = dest_dir.join(&artifact_name);
        let signature = dest_dir.join(format!("{artifact_name}{SIG_SUFFIX}"));

        let (artifact_url, accept) = self.resolve_download(&manifest.url).await?;
        let bytes =
            http::download_to(&self.client, &artifact_url, &artifact, accept, &progress).await?;

        let (sig_url, sig_accept) = self.resolve_download(&manifest.sig_url).await?;
        let sig_bytes = http::get_bytes(&self.client, &sig_url, sig_accept).await?;
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

/// Parse a version out of a tag carrying `prefix`, or `None` if the tag is not one of ours.
///
/// Free-standing because two prefixes now feed it — stable and staging — and a method reading
/// `self.tag_prefix` invited exactly the bug where a staging scan silently measured stable tags.
fn version_under(prefix: &str, tag: &str) -> Option<semver::Version> {
    semver::Version::parse(tag.strip_prefix(prefix)?).ok()
}

/// Extract a filename from a URL, refusing anything that could escape `dest_dir`.
///
/// A manifest is signed, but this runs *before* verification, and a compromised
/// publisher is exactly the case where writing to an arbitrary path would matter.
pub(crate) fn safe_file_name(url: &str) -> Result<String, Error> {
    let tail = url.rsplit('/').next().unwrap_or_default();
    let tail = tail.split(['?', '#']).next().unwrap_or_default();

    let looks_safe = !tail.is_empty()
        && tail != "."
        && tail != ".."
        && !tail.contains('/')
        && !tail.contains('\\')
        && !tail.contains('\0');

    if looks_safe {
        Ok(tail.to_owned())
    } else {
        Err(Error::Verification(format!(
            "manifest url {url:?} does not end in a usable filename"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The public hosts, as literals rather than read from `crate::config`'s defaults.
    ///
    /// The literals are the point. `crate::config` asserts that the serde defaults equal
    /// these two strings and this asserts that these two strings build the URLs the daemon
    /// has always sent; a single shared constant would let a typo satisfy both.
    fn public_hosts() -> Hosts {
        Hosts::new("https://api.github.com", "https://github.com")
    }

    fn source() -> GithubReleases {
        GithubReleases::new(
            "ORG/robot-daemon".into(),
            "daemon-v".into(),
            "manifest.json".into(),
            "daemon-dev-".into(),
            "daemon-staging-v".into(),
            public_hosts(),
        )
    }

    /// **A robot that configures no host sends exactly the URLs it sent before the fields
    /// existed.**
    ///
    /// Each of the four is a request this daemon makes. They are asserted as whole strings
    /// because the failure they guard is a default that drifts: it is invisible in a config
    /// file that never names the field, and on a board it surfaces as an update that never
    /// arrives rather than as anything about a host.
    #[test]
    fn the_public_hosts_build_exactly_the_historic_urls() {
        let s = source();
        assert_eq!(
            s.release_by_tag_url("daemon-v1.4.2"),
            "https://api.github.com/repos/ORG/robot-daemon/releases/tags/daemon-v1.4.2"
        );
        assert_eq!(
            s.release_by_id_url(7),
            "https://api.github.com/repos/ORG/robot-daemon/releases/7"
        );
        assert_eq!(
            s.releases_page_url(2),
            "https://api.github.com/repos/ORG/robot-daemon/releases?per_page=100&page=2"
        );
        assert_eq!(
            s.download_prefix(),
            "https://github.com/ORG/robot-daemon/releases/download/"
        );
    }

    /// **A configured host is where the request actually goes.**
    ///
    /// The whole reason the field exists is a robot that must not reach the public service,
    /// so a base that were echoed into a log but not into the URL would read as configured
    /// and egress anyway. The two hosts differ here on purpose: one value serving both would
    /// hide a `format!` that reached for the wrong field.
    #[test]
    fn a_configured_host_replaces_the_public_one_in_every_url() {
        let s = GithubReleases::new(
            "ORG/robot-daemon".into(),
            "daemon-v".into(),
            "manifest.json".into(),
            "daemon-dev-".into(),
            "daemon-staging-v".into(),
            Hosts::new("https://gh.internal/api/v3", "https://gh.internal/dl"),
        );

        assert_eq!(
            s.release_by_tag_url("daemon-v1.4.2"),
            "https://gh.internal/api/v3/repos/ORG/robot-daemon/releases/tags/daemon-v1.4.2"
        );
        assert_eq!(
            s.release_by_id_url(7),
            "https://gh.internal/api/v3/repos/ORG/robot-daemon/releases/7"
        );
        assert_eq!(
            s.releases_page_url(1),
            "https://gh.internal/api/v3/repos/ORG/robot-daemon/releases?per_page=100&page=1"
        );
        assert_eq!(
            s.download_prefix(),
            "https://gh.internal/dl/ORG/robot-daemon/releases/download/"
        );

        // Recognition of our own download URLs follows the configured host, because it is
        // matching rather than guessing — and that is what makes a private repository's
        // artifacts fetchable, since their `releases/download/...` URLs 404 even with a token.
        assert_eq!(
            s.split_release_url(
                "https://gh.internal/dl/ORG/robot-daemon/releases/download/daemon-v1.4.2/daemon-1.4.2.tar.zst"
            ),
            Some((
                "daemon-v1.4.2".to_owned(),
                "daemon-1.4.2.tar.zst".to_owned()
            ))
        );
        // The public host is now somebody else's, and `resolve_download` runs before the
        // manifest's signature is checked — so treating it as ours would aim an
        // authenticated API request at a URL an unverified manifest chose.
        assert_eq!(
            s.split_release_url(
                "https://github.com/ORG/robot-daemon/releases/download/daemon-v1.4.2/x.tar.zst"
            ),
            None
        );
    }

    /// **A base written with a trailing slash does not become a doubled slash.**
    ///
    /// `https://github.com/` is how half of everyone writes a host, and `format!("{base}/repos…")`
    /// turns that into `https://github.com//repos/…`. A server reads the doubled slash as a
    /// different path, so the request 404s while the config file looks right — and the only
    /// symptom is `no releases in ORG/… with tag prefix "daemon-v"`, which names neither the
    /// slash nor the field that carried it.
    #[test]
    fn a_trailing_slash_on_a_configured_host_is_not_doubled() {
        let s = GithubReleases::new(
            "ORG/robot-daemon".into(),
            "daemon-v".into(),
            "manifest.json".into(),
            "daemon-dev-".into(),
            "daemon-staging-v".into(),
            Hosts::new("https://gh.internal/", "https://gh.internal//"),
        );
        assert_eq!(
            s.release_by_tag_url("t"),
            "https://gh.internal/repos/ORG/robot-daemon/releases/tags/t"
        );
        assert_eq!(
            s.download_prefix(),
            "https://gh.internal/ORG/robot-daemon/releases/download/"
        );
    }

    #[test]
    fn tags_round_trip() {
        let s = source();
        let v = semver::Version::new(1, 4, 2);
        assert_eq!(s.tag_for(&v), "daemon-v1.4.2");
        assert_eq!(version_under(&s.tag_prefix, "daemon-v1.4.2"), Some(v));
    }

    #[test]
    fn staging_tags_round_trip() {
        let s = source();
        let v = semver::Version::new(0, 3, 0);
        assert_eq!(s.staging_tag_for(&v), "daemon-staging-v0.3.0");
        assert_eq!(
            version_under("daemon-staging-v", "daemon-staging-v0.3.0"),
            Some(v)
        );
    }

    /// The two channels must not read each other's tags.
    ///
    /// This is the failure that would matter and would not look like one: a staging scan that
    /// silently matched `daemon-v*` would report the newest *stable* release as the candidate,
    /// and `--staging` would install what a plain `apply` already installs while claiming to
    /// test something. It holds because `daemon-staging-v0.3.0` does not start with `daemon-v`
    /// — an accident of naming, so it is pinned here rather than left to be re-derived.
    #[test]
    fn the_two_channels_cannot_read_each_others_tags() {
        assert_eq!(version_under("daemon-v", "daemon-staging-v0.3.0"), None);
        assert_eq!(version_under("daemon-staging-v", "daemon-v0.3.0"), None);
        // And neither reads a dev build, which is what keeps `--staging` from resolving to a
        // branch someone pushed.
        assert_eq!(
            version_under("daemon-staging-v", "daemon-dev-my-branch"),
            None
        );
    }

    /// A ref becomes a dev tag, and the ref is appended verbatim.
    ///
    /// The slash case is the one that matters: `feature/foo` is a valid branch name, so
    /// anything that sanitised it would resolve to a tag nobody published and fail with
    /// "release not found" rather than anything that points at the cause.
    #[test]
    fn refs_become_dev_tags_verbatim() {
        let s = source();
        assert_eq!(s.ref_tag_for("my-branch"), "daemon-dev-my-branch");
        assert_eq!(s.ref_tag_for("feature/foo"), "daemon-dev-feature/foo");
    }

    /// **A dev tag must never be mistaken for a release.** `version_under` drives
    /// `newest_version`, which is what the fleet installs — so if a dev tag parsed as a
    /// version here, a branch build could become `latest` for every robot. That is the
    /// failure the two independent guards exist to prevent, and this is the first of them.
    #[test]
    fn a_dev_tag_is_not_a_release_version() {
        let s = source();
        assert_eq!(version_under(&s.tag_prefix, "daemon-dev-my-branch"), None);
        // Even when the dev tag ends in something version-shaped.
        assert_eq!(version_under(&s.tag_prefix, "daemon-dev-0.2.0"), None);
        // And the staging stream stays separate too.
        assert_eq!(version_under(&s.tag_prefix, "daemon-staging-v0.2.0"), None);
    }

    /// Another channel's tags in the same repo must be ignored, not misparsed.
    #[test]
    fn foreign_tags_are_ignored() {
        let s = source();
        assert_eq!(version_under(&s.tag_prefix, "model-v3.0.0"), None);
        assert_eq!(version_under(&s.tag_prefix, "v1.0.0"), None);
        assert_eq!(version_under(&s.tag_prefix, "daemon-vnot-a-version"), None);
    }

    #[test]
    fn asset_lookup_lists_what_was_available_on_failure() {
        let release = Release {
            id: 1,
            tag_name: "daemon-v1.0.0".into(),
            draft: false,
            prerelease: false,
            assets: vec![Asset {
                name: "other.txt".into(),
                url: "https://api.github.com/repos/ORG/robot-daemon/releases/assets/1".into(),
            }],
        };
        let err = source().asset_url(&release, "manifest.json").unwrap_err();
        // A support ticket needs to see what *was* there.
        assert!(err.to_string().contains("other.txt"), "{err}");
    }

    /// A release whose build has not uploaded yet must not read as a broken release.
    ///
    /// This is what an operator hits by running `update apply --staging` in the minutes after
    /// publishing, and the old answer — `network error: ... has no asset named "manifest.json"
    /// (has: )` — pointed at the two things that were not wrong: the release, and the network.
    #[test]
    fn an_empty_release_says_its_build_has_not_finished() {
        let release = Release {
            id: 1,
            tag_name: "daemon-staging-v0.5.1".into(),
            draft: false,
            prerelease: true,
            assets: vec![],
        };

        let err = source().asset_url(&release, "manifest.json").unwrap_err();
        assert!(
            matches!(err, Error::ReleaseNotReady { .. }),
            "an empty release is a wait, not a fetch failure, got {err:?}"
        );

        let msg = err.to_string();
        // The tag, so it is clear *which* release, and where to watch it land.
        assert!(msg.contains("daemon-staging-v0.5.1"), "{msg}");
        assert!(
            msg.contains("https://github.com/ORG/robot-daemon/releases/tag/daemon-staging-v0.5.1"),
            "{msg}"
        );
        // And none of the vocabulary that sent people debugging the wrong thing.
        assert!(!msg.contains("network"), "{msg}");
        assert!(!msg.contains("no asset named"), "{msg}");
    }

    /// **A manifest must not redirect the asset lookup at another repository.**
    ///
    /// `resolve_download` runs before the manifest's signature is checked, so a URL naming a
    /// foreign repo must be left alone rather than turned into an API request carrying our
    /// token.
    #[test]
    fn only_our_own_release_urls_are_split() {
        let s = source();

        assert_eq!(
            s.split_release_url(
                "https://github.com/ORG/robot-daemon/releases/download/daemon-dev-my-branch/daemon-0.2.0-dev.1.abc1234.tar.zst"
            ),
            Some((
                "daemon-dev-my-branch".to_owned(),
                "daemon-0.2.0-dev.1.abc1234.tar.zst".to_owned()
            ))
        );

        for foreign in [
            "https://github.com/attacker/repo/releases/download/v1/x.tar.zst",
            "https://cdn.example.com/daemon-1.0.0.tar.zst",
            // Ours, but not a release-download URL.
            "https://github.com/ORG/robot-daemon/archive/refs/heads/main.tar.gz",
        ] {
            assert_eq!(s.split_release_url(foreign), None, "{foreign}");
        }
    }

    #[test]
    fn accepts_a_plain_filename() {
        assert_eq!(
            safe_file_name("https://example.com/a/b/daemon-1.0.0.tar.zst").unwrap(),
            "daemon-1.0.0.tar.zst"
        );
        // Query strings are common on signed CDN URLs.
        assert_eq!(
            safe_file_name("https://example.com/x.tar.zst?token=abc").unwrap(),
            "x.tar.zst"
        );
    }

    /// The download path must not be steerable by a manifest.
    #[test]
    fn refuses_names_that_could_escape() {
        for url in [
            "https://example.com/",
            "https://example.com/..",
            "https://example.com/.",
            "https://example.com/a/",
        ] {
            assert!(safe_file_name(url).is_err(), "should refuse {url}");
        }
    }

    /// A dev build must never be selected as `latest`, even if whoever published it
    /// forgot to tick "prerelease". Fleet-wide auto-updates read `latest`.
    #[test]
    fn dev_versions_are_recognised_as_prereleases() {
        let s = source();
        let dev = version_under(&s.tag_prefix, "daemon-v0.2.0-dev.5.abc1234").unwrap();
        assert!(
            !dev.pre.is_empty(),
            "dev builds must carry a semver prerelease"
        );

        let stable = version_under(&s.tag_prefix, "daemon-v0.2.0").unwrap();
        assert!(stable.pre.is_empty());
        // And a dev build sorts *below* the release it precedes, so it can never look
        // like an upgrade from it.
        assert!(dev < stable);
    }

    /// GitHub adds response fields regularly; deserialisation must not break.
    #[test]
    fn unknown_release_fields_are_tolerated() {
        let json = serde_json::json!({
            "id": 7,
            "tag_name": "daemon-v1.0.0",
            "some_new_field": 42,
            "assets": [{
                "name": "manifest.json",
                "url": "https://api.github.com/repos/ORG/robot-daemon/releases/assets/7",
                "browser_download_url": "https://example/m.json",
                "another_new_field": true
            }]
        });
        let release: Release = serde_json::from_value(json).unwrap();
        assert_eq!(release.tag_name, "daemon-v1.0.0");
        assert!(!release.draft, "missing `draft` should default to false");
        // `url` is required, not defaulted: it is how assets are fetched, and a release whose
        // assets lack it is not something to paper over with an empty string that would fail
        // later as a confusing HTTP error.
        assert!(release.assets[0].url.contains("/releases/assets/"));
    }
}
