//! Curated model catalog. A row is downloaded only from an allowlisted host,
//! hashed while the bytes stream in, and promoted after the digest matches.
//! `library refresh` prints a changed sha256 and does not edit the catalog.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure};

use crate::compile::{compile_manifest_capped, write_model_image};
use crate::gguf::{canonical_llama_tensor, knolo_precision, parse_gguf_bytes_capped};
use crate::io::prefixed;
use crate::lockfile::{new_lockfile, pin_alias, read_lockfile, write_lockfile};
use crate::paths::check_relative_posix;
use crate::verify::verify_image;

const CATALOG_KIND: &str = "knolo.infer.library";

/// Llama 3 instruct turns: system, user, and assistant, with no tools and no images.
pub const LLAMA_CHAT_TEMPLATE: &str = "<|begin_of_text|>{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% endfor %}<|start_header_id|>assistant<|end_header_id|>\n\n";

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Catalog {
    pub kind: String,
    pub version: u32,
    pub models: Vec<CatalogRow>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogRow {
    pub id: String,
    pub display_name: String,
    pub architecture: String,
    pub adapter: String,
    pub repo: String,
    pub revision: String,
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub quant: String,
    pub context_reservation: u32,
    pub min_ram_bytes: u64,
    pub license_id: String,
    pub license_url: String,
    pub tags: Vec<String>,
    pub gated: bool,
    pub status: String,
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub tokenizer: Option<TokenizerSource>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TokenizerSource {
    pub repo: String,
    pub revision: String,
    pub filename: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone)]
pub struct FetchPolicy {
    pub allow_huggingface: bool,
    pub extra_hosts: Vec<String>,
    pub origin_override: Option<String>,
}

impl Default for FetchPolicy {
    fn default() -> Self {
        Self {
            allow_huggingface: true,
            extra_hosts: Vec::new(),
            origin_override: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InstalledModel {
    pub alias: String,
    pub model_image_root: String,
    pub path: String,
}

#[derive(Debug, Clone)]
pub struct RefreshChange {
    pub id: String,
    pub catalog_sha256: String,
    pub remote_sha256: String,
    pub remote_size: u64,
}

#[derive(Debug, Clone)]
pub struct PullOutcome {
    pub alias: String,
    pub model_image_root: String,
    pub path: String,
}

pub struct PullRequest<'a> {
    pub home: &'a Path,
    pub lock_path: &'a Path,
    pub row: &'a CatalogRow,
    pub policy: &'a FetchPolicy,
    pub accept_license: bool,
    pub ram_bytes: Option<u64>,
    pub token: Option<&'a str>,
    pub build_root: &'a DigestHex,
}

pub fn checked_in_catalog() -> Result<Catalog, InferFailure> {
    load_catalog(include_str!("../../../catalog/library.json"))
}

pub fn load_catalog(text: &str) -> Result<Catalog, InferFailure> {
    let value = crate::json::parse_strict_json(text, ErrorCode::ContractInvalid)?;
    let catalog: Catalog = serde_json::from_value(value).map_err(|err| {
        fail(
            ErrorCode::ContractInvalid,
            format!("model catalog: {err}"),
        )
    })?;
    validate_catalog(&catalog)?;
    Ok(catalog)
}

pub fn load_catalog_file(path: &Path) -> Result<Catalog, InferFailure> {
    let text = fs::read_to_string(path).map_err(|err| {
        fail(
            ErrorCode::ModelArtifactMissing,
            format!("cannot read model catalog: {err}"),
        )
    })?;
    load_catalog(&text)
}

pub fn find_row<'a>(catalog: &'a Catalog, id: &str) -> Option<&'a CatalogRow> {
    catalog.models.iter().find(|row| row.id == id)
}

/// Status shown by `library`. Architecture refusal wins over a RAM check.
pub fn display_status(row: &CatalogRow, ram_bytes: Option<u64>) -> &'static str {
    if row.status != "runnable" {
        return "not supported yet";
    }
    if let Some(ram) = ram_bytes {
        if row.min_ram_bytes > ram {
            return "too big for this machine";
        }
    }
    "runnable"
}

/// Real Llama rows have no quantized GPU kernel. The column stays honest about that.
pub fn gpu_label(row: &CatalogRow) -> &'static str {
    if row.status == "runnable" {
        "cpu only"
    } else {
        "not supported yet"
    }
}

pub fn host_allowed(host: &str, policy: &FetchPolicy) -> bool {
    if policy.extra_hosts.iter().any(|item| item == host) {
        return true;
    }
    if !policy.allow_huggingface {
        return false;
    }
    host == "huggingface.co"
        || host == "cdn-lfs.huggingface.co"
        || host == "cdn.hf.co"
        || host.ends_with(".cdn.hf.co")
}

pub fn list_installed(lock_path: &Path) -> Result<Vec<InstalledModel>, InferFailure> {
    let Some(lock) = read_lockfile(lock_path)? else {
        return Ok(Vec::new());
    };
    Ok(lock
        .models
        .iter()
        .map(|(alias, pin)| InstalledModel {
            alias: alias.clone(),
            model_image_root: pin.model_image_root.clone(),
            path: pin.model_image_path.clone(),
        })
        .collect())
}

pub fn remove_installed(home: &Path, lock_path: &Path, alias: &str) -> Result<(), InferFailure> {
    let Some(mut lock) = read_lockfile(lock_path)? else {
        return Err(fail(
            ErrorCode::ModelArtifactMissing,
            "infer lockfile is missing",
        ));
    };
    let pin = lock
        .models
        .get(alias)
        .ok_or_else(|| fail(ErrorCode::ModelArtifactMissing, "alias is not pinned"))?
        .clone();
    let kmodel = crate::paths::resolve_lock_relative(lock_path, &pin.model_image_path);
    let model_dir = installed_model_dir(home, &kmodel)?;
    fs::remove_dir_all(&model_dir).map_err(|err| {
        fail(
            ErrorCode::ModelArtifactMissing,
            format!("cannot remove {}: {err}", model_dir.display()),
        )
    })?;
    lock.models.remove(alias);
    write_lockfile(lock_path, &lock)
}

pub fn refresh_catalog(
    catalog: &Catalog,
    policy: &FetchPolicy,
    token: Option<&str>,
) -> Result<Vec<RefreshChange>, InferFailure> {
    let mut changes = Vec::new();
    for row in &catalog.models {
        if row.sha256.is_empty() {
            continue;
        }
        let url = file_url(policy, &row.repo, &row.revision, &row.filename, true)?;
        let body = curl_body(&resolve_url(&url, policy, token)?)?;
        let (remote_sha, remote_size) = parse_lfs_pointer(&body)?;
        let catalog_sha = digest_hex(&row.sha256)?;
        if remote_sha != catalog_sha.as_str() || remote_size != row.size_bytes {
            changes.push(RefreshChange {
                id: row.id.clone(),
                catalog_sha256: catalog_sha.as_str().to_string(),
                remote_sha256: remote_sha,
                remote_size,
            });
        }
    }
    Ok(changes)
}

pub fn pull_catalog_row(request: &PullRequest<'_>) -> Result<PullOutcome, InferFailure> {
    let row = request.row;
    if row.status != "runnable" || row.adapter != "knolo.llama.v1" {
        return Err(fail(
            ErrorCode::UnsupportedArchitecture,
            format!("{} is not supported yet", row.display_name),
        ));
    }
    if row.sha256.is_empty() {
        return Err(fail(
            ErrorCode::ModelArtifactMissing,
            "a catalog row without a sha256 cannot be pulled",
        ));
    }
    if let Some(ram) = request.ram_bytes {
        if row.min_ram_bytes > ram {
            return Err(fail(
                ErrorCode::InsufficientMemory,
                format!(
                    "{} needs {} bytes of RAM and this machine reports {ram}",
                    row.display_name, row.min_ram_bytes
                ),
            ));
        }
    }
    if !request.accept_license {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("license {} was not accepted", row.license_id),
        ));
    }
    if row.gated && request.token.is_none() {
        return Err(fail(
            ErrorCode::ModelArtifactMissing,
            "this model is gated; set HF_TOKEN",
        ));
    }
    let template = row.chat_template.as_deref().ok_or_else(|| {
        fail(
            ErrorCode::TemplateInvalid,
            "runnable catalog row is missing its chat template",
        )
    })?;
    let tokenizer = row.tokenizer.as_ref().ok_or_else(|| {
        fail(
            ErrorCode::TokenizerInvalid,
            "runnable catalog row is missing its tokenizer",
        )
    })?;
    let weight_url = file_url(
        request.policy,
        &row.repo,
        &row.revision,
        &row.filename,
        false,
    )?;
    // Refuse a host before creating directories. The download below uses this
    // already-checked URL, so a redirect is not probed twice.
    let weight_resolved = resolve_url(&weight_url, request.policy, request.token)?;
    same_directory(request.home, request.lock_path)?;
    fs::create_dir_all(request.home.join("models")).map_err(io_missing)?;
    let staging = request.home.join("staging").join(&row.id);
    if staging.exists() {
        // A previous attempt may have left a partial weight. Keep only the
        // weight file so resume still works, and drop a half-built image.
        for entry in fs::read_dir(&staging).map_err(io_missing)? {
            let entry = entry.map_err(io_missing)?;
            let name = entry.file_name();
            if name == row.filename.as_str() {
                continue;
            }
            let path = entry.path();
            if path.is_dir() {
                fs::remove_dir_all(&path).map_err(io_missing)?;
            } else {
                fs::remove_file(&path).map_err(io_missing)?;
            }
        }
    } else {
        fs::create_dir_all(&staging).map_err(io_missing)?;
    }
    let weight_path = staging.join(&row.filename);
    if let Err(err) = download_resolved(
        &weight_resolved,
        &weight_path,
        row.size_bytes,
        &row.sha256,
        request.token,
    ) {
        if err.message.contains("digest does not match") {
            let _ = fs::remove_file(&weight_path);
        }
        return Err(err);
    }
    let tokenizer_path = staging.join("tokenizer.json");
    download_file(
        &file_url(
            request.policy,
            &tokenizer.repo,
            &tokenizer.revision,
            &tokenizer.filename,
            false,
        )?,
        &tokenizer_path,
        tokenizer.size_bytes,
        &tokenizer.sha256,
        request.policy,
        request.token,
    )?;
    // The catalog digest names the raw Hugging Face file. The image embeds
    // the envelope so null stays inside the tokenizer object.
    install_tokenizer(&tokenizer_path)?;
    fs::write(staging.join("template.jinja"), template).map_err(io_missing)?;
    let weight_bytes = fs::read(&weight_path).map_err(io_missing)?;
    let parsed = parse_gguf_bytes_capped(&weight_bytes)?;
    let manifest = manifest_json(row, &parsed)?;
    fs::write(staging.join("manifest.json"), manifest).map_err(io_missing)?;
    let compiled = compile_manifest_capped(&staging.join("manifest.json"))?;
    write_model_image(&staging.join("model.kmodel"), &compiled.bytes)?;
    let dest = request.home.join("models").join(&row.id);
    promote_dir(&staging, &dest)?;
    let mut lock = match read_lockfile(request.lock_path)? {
        Some(existing) => existing,
        None => new_lockfile(request.build_root),
    };
    let image = verify_image(&compiled.bytes)?.image;
    pin_alias(
        &mut lock,
        &row.id,
        &format!("models/{}/model.kmodel", row.id),
        &image,
    )?;
    write_lockfile(request.lock_path, &lock)?;
    Ok(PullOutcome {
        alias: row.id.clone(),
        model_image_root: image.image_root()?.to_string(),
        path: format!("models/{}/model.kmodel", row.id),
    })
}

fn validate_catalog(catalog: &Catalog) -> Result<(), InferFailure> {
    if catalog.kind != CATALOG_KIND || catalog.version != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "model catalog kind and version must be knolo.infer.library 1",
        ));
    }
    if catalog.models.is_empty() || catalog.models.len() > 64 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "model catalog is empty or too large",
        ));
    }
    let mut seen = Vec::new();
    for row in &catalog.models {
        crate::lockfile::valid_alias(&row.id)?;
        if seen.iter().any(|id: &String| id == &row.id) {
            return Err(fail(
                ErrorCode::ContractInvalid,
                format!("duplicate catalog id {}", row.id),
            ));
        }
        seen.push(row.id.clone());
        if row.status != "runnable" && row.status != "not_supported" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "catalog status must be runnable or not_supported",
            ));
        }
        if !row.sha256.is_empty() {
            digest_hex(&row.sha256)?;
        }
        check_repo(&row.repo)?;
        check_token(&row.revision, "revision")?;
        check_relative_posix(&row.filename, ErrorCode::ContractInvalid)?;
        for tag in &row.tags {
            if !matches!(
                tag.as_str(),
                "instruct" | "base" | "uncensored" | "new"
            ) {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    format!("catalog tag {tag} is not allowlisted"),
                ));
            }
        }
        if row.status == "runnable" {
            if row.adapter != "knolo.llama.v1" {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "a runnable catalog row must use knolo.llama.v1",
                ));
            }
            if row.chat_template.as_deref() != Some(LLAMA_CHAT_TEMPLATE) {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "a runnable catalog row must use the Llama 3 chat template",
                ));
            }
            let Some(tokenizer) = &row.tokenizer else {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "a runnable catalog row needs a tokenizer",
                ));
            };
            check_repo(&tokenizer.repo)?;
            check_token(&tokenizer.revision, "revision")?;
            check_relative_posix(&tokenizer.filename, ErrorCode::ContractInvalid)?;
            digest_hex(&tokenizer.sha256)?;
            if tokenizer.size_bytes == 0 || row.size_bytes == 0 {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "a runnable catalog file size must be non-zero",
                ));
            }
            if !(16..=2048).contains(&row.context_reservation)
                || row.context_reservation % 16 != 0
            {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "catalog context reservation must be a multiple of 16 from 16 to 2048",
                ));
            }
        }
    }
    Ok(())
}

fn digest_hex(value: &str) -> Result<DigestHex, InferFailure> {
    DigestHex::parse(value)
}

fn check_repo(repo: &str) -> Result<(), InferFailure> {
    let Some((owner, name)) = repo.split_once('/') else {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "catalog repo must be owner/name",
        ));
    };
    if repo.matches('/').count() != 1
        || owner.is_empty()
        || name.is_empty()
        || owner == "."
        || owner == ".."
        || name == "."
        || name == ".."
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "catalog repo must be owner/name",
        ));
    }
    Ok(())
}

fn check_token(value: &str, label: &str) -> Result<(), InferFailure> {
    if (1..=64).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Ok(());
    }
    Err(fail(
        ErrorCode::ContractInvalid,
        format!("catalog {label} is invalid"),
    ))
}

fn file_url(
    policy: &FetchPolicy,
    repo: &str,
    revision: &str,
    filename: &str,
    raw: bool,
) -> Result<String, InferFailure> {
    check_repo(repo)?;
    check_token(revision, "revision")?;
    check_relative_posix(filename, ErrorCode::ContractInvalid)?;
    let origin = policy
        .origin_override
        .as_deref()
        .unwrap_or("https://huggingface.co");
    let kind = if raw { "raw" } else { "resolve" };
    Ok(format!("{origin}/{repo}/{kind}/{revision}/{filename}"))
}

fn download_file(
    url: &str,
    dest: &Path,
    expected_size: u64,
    expected_sha: &str,
    policy: &FetchPolicy,
    token: Option<&str>,
) -> Result<(), InferFailure> {
    let resolved = resolve_url(url, policy, token)?;
    download_resolved(&resolved, dest, expected_size, expected_sha, token)
}

fn download_resolved(
    resolved: &str,
    dest: &Path,
    expected_size: u64,
    expected_sha: &str,
    token: Option<&str>,
) -> Result<(), InferFailure> {
    let expected = digest_hex(expected_sha)?;
    let mut offset = dest.metadata().map(|meta| meta.len()).unwrap_or(0);
    if offset > expected_size {
        fs::remove_file(dest).map_err(io_missing)?;
        offset = 0;
    }
    if offset == expected_size && offset > 0 {
        let bytes = fs::read(dest).map_err(io_missing)?;
        if prefixed(sha256(&bytes)) == expected {
            return Ok(());
        }
        fs::remove_file(dest).map_err(io_missing)?;
        offset = 0;
    }
    let mut hasher = Sha256::new();
    if offset > 0 {
        let mut prefix = File::open(dest).map_err(io_missing)?;
        let mut buf = [0u8; 64 * 1024];
        let mut left = offset;
        while left > 0 {
            let want = usize::try_from(left.min(buf.len() as u64)).unwrap_or(buf.len());
            let read = prefix.read(&mut buf[..want]).map_err(io_missing)?;
            if read == 0 {
                return Err(fail(
                    ErrorCode::ModelArtifactMissing,
                    "staged download ended before its declared size",
                ));
            }
            hasher.update(&buf[..read]);
            left -= read as u64;
        }
    }
    let mut child = curl_command(token)
        .args(["--fail", "--max-redirs", "0", "--output", "-"])
        .args(continue_at(offset))
        .arg(resolved)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| {
            fail(
                ErrorCode::ModelArtifactMissing,
                format!("curl is required to download a model: {err}"),
            )
        })?;
    let mut stdout = child.stdout.take().ok_or_else(|| {
        fail(
            ErrorCode::ModelArtifactMissing,
            "curl did not return a download stream",
        )
    })?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(offset > 0)
        .write(true)
        .open(dest)
        .map_err(io_missing)?;
    if offset == 0 {
        file.set_len(0).map_err(io_missing)?;
    }
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = stdout.read(&mut buf).map_err(io_missing)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
        file.write_all(&buf[..read]).map_err(io_missing)?;
    }
    let status = child.wait().map_err(io_missing)?;
    let stderr = child
        .stderr
        .take()
        .and_then(|mut pipe| {
            let mut text = String::new();
            pipe.read_to_string(&mut text).ok()?;
            Some(text)
        })
        .unwrap_or_default();
    file.sync_all().map_err(io_missing)?;
    let size = file.metadata().map_err(io_missing)?.len();
    if !status.success() || size != expected_size {
        if size < expected_size {
            return Err(fail(
                ErrorCode::ModelArtifactMissing,
                format!(
                    "download interrupted at {size} of {expected_size} bytes{}",
                    curl_note(&stderr)
                ),
            ));
        }
        let _ = fs::remove_file(dest);
        return Err(fail(
            ErrorCode::ModelDigestMismatch,
            format!("download size does not match the catalog row{}", curl_note(&stderr)),
        ));
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&hasher.finalize());
    if prefixed(digest) != expected {
        let _ = fs::remove_file(dest);
        return Err(fail(
            ErrorCode::ModelDigestMismatch,
            "download digest does not match the catalog row",
        ));
    }
    Ok(())
}

fn continue_at(offset: u64) -> Vec<String> {
    if offset == 0 {
        Vec::new()
    } else {
        // An explicit Range keeps the remaining bytes on stdout. curl's
        // --continue-at flag refuses to resume when the output is a pipe.
        vec!["--header".into(), format!("Range: bytes={offset}-")]
    }
}

fn curl_note(stderr: &str) -> String {
    let trimmed = stderr.trim();
    if trimmed.is_empty() {
        String::new()
    } else {
        format!(": {}", trimmed.chars().take(240).collect::<String>())
    }
}

fn resolve_url(url: &str, policy: &FetchPolicy, token: Option<&str>) -> Result<String, InferFailure> {
    check_fetch_url(url, policy)?;
    let mut current = url.to_string();
    for _ in 0..5 {
        let headers = curl_head(&current, token)?;
        let status = header_status(&headers)?;
        if matches!(status, 301 | 302 | 303 | 307 | 308) {
            let location = header_value(&headers, "location").ok_or_else(|| {
                fail(
                    ErrorCode::ModelArtifactMissing,
                    "download redirect is missing a location",
                )
            })?;
            check_fetch_url(location, policy)?;
            current = location.to_string();
            continue;
        }
        if status == 200 || status == 206 {
            return Ok(current);
        }
        return Err(fail(
            ErrorCode::ModelArtifactMissing,
            format!("download request returned HTTP {status}"),
        ));
    }
    Err(fail(
        ErrorCode::ModelArtifactMissing,
        "download followed too many redirects",
    ))
}

fn check_fetch_url(url: &str, policy: &FetchPolicy) -> Result<(), InferFailure> {
    let host = url_host(url)?;
    if host_allowed(&host, policy) {
        return Ok(());
    }
    Err(fail(
        ErrorCode::ModelArtifactMissing,
        format!("download host is not allowlisted: {host}"),
    ))
}

fn url_host(url: &str) -> Result<String, InferFailure> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .ok_or_else(|| fail(ErrorCode::ModelArtifactMissing, "download URL must be http(s)"))?;
    if rest.contains('@') {
        return Err(fail(
            ErrorCode::ModelArtifactMissing,
            "download URL must not carry user info",
        ));
    }
    let authority = rest.split('/').next().unwrap_or(rest);
    if authority.is_empty() {
        return Err(fail(
            ErrorCode::ModelArtifactMissing,
            "download URL is missing a host",
        ));
    }
    let host = if let Some(host) = authority.strip_prefix('[') {
        host.split(']').next().unwrap_or(host)
    } else {
        authority.split(':').next().unwrap_or(authority)
    };
    if host.is_empty() {
        return Err(fail(
            ErrorCode::ModelArtifactMissing,
            "download URL is missing a host",
        ));
    }
    Ok(host.to_string())
}

fn curl_command(token: Option<&str>) -> Command {
    let mut command = Command::new("curl");
    command.args([
        "--silent",
        "--show-error",
        "--http1.1",
        "--noproxy",
        "*",
        "--max-time",
        "0",
        "--user-agent",
        "knolo-infer",
    ]);
    if let Some(token) = token {
        let header = format!("Authorization: Bearer {token}");
        command.arg("--header").arg(header);
    }
    command
}

fn curl_head(url: &str, token: Option<&str>) -> Result<String, InferFailure> {
    let output = curl_command(token)
        .args(["--head", "--max-redirs", "0", "--output", "/dev/null", "--dump-header", "-"])
        .arg(url)
        .output()
        .map_err(|err| {
            fail(
                ErrorCode::ModelArtifactMissing,
                format!("curl is required to download a model: {err}"),
            )
        })?;
    if !output.status.success() {
        return Err(fail(
            ErrorCode::ModelArtifactMissing,
            format!("download probe failed{}", curl_note(&String::from_utf8_lossy(&output.stderr))),
        ));
    }
    String::from_utf8(output.stdout).map_err(|_| {
        fail(
            ErrorCode::ModelArtifactMissing,
            "download probe headers are not UTF-8",
        )
    })
}

fn curl_body(url: &str) -> Result<String, InferFailure> {
    let output = curl_command(None)
        .args(["--fail", "--max-redirs", "0", "--max-time", "30"])
        .arg(url)
        .output()
        .map_err(|err| {
            fail(
                ErrorCode::ModelArtifactMissing,
                format!("curl is required to download a model: {err}"),
            )
        })?;
    if !output.status.success() {
        return Err(fail(
            ErrorCode::ModelArtifactMissing,
            format!(
                "catalog refresh failed{}",
                curl_note(&String::from_utf8_lossy(&output.stderr))
            ),
        ));
    }
    let text = String::from_utf8(output.stdout).map_err(|_| {
        fail(
            ErrorCode::ModelArtifactMissing,
            "catalog refresh body is not UTF-8",
        )
    })?;
    if text.len() > 4096 {
        return Err(fail(
            ErrorCode::ModelArtifactMissing,
            "catalog refresh expected an LFS pointer, not a weight file",
        ));
    }
    Ok(text)
}

fn header_status(headers: &str) -> Result<u16, InferFailure> {
    let line = headers.lines().find(|line| line.starts_with("HTTP/")).ok_or_else(|| {
        fail(
            ErrorCode::ModelArtifactMissing,
            "download probe did not return a status",
        )
    })?;
    let code = line.split_whitespace().nth(1).ok_or_else(|| {
        fail(
            ErrorCode::ModelArtifactMissing,
            "download probe status is invalid",
        )
    })?;
    code.parse::<u16>().map_err(|_| {
        fail(
            ErrorCode::ModelArtifactMissing,
            "download probe status is invalid",
        )
    })
}

fn header_value<'a>(headers: &'a str, name: &str) -> Option<&'a str> {
    headers.lines().rev().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        if key.eq_ignore_ascii_case(name) {
            Some(value.trim())
        } else {
            None
        }
    })
}

fn parse_lfs_pointer(text: &str) -> Result<(String, u64), InferFailure> {
    let mut oid = None;
    let mut size = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("oid sha256:") {
            oid = Some(format!("sha256-{}", rest.trim()));
        } else if let Some(rest) = line.strip_prefix("size ") {
            size = Some(rest.trim().parse::<u64>().map_err(|_| {
                fail(
                    ErrorCode::ModelArtifactMissing,
                    "LFS pointer size is invalid",
                )
            })?);
        }
    }
    match (oid, size) {
        (Some(oid), Some(size)) => Ok((oid, size)),
        _ => Err(fail(
            ErrorCode::ModelArtifactMissing,
            "catalog refresh did not receive an LFS pointer",
        )),
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(&hasher.finalize());
    out
}

/// Rewrite a raw Hugging Face `tokenizer.json` as `huggingface.tokenizers.v1`.
/// A Knolo token list, and a file that is already that envelope, stays as downloaded.
fn install_tokenizer(path: &Path) -> Result<(), InferFailure> {
    let bytes = fs::read(path).map_err(io_missing)?;
    let text = std::str::from_utf8(&bytes).map_err(|_| {
        fail(
            ErrorCode::TokenizerInvalid,
            "tokenizer bytes are not UTF-8",
        )
    })?;
    let parsed = match crate::json::parse_strict_json(text, ErrorCode::TokenizerInvalid) {
        Ok(value) => value,
        Err(err) if err.message == "JSON null is not allowed" => {
            crate::json::parse_strict_json_allowing_null(text, ErrorCode::TokenizerInvalid)?
        }
        Err(err) => return Err(err),
    };
    let kind = parsed.get("kind").and_then(|item| item.as_str());
    if matches!(
        kind,
        Some("knolo.llama.tokens.v1" | "knolo.micro.tokens.v1" | "huggingface.tokenizers.v1")
    ) {
        return Ok(());
    }
    if kind.is_some() {
        return Err(fail(
            ErrorCode::TokenizerInvalid,
            "tokenizer kind is not allowlisted",
        ));
    }
    if parsed.get("model").filter(|item| item.is_object()).is_none() {
        return Err(fail(
            ErrorCode::TokenizerInvalid,
            "huggingface tokenizer is missing its model",
        ));
    }
    let wrapped = json!({
        "kind": "huggingface.tokenizers.v1",
        "version": 1,
        "tokenizer": parsed,
    });
    let encoded = serde_json::to_vec(&wrapped).map_err(|_| {
        fail(
            ErrorCode::TokenizerInvalid,
            "huggingface tokenizer could not be encoded",
        )
    })?;
    if encoded.len() > infer_contracts::MAX_EMBEDDED_BYTES {
        return Err(fail(
            ErrorCode::TokenizerInvalid,
            "huggingface tokenizer exceeds the embedded size cap",
        ));
    }
    fs::write(path, &encoded).map_err(io_missing)?;
    Ok(())
}

fn manifest_json(row: &CatalogRow, file: &crate::gguf::GgufFile) -> Result<String, InferFailure> {
    let mut inventory = Vec::new();
    let mut precisions = Vec::new();
    for tensor in &file.tensors {
        if tensor.name == "rope_freqs.weight" {
            continue;
        }
        let mut shape = Vec::with_capacity(tensor.shape.len());
        for dim in &tensor.shape {
            shape.push(u32::try_from(*dim).map_err(|_| {
                fail(
                    ErrorCode::ModelImageInvalid,
                    "gguf tensor dimension exceeds u32",
                )
            })?);
        }
        let (name, shape) = canonical_llama_tensor(&tensor.name, &shape)?;
        let dtype = knolo_precision(tensor.tensor_type);
        if !precisions.iter().any(|item: &String| item == dtype) {
            precisions.push(dtype.to_string());
        }
        inventory.push(json!({
            "dtype": dtype,
            "name": name,
            "shape": shape,
        }));
    }
    if !inventory.iter().any(|item| item["name"] == "lm_head.weight") {
        if let Some(embed) = inventory
            .iter()
            .find(|item| item["name"] == "embed.weight")
            .cloned()
        {
            let mut head = embed;
            head["name"] = Value::String("lm_head.weight".into());
            inventory.push(head);
        }
    }
    precisions.sort();
    let mut extensions = serde_json::Map::new();
    if row.context_reservation > 16 {
        extensions.insert(
            "knolo.llama.context".into(),
            Value::from(row.context_reservation),
        );
    }
    let value = json!({
        "architecture": {"adapter": row.adapter, "family": row.architecture},
        "capabilities": ["text-generation"],
        "extensions": extensions,
        "generationDefaults": {
            "frequencyPenaltyMicros": 0,
            "maxOutputTokens": 64,
            "minPMillionths": 0,
            "presencePenaltyMicros": 0,
            "repetitionPenaltyMicros": 1000000,
            "temperatureMicros": 0,
            "topK": 1,
            "topPMillionths": 1000000
        },
        "kind": "knolo.infer.model-image",
        "license": {"acceptanceRequired": true, "id": row.license_id},
        "name": row.id,
        "precisions": precisions,
        "requirements": {"minimumRamBytes": row.min_ram_bytes, "minimumVramBytes": 0},
        "sources": [{
            "provider": "huggingface",
            "repository": row.repo,
            "revision": row.revision
        }],
        "specialTokens": {"additional": {}},
        "template": {"embedded": "template.jinja"},
        "tensorInventory": inventory,
        "tokenizer": {"embedded": "tokenizer.json"},
        "variant": row.quant,
        "version": 1,
        "weights": {
            "files": [{"path": row.filename, "sha256": row.sha256, "sizeBytes": row.size_bytes}],
            "format": "gguf"
        }
    });
    serde_json::to_string_pretty(&value)
        .map(|text| format!("{text}\n"))
        .map_err(|_| {
            fail(
                ErrorCode::ModelImageInvalid,
                "catalog manifest could not be encoded",
            )
        })
}

fn replacing_path(dest: &Path) -> PathBuf {
    let name = dest
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model");
    dest.with_file_name(format!("{name}.replacing"))
}

fn promote_dir(staging: &Path, dest: &Path) -> Result<(), InferFailure> {
    if dest.exists() {
        let backup = replacing_path(dest);
        let _ = fs::remove_dir_all(&backup);
        fs::rename(dest, &backup).map_err(io_missing)?;
        if let Err(err) = fs::rename(staging, dest) {
            let _ = fs::rename(&backup, dest);
            return Err(io_missing(err));
        }
        let _ = fs::remove_dir_all(&backup);
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(io_missing)?;
    }
    fs::rename(staging, dest).map_err(io_missing)
}

fn installed_model_dir(home: &Path, kmodel: &Path) -> Result<PathBuf, InferFailure> {
    let models = home.join("models");
    let models = fs::canonicalize(&models).map_err(|_| {
        fail(
            ErrorCode::ModelArtifactMissing,
            "rm refuses a path outside the home model directory",
        )
    })?;
    let file = fs::canonicalize(kmodel).map_err(|_| {
        fail(
            ErrorCode::ModelArtifactMissing,
            "rm refuses a path outside the home model directory",
        )
    })?;
    if !file.starts_with(&models) {
        return Err(fail(
            ErrorCode::ModelArtifactMissing,
            "rm refuses a path outside the home model directory",
        ));
    }
    let Some(dir) = file.parent() else {
        return Err(fail(
            ErrorCode::ModelArtifactMissing,
            "rm refuses a path outside the home model directory",
        ));
    };
    if dir.parent() != Some(models.as_path()) {
        return Err(fail(
            ErrorCode::ModelArtifactMissing,
            "rm refuses a path outside the home model directory",
        ));
    }
    Ok(dir.to_path_buf())
}

fn same_directory(home: &Path, lock_path: &Path) -> Result<(), InferFailure> {
    fs::create_dir_all(home).map_err(io_missing)?;
    let home = fs::canonicalize(home).map_err(io_missing)?;
    let parent = lock_path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(io_missing)?;
    let parent = fs::canonicalize(parent).map_err(io_missing)?;
    if parent != home {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "catalog pull writes the lockfile in the home directory",
        ));
    }
    Ok(())
}

fn io_missing(err: std::io::Error) -> InferFailure {
    fail(
        ErrorCode::ModelArtifactMissing,
        format!("catalog file could not be written: {err}"),
    )
}

#[cfg(test)]
mod tests {
    use super::{host_allowed, FetchPolicy};

    #[test]
    fn huggingface_cdn_host_is_allowlisted_without_a_download() {
        let policy = FetchPolicy::default();
        assert!(host_allowed("huggingface.co", &policy));
        assert!(host_allowed("cdn-lfs.huggingface.co", &policy));
        assert!(host_allowed("cdn.hf.co", &policy));
        assert!(host_allowed("us.aws.cdn.hf.co", &policy));
        assert!(!host_allowed("example.com", &policy));
        assert!(!host_allowed("notcdn.hf.co", &policy));
        assert!(!host_allowed("us.aws.cdn.hf.co.example.com", &policy));
    }

    #[test]
    fn raw_huggingface_tokenizer_is_wrapped_and_a_knolo_list_is_left_alone() {
        let dir = std::env::temp_dir().join(format!("knolo-tok-wrap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let raw_path = dir.join("tokenizer.json");
        std::fs::write(
            &raw_path,
            br#"{"model":{"type":"BPE","vocab":{"a":0}},"padding":null}"#,
        )
        .unwrap();
        super::install_tokenizer(&raw_path).unwrap();
        let wrapped: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&raw_path).unwrap()).unwrap();
        assert_eq!(wrapped["kind"], "huggingface.tokenizers.v1");
        assert_eq!(wrapped["version"], 1);
        assert!(wrapped["tokenizer"]["padding"].is_null());
        assert_eq!(wrapped["tokenizer"]["model"]["type"], "BPE");

        let list_path = dir.join("list.json");
        let list = br#"{"kind":"knolo.llama.tokens.v1","tokens":["a"],"version":1}"#;
        std::fs::write(&list_path, list).unwrap();
        super::install_tokenizer(&list_path).unwrap();
        assert_eq!(std::fs::read(&list_path).unwrap(), list);

        let envelope = br#"{"kind":"huggingface.tokenizers.v1","version":1,"tokenizer":{"model":{"type":"BPE"},"padding":null}}"#;
        let envelope_path = dir.join("envelope.json");
        std::fs::write(&envelope_path, envelope).unwrap();
        super::install_tokenizer(&envelope_path).unwrap();
        assert_eq!(std::fs::read(&envelope_path).unwrap(), envelope);

        std::fs::write(&raw_path, br#"{"kind":"other","version":1}"#).unwrap();
        assert!(super::install_tokenizer(&raw_path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
