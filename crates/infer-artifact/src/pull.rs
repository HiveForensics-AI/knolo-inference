//! Local `pull`. The pin names files already on disk. Bytes are hashed
//! before any staged rename. A mismatch leaves the destination untouched.
//! There is no network copy.

use std::fs;
use std::path::{Path, PathBuf};

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure};

use crate::io::write_atomic;
use crate::lockfile::read_lockfile;
use crate::paths::resolve_inside;
use crate::verify::{verify_image, verify_weights};

pub struct PullReport {
    pub image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub model_image_path: String,
}

pub fn pull_alias(
    lock_path: &Path,
    alias: &str,
    weights_dir: Option<&Path>,
) -> Result<PullReport, InferFailure> {
    let lock = read_lockfile(lock_path)?
        .ok_or_else(|| fail(ErrorCode::ModelArtifactMissing, "infer lockfile is missing"))?;
    let pin = lock
        .models
        .get(alias)
        .ok_or_else(|| fail(ErrorCode::ModelArtifactMissing, "alias is not pinned"))?;
    let cwd = std::env::current_dir().map_err(|err| {
        fail(
            ErrorCode::ModelArtifactMissing,
            format!("cannot resolve the working directory: {err}"),
        )
    })?;
    let kmodel = resolve_inside(
        &cwd,
        &pin.model_image_path,
        ErrorCode::ModelImageInvalid,
        ErrorCode::ModelArtifactMissing,
    )?;
    let image_bytes = fs::read(&kmodel).map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            fail(ErrorCode::ModelArtifactMissing, "model image is missing")
        } else {
            fail(
                ErrorCode::ModelImageInvalid,
                format!("model image could not be read: {err}"),
            )
        }
    })?;
    let verification = verify_image(&image_bytes)?;
    if verification.image_root.as_str() != pin.model_image_root
        || verification.artifact_root.as_str() != pin.artifact_root
    {
        return Err(fail(
            ErrorCode::ModelDigestMismatch,
            "pinned roots do not match the model image",
        ));
    }
    let weights = match weights_dir {
        Some(dir) => dir.to_path_buf(),
        None => kmodel
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
    };
    verify_weights(&verification.image, &weights)?;
    let mut staged = Vec::with_capacity(verification.image.files.len());
    for file in &verification.image.files {
        let path = resolve_inside(
            &weights,
            &file.path,
            ErrorCode::ModelImageInvalid,
            ErrorCode::ModelArtifactMissing,
        )?;
        let bytes = fs::read(&path).map_err(|err| {
            fail(
                ErrorCode::ModelArtifactMissing,
                format!("missing file {}: {err}", file.path),
            )
        })?;
        staged.push((path, bytes));
    }
    write_atomic(&kmodel, &image_bytes, ErrorCode::ModelArtifactMissing)?;
    for (path, bytes) in staged {
        write_atomic(&path, &bytes, ErrorCode::ModelArtifactMissing)?;
    }
    Ok(PullReport {
        image_root: verification.image_root,
        artifact_root: verification.artifact_root,
        model_image_path: pin.model_image_path.clone(),
    })
}

pub fn pull_destination(report: &PullReport) -> PathBuf {
    PathBuf::from(&report.model_image_path)
}
