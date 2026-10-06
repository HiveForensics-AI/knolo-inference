//! Exclusive CBOR writes for a caller-supplied report directory.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use infer_contracts::{fail, ErrorCode, InferFailure, PlacementPlanV1};

pub(crate) fn accept_micro_plan(plan: &PlacementPlanV1, noun: &str) -> Result<(), InferFailure> {
    plan.validate()?;
    if plan.graph_capture_mode != "off" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("graph capture is off for the {noun} report"),
        ));
    }
    if plan.rejection_reason.is_some() {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "placement was already rejected",
        ));
    }
    if plan.context_reservation_tokens != crate::MAX_CONTEXT
        || plan.kv_block_size != crate::BLOCK_SIZE
        || plan.kv_precision != "f32"
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("{noun} report is the micro fixture"),
        ));
    }
    Ok(())
}

pub(crate) fn accept_cold_single(
    noun: &str,
    execution_mode: &str,
    cache_policy: &str,
    concurrency: u32,
    run_count: u32,
    warm_state: &str,
    request_count: u32,
) -> Result<(), InferFailure> {
    if execution_mode == "throughput" {
        return Err(fail(
            ErrorCode::BackendNotAllowed,
            "throughput execution mode is not enabled",
        ));
    }
    if execution_mode != "isolated-replay" && execution_mode != "pinned" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "field executionMode has an unsupported value",
        ));
    }
    if cache_policy != "off" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("prefix cache is off for the {noun} report"),
        ));
    }
    if concurrency != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("{noun} concurrency is one"),
        ));
    }
    if run_count != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("{noun} run count is one"),
        ));
    }
    if warm_state != "cold" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("{noun} warm state is cold"),
        ));
    }
    if request_count != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("{noun} request count is one"),
        ));
    }
    Ok(())
}

pub(crate) fn same_root(
    message: &str,
    left: &infer_contracts::DigestHex,
    right: &infer_contracts::DigestHex,
) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

pub(crate) fn same_text(message: &str, left: &str, right: &str) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

pub(crate) fn same_u32(message: &str, left: u32, right: u32) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

pub(crate) fn same_u64(message: &str, left: u64, right: u64) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

pub(crate) fn same_bool(message: &str, left: bool, right: bool) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

pub(crate) fn write_exclusive(
    base: &Path,
    relative: &str,
    bytes: &[u8],
    noun: &str,
) -> Result<(), InferFailure> {
    let path = output_path(base, relative, noun)?;
    write_new(&path, bytes, noun)
}

fn output_path(base: &Path, relative: &str, noun: &str) -> Result<PathBuf, InferFailure> {
    infer_contracts::validate_relative_path(relative)?;
    let base = fs::canonicalize(base).map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            fail(
                ErrorCode::ContractInvalid,
                format!("{noun} directory does not exist"),
            )
        } else {
            fail(
                ErrorCode::ContractInvalid,
                format!("{noun} directory: {err}"),
            )
        }
    })?;
    if !base.is_dir() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("{noun} directory does not exist"),
        ));
    }
    let mut parent = base.clone();
    let mut parts = relative.split('/');
    let file_name = parts.next_back().expect("relative path has a file name");
    for segment in parts {
        parent.push(segment);
        let meta = match fs::symlink_metadata(&parent) {
            Ok(meta) => meta,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    format!("{noun} directory does not exist"),
                ));
            }
            Err(err) => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    format!("{noun} directory: {err}"),
                ));
            }
        };
        if meta.file_type().is_symlink() || !meta.is_dir() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                format!("{noun} path leaves the directory"),
            ));
        }
    }
    let parent = fs::canonicalize(&parent).map_err(|err| {
        fail(
            ErrorCode::ContractInvalid,
            format!("{noun} directory: {err}"),
        )
    })?;
    if parent != base && !parent.starts_with(&base) {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("{noun} path leaves the directory"),
        ));
    }
    let path = parent.join(file_name);
    match fs::symlink_metadata(&path) {
        Ok(_) => Err(fail(
            ErrorCode::ContractInvalid,
            format!("{noun} output already exists"),
        )),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(path),
        Err(err) => Err(fail(
            ErrorCode::ContractInvalid,
            format!("{noun} output: {err}"),
        )),
    }
}

fn write_new(path: &Path, bytes: &[u8], noun: &str) -> Result<(), InferFailure> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| {
            if err.kind() == std::io::ErrorKind::AlreadyExists {
                fail(
                    ErrorCode::ContractInvalid,
                    format!("{noun} output already exists"),
                )
            } else {
                fail(ErrorCode::ContractInvalid, format!("write: {err}"))
            }
        })?;
    file.write_all(bytes)
        .map_err(|err| fail(ErrorCode::ContractInvalid, format!("write: {err}")))?;
    file.sync_all()
        .map_err(|err| fail(ErrorCode::ContractInvalid, format!("write: {err}")))?;
    drop(file);
    if let Some(parent) = path.parent() {
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}
