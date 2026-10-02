//! Receipt overhead for one cold micro-fixture request.
//!
//! The measurement does not run a model, does not write a serve journal, and
//! does not read a GGUF payload. `dequant_gguf`, `quant_gemm`,
//! `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`,
//! `measure_micro_throughput`, `measure_micro_latency`,
//! `measure_corruption_fuzz`, `measure_cancellation_latency`, and
//! `measure_receipt_finalization` do not call this path. The layout is
//! specified in `spec/KIP-INFER-0033-receipt-overhead.md`.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use infer_contracts::{
    fail, output_token_root, prompt_token_root, receipt_overhead_nanos, DigestHex, ErrorCode,
    InferFailure, OverheadReportV1, PlacementPlanV1,
};

/// Counts and receipt write durations from one completed micro-fixture request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverheadObservation {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub receipt_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub finish_reason: String,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub accepted_write_nanos: u64,
    pub receipt_write_nanos: u64,
}

/// One micro-fixture observation and the report that records its receipt writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroOverhead {
    pub plan: PlacementPlanV1,
    pub observed: OverheadObservation,
    pub report: OverheadReportV1,
}

/// Record receipt overhead. No file is created.
pub fn measure_receipt_overhead(
    plan: &PlacementPlanV1,
    observed: &OverheadObservation,
) -> Result<MicroOverhead, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_receipt_overhead(&produced)?;
    Ok(produced)
}

/// Recompute the report from the stored plan and observation.
pub fn verify_receipt_overhead(measurement: &MicroOverhead) -> Result<(), InferFailure> {
    let report = &measurement.report;
    report.validate()?;
    same_root(
        "model image root does not match",
        &report.model_image_root,
        &measurement.observed.model_image_root,
    )?;
    same_root(
        "artifact root does not match",
        &report.artifact_root,
        &measurement.observed.artifact_root,
    )?;
    same_root(
        "engine build root does not match",
        &report.engine_build_root,
        &measurement.observed.engine_build_root,
    )?;
    same_root(
        "placement root does not match",
        &report.placement_root,
        &measurement.plan.root()?,
    )?;
    same_root(
        "prompt token root does not match",
        &report.prompt_token_root,
        &prompt_token_root(&measurement.observed.prompt_tokens)?,
    )?;
    same_root(
        "output token root does not match",
        &report.output_token_root,
        &output_token_root(&measurement.observed.output_tokens)?,
    )?;
    same_root(
        "receipt root does not match",
        &report.receipt_root,
        &measurement.observed.receipt_root,
    )?;
    same_text(
        "execution mode does not match",
        &report.execution_mode,
        &measurement.observed.execution_mode,
    )?;
    same_text(
        "cache policy does not match",
        &report.cache_policy,
        &measurement.observed.cache_policy,
    )?;
    same_u32(
        "overhead concurrency does not match",
        report.concurrency,
        measurement.observed.concurrency,
    )?;
    same_u32(
        "overhead run count does not match",
        report.run_count,
        measurement.observed.run_count,
    )?;
    same_text(
        "overhead warm state does not match",
        &report.warm_state,
        &measurement.observed.warm_state,
    )?;
    same_u32(
        "overhead request count does not match",
        report.request_count,
        measurement.observed.request_count,
    )?;
    same_text(
        "finish reason does not match",
        &report.finish_reason,
        &measurement.observed.finish_reason,
    )?;
    same_u32(
        "prefill token count does not match",
        report.prefill_tokens,
        measurement.observed.prefill_tokens,
    )?;
    same_u32(
        "decode token count does not match",
        report.decode_tokens,
        measurement.observed.decode_tokens,
    )?;
    same_u64(
        "accepted write time does not match",
        report.accepted_write_nanos,
        measurement.observed.accepted_write_nanos,
    )?;
    same_u64(
        "receipt write time does not match",
        report.receipt_write_nanos,
        measurement.observed.receipt_write_nanos,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "overhead validation did not match",
        ));
    }
    Ok(())
}

/// Write the report. The plan, the token ids, and the inference receipt are not written.
pub fn write_overhead_report(
    base: &Path,
    measurement: &MicroOverhead,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_receipt_overhead(measurement)?;
    let receipt_bytes = measurement.report.to_bytes()?;
    let receipt = output_path(base, receipt_path)?;
    write_new(&receipt, &receipt_bytes)
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &OverheadObservation,
) -> Result<MicroOverhead, InferFailure> {
    plan.validate()?;
    if plan.graph_capture_mode != "off" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "graph capture is off for the overhead report",
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
            "overhead report is the micro fixture",
        ));
    }
    if observed.execution_mode == "throughput" {
        return Err(fail(
            ErrorCode::BackendNotAllowed,
            "throughput execution mode is not enabled",
        ));
    }
    if observed.execution_mode != "isolated-replay" && observed.execution_mode != "pinned" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "field executionMode has an unsupported value",
        ));
    }
    if observed.cache_policy != "off" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "prefix cache is off for the overhead report",
        ));
    }
    if observed.concurrency != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "overhead concurrency is one",
        ));
    }
    if observed.run_count != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "overhead run count is one",
        ));
    }
    if observed.warm_state != "cold" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "overhead warm state is cold",
        ));
    }
    if observed.request_count != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "overhead request count is one",
        ));
    }
    if observed.finish_reason == "cancelled" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "a cancelled request stores no receipt",
        ));
    }
    if observed.finish_reason == "error" || observed.finish_reason == "timeout" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "a failed request stores no receipt",
        ));
    }
    if observed.finish_reason != "stop" && observed.finish_reason != "length" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "field finishReason has an unsupported value",
        ));
    }
    if observed.prefill_tokens == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "prefill token count is zero",
        ));
    }
    if observed.finish_reason == "stop" && observed.decode_tokens == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "a stop receipt has no output tokens",
        ));
    }
    if observed.prompt_tokens.len() != observed.prefill_tokens as usize {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "prefill token count does not match",
        ));
    }
    if observed.output_tokens.len() != observed.decode_tokens as usize {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "decode token count does not match",
        ));
    }
    let span = observed
        .prefill_tokens
        .checked_add(observed.decode_tokens)
        .ok_or_else(|| {
            fail(
                ErrorCode::ContextLimitExceeded,
                "micro context length overflows",
            )
        })?;
    if span > crate::MAX_CONTEXT {
        return Err(fail(
            ErrorCode::ContextLimitExceeded,
            "micro prompt does not fit in the reserved context",
        ));
    }
    in_vocab(&observed.prompt_tokens)?;
    in_vocab(&observed.output_tokens)?;
    let overhead =
        receipt_overhead_nanos(observed.accepted_write_nanos, observed.receipt_write_nanos)?;
    let report = OverheadReportV1 {
        model_image_root: observed.model_image_root.clone(),
        artifact_root: observed.artifact_root.clone(),
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        prompt_token_root: prompt_token_root(&observed.prompt_tokens)?,
        output_token_root: output_token_root(&observed.output_tokens)?,
        receipt_root: observed.receipt_root.clone(),
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        finish_reason: observed.finish_reason.clone(),
        prefill_tokens: observed.prefill_tokens,
        decode_tokens: observed.decode_tokens,
        accepted_write_nanos: observed.accepted_write_nanos,
        receipt_write_nanos: observed.receipt_write_nanos,
        receipt_overhead_nanos: overhead,
        validation_result: "recorded".into(),
        extensions: std::collections::BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroOverhead {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}

fn in_vocab(tokens: &[u32]) -> Result<(), InferFailure> {
    if tokens.iter().any(|id| *id as usize >= crate::VOCAB) {
        Err(fail(
            ErrorCode::ContractInvalid,
            "overhead token is outside the micro vocabulary",
        ))
    } else {
        Ok(())
    }
}

fn same_root(message: &str, left: &DigestHex, right: &DigestHex) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn same_text(message: &str, left: &str, right: &str) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn same_u32(message: &str, left: u32, right: u32) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn same_u64(message: &str, left: u64, right: u64) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn output_path(base: &Path, relative: &str) -> Result<PathBuf, InferFailure> {
    infer_contracts::validate_relative_path(relative)?;
    let base = fs::canonicalize(base).map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            fail(
                ErrorCode::ContractInvalid,
                "overhead directory does not exist",
            )
        } else {
            fail(
                ErrorCode::ContractInvalid,
                format!("overhead directory: {err}"),
            )
        }
    })?;
    if !base.is_dir() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "overhead directory does not exist",
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
                    "overhead directory does not exist",
                ));
            }
            Err(err) => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    format!("overhead directory: {err}"),
                ));
            }
        };
        if meta.file_type().is_symlink() || !meta.is_dir() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "overhead path leaves the directory",
            ));
        }
    }
    let parent = fs::canonicalize(&parent).map_err(|err| {
        fail(
            ErrorCode::ContractInvalid,
            format!("overhead directory: {err}"),
        )
    })?;
    if parent != base && !parent.starts_with(&base) {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "overhead path leaves the directory",
        ));
    }
    let path = parent.join(file_name);
    match fs::symlink_metadata(&path) {
        Ok(_) => Err(fail(
            ErrorCode::ContractInvalid,
            "overhead output already exists",
        )),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(path),
        Err(err) => Err(fail(
            ErrorCode::ContractInvalid,
            format!("overhead output: {err}"),
        )),
    }
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), InferFailure> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| {
            if err.kind() == std::io::ErrorKind::AlreadyExists {
                fail(ErrorCode::ContractInvalid, "overhead output already exists")
            } else {
                fail(ErrorCode::ContractInvalid, format!("write: {err}"))
            }
        })?;
    file.write_all(bytes)
        .map_err(|err| fail(ErrorCode::ContractInvalid, format!("write: {err}")))?;
    file.sync_all()
        .map_err(|err| fail(ErrorCode::ContractInvalid, format!("write: {err}")))?;
    drop(file);
    sync_parent(path);
    Ok(())
}

fn sync_parent(path: &Path) {
    let Some(parent) = path.parent() else {
        return;
    };
    if let Ok(dir) = File::open(parent) {
        let _ = dir.sync_all();
    }
}
