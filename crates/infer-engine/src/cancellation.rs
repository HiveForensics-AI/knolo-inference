//! Cancellation latency for one cold micro-fixture request.
//!
//! The measurement does not run a model, does not cancel a live request, and
//! does not read a GGUF payload. `dequant_gguf`, `quant_gemm`,
//! `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`,
//! `measure_micro_throughput`, `measure_micro_latency`, and
//! `measure_corruption_fuzz` do not call this path. The layout is specified
//! in `spec/KIP-INFER-0031-cancellation-latency.md`.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use infer_contracts::{
    cancellation_latency_nanos, fail, output_token_root, prompt_token_root, CancellationReportV1,
    DigestHex, ErrorCode, InferFailure, PlacementPlanV1,
};

/// Counts and cancel times from one completed micro-fixture request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CancellationObservation {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub cancel_stage: String,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub requested_nanos: u64,
    pub terminal_nanos: u64,
}

/// One micro-fixture observation and the report that records its cancel time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroCancellation {
    pub plan: PlacementPlanV1,
    pub observed: CancellationObservation,
    pub report: CancellationReportV1,
}

/// Record cancellation latency. No file is created.
pub fn measure_cancellation_latency(
    plan: &PlacementPlanV1,
    observed: &CancellationObservation,
) -> Result<MicroCancellation, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_cancellation_latency(&produced)?;
    Ok(produced)
}

/// Recompute the report from the stored plan and observation.
pub fn verify_cancellation_latency(measurement: &MicroCancellation) -> Result<(), InferFailure> {
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
        "cancellation concurrency does not match",
        report.concurrency,
        measurement.observed.concurrency,
    )?;
    same_u32(
        "cancellation run count does not match",
        report.run_count,
        measurement.observed.run_count,
    )?;
    same_text(
        "cancellation warm state does not match",
        &report.warm_state,
        &measurement.observed.warm_state,
    )?;
    same_u32(
        "cancellation request count does not match",
        report.request_count,
        measurement.observed.request_count,
    )?;
    same_text(
        "cancel stage does not match",
        &report.cancel_stage,
        &measurement.observed.cancel_stage,
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
        "cancel request time does not match",
        report.requested_nanos,
        measurement.observed.requested_nanos,
    )?;
    same_u64(
        "cancel terminal time does not match",
        report.terminal_nanos,
        measurement.observed.terminal_nanos,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "cancellation validation did not match",
        ));
    }
    Ok(())
}

/// Write the report. The plan and the token ids are not written.
pub fn write_cancellation_report(
    base: &Path,
    measurement: &MicroCancellation,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_cancellation_latency(measurement)?;
    let receipt_bytes = measurement.report.to_bytes()?;
    let receipt = output_path(base, receipt_path)?;
    write_new(&receipt, &receipt_bytes)
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &CancellationObservation,
) -> Result<MicroCancellation, InferFailure> {
    plan.validate()?;
    if plan.graph_capture_mode != "off" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "graph capture is off for the cancellation report",
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
            "cancellation report is the micro fixture",
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
            "prefix cache is off for the cancellation report",
        ));
    }
    if observed.concurrency != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "cancellation concurrency is one",
        ));
    }
    if observed.run_count != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "cancellation run count is one",
        ));
    }
    if observed.warm_state != "cold" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "cancellation warm state is cold",
        ));
    }
    if observed.request_count != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "cancellation request count is one",
        ));
    }
    if observed.cancel_stage != "prefill" && observed.cancel_stage != "decode" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "field cancelStage has an unsupported value",
        ));
    }
    if observed.prefill_tokens == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "prefill token count is zero",
        ));
    }
    if observed.cancel_stage == "prefill" && observed.decode_tokens != 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "prefill cancel has output tokens",
        ));
    }
    if observed.cancel_stage == "decode" && observed.decode_tokens == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "decode cancel has no output tokens",
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
    let latency = cancellation_latency_nanos(observed.requested_nanos, observed.terminal_nanos)?;
    let report = CancellationReportV1 {
        model_image_root: observed.model_image_root.clone(),
        artifact_root: observed.artifact_root.clone(),
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        prompt_token_root: prompt_token_root(&observed.prompt_tokens)?,
        output_token_root: output_token_root(&observed.output_tokens)?,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        cancel_stage: observed.cancel_stage.clone(),
        prefill_tokens: observed.prefill_tokens,
        decode_tokens: observed.decode_tokens,
        requested_nanos: observed.requested_nanos,
        terminal_nanos: observed.terminal_nanos,
        cancellation_latency_nanos: latency,
        validation_result: "recorded".into(),
        extensions: std::collections::BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroCancellation {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}

fn in_vocab(tokens: &[u32]) -> Result<(), InferFailure> {
    if tokens.iter().any(|id| *id as usize >= crate::VOCAB) {
        Err(fail(
            ErrorCode::ContractInvalid,
            "cancellation token is outside the micro vocabulary",
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
                "cancellation directory does not exist",
            )
        } else {
            fail(
                ErrorCode::ContractInvalid,
                format!("cancellation directory: {err}"),
            )
        }
    })?;
    if !base.is_dir() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "cancellation directory does not exist",
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
                    "cancellation directory does not exist",
                ));
            }
            Err(err) => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    format!("cancellation directory: {err}"),
                ));
            }
        };
        if meta.file_type().is_symlink() || !meta.is_dir() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "cancellation path leaves the directory",
            ));
        }
    }
    let parent = fs::canonicalize(&parent).map_err(|err| {
        fail(
            ErrorCode::ContractInvalid,
            format!("cancellation directory: {err}"),
        )
    })?;
    if parent != base && !parent.starts_with(&base) {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "cancellation path leaves the directory",
        ));
    }
    let path = parent.join(file_name);
    match fs::symlink_metadata(&path) {
        Ok(_) => Err(fail(
            ErrorCode::ContractInvalid,
            "cancellation output already exists",
        )),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(path),
        Err(err) => Err(fail(
            ErrorCode::ContractInvalid,
            format!("cancellation output: {err}"),
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
                fail(
                    ErrorCode::ContractInvalid,
                    "cancellation output already exists",
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
