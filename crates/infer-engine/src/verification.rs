//! Model verification time for one cold check of the micro fixture.
//!
//! The measurement does not open a model file. `dequant_gguf`, `quant_gemm`,
//! `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`,
//! `measure_micro_throughput`, `measure_micro_latency`,
//! `measure_corruption_fuzz`, `measure_cancellation_latency`,
//! `measure_receipt_finalization`, `measure_receipt_overhead`, and
//! `measure_model_swap` do not call this path. Load does not call it. The
//! layout is specified in `spec/KIP-INFER-0035-model-verification.md`.

use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, VerificationReportV1,
    MAX_VERIFIED_BYTES,
};

use crate::report_io::{accept_cold_single, accept_micro_plan, write_exclusive};

/// Roots, byte count, and duration from one completed cold verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationObservation {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub verified_bytes: u64,
    pub model_verification_nanos: u64,
}

/// One micro-fixture observation and the report that records its verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroVerification {
    pub plan: PlacementPlanV1,
    pub observed: VerificationObservation,
    pub report: VerificationReportV1,
}

/// Record model verification time. No file is created.
pub fn measure_model_verification(
    plan: &PlacementPlanV1,
    observed: &VerificationObservation,
) -> Result<MicroVerification, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_model_verification(&produced)?;
    Ok(produced)
}

/// Recompute the report from the stored plan and observation.
pub fn verify_model_verification(measurement: &MicroVerification) -> Result<(), InferFailure> {
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
        "verification concurrency does not match",
        report.concurrency,
        measurement.observed.concurrency,
    )?;
    same_u32(
        "verification run count does not match",
        report.run_count,
        measurement.observed.run_count,
    )?;
    same_text(
        "verification warm state does not match",
        &report.warm_state,
        &measurement.observed.warm_state,
    )?;
    same_u32(
        "verification request count does not match",
        report.request_count,
        measurement.observed.request_count,
    )?;
    same_u64(
        "verified byte count does not match",
        report.verified_bytes,
        measurement.observed.verified_bytes,
    )?;
    same_u64(
        "model verification time does not match",
        report.model_verification_nanos,
        measurement.observed.model_verification_nanos,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "verification validation did not match",
        ));
    }
    Ok(())
}

/// Write the report. The plan and the model bytes are not written.
pub fn write_verification_report(
    base: &Path,
    measurement: &MicroVerification,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_model_verification(measurement)?;
    let receipt_bytes = measurement.report.to_bytes()?;
    write_exclusive(base, receipt_path, &receipt_bytes, "verification")
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &VerificationObservation,
) -> Result<MicroVerification, InferFailure> {
    accept_micro_plan(plan, "verification")?;
    accept_cold_single(
        "verification",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.verified_bytes == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "verified byte count is zero",
        ));
    }
    if observed.verified_bytes > MAX_VERIFIED_BYTES {
        return Err(fail(
            ErrorCode::InsufficientMemory,
            "verified bytes exceed the parser cap",
        ));
    }
    if observed.model_verification_nanos == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "model verification time is zero",
        ));
    }
    let report = VerificationReportV1 {
        model_image_root: observed.model_image_root.clone(),
        artifact_root: observed.artifact_root.clone(),
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        verified_bytes: observed.verified_bytes,
        model_verification_nanos: observed.model_verification_nanos,
        validation_result: "recorded".into(),
        extensions: std::collections::BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroVerification {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
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
