//! Model swap time for one cold replacement onto the micro fixture.
//!
//! The measurement does not unload a worker and does not open a model file.
//! `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`, `perplexity_delta`,
//! `measure_placement_memory`, `measure_micro_throughput`,
//! `measure_micro_latency`, `measure_corruption_fuzz`,
//! `measure_cancellation_latency`, `measure_receipt_finalization`, and
//! `measure_receipt_overhead` do not call this path. Verification and load
//! do not call it. The layout is specified in `spec/KIP-INFER-0034-model-swap.md`.

use std::path::Path;

use infer_contracts::{
    fail, model_swap_nanos, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, SwapReportV1,
};

use crate::report_io::{accept_cold_single, accept_micro_plan, write_exclusive};

/// Roots and durations from one completed cold swap onto the micro fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapObservation {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub resident_model_image_root: DigestHex,
    pub resident_artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub unload_nanos: u64,
    pub incoming_verification_nanos: u64,
    pub incoming_load_nanos: u64,
}

/// One micro-fixture observation and the report that records its swap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroSwap {
    pub plan: PlacementPlanV1,
    pub observed: SwapObservation,
    pub report: SwapReportV1,
}

/// Record model swap time. No file is created.
pub fn measure_model_swap(
    plan: &PlacementPlanV1,
    observed: &SwapObservation,
) -> Result<MicroSwap, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_model_swap(&produced)?;
    Ok(produced)
}

/// Recompute the report from the stored plan and observation.
pub fn verify_model_swap(measurement: &MicroSwap) -> Result<(), InferFailure> {
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
        "resident model image root does not match",
        &report.resident_model_image_root,
        &measurement.observed.resident_model_image_root,
    )?;
    same_root(
        "resident artifact root does not match",
        &report.resident_artifact_root,
        &measurement.observed.resident_artifact_root,
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
        "swap concurrency does not match",
        report.concurrency,
        measurement.observed.concurrency,
    )?;
    same_u32(
        "swap run count does not match",
        report.run_count,
        measurement.observed.run_count,
    )?;
    same_text(
        "swap warm state does not match",
        &report.warm_state,
        &measurement.observed.warm_state,
    )?;
    same_u32(
        "swap request count does not match",
        report.request_count,
        measurement.observed.request_count,
    )?;
    same_u64(
        "unload time does not match",
        report.unload_nanos,
        measurement.observed.unload_nanos,
    )?;
    same_u64(
        "incoming verification time does not match",
        report.incoming_verification_nanos,
        measurement.observed.incoming_verification_nanos,
    )?;
    same_u64(
        "incoming load time does not match",
        report.incoming_load_nanos,
        measurement.observed.incoming_load_nanos,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "swap validation did not match",
        ));
    }
    Ok(())
}

/// Write the report. The plan and the resident bytes are not written.
pub fn write_swap_report(
    base: &Path,
    measurement: &MicroSwap,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_model_swap(measurement)?;
    let receipt_bytes = measurement.report.to_bytes()?;
    write_exclusive(base, receipt_path, &receipt_bytes, "swap")
}

fn assemble(plan: &PlacementPlanV1, observed: &SwapObservation) -> Result<MicroSwap, InferFailure> {
    accept_micro_plan(plan, "swap")?;
    accept_cold_single(
        "swap",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.resident_model_image_root == observed.model_image_root {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "a swap replaces a different model image",
        ));
    }
    if observed.resident_artifact_root == observed.artifact_root {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "a swap replaces a different artifact",
        ));
    }
    let swap = model_swap_nanos(
        observed.unload_nanos,
        observed.incoming_verification_nanos,
        observed.incoming_load_nanos,
    )?;
    let report = SwapReportV1 {
        model_image_root: observed.model_image_root.clone(),
        artifact_root: observed.artifact_root.clone(),
        resident_model_image_root: observed.resident_model_image_root.clone(),
        resident_artifact_root: observed.resident_artifact_root.clone(),
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        unload_nanos: observed.unload_nanos,
        incoming_verification_nanos: observed.incoming_verification_nanos,
        incoming_load_nanos: observed.incoming_load_nanos,
        model_swap_nanos: swap,
        validation_result: "recorded".into(),
        extensions: std::collections::BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroSwap {
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
