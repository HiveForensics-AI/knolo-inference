//! Model load time for one cold load of the micro fixture.
//!
//! The measurement does not open a model file. Verification time stays on
//! the verification report. `dequant_gguf`, `quant_gemm`,
//! `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`,
//! `measure_micro_throughput`, `measure_micro_latency`,
//! `measure_corruption_fuzz`, `measure_cancellation_latency`,
//! `measure_receipt_finalization`, `measure_receipt_overhead`,
//! `measure_model_swap`, and `measure_model_verification` do not call this
//! path. The layout is specified in `spec/KIP-INFER-0036-model-load.md`.

use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, LoadReportV1, PlacementPlanV1};

use crate::report_io::{accept_cold_single, accept_micro_plan, write_exclusive};

/// Roots and duration from one completed cold load of the micro fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadObservation {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub model_load_nanos: u64,
}

/// One micro-fixture observation and the report that records its load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroLoad {
    pub plan: PlacementPlanV1,
    pub observed: LoadObservation,
    pub report: LoadReportV1,
}

/// Record model load time. No file is created.
pub fn measure_model_load(
    plan: &PlacementPlanV1,
    observed: &LoadObservation,
) -> Result<MicroLoad, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_model_load(&produced)?;
    Ok(produced)
}

/// Recompute the report from the stored plan and observation.
pub fn verify_model_load(measurement: &MicroLoad) -> Result<(), InferFailure> {
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
        "load concurrency does not match",
        report.concurrency,
        measurement.observed.concurrency,
    )?;
    same_u32(
        "load run count does not match",
        report.run_count,
        measurement.observed.run_count,
    )?;
    same_text(
        "load warm state does not match",
        &report.warm_state,
        &measurement.observed.warm_state,
    )?;
    same_u32(
        "load request count does not match",
        report.request_count,
        measurement.observed.request_count,
    )?;
    same_u64(
        "model load time does not match",
        report.model_load_nanos,
        measurement.observed.model_load_nanos,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "load validation did not match",
        ));
    }
    Ok(())
}

/// Write the report. The plan and the model bytes are not written.
pub fn write_load_report(
    base: &Path,
    measurement: &MicroLoad,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_model_load(measurement)?;
    let receipt_bytes = measurement.report.to_bytes()?;
    write_exclusive(base, receipt_path, &receipt_bytes, "load")
}

fn assemble(plan: &PlacementPlanV1, observed: &LoadObservation) -> Result<MicroLoad, InferFailure> {
    accept_micro_plan(plan, "load")?;
    accept_cold_single(
        "load",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.model_load_nanos == 0 {
        return Err(fail(ErrorCode::ContractInvalid, "model load time is zero"));
    }
    let report = LoadReportV1 {
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
        model_load_nanos: observed.model_load_nanos,
        validation_result: "recorded".into(),
        extensions: std::collections::BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroLoad {
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
