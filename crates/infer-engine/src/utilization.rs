//! KV utilization for one cold run of the micro fixture.
//!
//! The measurement does not allocate a page pool. The micro fixture occupies
//! one page of the eight-page pool. `dequant_gguf`, `quant_gemm`,
//! `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`,
//! `measure_micro_throughput`, `measure_micro_latency`,
//! `measure_corruption_fuzz`, `measure_cancellation_latency`,
//! `measure_receipt_finalization`, `measure_receipt_overhead`,
//! `measure_model_swap`, `measure_model_verification`, `measure_model_load`,
//! and `measure_peak_memory` do not call this path. The layout is specified
//! in `spec/KIP-INFER-0038-kv-utilization.md`.

use std::path::Path;

use infer_contracts::{
    fail, kv_utilization_millionths, DigestHex, ErrorCode, InferFailure, KvReportV1,
    PlacementPlanV1, MICRO_KV_PAGES, MICRO_KV_PAGE_TOKENS,
};

use crate::report_io::{accept_cold_single, accept_micro_plan, write_exclusive};

/// Roots and occupancy from one completed cold run of the micro fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvObservation {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub peak_pages: u32,
    pub peak_tokens: u32,
}

/// One micro-fixture observation and the report that records its occupancy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroKvUtilization {
    pub plan: PlacementPlanV1,
    pub observed: KvObservation,
    pub report: KvReportV1,
}

/// Record KV utilization. No file is created and no page pool is allocated.
pub fn measure_kv_utilization(
    plan: &PlacementPlanV1,
    observed: &KvObservation,
) -> Result<MicroKvUtilization, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_kv_utilization(&produced)?;
    Ok(produced)
}

/// Recompute the report from the stored plan and observation.
pub fn verify_kv_utilization(measurement: &MicroKvUtilization) -> Result<(), InferFailure> {
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
        "kv concurrency does not match",
        report.concurrency,
        measurement.observed.concurrency,
    )?;
    same_u32(
        "kv run count does not match",
        report.run_count,
        measurement.observed.run_count,
    )?;
    same_text(
        "kv warm state does not match",
        &report.warm_state,
        &measurement.observed.warm_state,
    )?;
    same_u32(
        "kv request count does not match",
        report.request_count,
        measurement.observed.request_count,
    )?;
    same_u32(
        "kv peak pages do not match",
        report.peak_pages,
        measurement.observed.peak_pages,
    )?;
    same_u32(
        "kv peak tokens do not match",
        report.peak_tokens,
        measurement.observed.peak_tokens,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "kv validation did not match",
        ));
    }
    Ok(())
}

/// Write the report. The plan and the page pool are not written.
pub fn write_kv_report(
    base: &Path,
    measurement: &MicroKvUtilization,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_kv_utilization(measurement)?;
    let receipt_bytes = measurement.report.to_bytes()?;
    write_exclusive(base, receipt_path, &receipt_bytes, "kv")
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &KvObservation,
) -> Result<MicroKvUtilization, InferFailure> {
    accept_micro_plan(plan, "kv")?;
    accept_cold_single(
        "kv",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.peak_tokens == 0 {
        return Err(fail(ErrorCode::ContractInvalid, "kv peak tokens are zero"));
    }
    if observed.peak_tokens > MICRO_KV_PAGE_TOKENS {
        return Err(fail(
            ErrorCode::ContextLimitExceeded,
            "kv peak tokens exceed the micro context",
        ));
    }
    if observed.peak_pages != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "the micro fixture occupies one page",
        ));
    }
    let utilization = kv_utilization_millionths(observed.peak_tokens, observed.peak_pages)?;
    let report = KvReportV1 {
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
        page_total: MICRO_KV_PAGES,
        page_size_tokens: MICRO_KV_PAGE_TOKENS,
        peak_pages: observed.peak_pages,
        peak_tokens: observed.peak_tokens,
        utilization_millionths: utilization,
        validation_result: "recorded".into(),
        extensions: std::collections::BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroKvUtilization {
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
