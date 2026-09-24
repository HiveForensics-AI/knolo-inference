//! Prefix reuse for one cold run of the micro fixture.
//!
//! The measurement does not allocate a prefix index. Cache policy stays
//! `off`, so every count is zero. `dequant_gguf`, `quant_gemm`,
//! `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`,
//! `measure_micro_throughput`, `measure_micro_latency`,
//! `measure_corruption_fuzz`, `measure_cancellation_latency`,
//! `measure_receipt_finalization`, `measure_receipt_overhead`,
//! `measure_model_swap`, `measure_model_verification`, `measure_model_load`,
//! `measure_peak_memory`, and `measure_kv_utilization` do not call this
//! path. The layout is specified in `spec/KIP-INFER-0039-prefix-reuse.md`.

use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, PrefixReportV1};

use crate::report_io::{accept_cold_single, accept_micro_plan, write_exclusive};

/// Roots and reuse counts from one completed cold run of the micro fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixObservation {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub lookup_count: u32,
    pub hit_count: u32,
    pub miss_count: u32,
    pub reused_tokens: u32,
}

/// One micro-fixture observation and the report that records its reuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroPrefix {
    pub plan: PlacementPlanV1,
    pub observed: PrefixObservation,
    pub report: PrefixReportV1,
}

/// Record prefix reuse. No file is created and no prefix index is allocated.
pub fn measure_prefix_reuse(
    plan: &PlacementPlanV1,
    observed: &PrefixObservation,
) -> Result<MicroPrefix, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_prefix_reuse(&produced)?;
    Ok(produced)
}

/// Recompute the report from the stored plan and observation.
pub fn verify_prefix_reuse(measurement: &MicroPrefix) -> Result<(), InferFailure> {
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
        "prefix concurrency does not match",
        report.concurrency,
        measurement.observed.concurrency,
    )?;
    same_u32(
        "prefix run count does not match",
        report.run_count,
        measurement.observed.run_count,
    )?;
    same_text(
        "prefix warm state does not match",
        &report.warm_state,
        &measurement.observed.warm_state,
    )?;
    same_u32(
        "prefix request count does not match",
        report.request_count,
        measurement.observed.request_count,
    )?;
    same_u32(
        "prefix lookups do not match",
        report.lookup_count,
        measurement.observed.lookup_count,
    )?;
    same_u32(
        "prefix hits do not match",
        report.hit_count,
        measurement.observed.hit_count,
    )?;
    same_u32(
        "prefix misses do not match",
        report.miss_count,
        measurement.observed.miss_count,
    )?;
    same_u32(
        "reused tokens do not match",
        report.reused_tokens,
        measurement.observed.reused_tokens,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "prefix validation did not match",
        ));
    }
    Ok(())
}

/// Write the report. The plan and the token ids are not written.
pub fn write_prefix_report(
    base: &Path,
    measurement: &MicroPrefix,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_prefix_reuse(measurement)?;
    let receipt_bytes = measurement.report.to_bytes()?;
    write_exclusive(base, receipt_path, &receipt_bytes, "prefix")
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &PrefixObservation,
) -> Result<MicroPrefix, InferFailure> {
    accept_micro_plan(plan, "prefix")?;
    accept_cold_single(
        "prefix",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.lookup_count != 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "prefix lookups are zero while the cache is off",
        ));
    }
    if observed.hit_count != 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "prefix hits are zero while the cache is off",
        ));
    }
    if observed.miss_count != 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "prefix misses are zero while the cache is off",
        ));
    }
    if observed.reused_tokens != 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "reused tokens are zero while the cache is off",
        ));
    }
    let report = PrefixReportV1 {
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
        lookup_count: observed.lookup_count,
        hit_count: observed.hit_count,
        miss_count: observed.miss_count,
        reused_tokens: observed.reused_tokens,
        validation_result: "recorded".into(),
        extensions: std::collections::BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroPrefix {
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
