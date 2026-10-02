//! One extreme-context cache the engine did not apply for a cold micro fixture.
//!
//! The report stores the reason and the count. This measurement does
//! not run the feature.
//!
//! The layout is specified in `spec/KIP-INFER-0134-extreme-context-cache.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, ExtremeCacheReportV1, InferFailure, PlacementPlanV1,
    MAX_EXTREME_TOKENS,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied extreme-cache record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtremeCacheObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub requested_tokens: u32,
    pub cached_tokens: u32,
    pub code: String,
    pub retryable: bool,
    pub windowed: bool,
    pub yarn_applied: bool,
    pub paged: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the extreme-cache report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroExtremeCache {
    pub plan: PlacementPlanV1,
    pub observed: ExtremeCacheObservation,
    pub report: ExtremeCacheReportV1,
}

/// Record the extreme-cache measurement.
pub fn measure_extreme_cache(
    plan: &PlacementPlanV1,
    observed: &ExtremeCacheObservation,
) -> Result<MicroExtremeCache, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_extreme_cache(&produced)?;
    Ok(produced)
}

/// Recompute the extreme-cache report from the stored plan and observation.
pub fn verify_extreme_cache(measurement: &MicroExtremeCache) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "engine build root does not match",
        &report.engine_build_root,
        &observed.engine_build_root,
    )?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_u32(
        "requested tokens do not match",
        report.requested_tokens,
        observed.requested_tokens,
    )?;
    same_u32(
        "cached tokens do not match",
        report.cached_tokens,
        observed.cached_tokens,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "windowed does not match",
        report.windowed,
        observed.windowed,
    )?;
    same_bool(
        "yarn applied does not match",
        report.yarn_applied,
        observed.yarn_applied,
    )?;
    same_bool("paged does not match", report.paged, observed.paged)?;
    same_bool(
        "forward ran does not match",
        report.forward_ran,
        observed.forward_ran,
    )?;
    same_bool(
        "receipt stored does not match",
        report.receipt_stored,
        observed.receipt_stored,
    )?;
    same_text(
        "execution mode does not match",
        &report.execution_mode,
        &observed.execution_mode,
    )?;
    same_text(
        "cache policy does not match",
        &report.cache_policy,
        &observed.cache_policy,
    )?;
    same_u32(
        "extreme-cache concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "extreme-cache run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "extreme-cache warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "extreme-cache request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    same_root(
        "placement root does not match",
        &report.placement_root,
        &measurement.plan.root()?,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "extreme-cache validation did not match",
        ));
    }
    Ok(())
}

/// Write the extreme-cache report.
pub fn write_extreme_cache_report(
    base: &Path,
    measurement: &MicroExtremeCache,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_extreme_cache(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "extreme-cache",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ExtremeCacheObservation,
) -> Result<MicroExtremeCache, InferFailure> {
    accept_micro_plan(plan, "extreme-cache")?;
    accept_cold_single(
        "extreme-cache",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.requested_tokens > MAX_EXTREME_TOKENS {
        return Err(fail(
            ErrorCode::ContextLimitExceeded,
            "an extreme-context request exceeds the context",
        ));
    }
    let report = ExtremeCacheReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        reason: observed.reason.clone(),
        requested_tokens: observed.requested_tokens,
        cached_tokens: observed.cached_tokens,
        code: observed.code.clone(),
        retryable: observed.retryable,
        windowed: observed.windowed,
        yarn_applied: observed.yarn_applied,
        paged: observed.paged,
        forward_ran: observed.forward_ran,
        receipt_stored: observed.receipt_stored,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        placement_root: plan.root()?,
        validation_result: ("recorded").into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroExtremeCache {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
