//! Prefix-eviction record for one cold micro fixture under load.
//!
//! The report stores the load count and the zero eviction counts. This
//! measurement does not allocate a prefix index. The layout is specified in
//! `spec/KIP-INFER-0075-prefix-eviction.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, EvictionReportV1, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied prefix eviction for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictionObservation {
    pub engine_build_root: DigestHex,
    pub load_requests: u32,
    pub evicted_pages: u32,
    pub evicted_tokens: u32,
    pub active_evicted: bool,
    pub prefix_allocated: bool,
    pub listener_up: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the prefix-eviction report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroEviction {
    pub plan: PlacementPlanV1,
    pub observed: EvictionObservation,
    pub report: EvictionReportV1,
}

/// Record prefix eviction under load. The index is not allocated.
pub fn measure_prefix_eviction(
    plan: &PlacementPlanV1,
    observed: &EvictionObservation,
) -> Result<MicroEviction, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_prefix_eviction(&produced)?;
    Ok(produced)
}

/// Recompute the prefix-eviction report from the stored plan and observation.
pub fn verify_prefix_eviction(measurement: &MicroEviction) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "engine build root does not match",
        &report.engine_build_root,
        &observed.engine_build_root,
    )?;
    same_root(
        "placement root does not match",
        &report.placement_root,
        &measurement.plan.root()?,
    )?;
    same_u32(
        "load requests do not match",
        report.load_requests,
        observed.load_requests,
    )?;
    same_u32(
        "evicted pages do not match",
        report.evicted_pages,
        observed.evicted_pages,
    )?;
    same_u32(
        "evicted tokens do not match",
        report.evicted_tokens,
        observed.evicted_tokens,
    )?;
    same_bool(
        "active evicted does not match",
        report.active_evicted,
        observed.active_evicted,
    )?;
    same_bool(
        "prefix allocated does not match",
        report.prefix_allocated,
        observed.prefix_allocated,
    )?;
    same_bool(
        "listener up does not match",
        report.listener_up,
        observed.listener_up,
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
        "eviction concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "eviction run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "eviction warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "eviction request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "eviction validation did not match",
        ));
    }
    Ok(())
}

/// Write the prefix-eviction report. The index is not allocated.
pub fn write_eviction_report(
    base: &Path,
    measurement: &MicroEviction,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_prefix_eviction(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "eviction",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &EvictionObservation,
) -> Result<MicroEviction, InferFailure> {
    accept_micro_plan(plan, "eviction")?;
    accept_cold_single(
        "eviction",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = EvictionReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        load_requests: observed.load_requests,
        evicted_pages: observed.evicted_pages,
        evicted_tokens: observed.evicted_tokens,
        active_evicted: observed.active_evicted,
        prefix_allocated: observed.prefix_allocated,
        listener_up: observed.listener_up,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        validation_result: "recorded".into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroEviction {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
