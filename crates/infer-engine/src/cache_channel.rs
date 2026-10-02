//! Cache side-channel policy for one cold micro fixture.
//!
//! The report stores the tenant scope and the disclosure rule. This
//! measurement does not allocate a prefix index. The layout is specified in
//! `spec/KIP-INFER-0060-cache-channel.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, CacheChannelReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied cache policy for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheChannelObservation {
    pub engine_build_root: DigestHex,
    pub tenant_root: DigestHex,
    pub project_root: DigestHex,
    pub sharing_policy: String,
    pub cross_tenant: bool,
    pub existence_disclosure: String,
    pub metrics_scope: String,
    pub prefix_allocated: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the cache side-channel report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroCacheChannel {
    pub plan: PlacementPlanV1,
    pub observed: CacheChannelObservation,
    pub report: CacheChannelReportV1,
}

/// Record the cache side-channel policy. The prefix index is not allocated.
pub fn measure_cache_channel(
    plan: &PlacementPlanV1,
    observed: &CacheChannelObservation,
) -> Result<MicroCacheChannel, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_cache_channel(&produced)?;
    Ok(produced)
}

/// Recompute the cache side-channel report from the stored plan and observation.
pub fn verify_cache_channel(measurement: &MicroCacheChannel) -> Result<(), InferFailure> {
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
    same_root(
        "tenant root does not match",
        &report.tenant_root,
        &observed.tenant_root,
    )?;
    same_root(
        "project root does not match",
        &report.project_root,
        &observed.project_root,
    )?;
    same_text(
        "sharing policy does not match",
        &report.sharing_policy,
        &observed.sharing_policy,
    )?;
    same_bool(
        "cross tenant does not match",
        report.cross_tenant,
        observed.cross_tenant,
    )?;
    same_text(
        "existence disclosure does not match",
        &report.existence_disclosure,
        &observed.existence_disclosure,
    )?;
    same_text(
        "metrics scope does not match",
        &report.metrics_scope,
        &observed.metrics_scope,
    )?;
    same_bool(
        "prefix allocated does not match",
        report.prefix_allocated,
        observed.prefix_allocated,
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
        "cache channel concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "cache channel run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "cache channel warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "cache channel request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "cache channel validation did not match",
        ));
    }
    Ok(())
}

/// Write the cache side-channel report. The prefix index is not allocated.
pub fn write_cache_channel_report(
    base: &Path,
    measurement: &MicroCacheChannel,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_cache_channel(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "cache channel",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &CacheChannelObservation,
) -> Result<MicroCacheChannel, InferFailure> {
    accept_micro_plan(plan, "cache channel")?;
    accept_cold_single(
        "cache channel",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = CacheChannelReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        tenant_root: observed.tenant_root.clone(),
        project_root: observed.project_root.clone(),
        sharing_policy: observed.sharing_policy.clone(),
        cross_tenant: observed.cross_tenant,
        existence_disclosure: observed.existence_disclosure.clone(),
        metrics_scope: observed.metrics_scope.clone(),
        prefix_allocated: observed.prefix_allocated,
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
    Ok(MicroCacheChannel {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
