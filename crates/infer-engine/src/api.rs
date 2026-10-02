//! API boundary for one cold micro fixture.
//!
//! The report stores the bind, the hook, and the limits. This measurement
//! does not bind a socket. The layout is specified in
//! `spec/KIP-INFER-0059-api-boundary.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, ApiReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, MAX_API_BODY,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32, same_u64,
    write_exclusive,
};

/// Caller-supplied API boundary for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiObservation {
    pub engine_build_root: DigestHex,
    pub tenant_root: DigestHex,
    pub bind: String,
    pub remote_explicit: bool,
    pub auth: String,
    pub body_limit_bytes: u64,
    pub rate_per_minute: u32,
    pub concurrency_limit: u32,
    pub prompt_log: String,
    pub metrics_labels: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the API report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroApi {
    pub plan: PlacementPlanV1,
    pub observed: ApiObservation,
    pub report: ApiReportV1,
}

/// Record the API boundary. The listener is not bound.
pub fn measure_api_boundary(
    plan: &PlacementPlanV1,
    observed: &ApiObservation,
) -> Result<MicroApi, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_api_boundary(&produced)?;
    Ok(produced)
}

/// Recompute the API report from the stored plan and observation.
pub fn verify_api_boundary(measurement: &MicroApi) -> Result<(), InferFailure> {
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
    same_text("bind does not match", &report.bind, &observed.bind)?;
    same_bool(
        "remote explicit does not match",
        report.remote_explicit,
        observed.remote_explicit,
    )?;
    same_text("auth does not match", &report.auth, &observed.auth)?;
    same_u64(
        "body limit does not match",
        report.body_limit_bytes,
        observed.body_limit_bytes,
    )?;
    same_u32(
        "api rate does not match",
        report.rate_per_minute,
        observed.rate_per_minute,
    )?;
    same_u32(
        "concurrency limit does not match",
        report.concurrency_limit,
        observed.concurrency_limit,
    )?;
    same_text(
        "prompt log does not match",
        &report.prompt_log,
        &observed.prompt_log,
    )?;
    same_text(
        "metrics labels do not match",
        &report.metrics_labels,
        &observed.metrics_labels,
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
        "api concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "api run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "api warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "api request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "api validation did not match",
        ));
    }
    Ok(())
}

/// Write the API report. The listener is not bound.
pub fn write_api_report(
    base: &Path,
    measurement: &MicroApi,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_api_boundary(measurement)?;
    write_exclusive(base, receipt_path, &measurement.report.to_bytes()?, "api")
}

fn assemble(plan: &PlacementPlanV1, observed: &ApiObservation) -> Result<MicroApi, InferFailure> {
    accept_micro_plan(plan, "api")?;
    accept_cold_single(
        "api",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.body_limit_bytes > MAX_API_BODY {
        return Err(fail(
            ErrorCode::InsufficientMemory,
            "the request body exceeds 1 MiB",
        ));
    }
    let report = ApiReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        tenant_root: observed.tenant_root.clone(),
        bind: observed.bind.clone(),
        remote_explicit: observed.remote_explicit,
        auth: observed.auth.clone(),
        body_limit_bytes: observed.body_limit_bytes,
        rate_per_minute: observed.rate_per_minute,
        concurrency_limit: observed.concurrency_limit,
        prompt_log: observed.prompt_log.clone(),
        metrics_labels: observed.metrics_labels.clone(),
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
    Ok(MicroApi {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
