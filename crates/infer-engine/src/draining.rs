//! Draining refusal for one cold micro fixture.
//!
//! The report stores that a new completion was refused. This measurement does
//! not drain the listener and does not parse a body. The layout is specified
//! in `spec/KIP-INFER-0085-service-draining.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, DrainingReportV1, ErrorCode, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied draining refusal for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrainingObservation {
    pub engine_build_root: DigestHex,
    pub lifecycle: String,
    pub code: String,
    pub retryable: bool,
    pub receipt_stored: bool,
    pub body_parsed: bool,
    pub listener_up: bool,
    pub worker_loaded: bool,
    pub restart_counted: bool,
    pub http_status: u32,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the draining report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroDraining {
    pub plan: PlacementPlanV1,
    pub observed: DrainingObservation,
    pub report: DrainingReportV1,
}

/// Record the draining refusal. The body is not parsed.
pub fn measure_draining(
    plan: &PlacementPlanV1,
    observed: &DrainingObservation,
) -> Result<MicroDraining, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_draining(&produced)?;
    Ok(produced)
}

/// Recompute the draining report from the stored plan and observation.
pub fn verify_draining(measurement: &MicroDraining) -> Result<(), InferFailure> {
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
    same_text(
        "lifecycle does not match",
        &report.lifecycle,
        &observed.lifecycle,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "receipt stored does not match",
        report.receipt_stored,
        observed.receipt_stored,
    )?;
    same_bool(
        "body parsed does not match",
        report.body_parsed,
        observed.body_parsed,
    )?;
    same_bool(
        "listener up does not match",
        report.listener_up,
        observed.listener_up,
    )?;
    same_bool(
        "worker loaded does not match",
        report.worker_loaded,
        observed.worker_loaded,
    )?;
    same_bool(
        "restart counted does not match",
        report.restart_counted,
        observed.restart_counted,
    )?;
    same_u32(
        "http status does not match",
        report.http_status,
        observed.http_status,
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
        "draining concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "draining run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "draining warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "draining request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "draining validation did not match",
        ));
    }
    Ok(())
}

/// Write the draining report. The body is not parsed.
pub fn write_draining_report(
    base: &Path,
    measurement: &MicroDraining,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_draining(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "draining",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &DrainingObservation,
) -> Result<MicroDraining, InferFailure> {
    accept_micro_plan(plan, "draining")?;
    accept_cold_single(
        "draining",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = DrainingReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        lifecycle: observed.lifecycle.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        receipt_stored: observed.receipt_stored,
        body_parsed: observed.body_parsed,
        listener_up: observed.listener_up,
        worker_loaded: observed.worker_loaded,
        restart_counted: observed.restart_counted,
        http_status: observed.http_status,
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
    Ok(MicroDraining {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
