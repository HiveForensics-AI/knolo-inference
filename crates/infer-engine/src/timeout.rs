//! Request-timeout record for one cold micro fixture.
//!
//! The report stores the stage and the wait. This measurement does not arm a
//! timer. The layout is specified in `spec/KIP-INFER-0079-request-timeout.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, TimeoutReportV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32, same_u64,
    write_exclusive,
};

/// Caller-supplied request timeout for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeoutObservation {
    pub engine_build_root: DigestHex,
    pub request_id: String,
    pub timeout_stage: String,
    pub waited_nanos: u64,
    pub code: String,
    pub retryable: bool,
    pub receipt_stored: bool,
    pub listener_up: bool,
    pub worker_lost: bool,
    pub http_status: u32,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the request-timeout report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroTimeout {
    pub plan: PlacementPlanV1,
    pub observed: TimeoutObservation,
    pub report: TimeoutReportV1,
}

/// Record a request timeout. No timer is armed.
pub fn measure_timeout(
    plan: &PlacementPlanV1,
    observed: &TimeoutObservation,
) -> Result<MicroTimeout, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_timeout(&produced)?;
    Ok(produced)
}

/// Recompute the request-timeout report from the stored plan and observation.
pub fn verify_timeout(measurement: &MicroTimeout) -> Result<(), InferFailure> {
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
        "request id does not match",
        &report.request_id,
        &observed.request_id,
    )?;
    same_text(
        "timeout stage does not match",
        &report.timeout_stage,
        &observed.timeout_stage,
    )?;
    same_u64(
        "waited nanos do not match",
        report.waited_nanos,
        observed.waited_nanos,
    )?;
    same_text("timeout code does not match", &report.code, &observed.code)?;
    same_bool(
        "timeout retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "receipt stored does not match",
        report.receipt_stored,
        observed.receipt_stored,
    )?;
    same_bool(
        "listener up does not match",
        report.listener_up,
        observed.listener_up,
    )?;
    same_bool(
        "worker lost does not match",
        report.worker_lost,
        observed.worker_lost,
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
        "timeout concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "timeout run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "timeout warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "timeout request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "timeout validation did not match",
        ));
    }
    Ok(())
}

/// Write the request-timeout report. No timer is armed.
pub fn write_timeout_report(
    base: &Path,
    measurement: &MicroTimeout,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_timeout(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "timeout",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &TimeoutObservation,
) -> Result<MicroTimeout, InferFailure> {
    accept_micro_plan(plan, "timeout")?;
    accept_cold_single(
        "timeout",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = TimeoutReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        request_id: observed.request_id.clone(),
        timeout_stage: observed.timeout_stage.clone(),
        waited_nanos: observed.waited_nanos,
        code: observed.code.clone(),
        retryable: observed.retryable,
        receipt_stored: observed.receipt_stored,
        listener_up: observed.listener_up,
        worker_lost: observed.worker_lost,
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
    Ok(MicroTimeout {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
