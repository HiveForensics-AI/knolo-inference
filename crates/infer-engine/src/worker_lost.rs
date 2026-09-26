//! Worker-lost record for one cold micro fixture.
//!
//! The report stores that a ready worker exited. This measurement does not
//! kill a process. The layout is specified in `spec/KIP-INFER-0084-worker-lost.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, WorkerLostReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied worker loss for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerLostObservation {
    pub engine_build_root: DigestHex,
    pub request_id: String,
    pub code: String,
    pub retryable: bool,
    pub receipt_stored: bool,
    pub listener_up: bool,
    pub supervisor_exited: bool,
    pub http_status: u32,
    pub journal_sealed: bool,
    pub restart_counted: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the worker-lost report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroWorkerLost {
    pub plan: PlacementPlanV1,
    pub observed: WorkerLostObservation,
    pub report: WorkerLostReportV1,
}

/// Record the lost worker. No process is killed.
pub fn measure_worker_lost(
    plan: &PlacementPlanV1,
    observed: &WorkerLostObservation,
) -> Result<MicroWorkerLost, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_worker_lost(&produced)?;
    Ok(produced)
}

/// Recompute the worker-lost report from the stored plan and observation.
pub fn verify_worker_lost(measurement: &MicroWorkerLost) -> Result<(), InferFailure> {
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
        "listener up does not match",
        report.listener_up,
        observed.listener_up,
    )?;
    same_bool(
        "supervisor exited does not match",
        report.supervisor_exited,
        observed.supervisor_exited,
    )?;
    same_u32(
        "http status does not match",
        report.http_status,
        observed.http_status,
    )?;
    same_bool(
        "journal sealed does not match",
        report.journal_sealed,
        observed.journal_sealed,
    )?;
    same_bool(
        "restart counted does not match",
        report.restart_counted,
        observed.restart_counted,
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
        "worker-lost concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "worker-lost run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "worker-lost warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "worker-lost request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "worker-lost validation did not match",
        ));
    }
    Ok(())
}

/// Write the worker-lost report. No process is killed.
pub fn write_worker_lost_report(
    base: &Path,
    measurement: &MicroWorkerLost,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_worker_lost(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "worker-lost",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &WorkerLostObservation,
) -> Result<MicroWorkerLost, InferFailure> {
    accept_micro_plan(plan, "worker-lost")?;
    accept_cold_single(
        "worker-lost",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = WorkerLostReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        request_id: observed.request_id.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        receipt_stored: observed.receipt_stored,
        listener_up: observed.listener_up,
        supervisor_exited: observed.supervisor_exited,
        http_status: observed.http_status,
        journal_sealed: observed.journal_sealed,
        restart_counted: observed.restart_counted,
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
    Ok(MicroWorkerLost {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
