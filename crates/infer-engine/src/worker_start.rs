//! Worker-start failure for one cold micro fixture.
//!
//! The report stores why the worker did not become ready. This measurement
//! does not spawn a process. The layout is specified in
//! `spec/KIP-INFER-0080-worker-start.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, WorkerStartReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied worker-start failure for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerStartObservation {
    pub engine_build_root: DigestHex,
    pub start_failure: String,
    pub code: String,
    pub retryable: bool,
    pub receipt_stored: bool,
    pub listener_up: bool,
    pub process_spawned: bool,
    pub worker_ready: bool,
    pub restart_count: u32,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the worker-start report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroWorkerStart {
    pub plan: PlacementPlanV1,
    pub observed: WorkerStartObservation,
    pub report: WorkerStartReportV1,
}

/// Record a worker-start failure. No process is spawned.
pub fn measure_worker_start(
    plan: &PlacementPlanV1,
    observed: &WorkerStartObservation,
) -> Result<MicroWorkerStart, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_worker_start(&produced)?;
    Ok(produced)
}

/// Recompute the worker-start report from the stored plan and observation.
pub fn verify_worker_start(measurement: &MicroWorkerStart) -> Result<(), InferFailure> {
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
        "start failure does not match",
        &report.start_failure,
        &observed.start_failure,
    )?;
    same_text(
        "worker-start code does not match",
        &report.code,
        &observed.code,
    )?;
    same_bool(
        "worker-start retryable does not match",
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
        "process spawned does not match",
        report.process_spawned,
        observed.process_spawned,
    )?;
    same_bool(
        "worker ready does not match",
        report.worker_ready,
        observed.worker_ready,
    )?;
    same_u32(
        "restart count does not match",
        report.restart_count,
        observed.restart_count,
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
        "worker-start concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "worker-start run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "worker-start warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "worker-start request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "worker-start validation did not match",
        ));
    }
    Ok(())
}

/// Write the worker-start report. No process is spawned.
pub fn write_worker_start_report(
    base: &Path,
    measurement: &MicroWorkerStart,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_worker_start(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "worker-start",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &WorkerStartObservation,
) -> Result<MicroWorkerStart, InferFailure> {
    accept_micro_plan(plan, "worker-start")?;
    accept_cold_single(
        "worker-start",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = WorkerStartReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        start_failure: observed.start_failure.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        receipt_stored: observed.receipt_stored,
        listener_up: observed.listener_up,
        process_spawned: observed.process_spawned,
        worker_ready: observed.worker_ready,
        restart_count: observed.restart_count,
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
    Ok(MicroWorkerStart {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
