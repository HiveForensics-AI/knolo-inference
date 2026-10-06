//! Daemon restart record for one cold micro fixture.
//!
//! The report stores the owner roots and the reason. This measurement does
//! not spawn a process. The layout is specified in
//! `spec/KIP-INFER-0071-daemon-restart.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, RestartReportV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied daemon restart for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartObservation {
    pub engine_build_root: DigestHex,
    pub previous_owner_root: DigestHex,
    pub incoming_owner_root: DigestHex,
    pub reason: String,
    pub lock_replaced: bool,
    pub journals_sealed: bool,
    pub listener_up: bool,
    pub worker_restart_count: u32,
    pub process_spawned: bool,
    pub second_worker: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the daemon-restart report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroRestart {
    pub plan: PlacementPlanV1,
    pub observed: RestartObservation,
    pub report: RestartReportV1,
}

/// Record the daemon restart. A process is not spawned.
pub fn measure_daemon_restart(
    plan: &PlacementPlanV1,
    observed: &RestartObservation,
) -> Result<MicroRestart, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_daemon_restart(&produced)?;
    Ok(produced)
}

/// Recompute the daemon-restart report from the stored plan and observation.
pub fn verify_daemon_restart(measurement: &MicroRestart) -> Result<(), InferFailure> {
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
        "previous owner root does not match",
        &report.previous_owner_root,
        &observed.previous_owner_root,
    )?;
    same_root(
        "incoming owner root does not match",
        &report.incoming_owner_root,
        &observed.incoming_owner_root,
    )?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_bool(
        "lock replaced does not match",
        report.lock_replaced,
        observed.lock_replaced,
    )?;
    same_bool(
        "journals sealed does not match",
        report.journals_sealed,
        observed.journals_sealed,
    )?;
    same_bool(
        "listener up does not match",
        report.listener_up,
        observed.listener_up,
    )?;
    same_u32(
        "worker restart count does not match",
        report.worker_restart_count,
        observed.worker_restart_count,
    )?;
    same_bool(
        "process spawned does not match",
        report.process_spawned,
        observed.process_spawned,
    )?;
    same_bool(
        "second worker does not match",
        report.second_worker,
        observed.second_worker,
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
        "restart concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "restart run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "restart warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "restart request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "restart validation did not match",
        ));
    }
    Ok(())
}

/// Write the daemon-restart report. A process is not spawned.
pub fn write_restart_report(
    base: &Path,
    measurement: &MicroRestart,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_daemon_restart(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "restart",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &RestartObservation,
) -> Result<MicroRestart, InferFailure> {
    accept_micro_plan(plan, "restart")?;
    accept_cold_single(
        "restart",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = RestartReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        previous_owner_root: observed.previous_owner_root.clone(),
        incoming_owner_root: observed.incoming_owner_root.clone(),
        reason: observed.reason.clone(),
        lock_replaced: observed.lock_replaced,
        journals_sealed: observed.journals_sealed,
        listener_up: observed.listener_up,
        worker_restart_count: observed.worker_restart_count,
        process_spawned: observed.process_spawned,
        second_worker: observed.second_worker,
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
    Ok(MicroRestart {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
