//! Concurrent load record for one cold micro fixture.
//!
//! The report stores the phase and the worker flags. This measurement does
//! not start a worker. The layout is specified in
//! `spec/KIP-INFER-0074-concurrent-load.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, ConcurrentLoadReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied concurrent load for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcurrentLoadObservation {
    pub engine_build_root: DigestHex,
    pub phase: String,
    pub lifecycle: String,
    pub inflight: u32,
    pub worker_started: bool,
    pub second_worker: bool,
    pub worker_replaced: bool,
    pub restart_count: u32,
    pub listener_up: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the concurrent-load report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroConcurrentLoad {
    pub plan: PlacementPlanV1,
    pub observed: ConcurrentLoadObservation,
    pub report: ConcurrentLoadReportV1,
}

/// Record the concurrent load. A worker is not started.
pub fn measure_concurrent_load(
    plan: &PlacementPlanV1,
    observed: &ConcurrentLoadObservation,
) -> Result<MicroConcurrentLoad, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_concurrent_load(&produced)?;
    Ok(produced)
}

/// Recompute the concurrent-load report from the stored plan and observation.
pub fn verify_concurrent_load(measurement: &MicroConcurrentLoad) -> Result<(), InferFailure> {
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
    same_text("phase does not match", &report.phase, &observed.phase)?;
    same_text(
        "lifecycle does not match",
        &report.lifecycle,
        &observed.lifecycle,
    )?;
    same_u32(
        "inflight does not match",
        report.inflight,
        observed.inflight,
    )?;
    same_bool(
        "worker started does not match",
        report.worker_started,
        observed.worker_started,
    )?;
    same_bool(
        "second worker does not match",
        report.second_worker,
        observed.second_worker,
    )?;
    same_bool(
        "worker replaced does not match",
        report.worker_replaced,
        observed.worker_replaced,
    )?;
    same_u32(
        "restart count does not match",
        report.restart_count,
        observed.restart_count,
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
        "concurrent-load concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "concurrent-load run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "concurrent-load warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "concurrent-load request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "concurrent-load validation did not match",
        ));
    }
    Ok(())
}

/// Write the concurrent-load report. A worker is not started.
pub fn write_concurrent_load_report(
    base: &Path,
    measurement: &MicroConcurrentLoad,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_concurrent_load(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "concurrent-load",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ConcurrentLoadObservation,
) -> Result<MicroConcurrentLoad, InferFailure> {
    accept_micro_plan(plan, "concurrent-load")?;
    accept_cold_single(
        "concurrent-load",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = ConcurrentLoadReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        phase: observed.phase.clone(),
        lifecycle: observed.lifecycle.clone(),
        inflight: observed.inflight,
        worker_started: observed.worker_started,
        second_worker: observed.second_worker,
        worker_replaced: observed.worker_replaced,
        restart_count: observed.restart_count,
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
    Ok(MicroConcurrentLoad {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
