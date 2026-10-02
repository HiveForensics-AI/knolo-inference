//! Unload under a queued request for one cold micro fixture.
//!
//! The report stores the queue and the lifecycle. This measurement does not
//! unload a worker. The layout is specified in
//! `spec/KIP-INFER-0070-queued-unload.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, UnloadReportV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied queued unload for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnloadObservation {
    pub engine_build_root: DigestHex,
    pub request_id: String,
    pub queued_requests: u32,
    pub admitted_requests: u32,
    pub queued_started: bool,
    pub admitted_finished: bool,
    pub lifecycle: String,
    pub worker_exited: bool,
    pub code: String,
    pub retryable: bool,
    pub listener_up: bool,
    pub model_reloaded: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the queued-unload report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroUnload {
    pub plan: PlacementPlanV1,
    pub observed: UnloadObservation,
    pub report: UnloadReportV1,
}

/// Record the queued unload. The worker is not unloaded.
pub fn measure_queued_unload(
    plan: &PlacementPlanV1,
    observed: &UnloadObservation,
) -> Result<MicroUnload, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_queued_unload(&produced)?;
    Ok(produced)
}

/// Recompute the queued-unload report from the stored plan and observation.
pub fn verify_queued_unload(measurement: &MicroUnload) -> Result<(), InferFailure> {
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
    same_u32(
        "queued requests do not match",
        report.queued_requests,
        observed.queued_requests,
    )?;
    same_u32(
        "admitted requests do not match",
        report.admitted_requests,
        observed.admitted_requests,
    )?;
    same_bool(
        "queued started does not match",
        report.queued_started,
        observed.queued_started,
    )?;
    same_bool(
        "admitted finished does not match",
        report.admitted_finished,
        observed.admitted_finished,
    )?;
    same_text(
        "lifecycle does not match",
        &report.lifecycle,
        &observed.lifecycle,
    )?;
    same_bool(
        "worker exited does not match",
        report.worker_exited,
        observed.worker_exited,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "listener up does not match",
        report.listener_up,
        observed.listener_up,
    )?;
    same_bool(
        "model reloaded does not match",
        report.model_reloaded,
        observed.model_reloaded,
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
        "unload concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "unload run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "unload warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "unload request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "unload validation did not match",
        ));
    }
    Ok(())
}

/// Write the queued-unload report. The worker is not unloaded.
pub fn write_unload_report(
    base: &Path,
    measurement: &MicroUnload,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_queued_unload(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "unload",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &UnloadObservation,
) -> Result<MicroUnload, InferFailure> {
    accept_micro_plan(plan, "unload")?;
    accept_cold_single(
        "unload",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = UnloadReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        request_id: observed.request_id.clone(),
        queued_requests: observed.queued_requests,
        admitted_requests: observed.admitted_requests,
        queued_started: observed.queued_started,
        admitted_finished: observed.admitted_finished,
        lifecycle: observed.lifecycle.clone(),
        worker_exited: observed.worker_exited,
        code: observed.code.clone(),
        retryable: observed.retryable,
        listener_up: observed.listener_up,
        model_reloaded: observed.model_reloaded,
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
    Ok(MicroUnload {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
