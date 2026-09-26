//! One workstation recipe that is not blessed for a cold micro fixture.
//!
//! The report stores the reason. This measurement does not run a benchmark
//! and does not bless a recipe. The layout is specified in
//! `spec/KIP-INFER-0123-workstation-recipe.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, WorkstationReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied workstation record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkstationObservation {
    pub engine_build_root: DigestHex,
    pub profile_root: DigestHex,
    pub benchmark_root: DigestHex,
    pub reason: String,
    pub benchmark_count: u32,
    pub blessed: bool,
    pub benchmark_recorded: bool,
    pub fallback_selected: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the workstation report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroWorkstation {
    pub plan: PlacementPlanV1,
    pub observed: WorkstationObservation,
    pub report: WorkstationReportV1,
}

/// Record the workstation measurement.
pub fn measure_workstation(
    plan: &PlacementPlanV1,
    observed: &WorkstationObservation,
) -> Result<MicroWorkstation, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_workstation(&produced)?;
    Ok(produced)
}

/// Recompute the workstation report from the stored plan and observation.
pub fn verify_workstation(measurement: &MicroWorkstation) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "engine build root does not match",
        &report.engine_build_root,
        &observed.engine_build_root,
    )?;
    same_root(
        "profile root does not match",
        &report.profile_root,
        &observed.profile_root,
    )?;
    same_root(
        "benchmark root does not match",
        &report.benchmark_root,
        &observed.benchmark_root,
    )?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_u32(
        "benchmark count does not match",
        report.benchmark_count,
        observed.benchmark_count,
    )?;
    same_bool("blessed does not match", report.blessed, observed.blessed)?;
    same_bool(
        "benchmark recorded does not match",
        report.benchmark_recorded,
        observed.benchmark_recorded,
    )?;
    same_bool(
        "fallback selected does not match",
        report.fallback_selected,
        observed.fallback_selected,
    )?;
    same_bool(
        "forward ran does not match",
        report.forward_ran,
        observed.forward_ran,
    )?;
    same_bool(
        "receipt stored does not match",
        report.receipt_stored,
        observed.receipt_stored,
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
        "workstation concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "workstation run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "workstation warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "workstation request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    same_root(
        "placement root does not match",
        &report.placement_root,
        &measurement.plan.root()?,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "workstation validation did not match",
        ));
    }
    Ok(())
}

/// Write the workstation report.
pub fn write_workstation_report(
    base: &Path,
    measurement: &MicroWorkstation,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_workstation(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "workstation",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &WorkstationObservation,
) -> Result<MicroWorkstation, InferFailure> {
    accept_micro_plan(plan, "workstation")?;
    accept_cold_single(
        "workstation",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = WorkstationReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        profile_root: observed.profile_root.clone(),
        benchmark_root: observed.benchmark_root.clone(),
        reason: observed.reason.clone(),
        benchmark_count: observed.benchmark_count,
        blessed: observed.blessed,
        benchmark_recorded: observed.benchmark_recorded,
        fallback_selected: observed.fallback_selected,
        forward_ran: observed.forward_ran,
        receipt_stored: observed.receipt_stored,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        validation_result: ("recorded").into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroWorkstation {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
