//! One router comparison the engine did not run for a cold micro fixture.
//!
//! The report stores the reason and the sample count. This measurement does
//! not compare router outputs. The layout is specified in
//! `spec/KIP-INFER-0121-router-conformance.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, RouterReportV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied router record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterObservation {
    pub engine_build_root: DigestHex,
    pub reference_root: DigestHex,
    pub candidate_root: DigestHex,
    pub reason: String,
    pub sample_count: u32,
    pub compared: bool,
    pub parity: bool,
    pub load_recorded: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the router report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroRouter {
    pub plan: PlacementPlanV1,
    pub observed: RouterObservation,
    pub report: RouterReportV1,
}

/// Record the router measurement.
pub fn measure_router(
    plan: &PlacementPlanV1,
    observed: &RouterObservation,
) -> Result<MicroRouter, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_router(&produced)?;
    Ok(produced)
}

/// Recompute the router report from the stored plan and observation.
pub fn verify_router(measurement: &MicroRouter) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "engine build root does not match",
        &report.engine_build_root,
        &observed.engine_build_root,
    )?;
    same_root(
        "reference root does not match",
        &report.reference_root,
        &observed.reference_root,
    )?;
    same_root(
        "candidate root does not match",
        &report.candidate_root,
        &observed.candidate_root,
    )?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_u32(
        "sample count does not match",
        report.sample_count,
        observed.sample_count,
    )?;
    same_bool(
        "compared does not match",
        report.compared,
        observed.compared,
    )?;
    same_bool("parity does not match", report.parity, observed.parity)?;
    same_bool(
        "load recorded does not match",
        report.load_recorded,
        observed.load_recorded,
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
        "router concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "router run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "router warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "router request count does not match",
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
            "router validation did not match",
        ));
    }
    Ok(())
}

/// Write the router report.
pub fn write_router_report(
    base: &Path,
    measurement: &MicroRouter,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_router(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "router",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &RouterObservation,
) -> Result<MicroRouter, InferFailure> {
    accept_micro_plan(plan, "router")?;
    accept_cold_single(
        "router",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = RouterReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reference_root: observed.reference_root.clone(),
        candidate_root: observed.candidate_root.clone(),
        reason: observed.reason.clone(),
        sample_count: observed.sample_count,
        compared: observed.compared,
        parity: observed.parity,
        load_recorded: observed.load_recorded,
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
    Ok(MicroRouter {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
