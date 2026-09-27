//! One large expert placement the engine did not apply for a cold micro fixture.
//!
//! The report stores the reason and the count. This measurement does
//! not run the feature.
//!
//! The layout is specified in `spec/KIP-INFER-0133-large-expert-placement.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, LargePlacementReportV1, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied large-placement record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LargePlacementObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub requested_experts: u32,
    pub placed_experts: u32,
    pub code: String,
    pub retryable: bool,
    pub spanned: bool,
    pub sharded: bool,
    pub replicated: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the large-placement report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroLargePlacement {
    pub plan: PlacementPlanV1,
    pub observed: LargePlacementObservation,
    pub report: LargePlacementReportV1,
}

/// Record the large-placement measurement.
pub fn measure_large_placement(
    plan: &PlacementPlanV1,
    observed: &LargePlacementObservation,
) -> Result<MicroLargePlacement, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_large_placement(&produced)?;
    Ok(produced)
}

/// Recompute the large-placement report from the stored plan and observation.
pub fn verify_large_placement(measurement: &MicroLargePlacement) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "engine build root does not match",
        &report.engine_build_root,
        &observed.engine_build_root,
    )?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_u32(
        "requested experts do not match",
        report.requested_experts,
        observed.requested_experts,
    )?;
    same_u32(
        "placed experts do not match",
        report.placed_experts,
        observed.placed_experts,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool("spanned does not match", report.spanned, observed.spanned)?;
    same_bool("sharded does not match", report.sharded, observed.sharded)?;
    same_bool(
        "replicated does not match",
        report.replicated,
        observed.replicated,
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
        "large-placement concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "large-placement run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "large-placement warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "large-placement request count does not match",
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
            "large-placement validation did not match",
        ));
    }
    Ok(())
}

/// Write the large-placement report.
pub fn write_large_placement_report(
    base: &Path,
    measurement: &MicroLargePlacement,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_large_placement(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "large-placement",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &LargePlacementObservation,
) -> Result<MicroLargePlacement, InferFailure> {
    accept_micro_plan(plan, "large-placement")?;
    accept_cold_single(
        "large-placement",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = LargePlacementReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        reason: observed.reason.clone(),
        requested_experts: observed.requested_experts,
        placed_experts: observed.placed_experts,
        code: observed.code.clone(),
        retryable: observed.retryable,
        spanned: observed.spanned,
        sharded: observed.sharded,
        replicated: observed.replicated,
        forward_ran: observed.forward_ran,
        receipt_stored: observed.receipt_stored,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        placement_root: plan.root()?,
        validation_result: ("recorded").into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroLargePlacement {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
