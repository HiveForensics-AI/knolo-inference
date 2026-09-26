//! One mixed placement the planner did not select for a cold micro fixture.
//!
//! The report stores the reason and the device count. This measurement does
//! not place a second device and does not fall back. The layout is specified
//! in `spec/KIP-INFER-0124-mixed-placement.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, MixedPlacementReportV1, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied mixed-placement record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedPlacementObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub device_count: u32,
    pub code: String,
    pub retryable: bool,
    pub single_recorded: bool,
    pub mixed_selected: bool,
    pub automatic_fallback: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the mixed-placement report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroMixedPlacement {
    pub plan: PlacementPlanV1,
    pub observed: MixedPlacementObservation,
    pub report: MixedPlacementReportV1,
}

/// Record the mixed-placement measurement.
pub fn measure_mixed_placement(
    plan: &PlacementPlanV1,
    observed: &MixedPlacementObservation,
) -> Result<MicroMixedPlacement, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_mixed_placement(&produced)?;
    Ok(produced)
}

/// Recompute the mixed-placement report from the stored plan and observation.
pub fn verify_mixed_placement(measurement: &MicroMixedPlacement) -> Result<(), InferFailure> {
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
        "device count does not match",
        report.device_count,
        observed.device_count,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "single recorded does not match",
        report.single_recorded,
        observed.single_recorded,
    )?;
    same_bool(
        "mixed selected does not match",
        report.mixed_selected,
        observed.mixed_selected,
    )?;
    same_bool(
        "automatic fallback does not match",
        report.automatic_fallback,
        observed.automatic_fallback,
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
        "mixed-placement concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "mixed-placement run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "mixed-placement warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "mixed-placement request count does not match",
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
            "mixed-placement validation did not match",
        ));
    }
    Ok(())
}

/// Write the mixed-placement report.
pub fn write_mixed_placement_report(
    base: &Path,
    measurement: &MicroMixedPlacement,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_mixed_placement(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "mixed-placement",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &MixedPlacementObservation,
) -> Result<MicroMixedPlacement, InferFailure> {
    accept_micro_plan(plan, "mixed-placement")?;
    accept_cold_single(
        "mixed-placement",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = MixedPlacementReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        device_count: observed.device_count,
        code: observed.code.clone(),
        retryable: observed.retryable,
        single_recorded: observed.single_recorded,
        mixed_selected: observed.mixed_selected,
        automatic_fallback: observed.automatic_fallback,
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
    Ok(MicroMixedPlacement {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
