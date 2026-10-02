//! Unsatisfiable placement for one cold micro fixture.
//!
//! The report stores why the placement was refused. This measurement does
//! not open a device and does not fall back to another device. The layout is
//! specified in `spec/KIP-INFER-0094-placement-unsatisfiable.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, PlacementRefusalReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied unsatisfiable placement for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementRefusalObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub code: String,
    pub retryable: bool,
    pub probe_reached: bool,
    pub slot_visible: bool,
    pub device_opened: bool,
    pub cpu_fallback: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the placement-refusal report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroPlacementRefusal {
    pub plan: PlacementPlanV1,
    pub observed: PlacementRefusalObservation,
    pub report: PlacementRefusalReportV1,
}

/// Record the unsatisfiable placement. The device is not opened.
pub fn measure_placement_refusal(
    plan: &PlacementPlanV1,
    observed: &PlacementRefusalObservation,
) -> Result<MicroPlacementRefusal, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_placement_refusal(&produced)?;
    Ok(produced)
}

/// Recompute the placement-refusal report from the stored plan and observation.
pub fn verify_placement_refusal(measurement: &MicroPlacementRefusal) -> Result<(), InferFailure> {
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
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "probe reached does not match",
        report.probe_reached,
        observed.probe_reached,
    )?;
    same_bool(
        "slot visible does not match",
        report.slot_visible,
        observed.slot_visible,
    )?;
    same_bool(
        "device opened does not match",
        report.device_opened,
        observed.device_opened,
    )?;
    same_bool(
        "cpu fallback does not match",
        report.cpu_fallback,
        observed.cpu_fallback,
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
        "placement-refusal concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "placement-refusal run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "placement-refusal warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "placement-refusal request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "placement-refusal validation did not match",
        ));
    }
    Ok(())
}

/// Write the placement-refusal report. The device is not opened.
pub fn write_placement_refusal_report(
    base: &Path,
    measurement: &MicroPlacementRefusal,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_placement_refusal(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "placement-refusal",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &PlacementRefusalObservation,
) -> Result<MicroPlacementRefusal, InferFailure> {
    accept_micro_plan(plan, "placement-refusal")?;
    accept_cold_single(
        "placement-refusal",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = PlacementRefusalReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        probe_reached: observed.probe_reached,
        slot_visible: observed.slot_visible,
        device_opened: observed.device_opened,
        cpu_fallback: observed.cpu_fallback,
        forward_ran: observed.forward_ran,
        receipt_stored: observed.receipt_stored,
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
    Ok(MicroPlacementRefusal {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
