//! One expert the planner did not place for a cold micro fixture.
//!
//! The report stores the reason and the device count. This measurement does
//! not place an expert. The layout is specified in
//! `spec/KIP-INFER-0119-expert-placement.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, ExpertPlacementReportV1, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied expert-placement record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertPlacementObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub device_count: u32,
    pub experts_placed: u32,
    pub code: String,
    pub retryable: bool,
    pub cpu_offload: bool,
    pub resident_moved: bool,
    pub tensor_parallel: bool,
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

/// One micro-fixture observation and the expert-placement report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroExpertPlacement {
    pub plan: PlacementPlanV1,
    pub observed: ExpertPlacementObservation,
    pub report: ExpertPlacementReportV1,
}

/// Record the expert-placement measurement.
pub fn measure_expert_placement(
    plan: &PlacementPlanV1,
    observed: &ExpertPlacementObservation,
) -> Result<MicroExpertPlacement, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_expert_placement(&produced)?;
    Ok(produced)
}

/// Recompute the expert-placement report from the stored plan and observation.
pub fn verify_expert_placement(measurement: &MicroExpertPlacement) -> Result<(), InferFailure> {
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
    same_u32(
        "experts placed do not match",
        report.experts_placed,
        observed.experts_placed,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "cpu offload does not match",
        report.cpu_offload,
        observed.cpu_offload,
    )?;
    same_bool(
        "resident moved does not match",
        report.resident_moved,
        observed.resident_moved,
    )?;
    same_bool(
        "tensor parallel does not match",
        report.tensor_parallel,
        observed.tensor_parallel,
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
        "expert-placement concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "expert-placement run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "expert-placement warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "expert-placement request count does not match",
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
            "expert-placement validation did not match",
        ));
    }
    Ok(())
}

/// Write the expert-placement report.
pub fn write_expert_placement_report(
    base: &Path,
    measurement: &MicroExpertPlacement,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_expert_placement(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "expert-placement",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ExpertPlacementObservation,
) -> Result<MicroExpertPlacement, InferFailure> {
    accept_micro_plan(plan, "expert-placement")?;
    accept_cold_single(
        "expert-placement",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = ExpertPlacementReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        device_count: observed.device_count,
        experts_placed: observed.experts_placed,
        code: observed.code.clone(),
        retryable: observed.retryable,
        cpu_offload: observed.cpu_offload,
        resident_moved: observed.resident_moved,
        tensor_parallel: observed.tensor_parallel,
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
    Ok(MicroExpertPlacement {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
