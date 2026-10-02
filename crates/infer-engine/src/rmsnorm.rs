//! One rmsnorm kernel the engine did not apply for a cold micro fixture.
//!
//! The report stores the reason and the count. This measurement does
//! not run the feature.
//!
//! The layout is specified in `spec/KIP-INFER-0138-rmsnorm.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, RmsnormReportV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied rmsnorm record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmsnormObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub requested_layers: u32,
    pub applied_layers: u32,
    pub code: String,
    pub retryable: bool,
    pub norm_applied: bool,
    pub weight_applied: bool,
    pub eps_applied: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the rmsnorm report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroRmsnorm {
    pub plan: PlacementPlanV1,
    pub observed: RmsnormObservation,
    pub report: RmsnormReportV1,
}

/// Record the rmsnorm measurement.
pub fn measure_rmsnorm(
    plan: &PlacementPlanV1,
    observed: &RmsnormObservation,
) -> Result<MicroRmsnorm, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_rmsnorm(&produced)?;
    Ok(produced)
}

/// Recompute the rmsnorm report from the stored plan and observation.
pub fn verify_rmsnorm(measurement: &MicroRmsnorm) -> Result<(), InferFailure> {
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
        "requested layers do not match",
        report.requested_layers,
        observed.requested_layers,
    )?;
    same_u32(
        "applied layers do not match",
        report.applied_layers,
        observed.applied_layers,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "norm applied does not match",
        report.norm_applied,
        observed.norm_applied,
    )?;
    same_bool(
        "weight applied does not match",
        report.weight_applied,
        observed.weight_applied,
    )?;
    same_bool(
        "eps applied does not match",
        report.eps_applied,
        observed.eps_applied,
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
        "rmsnorm concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "rmsnorm run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "rmsnorm warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "rmsnorm request count does not match",
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
            "rmsnorm validation did not match",
        ));
    }
    Ok(())
}

/// Write the rmsnorm report.
pub fn write_rmsnorm_report(
    base: &Path,
    measurement: &MicroRmsnorm,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_rmsnorm(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "rmsnorm",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &RmsnormObservation,
) -> Result<MicroRmsnorm, InferFailure> {
    accept_micro_plan(plan, "rmsnorm")?;
    accept_cold_single(
        "rmsnorm",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = RmsnormReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        reason: observed.reason.clone(),
        requested_layers: observed.requested_layers,
        applied_layers: observed.applied_layers,
        code: observed.code.clone(),
        retryable: observed.retryable,
        norm_applied: observed.norm_applied,
        weight_applied: observed.weight_applied,
        eps_applied: observed.eps_applied,
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
    Ok(MicroRmsnorm {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
