//! One fused residual the engine did not apply for a cold micro fixture.
//!
//! The report stores the reason and the count. This measurement does
//! not run the feature.
//!
//! The layout is specified in `spec/KIP-INFER-0140-fused-residual.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, ResidualReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied residual record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub requested_layers: u32,
    pub applied_layers: u32,
    pub code: String,
    pub retryable: bool,
    pub fused_applied: bool,
    pub added: bool,
    pub norm_folded: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the residual report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroResidual {
    pub plan: PlacementPlanV1,
    pub observed: ResidualObservation,
    pub report: ResidualReportV1,
}

/// Record the residual measurement.
pub fn measure_residual(
    plan: &PlacementPlanV1,
    observed: &ResidualObservation,
) -> Result<MicroResidual, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_residual(&produced)?;
    Ok(produced)
}

/// Recompute the residual report from the stored plan and observation.
pub fn verify_residual(measurement: &MicroResidual) -> Result<(), InferFailure> {
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
        "fused applied does not match",
        report.fused_applied,
        observed.fused_applied,
    )?;
    same_bool("added does not match", report.added, observed.added)?;
    same_bool(
        "norm folded does not match",
        report.norm_folded,
        observed.norm_folded,
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
        "residual concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "residual run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "residual warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "residual request count does not match",
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
            "residual validation did not match",
        ));
    }
    Ok(())
}

/// Write the residual report.
pub fn write_residual_report(
    base: &Path,
    measurement: &MicroResidual,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_residual(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "residual",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ResidualObservation,
) -> Result<MicroResidual, InferFailure> {
    accept_micro_plan(plan, "residual")?;
    accept_cold_single(
        "residual",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = ResidualReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        reason: observed.reason.clone(),
        requested_layers: observed.requested_layers,
        applied_layers: observed.applied_layers,
        code: observed.code.clone(),
        retryable: observed.retryable,
        fused_applied: observed.fused_applied,
        added: observed.added,
        norm_folded: observed.norm_folded,
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
    Ok(MicroResidual {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
