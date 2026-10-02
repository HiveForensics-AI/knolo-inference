//! One positional extension the engine did not apply for a cold micro fixture.
//!
//! The report stores the reason and the count. This measurement does
//! not run the feature.
//!
//! The layout is specified in `spec/KIP-INFER-0136-positional-extension.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, PositionalReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied positional record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionalObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub requested_scale: u32,
    pub applied_scale: u32,
    pub code: String,
    pub retryable: bool,
    pub yarn_applied: bool,
    pub ntk_applied: bool,
    pub rope_scaled: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the positional report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroPositional {
    pub plan: PlacementPlanV1,
    pub observed: PositionalObservation,
    pub report: PositionalReportV1,
}

/// Record the positional measurement.
pub fn measure_positional(
    plan: &PlacementPlanV1,
    observed: &PositionalObservation,
) -> Result<MicroPositional, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_positional(&produced)?;
    Ok(produced)
}

/// Recompute the positional report from the stored plan and observation.
pub fn verify_positional(measurement: &MicroPositional) -> Result<(), InferFailure> {
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
        "requested scale does not match",
        report.requested_scale,
        observed.requested_scale,
    )?;
    same_u32(
        "applied scale does not match",
        report.applied_scale,
        observed.applied_scale,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "yarn applied does not match",
        report.yarn_applied,
        observed.yarn_applied,
    )?;
    same_bool(
        "ntk applied does not match",
        report.ntk_applied,
        observed.ntk_applied,
    )?;
    same_bool(
        "rope scaled does not match",
        report.rope_scaled,
        observed.rope_scaled,
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
        "positional concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "positional run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "positional warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "positional request count does not match",
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
            "positional validation did not match",
        ));
    }
    Ok(())
}

/// Write the positional report.
pub fn write_positional_report(
    base: &Path,
    measurement: &MicroPositional,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_positional(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "positional",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &PositionalObservation,
) -> Result<MicroPositional, InferFailure> {
    accept_micro_plan(plan, "positional")?;
    accept_cold_single(
        "positional",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = PositionalReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        reason: observed.reason.clone(),
        requested_scale: observed.requested_scale,
        applied_scale: observed.applied_scale,
        code: observed.code.clone(),
        retryable: observed.retryable,
        yarn_applied: observed.yarn_applied,
        ntk_applied: observed.ntk_applied,
        rope_scaled: observed.rope_scaled,
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
    Ok(MicroPositional {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
