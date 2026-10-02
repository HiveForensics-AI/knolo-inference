//! One linear-attention request the engine did not apply for a cold micro fixture.
//!
//! The report stores the reason and the state span. This measurement does
//! not run linear attention.
//!
//! The layout is specified in `spec/KIP-INFER-0128-linear-attention.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, LinearReportV1, PlacementPlanV1, MAX_LINEAR_TOKENS,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied linear record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub state_tokens: u32,
    pub code: String,
    pub retryable: bool,
    pub attention_exact: bool,
    pub linear_applied: bool,
    pub kda_selected: bool,
    pub decay_applied: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the linear report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroLinear {
    pub plan: PlacementPlanV1,
    pub observed: LinearObservation,
    pub report: LinearReportV1,
}

/// Record the linear measurement.
pub fn measure_linear(
    plan: &PlacementPlanV1,
    observed: &LinearObservation,
) -> Result<MicroLinear, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_linear(&produced)?;
    Ok(produced)
}

/// Recompute the linear report from the stored plan and observation.
pub fn verify_linear(measurement: &MicroLinear) -> Result<(), InferFailure> {
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
        "state tokens do not match",
        report.state_tokens,
        observed.state_tokens,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "attention exact does not match",
        report.attention_exact,
        observed.attention_exact,
    )?;
    same_bool(
        "linear applied does not match",
        report.linear_applied,
        observed.linear_applied,
    )?;
    same_bool(
        "kda selected does not match",
        report.kda_selected,
        observed.kda_selected,
    )?;
    same_bool(
        "decay applied does not match",
        report.decay_applied,
        observed.decay_applied,
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
        "linear concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "linear run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "linear warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "linear request count does not match",
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
            "linear validation did not match",
        ));
    }
    Ok(())
}

/// Write the linear report.
pub fn write_linear_report(
    base: &Path,
    measurement: &MicroLinear,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_linear(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "linear",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &LinearObservation,
) -> Result<MicroLinear, InferFailure> {
    accept_micro_plan(plan, "linear")?;
    accept_cold_single(
        "linear",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.reason == "state" && observed.state_tokens > MAX_LINEAR_TOKENS {
        return Err(fail(
            ErrorCode::ContextLimitExceeded,
            "a linear state exceeds the context",
        ));
    }

    let report = LinearReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        state_tokens: observed.state_tokens,
        code: observed.code.clone(),
        retryable: observed.retryable,
        attention_exact: observed.attention_exact,
        linear_applied: observed.linear_applied,
        kda_selected: observed.kda_selected,
        decay_applied: observed.decay_applied,
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
    Ok(MicroLinear {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
