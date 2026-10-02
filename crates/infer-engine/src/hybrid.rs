//! One hybrid-attention state the engine did not allocate for a cold micro fixture.
//!
//! The report stores the reason and the state span. This measurement does
//! not allocate hybrid state.
//!
//! The layout is specified in `spec/KIP-INFER-0127-hybrid-attention.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, HybridReportV1, InferFailure, PlacementPlanV1, MAX_HYBRID_TOKENS,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied hybrid record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub state_tokens: u32,
    pub code: String,
    pub retryable: bool,
    pub attention_exact: bool,
    pub hybrid_allocated: bool,
    pub sliding_applied: bool,
    pub full_mixed: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the hybrid report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroHybrid {
    pub plan: PlacementPlanV1,
    pub observed: HybridObservation,
    pub report: HybridReportV1,
}

/// Record the hybrid measurement.
pub fn measure_hybrid(
    plan: &PlacementPlanV1,
    observed: &HybridObservation,
) -> Result<MicroHybrid, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_hybrid(&produced)?;
    Ok(produced)
}

/// Recompute the hybrid report from the stored plan and observation.
pub fn verify_hybrid(measurement: &MicroHybrid) -> Result<(), InferFailure> {
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
        "hybrid allocated does not match",
        report.hybrid_allocated,
        observed.hybrid_allocated,
    )?;
    same_bool(
        "sliding applied does not match",
        report.sliding_applied,
        observed.sliding_applied,
    )?;
    same_bool(
        "full mixed does not match",
        report.full_mixed,
        observed.full_mixed,
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
        "hybrid concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "hybrid run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "hybrid warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "hybrid request count does not match",
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
            "hybrid validation did not match",
        ));
    }
    Ok(())
}

/// Write the hybrid report.
pub fn write_hybrid_report(
    base: &Path,
    measurement: &MicroHybrid,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_hybrid(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "hybrid",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &HybridObservation,
) -> Result<MicroHybrid, InferFailure> {
    accept_micro_plan(plan, "hybrid")?;
    accept_cold_single(
        "hybrid",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.reason == "state" && observed.state_tokens > MAX_HYBRID_TOKENS {
        return Err(fail(
            ErrorCode::ContextLimitExceeded,
            "a hybrid state exceeds the context",
        ));
    }

    let report = HybridReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        state_tokens: observed.state_tokens,
        code: observed.code.clone(),
        retryable: observed.retryable,
        attention_exact: observed.attention_exact,
        hybrid_allocated: observed.hybrid_allocated,
        sliding_applied: observed.sliding_applied,
        full_mixed: observed.full_mixed,
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
    Ok(MicroHybrid {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
