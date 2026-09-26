//! One speculative request the engine did not run for a cold micro fixture.
//!
//! The report stores the reason and the proposal length. This measurement
//! does not run a draft model or an MTP head. The layout is specified in
//! `spec/KIP-INFER-0112-speculative-decoding.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, SpeculativeReportV1,
    MAX_PROPOSAL_TOKENS,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied speculative refusal for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeObservation {
    pub engine_build_root: DigestHex,
    pub target_root: DigestHex,
    pub proposal_root: DigestHex,
    pub reason: String,
    pub proposal_tokens: u32,
    pub accepted_tokens: u32,
    pub rejected_tokens: u32,
    pub code: String,
    pub retryable: bool,
    pub speculated: bool,
    pub distribution_changed: bool,
    pub cache_affected: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the speculative report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroSpeculative {
    pub plan: PlacementPlanV1,
    pub observed: SpeculativeObservation,
    pub report: SpeculativeReportV1,
}

/// Record the speculative refusal. Speculation does not run.
pub fn measure_speculative(
    plan: &PlacementPlanV1,
    observed: &SpeculativeObservation,
) -> Result<MicroSpeculative, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_speculative(&produced)?;
    Ok(produced)
}

/// Recompute the speculative report from the stored plan and observation.
pub fn verify_speculative(measurement: &MicroSpeculative) -> Result<(), InferFailure> {
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
    same_root(
        "target root does not match",
        &report.target_root,
        &observed.target_root,
    )?;
    same_root(
        "proposal root does not match",
        &report.proposal_root,
        &observed.proposal_root,
    )?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_u32(
        "proposal tokens do not match",
        report.proposal_tokens,
        observed.proposal_tokens,
    )?;
    same_u32(
        "accepted tokens do not match",
        report.accepted_tokens,
        observed.accepted_tokens,
    )?;
    same_u32(
        "rejected tokens do not match",
        report.rejected_tokens,
        observed.rejected_tokens,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "speculated does not match",
        report.speculated,
        observed.speculated,
    )?;
    same_bool(
        "distribution changed does not match",
        report.distribution_changed,
        observed.distribution_changed,
    )?;
    same_bool(
        "cache affected does not match",
        report.cache_affected,
        observed.cache_affected,
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
        "speculative concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "speculative run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "speculative warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "speculative request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "speculative validation did not match",
        ));
    }
    Ok(())
}

/// Write the speculative report. Speculation does not run.
pub fn write_speculative_report(
    base: &Path,
    measurement: &MicroSpeculative,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_speculative(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "speculative",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &SpeculativeObservation,
) -> Result<MicroSpeculative, InferFailure> {
    accept_micro_plan(plan, "speculative")?;
    accept_cold_single(
        "speculative",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if matches!(observed.reason.as_str(), "draft" | "mtp")
        && observed.proposal_tokens > MAX_PROPOSAL_TOKENS
    {
        return Err(fail(
            ErrorCode::ContextLimitExceeded,
            "a speculative proposal exceeds the context",
        ));
    }
    let report = SpeculativeReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        target_root: observed.target_root.clone(),
        proposal_root: observed.proposal_root.clone(),
        reason: observed.reason.clone(),
        proposal_tokens: observed.proposal_tokens,
        accepted_tokens: observed.accepted_tokens,
        rejected_tokens: observed.rejected_tokens,
        code: observed.code.clone(),
        retryable: observed.retryable,
        speculated: observed.speculated,
        distribution_changed: observed.distribution_changed,
        cache_affected: observed.cache_affected,
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
    Ok(MicroSpeculative {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
