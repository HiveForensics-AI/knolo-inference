//! One MTP head the engine did not run for a cold micro fixture.
//!
//! The report stores the reason and the proposal length. This measurement
//! does not run an MTP head.
//!
//! The layout is specified in `spec/KIP-INFER-0130-mtp.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, MtpReportV1, PlacementPlanV1, MAX_MTP_TOKENS,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied mtp record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub proposal_tokens: u32,
    pub accepted_tokens: u32,
    pub code: String,
    pub retryable: bool,
    pub head_ran: bool,
    pub drafted: bool,
    pub accepted: bool,
    pub speculated: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the mtp report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroMtp {
    pub plan: PlacementPlanV1,
    pub observed: MtpObservation,
    pub report: MtpReportV1,
}

/// Record the mtp measurement.
pub fn measure_mtp(
    plan: &PlacementPlanV1,
    observed: &MtpObservation,
) -> Result<MicroMtp, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_mtp(&produced)?;
    Ok(produced)
}

/// Recompute the mtp report from the stored plan and observation.
pub fn verify_mtp(measurement: &MicroMtp) -> Result<(), InferFailure> {
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
        "proposal tokens do not match",
        report.proposal_tokens,
        observed.proposal_tokens,
    )?;
    same_u32(
        "accepted tokens do not match",
        report.accepted_tokens,
        observed.accepted_tokens,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "head ran does not match",
        report.head_ran,
        observed.head_ran,
    )?;
    same_bool("drafted does not match", report.drafted, observed.drafted)?;
    same_bool(
        "accepted does not match",
        report.accepted,
        observed.accepted,
    )?;
    same_bool(
        "speculated does not match",
        report.speculated,
        observed.speculated,
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
        "mtp concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "mtp run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "mtp warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "mtp request count does not match",
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
            "mtp validation did not match",
        ));
    }
    Ok(())
}

/// Write the mtp report.
pub fn write_mtp_report(
    base: &Path,
    measurement: &MicroMtp,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_mtp(measurement)?;
    write_exclusive(base, receipt_path, &measurement.report.to_bytes()?, "mtp")
}

fn assemble(plan: &PlacementPlanV1, observed: &MtpObservation) -> Result<MicroMtp, InferFailure> {
    accept_micro_plan(plan, "mtp")?;
    accept_cold_single(
        "mtp",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.proposal_tokens > MAX_MTP_TOKENS {
        return Err(fail(
            ErrorCode::ContextLimitExceeded,
            "an MTP proposal exceeds the context",
        ));
    }

    let report = MtpReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        proposal_tokens: observed.proposal_tokens,
        accepted_tokens: observed.accepted_tokens,
        code: observed.code.clone(),
        retryable: observed.retryable,
        head_ran: observed.head_ran,
        drafted: observed.drafted,
        accepted: observed.accepted,
        speculated: observed.speculated,
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
    Ok(MicroMtp {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
