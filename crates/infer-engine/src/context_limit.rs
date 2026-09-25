//! Context limit for one cold micro fixture.
//!
//! The report stores the token counts. This measurement does not truncate
//! the prompt and does not run the forward. The layout is specified in
//! `spec/KIP-INFER-0097-context-limit.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, ContextLimitReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1,
    MAX_CONTEXT_PROMPT,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied context limit for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextLimitObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub prompt_tokens: u32,
    pub reserved_tokens: u32,
    pub context_tokens: u32,
    pub truncated: bool,
    pub code: String,
    pub retryable: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the context-limit report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroContextLimit {
    pub plan: PlacementPlanV1,
    pub observed: ContextLimitObservation,
    pub report: ContextLimitReportV1,
}

/// Record the context limit. Tokens are not truncated.
pub fn measure_context_limit(
    plan: &PlacementPlanV1,
    observed: &ContextLimitObservation,
) -> Result<MicroContextLimit, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_context_limit(&produced)?;
    Ok(produced)
}

/// Recompute the context-limit report from the stored plan and observation.
pub fn verify_context_limit(measurement: &MicroContextLimit) -> Result<(), InferFailure> {
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
    same_u32(
        "prompt tokens do not match",
        report.prompt_tokens,
        observed.prompt_tokens,
    )?;
    same_u32(
        "reserved tokens do not match",
        report.reserved_tokens,
        observed.reserved_tokens,
    )?;
    same_u32(
        "context tokens do not match",
        report.context_tokens,
        observed.context_tokens,
    )?;
    same_bool(
        "truncated does not match",
        report.truncated,
        observed.truncated,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
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
        "context-limit concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "context-limit run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "context-limit warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "context-limit request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "context-limit validation did not match",
        ));
    }
    Ok(())
}

/// Write the context-limit report. Tokens are not truncated.
pub fn write_context_limit_report(
    base: &Path,
    measurement: &MicroContextLimit,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_context_limit(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "context-limit",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ContextLimitObservation,
) -> Result<MicroContextLimit, InferFailure> {
    accept_micro_plan(plan, "context-limit")?;
    accept_cold_single(
        "context-limit",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.reason == "prompt" && observed.prompt_tokens > MAX_CONTEXT_PROMPT {
        return Err(fail(
            ErrorCode::ContextLimitExceeded,
            "prompt tokens exceed the record cap",
        ));
    }
    let report = ContextLimitReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        prompt_tokens: observed.prompt_tokens,
        reserved_tokens: observed.reserved_tokens,
        context_tokens: observed.context_tokens,
        truncated: observed.truncated,
        code: observed.code.clone(),
        retryable: observed.retryable,
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
    Ok(MicroContextLimit {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
