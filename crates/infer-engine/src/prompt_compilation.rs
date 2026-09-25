//! Prompt compilation failure for one cold micro fixture.
//!
//! The report stores the failure. This measurement does not compile a
//! prompt and does not run the forward. The layout is specified in
//! `spec/KIP-INFER-0098-prompt-compilation.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, PromptCompilationReportV1,
    MAX_REJECTED_TOKEN, MICRO_CONTEXT,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied prompt compilation failure for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptCompilationObservation {
    pub engine_build_root: DigestHex,
    pub failure: String,
    pub token_count: u32,
    pub rejected_token: u32,
    pub code: String,
    pub retryable: bool,
    pub template_rendered: bool,
    pub tokenizer_parsed: bool,
    pub prompt_compiled: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the prompt-compilation report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroPromptCompilation {
    pub plan: PlacementPlanV1,
    pub observed: PromptCompilationObservation,
    pub report: PromptCompilationReportV1,
}

/// Record the prompt compilation failure. The prompt is not compiled.
pub fn measure_prompt_compilation(
    plan: &PlacementPlanV1,
    observed: &PromptCompilationObservation,
) -> Result<MicroPromptCompilation, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_prompt_compilation(&produced)?;
    Ok(produced)
}

/// Recompute the prompt-compilation report from the stored plan and observation.
pub fn verify_prompt_compilation(measurement: &MicroPromptCompilation) -> Result<(), InferFailure> {
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
    same_text("failure does not match", &report.failure, &observed.failure)?;
    same_u32(
        "token count does not match",
        report.token_count,
        observed.token_count,
    )?;
    same_u32(
        "rejected token does not match",
        report.rejected_token,
        observed.rejected_token,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "template rendered does not match",
        report.template_rendered,
        observed.template_rendered,
    )?;
    same_bool(
        "tokenizer parsed does not match",
        report.tokenizer_parsed,
        observed.tokenizer_parsed,
    )?;
    same_bool(
        "prompt compiled does not match",
        report.prompt_compiled,
        observed.prompt_compiled,
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
        "prompt-compilation concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "prompt-compilation run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "prompt-compilation warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "prompt-compilation request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "prompt-compilation validation did not match",
        ));
    }
    Ok(())
}

/// Write the prompt-compilation report. The prompt is not compiled.
pub fn write_prompt_compilation_report(
    base: &Path,
    measurement: &MicroPromptCompilation,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_prompt_compilation(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "prompt-compilation",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &PromptCompilationObservation,
) -> Result<MicroPromptCompilation, InferFailure> {
    accept_micro_plan(plan, "prompt-compilation")?;
    accept_cold_single(
        "prompt-compilation",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.failure == "vocab" && observed.token_count > MICRO_CONTEXT {
        return Err(fail(
            ErrorCode::ContextLimitExceeded,
            "the prompt compilation is the micro fixture",
        ));
    }
    if observed.failure == "vocab" && observed.rejected_token > MAX_REJECTED_TOKEN {
        return Err(fail(
            ErrorCode::PromptCompilationFailed,
            "a rejected token exceeds the record cap",
        ));
    }
    let report = PromptCompilationReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        failure: observed.failure.clone(),
        token_count: observed.token_count,
        rejected_token: observed.rejected_token,
        code: observed.code.clone(),
        retryable: observed.retryable,
        template_rendered: observed.template_rendered,
        tokenizer_parsed: observed.tokenizer_parsed,
        prompt_compiled: observed.prompt_compiled,
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
    Ok(MicroPromptCompilation {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
