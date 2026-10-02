//! Tokenizer refusal for one cold micro fixture.
//!
//! The report stores that the prompt compiler refused the tokenizer. This
//! measurement does not parse a tokenizer. The layout is specified in
//! `spec/KIP-INFER-0088-tokenizer-invalid.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, TokenizerInvalidReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied tokenizer refusal for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerInvalidObservation {
    pub engine_build_root: DigestHex,
    pub tokenizer_root: DigestHex,
    pub failure: String,
    pub code: String,
    pub retryable: bool,
    pub tokenizer_parsed: bool,
    pub template_rendered: bool,
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

/// One micro-fixture observation and the tokenizer-invalid report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroTokenizerInvalid {
    pub plan: PlacementPlanV1,
    pub observed: TokenizerInvalidObservation,
    pub report: TokenizerInvalidReportV1,
}

/// Record the tokenizer refusal. The tokenizer is not parsed.
pub fn measure_tokenizer_invalid(
    plan: &PlacementPlanV1,
    observed: &TokenizerInvalidObservation,
) -> Result<MicroTokenizerInvalid, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_tokenizer_invalid(&produced)?;
    Ok(produced)
}

/// Recompute the tokenizer-invalid report from the stored plan and observation.
pub fn verify_tokenizer_invalid(measurement: &MicroTokenizerInvalid) -> Result<(), InferFailure> {
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
        "tokenizer root does not match",
        &report.tokenizer_root,
        &observed.tokenizer_root,
    )?;
    same_text("failure does not match", &report.failure, &observed.failure)?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "tokenizer parsed does not match",
        report.tokenizer_parsed,
        observed.tokenizer_parsed,
    )?;
    same_bool(
        "template rendered does not match",
        report.template_rendered,
        observed.template_rendered,
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
        "tokenizer concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "tokenizer run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "tokenizer warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "tokenizer request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "tokenizer validation did not match",
        ));
    }
    Ok(())
}

/// Write the tokenizer-invalid report. The tokenizer is not parsed.
pub fn write_tokenizer_invalid_report(
    base: &Path,
    measurement: &MicroTokenizerInvalid,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_tokenizer_invalid(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "tokenizer",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &TokenizerInvalidObservation,
) -> Result<MicroTokenizerInvalid, InferFailure> {
    accept_micro_plan(plan, "tokenizer")?;
    accept_cold_single(
        "tokenizer",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = TokenizerInvalidReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        tokenizer_root: observed.tokenizer_root.clone(),
        failure: observed.failure.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        tokenizer_parsed: observed.tokenizer_parsed,
        template_rendered: observed.template_rendered,
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
    Ok(MicroTokenizerInvalid {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
