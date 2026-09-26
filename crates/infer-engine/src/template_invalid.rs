//! Template refusal for one cold micro fixture.
//!
//! The report stores that the prompt compiler refused the template. This
//! measurement does not render a template. The layout is specified in
//! `spec/KIP-INFER-0089-template-invalid.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, TemplateInvalidReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied template refusal for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateInvalidObservation {
    pub engine_build_root: DigestHex,
    pub template_root: DigestHex,
    pub failure: String,
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

/// One micro-fixture observation and the template-invalid report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroTemplateInvalid {
    pub plan: PlacementPlanV1,
    pub observed: TemplateInvalidObservation,
    pub report: TemplateInvalidReportV1,
}

/// Record the template refusal. The template is not rendered.
pub fn measure_template_invalid(
    plan: &PlacementPlanV1,
    observed: &TemplateInvalidObservation,
) -> Result<MicroTemplateInvalid, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_template_invalid(&produced)?;
    Ok(produced)
}

/// Recompute the template-invalid report from the stored plan and observation.
pub fn verify_template_invalid(measurement: &MicroTemplateInvalid) -> Result<(), InferFailure> {
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
        "template root does not match",
        &report.template_root,
        &observed.template_root,
    )?;
    same_text("failure does not match", &report.failure, &observed.failure)?;
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
        "template concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "template run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "template warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "template request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "template validation did not match",
        ));
    }
    Ok(())
}

/// Write the template-invalid report. The template is not rendered.
pub fn write_template_invalid_report(
    base: &Path,
    measurement: &MicroTemplateInvalid,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_template_invalid(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "template",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &TemplateInvalidObservation,
) -> Result<MicroTemplateInvalid, InferFailure> {
    accept_micro_plan(plan, "template")?;
    accept_cold_single(
        "template",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = TemplateInvalidReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        template_root: observed.template_root.clone(),
        failure: observed.failure.clone(),
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
    Ok(MicroTemplateInvalid {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
