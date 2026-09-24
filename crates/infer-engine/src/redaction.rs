//! Redacted completion log for one cold micro fixture.
//!
//! The report stores the stage and the roots. This measurement does not
//! write a log line. The layout is specified in
//! `spec/KIP-INFER-0063-redacted-log.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, RedactionReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied redacted log for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedactionObservation {
    pub engine_build_root: DigestHex,
    pub stage: String,
    pub request_id: String,
    pub prompt_plan_root: DigestHex,
    pub receipt_root: DigestHex,
    pub format: String,
    pub redacted: bool,
    pub prompt: String,
    pub output: String,
    pub token_ids: String,
    pub alias: String,
    pub path: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the redaction report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroRedaction {
    pub plan: PlacementPlanV1,
    pub observed: RedactionObservation,
    pub report: RedactionReportV1,
}

/// Record the redacted log. The line is not written.
pub fn measure_redacted_log(
    plan: &PlacementPlanV1,
    observed: &RedactionObservation,
) -> Result<MicroRedaction, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_redacted_log(&produced)?;
    Ok(produced)
}

/// Recompute the redaction report from the stored plan and observation.
pub fn verify_redacted_log(measurement: &MicroRedaction) -> Result<(), InferFailure> {
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
    same_text("stage does not match", &report.stage, &observed.stage)?;
    same_text(
        "request id does not match",
        &report.request_id,
        &observed.request_id,
    )?;
    same_root(
        "prompt plan root does not match",
        &report.prompt_plan_root,
        &observed.prompt_plan_root,
    )?;
    same_root(
        "receipt root does not match",
        &report.receipt_root,
        &observed.receipt_root,
    )?;
    same_text("format does not match", &report.format, &observed.format)?;
    same_bool(
        "redacted does not match",
        report.redacted,
        observed.redacted,
    )?;
    same_text("prompt does not match", &report.prompt, &observed.prompt)?;
    same_text("output does not match", &report.output, &observed.output)?;
    same_text(
        "token ids do not match",
        &report.token_ids,
        &observed.token_ids,
    )?;
    same_text("alias does not match", &report.alias, &observed.alias)?;
    same_text("path does not match", &report.path, &observed.path)?;
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
        "redaction concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "redaction run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "redaction warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "redaction request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "redaction validation did not match",
        ));
    }
    Ok(())
}

/// Write the redaction report. The line is not written.
pub fn write_redaction_report(
    base: &Path,
    measurement: &MicroRedaction,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_redacted_log(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "redaction",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &RedactionObservation,
) -> Result<MicroRedaction, InferFailure> {
    accept_micro_plan(plan, "redaction")?;
    accept_cold_single(
        "redaction",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = RedactionReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        stage: observed.stage.clone(),
        request_id: observed.request_id.clone(),
        prompt_plan_root: observed.prompt_plan_root.clone(),
        receipt_root: observed.receipt_root.clone(),
        format: observed.format.clone(),
        redacted: observed.redacted,
        prompt: observed.prompt.clone(),
        output: observed.output.clone(),
        token_ids: observed.token_ids.clone(),
        alias: observed.alias.clone(),
        path: observed.path.clone(),
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
    Ok(MicroRedaction {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
