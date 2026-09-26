//! Safe error envelope for one cold micro fixture.
//!
//! The report stores the stable code and whether it can be retried. This
//! measurement does not return a prompt. The layout is specified in
//! `spec/KIP-INFER-0062-safe-error.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, SafeErrorReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied safe error for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeErrorObservation {
    pub engine_build_root: DigestHex,
    pub code: String,
    pub message: String,
    pub retryable: bool,
    pub request_id: String,
    pub attempt: u32,
    pub prompt_present: bool,
    pub secret_present: bool,
    pub partial_receipt: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the safe error report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroSafeError {
    pub plan: PlacementPlanV1,
    pub observed: SafeErrorObservation,
    pub report: SafeErrorReportV1,
}

/// Record the safe error. The prompt is not returned.
pub fn measure_safe_error(
    plan: &PlacementPlanV1,
    observed: &SafeErrorObservation,
) -> Result<MicroSafeError, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_safe_error(&produced)?;
    Ok(produced)
}

/// Recompute the safe error report from the stored plan and observation.
pub fn verify_safe_error(measurement: &MicroSafeError) -> Result<(), InferFailure> {
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
    same_text("code does not match", &report.code, &observed.code)?;
    same_text("message does not match", &report.message, &observed.message)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_text(
        "request id does not match",
        &report.request_id,
        &observed.request_id,
    )?;
    same_u32("attempt does not match", report.attempt, observed.attempt)?;
    same_bool(
        "prompt present does not match",
        report.prompt_present,
        observed.prompt_present,
    )?;
    same_bool(
        "secret present does not match",
        report.secret_present,
        observed.secret_present,
    )?;
    same_text(
        "partial receipt does not match",
        &report.partial_receipt,
        &observed.partial_receipt,
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
        "safe error concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "safe error run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "safe error warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "safe error request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "safe error validation did not match",
        ));
    }
    Ok(())
}

/// Write the safe error report. The prompt is not returned.
pub fn write_safe_error_report(
    base: &Path,
    measurement: &MicroSafeError,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_safe_error(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "safe error",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &SafeErrorObservation,
) -> Result<MicroSafeError, InferFailure> {
    accept_micro_plan(plan, "safe error")?;
    accept_cold_single(
        "safe error",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = SafeErrorReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        code: observed.code.clone(),
        message: observed.message.clone(),
        retryable: observed.retryable,
        request_id: observed.request_id.clone(),
        attempt: observed.attempt,
        prompt_present: observed.prompt_present,
        secret_present: observed.secret_present,
        partial_receipt: observed.partial_receipt.clone(),
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
    Ok(MicroSafeError {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
