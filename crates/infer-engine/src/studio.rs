//! Receipt view for one cold micro fixture.
//!
//! The view stores roots and counts. Prompt text and output text are not
//! fields. This measurement does not open a receipt file and does not render
//! a panel. The layout is specified in `spec/KIP-INFER-0047-receipt-view.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, StudioReportV1};

use crate::report_io::{accept_cold_single, accept_micro_plan, write_exclusive};

/// Roots and counts from one stored micro-fixture receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StudioObservation {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub receipt_root: DigestHex,
    pub prompt_token_root: DigestHex,
    pub output_token_root: DigestHex,
    pub output_text_root: DigestHex,
    pub knowledge_image_root: DigestHex,
    pub evidence_root: DigestHex,
    pub prompt_token_count: u32,
    pub output_token_count: u32,
    pub finish_reason: String,
    pub assurance: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the receipt view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroStudio {
    pub plan: PlacementPlanV1,
    pub observed: StudioObservation,
    pub report: StudioReportV1,
}

/// Record the receipt view. No receipt file is opened and no file is created.
pub fn measure_receipt_view(
    plan: &PlacementPlanV1,
    observed: &StudioObservation,
) -> Result<MicroStudio, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_receipt_view(&produced)?;
    Ok(produced)
}

/// Recompute the receipt view from the stored plan and observation.
pub fn verify_receipt_view(measurement: &MicroStudio) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "model image root does not match",
        &report.model_image_root,
        &observed.model_image_root,
    )?;
    same_root(
        "artifact root does not match",
        &report.artifact_root,
        &observed.artifact_root,
    )?;
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
        "receipt root does not match",
        &report.receipt_root,
        &observed.receipt_root,
    )?;
    same_root(
        "prompt token root does not match",
        &report.prompt_token_root,
        &observed.prompt_token_root,
    )?;
    same_root(
        "output token root does not match",
        &report.output_token_root,
        &observed.output_token_root,
    )?;
    same_root(
        "output text root does not match",
        &report.output_text_root,
        &observed.output_text_root,
    )?;
    same_root(
        "knowledge image root does not match",
        &report.knowledge_image_root,
        &observed.knowledge_image_root,
    )?;
    same_root(
        "evidence root does not match",
        &report.evidence_root,
        &observed.evidence_root,
    )?;
    same_u32(
        "prompt token count does not match",
        report.prompt_token_count,
        observed.prompt_token_count,
    )?;
    same_u32(
        "output token count does not match",
        report.output_token_count,
        observed.output_token_count,
    )?;
    same_text(
        "finish reason does not match",
        &report.finish_reason,
        &observed.finish_reason,
    )?;
    same_text(
        "assurance does not match",
        &report.assurance,
        &observed.assurance,
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
        "studio concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "studio run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "studio warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "studio request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "studio validation did not match",
        ));
    }
    Ok(())
}

/// Write the receipt view. The inference receipt is not written.
pub fn write_studio_report(
    base: &Path,
    measurement: &MicroStudio,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_receipt_view(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "studio",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &StudioObservation,
) -> Result<MicroStudio, InferFailure> {
    accept_micro_plan(plan, "studio")?;
    accept_cold_single(
        "studio",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = StudioReportV1 {
        model_image_root: observed.model_image_root.clone(),
        artifact_root: observed.artifact_root.clone(),
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        receipt_root: observed.receipt_root.clone(),
        prompt_token_root: observed.prompt_token_root.clone(),
        output_token_root: observed.output_token_root.clone(),
        output_text_root: observed.output_text_root.clone(),
        knowledge_image_root: observed.knowledge_image_root.clone(),
        evidence_root: observed.evidence_root.clone(),
        prompt_token_count: observed.prompt_token_count,
        output_token_count: observed.output_token_count,
        finish_reason: observed.finish_reason.clone(),
        assurance: observed.assurance.clone(),
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
    Ok(MicroStudio {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}

fn same_root(message: &str, left: &DigestHex, right: &DigestHex) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn same_text(message: &str, left: &str, right: &str) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn same_u32(message: &str, left: u32, right: u32) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}
