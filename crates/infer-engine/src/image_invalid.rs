//! Invalid model image for one cold micro fixture.
//!
//! The report stores the refusal. This measurement does not open a weight
//! file and does not run the forward. The layout is specified in
//! `spec/KIP-INFER-0099-model-image-invalid.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, ImageInvalidReportV1, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied invalid model image for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInvalidObservation {
    pub engine_build_root: DigestHex,
    pub image_root: DigestHex,
    pub reason: String,
    pub code: String,
    pub retryable: bool,
    pub image_parsed: bool,
    pub weights_opened: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the image-invalid report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroImageInvalid {
    pub plan: PlacementPlanV1,
    pub observed: ImageInvalidObservation,
    pub report: ImageInvalidReportV1,
}

/// Record the invalid image. Weights are not opened by this measurement.
pub fn measure_image_invalid(
    plan: &PlacementPlanV1,
    observed: &ImageInvalidObservation,
) -> Result<MicroImageInvalid, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_image_invalid(&produced)?;
    Ok(produced)
}

/// Recompute the image-invalid report from the stored plan and observation.
pub fn verify_image_invalid(measurement: &MicroImageInvalid) -> Result<(), InferFailure> {
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
        "image root does not match",
        &report.image_root,
        &observed.image_root,
    )?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "image parsed does not match",
        report.image_parsed,
        observed.image_parsed,
    )?;
    same_bool(
        "weights opened does not match",
        report.weights_opened,
        observed.weights_opened,
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
        "image-invalid concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "image-invalid run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "image-invalid warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "image-invalid request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "image-invalid validation did not match",
        ));
    }
    Ok(())
}

/// Write the image-invalid report. Weights are not opened by this measurement.
pub fn write_image_invalid_report(
    base: &Path,
    measurement: &MicroImageInvalid,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_image_invalid(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "image-invalid",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ImageInvalidObservation,
) -> Result<MicroImageInvalid, InferFailure> {
    accept_micro_plan(plan, "image-invalid")?;
    accept_cold_single(
        "image-invalid",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = ImageInvalidReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        image_root: observed.image_root.clone(),
        reason: observed.reason.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        image_parsed: observed.image_parsed,
        weights_opened: observed.weights_opened,
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
    Ok(MicroImageInvalid {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
