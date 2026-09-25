//! Invalid model-image signature block for one cold micro fixture.
//!
//! The report stores the refusal. This measurement does not read key bytes
//! and does not open a weight file. The layout is specified in
//! `spec/KIP-INFER-0100-model-image-signature.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, ImageSignatureReportV1, InferFailure, PlacementPlanV1,
    MAX_SIGNATURE_RECORD_BYTES, MAX_SIGNATURE_RECORD_COUNT,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied invalid signature block for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSignatureObservation {
    pub engine_build_root: DigestHex,
    pub image_root: DigestHex,
    pub reason: String,
    pub signature_count: u32,
    pub signature_bytes: u32,
    pub code: String,
    pub retryable: bool,
    pub algorithm_accepted: bool,
    pub weights_opened: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub key_material_present: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the image-signature report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroImageSignature {
    pub plan: PlacementPlanV1,
    pub observed: ImageSignatureObservation,
    pub report: ImageSignatureReportV1,
}

/// Record the invalid signature block. The key is not read.
pub fn measure_image_signature(
    plan: &PlacementPlanV1,
    observed: &ImageSignatureObservation,
) -> Result<MicroImageSignature, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_image_signature(&produced)?;
    Ok(produced)
}

/// Recompute the image-signature report from the stored plan and observation.
pub fn verify_image_signature(measurement: &MicroImageSignature) -> Result<(), InferFailure> {
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
    same_u32(
        "signature count does not match",
        report.signature_count,
        observed.signature_count,
    )?;
    same_u32(
        "signature bytes do not match",
        report.signature_bytes,
        observed.signature_bytes,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "algorithm accepted does not match",
        report.algorithm_accepted,
        observed.algorithm_accepted,
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
    same_bool(
        "key material present does not match",
        report.key_material_present,
        observed.key_material_present,
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
        "image-signature concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "image-signature run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "image-signature warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "image-signature request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "image-signature validation did not match",
        ));
    }
    Ok(())
}

/// Write the image-signature report. The key is not read.
pub fn write_image_signature_report(
    base: &Path,
    measurement: &MicroImageSignature,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_image_signature(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "image-signature",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ImageSignatureObservation,
) -> Result<MicroImageSignature, InferFailure> {
    accept_micro_plan(plan, "image-signature")?;
    accept_cold_single(
        "image-signature",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.reason == "length" && observed.signature_bytes > MAX_SIGNATURE_RECORD_BYTES {
        return Err(fail(
            ErrorCode::ModelImageSignatureInvalid,
            "signature bytes exceed the record cap",
        ));
    }
    if observed.reason == "count" && observed.signature_count > MAX_SIGNATURE_RECORD_COUNT {
        return Err(fail(
            ErrorCode::ModelImageSignatureInvalid,
            "signature count exceeds the record cap",
        ));
    }
    let report = ImageSignatureReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        image_root: observed.image_root.clone(),
        reason: observed.reason.clone(),
        signature_count: observed.signature_count,
        signature_bytes: observed.signature_bytes,
        code: observed.code.clone(),
        retryable: observed.retryable,
        algorithm_accepted: observed.algorithm_accepted,
        weights_opened: observed.weights_opened,
        forward_ran: observed.forward_ran,
        receipt_stored: observed.receipt_stored,
        key_material_present: observed.key_material_present,
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
    Ok(MicroImageSignature {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
