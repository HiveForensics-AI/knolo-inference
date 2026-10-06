//! Host-supplied Ed25519 receipt signature for one cold micro fixture.
//!
//! The report stores the status and the byte counts. This measurement does
//! not read key bytes and does not verify a receipt. The layout is specified
//! in `spec/KIP-INFER-0106-receipt-signing.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, ReceiptSignReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied receipt signature for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptSignObservation {
    pub engine_build_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub sign_status: String,
    pub receipt_signed: bool,
    pub receipt_verified: bool,
    pub public_key_bytes: u32,
    pub signature_bytes: u32,
    pub scalar_bytes: u32,
    pub key_material_present: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the receipt-sign report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroReceiptSign {
    pub plan: PlacementPlanV1,
    pub observed: ReceiptSignObservation,
    pub report: ReceiptSignReportV1,
}

/// Record the receipt signature. The receipt is not signed.
pub fn measure_receipt_sign(
    plan: &PlacementPlanV1,
    observed: &ReceiptSignObservation,
) -> Result<MicroReceiptSign, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_receipt_sign(&produced)?;
    Ok(produced)
}

/// Recompute the receipt-sign report from the stored plan and observation.
pub fn verify_receipt_sign(measurement: &MicroReceiptSign) -> Result<(), InferFailure> {
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
        "release root does not match",
        &report.release_root,
        &observed.release_root,
    )?;
    same_root(
        "message root does not match",
        &report.message_root,
        &observed.message_root,
    )?;
    same_text(
        "sign status does not match",
        &report.sign_status,
        &observed.sign_status,
    )?;
    same_bool(
        "receipt signed does not match",
        report.receipt_signed,
        observed.receipt_signed,
    )?;
    same_bool(
        "receipt verified does not match",
        report.receipt_verified,
        observed.receipt_verified,
    )?;
    same_u32(
        "public key bytes do not match",
        report.public_key_bytes,
        observed.public_key_bytes,
    )?;
    same_u32(
        "signature bytes do not match",
        report.signature_bytes,
        observed.signature_bytes,
    )?;
    same_u32(
        "scalar bytes do not match",
        report.scalar_bytes,
        observed.scalar_bytes,
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
        "receipt-sign concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "receipt-sign run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "receipt-sign warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "receipt-sign request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "receipt-sign validation did not match",
        ));
    }
    Ok(())
}

/// Write the receipt-sign report. The receipt is not signed.
pub fn write_receipt_sign_report(
    base: &Path,
    measurement: &MicroReceiptSign,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_receipt_sign(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "receipt-sign",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ReceiptSignObservation,
) -> Result<MicroReceiptSign, InferFailure> {
    accept_micro_plan(plan, "receipt-sign")?;
    accept_cold_single(
        "receipt-sign",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let validation_result = if observed.sign_status == "signed" {
        "verified"
    } else {
        "recorded"
    };
    let report = ReceiptSignReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        release_root: observed.release_root.clone(),
        message_root: observed.message_root.clone(),
        sign_status: observed.sign_status.clone(),
        receipt_signed: observed.receipt_signed,
        receipt_verified: observed.receipt_verified,
        public_key_bytes: observed.public_key_bytes,
        signature_bytes: observed.signature_bytes,
        scalar_bytes: observed.scalar_bytes,
        key_material_present: observed.key_material_present,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        validation_result: validation_result.into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroReceiptSign {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
