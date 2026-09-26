//! Host-supplied Ed25519 signature check for one cold micro fixture.
//!
//! The report stores the status and the byte counts. This measurement does
//! not read key bytes and does not clear the cofactor. The layout is
//! specified in `spec/KIP-INFER-0096-signature-check.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, SignatureCheckReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied signature check for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureCheckObservation {
    pub engine_build_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub check_status: String,
    pub signature_checked: bool,
    pub cofactor_cleared: bool,
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

/// One micro-fixture observation and the signature-check report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroSignatureCheck {
    pub plan: PlacementPlanV1,
    pub observed: SignatureCheckObservation,
    pub report: SignatureCheckReportV1,
}

/// Record the signature check. The signature is not checked.
pub fn measure_signature_check(
    plan: &PlacementPlanV1,
    observed: &SignatureCheckObservation,
) -> Result<MicroSignatureCheck, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_signature_check(&produced)?;
    Ok(produced)
}

/// Recompute the signature-check report from the stored plan and observation.
pub fn verify_signature_check(measurement: &MicroSignatureCheck) -> Result<(), InferFailure> {
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
        "check status does not match",
        &report.check_status,
        &observed.check_status,
    )?;
    same_bool(
        "signature checked does not match",
        report.signature_checked,
        observed.signature_checked,
    )?;
    same_bool(
        "cofactor cleared does not match",
        report.cofactor_cleared,
        observed.cofactor_cleared,
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
        "signature-check concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "signature-check run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "signature-check warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "signature-check request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "signature-check validation did not match",
        ));
    }
    Ok(())
}

/// Write the signature-check report. The signature is not checked.
pub fn write_signature_check_report(
    base: &Path,
    measurement: &MicroSignatureCheck,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_signature_check(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "signature-check",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &SignatureCheckObservation,
) -> Result<MicroSignatureCheck, InferFailure> {
    accept_micro_plan(plan, "signature-check")?;
    accept_cold_single(
        "signature-check",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let validation_result = if observed.check_status == "checked" {
        "verified"
    } else {
        "recorded"
    };
    let report = SignatureCheckReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        release_root: observed.release_root.clone(),
        message_root: observed.message_root.clone(),
        check_status: observed.check_status.clone(),
        signature_checked: observed.signature_checked,
        cofactor_cleared: observed.cofactor_cleared,
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
    Ok(MicroSignatureCheck {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
