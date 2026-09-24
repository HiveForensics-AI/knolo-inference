//! Signature shape gate for one cold micro fixture.
//!
//! The report stores a status, a key id, and a byte count. This measurement
//! does not read the signature and does not verify a key. The layout is
//! specified in `spec/KIP-INFER-0055-signature-gate.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, SignatureReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied signature shape for one release.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureObservation {
    pub engine_build_root: DigestHex,
    pub release_root: DigestHex,
    pub signature_status: String,
    pub signature_count: u32,
    pub key_id: String,
    pub signature_bytes: u32,
    pub key_verified: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the signature report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroSignature {
    pub plan: PlacementPlanV1,
    pub observed: SignatureObservation,
    pub report: SignatureReportV1,
}

/// Record the signature shape. The key is not checked.
pub fn measure_signature_gate(
    plan: &PlacementPlanV1,
    observed: &SignatureObservation,
) -> Result<MicroSignature, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_signature_gate(&produced)?;
    Ok(produced)
}

/// Recompute the signature report from the stored plan and observation.
pub fn verify_signature_gate(measurement: &MicroSignature) -> Result<(), InferFailure> {
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
    same_text(
        "signature status does not match",
        &report.signature_status,
        &observed.signature_status,
    )?;
    same_u32(
        "signature count does not match",
        report.signature_count,
        observed.signature_count,
    )?;
    same_text("key id does not match", &report.key_id, &observed.key_id)?;
    same_u32(
        "signature bytes do not match",
        report.signature_bytes,
        observed.signature_bytes,
    )?;
    same_bool(
        "key verified does not match",
        report.key_verified,
        observed.key_verified,
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
        "signature concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "signature run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "signature warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "signature request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "signature validation did not match",
        ));
    }
    Ok(())
}

/// Write the signature report. The signature bytes are not written.
pub fn write_signature_report(
    base: &Path,
    measurement: &MicroSignature,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_signature_gate(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "signature",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &SignatureObservation,
) -> Result<MicroSignature, InferFailure> {
    accept_micro_plan(plan, "signature")?;
    accept_cold_single(
        "signature",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = SignatureReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        release_root: observed.release_root.clone(),
        signature_status: observed.signature_status.clone(),
        signature_count: observed.signature_count,
        key_id: observed.key_id.clone(),
        signature_bytes: observed.signature_bytes,
        key_verified: observed.key_verified,
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
    Ok(MicroSignature {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
