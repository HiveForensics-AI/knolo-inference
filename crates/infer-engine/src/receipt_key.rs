//! Receipt-key custody for one cold micro fixture.
//!
//! The report stores where the signing key lives. This measurement does not
//! load a secret. The layout is specified in `spec/KIP-INFER-0057-receipt-key.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, ReceiptKeyReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied custody for one receipt signing key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptKeyObservation {
    pub engine_build_root: DigestHex,
    pub receipt_root: DigestHex,
    pub custody: String,
    pub key_material_serialized: bool,
    pub key_id: String,
    pub signature_status: String,
    pub signature_bytes: u32,
    pub rotation: String,
    pub previous_key_id: String,
    pub trusted_metadata: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the receipt-key report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroReceiptKey {
    pub plan: PlacementPlanV1,
    pub observed: ReceiptKeyObservation,
    pub report: ReceiptKeyReportV1,
}

/// Record receipt-key custody. The secret is not loaded.
pub fn measure_receipt_key(
    plan: &PlacementPlanV1,
    observed: &ReceiptKeyObservation,
) -> Result<MicroReceiptKey, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_receipt_key(&produced)?;
    Ok(produced)
}

/// Recompute the receipt-key report from the stored plan and observation.
pub fn verify_receipt_key(measurement: &MicroReceiptKey) -> Result<(), InferFailure> {
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
        "receipt root does not match",
        &report.receipt_root,
        &observed.receipt_root,
    )?;
    same_text("custody does not match", &report.custody, &observed.custody)?;
    same_bool(
        "key material does not match",
        report.key_material_serialized,
        observed.key_material_serialized,
    )?;
    same_text("key id does not match", &report.key_id, &observed.key_id)?;
    same_text(
        "signature status does not match",
        &report.signature_status,
        &observed.signature_status,
    )?;
    same_u32(
        "signature bytes do not match",
        report.signature_bytes,
        observed.signature_bytes,
    )?;
    same_text(
        "rotation does not match",
        &report.rotation,
        &observed.rotation,
    )?;
    same_text(
        "previous key id does not match",
        &report.previous_key_id,
        &observed.previous_key_id,
    )?;
    same_bool(
        "trusted metadata does not match",
        report.trusted_metadata,
        observed.trusted_metadata,
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
        "receipt key concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "receipt key run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "receipt key warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "receipt key request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "receipt key validation did not match",
        ));
    }
    Ok(())
}

/// Write the receipt-key report. The secret is not written.
pub fn write_receipt_key_report(
    base: &Path,
    measurement: &MicroReceiptKey,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_receipt_key(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "receipt key",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ReceiptKeyObservation,
) -> Result<MicroReceiptKey, InferFailure> {
    accept_micro_plan(plan, "receipt key")?;
    accept_cold_single(
        "receipt key",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = ReceiptKeyReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        receipt_root: observed.receipt_root.clone(),
        custody: observed.custody.clone(),
        key_material_serialized: observed.key_material_serialized,
        key_id: observed.key_id.clone(),
        signature_status: observed.signature_status.clone(),
        signature_bytes: observed.signature_bytes,
        rotation: observed.rotation.clone(),
        previous_key_id: observed.previous_key_id.clone(),
        trusted_metadata: observed.trusted_metadata,
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
    Ok(MicroReceiptKey {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
