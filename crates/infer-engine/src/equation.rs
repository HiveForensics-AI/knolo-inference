//! Host-supplied Ed25519 equation status for one cold micro fixture.
//!
//! The report stores the status and the byte counts. This measurement does
//! not read key bytes and does not compute the curve. The layout is
//! specified in `spec/KIP-INFER-0061-signature-equation.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, EquationReportV1, ErrorCode, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied equation status for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EquationObservation {
    pub engine_build_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub equation_status: String,
    pub equation_evaluated: bool,
    pub public_key_bytes: u32,
    pub signature_bytes: u32,
    pub key_material_present: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the equation report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroEquation {
    pub plan: PlacementPlanV1,
    pub observed: EquationObservation,
    pub report: EquationReportV1,
}

/// Record the equation status. The curve is not computed.
pub fn measure_signature_equation(
    plan: &PlacementPlanV1,
    observed: &EquationObservation,
) -> Result<MicroEquation, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_signature_equation(&produced)?;
    Ok(produced)
}

/// Recompute the equation report from the stored plan and observation.
pub fn verify_signature_equation(measurement: &MicroEquation) -> Result<(), InferFailure> {
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
        "equation status does not match",
        &report.equation_status,
        &observed.equation_status,
    )?;
    same_bool(
        "equation evaluated does not match",
        report.equation_evaluated,
        observed.equation_evaluated,
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
        "equation concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "equation run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "equation warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "equation request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "equation validation did not match",
        ));
    }
    Ok(())
}

/// Write the equation report. The curve is not computed.
pub fn write_equation_report(
    base: &Path,
    measurement: &MicroEquation,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_signature_equation(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "equation",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &EquationObservation,
) -> Result<MicroEquation, InferFailure> {
    accept_micro_plan(plan, "equation")?;
    accept_cold_single(
        "equation",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let validation_result = if observed.equation_status == "accepted" {
        "verified"
    } else {
        "recorded"
    };
    let report = EquationReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        release_root: observed.release_root.clone(),
        message_root: observed.message_root.clone(),
        equation_status: observed.equation_status.clone(),
        equation_evaluated: observed.equation_evaluated,
        public_key_bytes: observed.public_key_bytes,
        signature_bytes: observed.signature_bytes,
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
    Ok(MicroEquation {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
