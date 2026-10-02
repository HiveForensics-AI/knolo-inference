//! Host-supplied Ed25519 scalar reduction for one cold micro fixture.
//!
//! The report stores the status and the byte counts. This measurement does
//! not read key bytes and does not multiply the public key. The layout is
//! specified in `spec/KIP-INFER-0086-scalar-reduction.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, ScalarReportV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied scalar reduction for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarObservation {
    pub engine_build_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub scalar_status: String,
    pub scalar_reduced: bool,
    pub public_multiplied: bool,
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

/// One micro-fixture observation and the scalar-reduction report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroScalar {
    pub plan: PlacementPlanV1,
    pub observed: ScalarObservation,
    pub report: ScalarReportV1,
}

/// Record the scalar reduction. The scalar is not reduced.
pub fn measure_scalar(
    plan: &PlacementPlanV1,
    observed: &ScalarObservation,
) -> Result<MicroScalar, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_scalar(&produced)?;
    Ok(produced)
}

/// Recompute the scalar-reduction report from the stored plan and observation.
pub fn verify_scalar(measurement: &MicroScalar) -> Result<(), InferFailure> {
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
        "scalar status does not match",
        &report.scalar_status,
        &observed.scalar_status,
    )?;
    same_bool(
        "scalar reduced does not match",
        report.scalar_reduced,
        observed.scalar_reduced,
    )?;
    same_bool(
        "public multiplied does not match",
        report.public_multiplied,
        observed.public_multiplied,
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
        "scalar concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "scalar run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "scalar warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "scalar request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "scalar validation did not match",
        ));
    }
    Ok(())
}

/// Write the scalar-reduction report. The scalar is not reduced.
pub fn write_scalar_report(
    base: &Path,
    measurement: &MicroScalar,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_scalar(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "scalar",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ScalarObservation,
) -> Result<MicroScalar, InferFailure> {
    accept_micro_plan(plan, "scalar")?;
    accept_cold_single(
        "scalar",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let validation_result = if observed.scalar_status == "reduced" {
        "verified"
    } else {
        "recorded"
    };
    let report = ScalarReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        release_root: observed.release_root.clone(),
        message_root: observed.message_root.clone(),
        scalar_status: observed.scalar_status.clone(),
        scalar_reduced: observed.scalar_reduced,
        public_multiplied: observed.public_multiplied,
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
    Ok(MicroScalar {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
