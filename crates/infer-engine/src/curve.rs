//! Host-supplied Ed25519 curve result for one cold micro fixture.
//!
//! The report stores the status and the byte counts. This measurement does
//! not read key bytes and does not multiply the base point. The layout is
//! specified in `spec/KIP-INFER-0064-curve.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, CurveReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied curve result for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurveObservation {
    pub engine_build_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub curve_status: String,
    pub curve_computed: bool,
    pub public_key_bytes: u32,
    pub signature_bytes: u32,
    pub point_checked: bool,
    pub base_multiplied: bool,
    pub key_material_present: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the curve report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroCurve {
    pub plan: PlacementPlanV1,
    pub observed: CurveObservation,
    pub report: CurveReportV1,
}

/// Record the curve result. The base point is not multiplied.
pub fn measure_curve(
    plan: &PlacementPlanV1,
    observed: &CurveObservation,
) -> Result<MicroCurve, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_curve(&produced)?;
    Ok(produced)
}

/// Recompute the curve report from the stored plan and observation.
pub fn verify_curve(measurement: &MicroCurve) -> Result<(), InferFailure> {
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
        "curve status does not match",
        &report.curve_status,
        &observed.curve_status,
    )?;
    same_bool(
        "curve computed does not match",
        report.curve_computed,
        observed.curve_computed,
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
        "point checked does not match",
        report.point_checked,
        observed.point_checked,
    )?;
    same_bool(
        "base multiplied does not match",
        report.base_multiplied,
        observed.base_multiplied,
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
        "curve concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "curve run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "curve warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "curve request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "curve validation did not match",
        ));
    }
    Ok(())
}

/// Write the curve report. The base point is not multiplied.
pub fn write_curve_report(
    base: &Path,
    measurement: &MicroCurve,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_curve(measurement)?;
    write_exclusive(base, receipt_path, &measurement.report.to_bytes()?, "curve")
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &CurveObservation,
) -> Result<MicroCurve, InferFailure> {
    accept_micro_plan(plan, "curve")?;
    accept_cold_single(
        "curve",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let validation_result = if observed.curve_status == "on-curve" {
        "verified"
    } else {
        "recorded"
    };
    let report = CurveReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        release_root: observed.release_root.clone(),
        message_root: observed.message_root.clone(),
        curve_status: observed.curve_status.clone(),
        curve_computed: observed.curve_computed,
        public_key_bytes: observed.public_key_bytes,
        signature_bytes: observed.signature_bytes,
        point_checked: observed.point_checked,
        base_multiplied: observed.base_multiplied,
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
    Ok(MicroCurve {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
