//! Host-supplied Ed25519 point comparison for one cold micro fixture.
//!
//! The report stores the status and the byte counts. This measurement does
//! not read key bytes and does not hash the challenge. The layout is
//! specified in `spec/KIP-INFER-0076-point-comparison.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, EqualityReportV1, ErrorCode, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied point comparison for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EqualityObservation {
    pub engine_build_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub equality_status: String,
    pub points_compared: bool,
    pub points_equal: bool,
    pub challenge_hashed: bool,
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

/// One micro-fixture observation and the point-comparison report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroEquality {
    pub plan: PlacementPlanV1,
    pub observed: EqualityObservation,
    pub report: EqualityReportV1,
}

/// Record the point comparison. The points are not read.
pub fn measure_point_equality(
    plan: &PlacementPlanV1,
    observed: &EqualityObservation,
) -> Result<MicroEquality, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_point_equality(&produced)?;
    Ok(produced)
}

/// Recompute the point-comparison report from the stored plan and observation.
pub fn verify_point_equality(measurement: &MicroEquality) -> Result<(), InferFailure> {
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
        "equality status does not match",
        &report.equality_status,
        &observed.equality_status,
    )?;
    same_bool(
        "points compared does not match",
        report.points_compared,
        observed.points_compared,
    )?;
    same_bool(
        "points equal does not match",
        report.points_equal,
        observed.points_equal,
    )?;
    same_bool(
        "challenge hashed does not match",
        report.challenge_hashed,
        observed.challenge_hashed,
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
        "equality concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "equality run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "equality warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "equality request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "equality validation did not match",
        ));
    }
    Ok(())
}

/// Write the point-comparison report. The points are not read.
pub fn write_equality_report(
    base: &Path,
    measurement: &MicroEquality,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_point_equality(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "equality",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &EqualityObservation,
) -> Result<MicroEquality, InferFailure> {
    accept_micro_plan(plan, "equality")?;
    accept_cold_single(
        "equality",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let validation_result = if observed.equality_status == "equal" {
        "verified"
    } else {
        "recorded"
    };
    let report = EqualityReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        release_root: observed.release_root.clone(),
        message_root: observed.message_root.clone(),
        equality_status: observed.equality_status.clone(),
        points_compared: observed.points_compared,
        points_equal: observed.points_equal,
        challenge_hashed: observed.challenge_hashed,
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
    Ok(MicroEquality {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
