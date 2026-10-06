//! Host-supplied Ed25519 base-point multiplication for one cold micro fixture.
//!
//! The report stores the status and the byte counts. This measurement does
//! not read key bytes and does not add the public-key point. The layout is
//! specified in `spec/KIP-INFER-0068-base-point.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, BaseReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied base-point result for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseObservation {
    pub engine_build_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub base_status: String,
    pub base_multiplied: bool,
    pub public_key_bytes: u32,
    pub signature_bytes: u32,
    pub scalar_bytes: u32,
    pub point_added: bool,
    pub key_material_present: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the base-point report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroBase {
    pub plan: PlacementPlanV1,
    pub observed: BaseObservation,
    pub report: BaseReportV1,
}

/// Record the base-point result. The public-key point is not added.
pub fn measure_base(
    plan: &PlacementPlanV1,
    observed: &BaseObservation,
) -> Result<MicroBase, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_base(&produced)?;
    Ok(produced)
}

/// Recompute the base-point report from the stored plan and observation.
pub fn verify_base(measurement: &MicroBase) -> Result<(), InferFailure> {
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
        "base status does not match",
        &report.base_status,
        &observed.base_status,
    )?;
    same_bool(
        "base multiplied does not match",
        report.base_multiplied,
        observed.base_multiplied,
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
        "point added does not match",
        report.point_added,
        observed.point_added,
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
        "base concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "base run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "base warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "base request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "base validation did not match",
        ));
    }
    Ok(())
}

/// Write the base-point report. The public-key point is not added.
pub fn write_base_report(
    base: &Path,
    measurement: &MicroBase,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_base(measurement)?;
    write_exclusive(base, receipt_path, &measurement.report.to_bytes()?, "base")
}

fn assemble(plan: &PlacementPlanV1, observed: &BaseObservation) -> Result<MicroBase, InferFailure> {
    accept_micro_plan(plan, "base")?;
    accept_cold_single(
        "base",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let validation_result = if observed.base_status == "multiplied" {
        "verified"
    } else {
        "recorded"
    };
    let report = BaseReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        release_root: observed.release_root.clone(),
        message_root: observed.message_root.clone(),
        base_status: observed.base_status.clone(),
        base_multiplied: observed.base_multiplied,
        public_key_bytes: observed.public_key_bytes,
        signature_bytes: observed.signature_bytes,
        scalar_bytes: observed.scalar_bytes,
        point_added: observed.point_added,
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
    Ok(MicroBase {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
