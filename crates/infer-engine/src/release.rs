//! Signed release manifest for one cold micro fixture.
//!
//! The report stores roots and a signature status. This measurement does not
//! write a manifest and does not verify an Ed25519 key. The layout is
//! specified in `spec/KIP-INFER-0052-release-manifest.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, ReleaseReportV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_root, same_text, same_u32, write_exclusive,
};

/// Caller-supplied manifest for the engine build that would run the fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseObservation {
    pub engine_build_root: DigestHex,
    pub notice_root: DigestHex,
    pub sbom_root: DigestHex,
    pub binary_set_root: DigestHex,
    pub signature_status: String,
    pub signature_count: u32,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the release report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroRelease {
    pub plan: PlacementPlanV1,
    pub observed: ReleaseObservation,
    pub report: ReleaseReportV1,
}

/// Record the release manifest. No manifest file is written.
pub fn measure_release_manifest(
    plan: &PlacementPlanV1,
    observed: &ReleaseObservation,
) -> Result<MicroRelease, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_release_manifest(&produced)?;
    Ok(produced)
}

/// Recompute the release report from the stored plan and observation.
pub fn verify_release_manifest(measurement: &MicroRelease) -> Result<(), InferFailure> {
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
        "notice root does not match",
        &report.notice_root,
        &observed.notice_root,
    )?;
    same_root(
        "sbom root does not match",
        &report.sbom_root,
        &observed.sbom_root,
    )?;
    same_root(
        "binary set root does not match",
        &report.binary_set_root,
        &observed.binary_set_root,
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
        "release concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "release run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "release warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "release request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "release validation did not match",
        ));
    }
    Ok(())
}

/// Write the release report. The manifest file is not written.
pub fn write_release_report(
    base: &Path,
    measurement: &MicroRelease,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_release_manifest(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "release",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ReleaseObservation,
) -> Result<MicroRelease, InferFailure> {
    accept_micro_plan(plan, "release")?;
    accept_cold_single(
        "release",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = ReleaseReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        notice_root: observed.notice_root.clone(),
        sbom_root: observed.sbom_root.clone(),
        binary_set_root: observed.binary_set_root.clone(),
        signature_status: observed.signature_status.clone(),
        signature_count: observed.signature_count,
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
    Ok(MicroRelease {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
