//! Missing pinned artifact for one cold micro fixture.
//!
//! The report stores the refusal. This measurement does not open the
//! lockfile, the image, or the weights, and it does not download. The
//! layout is specified in `spec/KIP-INFER-0102-artifact-missing.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, ArtifactMissingReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied missing artifact for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactMissingObservation {
    pub engine_build_root: DigestHex,
    pub image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub reason: String,
    pub code: String,
    pub retryable: bool,
    pub source_provider: String,
    pub lock_present: bool,
    pub alias_pinned: bool,
    pub image_opened: bool,
    pub weights_opened: bool,
    pub downloaded: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the artifact-missing report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroArtifactMissing {
    pub plan: PlacementPlanV1,
    pub observed: ArtifactMissingObservation,
    pub report: ArtifactMissingReportV1,
}

/// Record the missing artifact. The file is not opened.
pub fn measure_artifact_missing(
    plan: &PlacementPlanV1,
    observed: &ArtifactMissingObservation,
) -> Result<MicroArtifactMissing, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_artifact_missing(&produced)?;
    Ok(produced)
}

/// Recompute the artifact-missing report from the stored plan and observation.
pub fn verify_artifact_missing(measurement: &MicroArtifactMissing) -> Result<(), InferFailure> {
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
        "image root does not match",
        &report.image_root,
        &observed.image_root,
    )?;
    same_root(
        "artifact root does not match",
        &report.artifact_root,
        &observed.artifact_root,
    )?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_text(
        "source provider does not match",
        &report.source_provider,
        &observed.source_provider,
    )?;
    same_bool(
        "lock present does not match",
        report.lock_present,
        observed.lock_present,
    )?;
    same_bool(
        "alias pinned does not match",
        report.alias_pinned,
        observed.alias_pinned,
    )?;
    same_bool(
        "image opened does not match",
        report.image_opened,
        observed.image_opened,
    )?;
    same_bool(
        "weights opened does not match",
        report.weights_opened,
        observed.weights_opened,
    )?;
    same_bool(
        "downloaded does not match",
        report.downloaded,
        observed.downloaded,
    )?;
    same_bool(
        "forward ran does not match",
        report.forward_ran,
        observed.forward_ran,
    )?;
    same_bool(
        "receipt stored does not match",
        report.receipt_stored,
        observed.receipt_stored,
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
        "artifact-missing concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "artifact-missing run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "artifact-missing warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "artifact-missing request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "artifact-missing validation did not match",
        ));
    }
    Ok(())
}

/// Write the artifact-missing report. The file is not opened.
pub fn write_artifact_missing_report(
    base: &Path,
    measurement: &MicroArtifactMissing,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_artifact_missing(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "artifact-missing",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ArtifactMissingObservation,
) -> Result<MicroArtifactMissing, InferFailure> {
    accept_micro_plan(plan, "artifact-missing")?;
    accept_cold_single(
        "artifact-missing",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = ArtifactMissingReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        image_root: observed.image_root.clone(),
        artifact_root: observed.artifact_root.clone(),
        reason: observed.reason.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        source_provider: observed.source_provider.clone(),
        lock_present: observed.lock_present,
        alias_pinned: observed.alias_pinned,
        image_opened: observed.image_opened,
        weights_opened: observed.weights_opened,
        downloaded: observed.downloaded,
        forward_ran: observed.forward_ran,
        receipt_stored: observed.receipt_stored,
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
    Ok(MicroArtifactMissing {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
