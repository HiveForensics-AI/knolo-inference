//! Hub installation check for one cold micro fixture.
//!
//! Every pinned artifact must be present. This measurement does not download
//! weights and does not open the files. The layout is specified in
//! `spec/KIP-INFER-0050-hub-install.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, InstallReportV1, PlacementPlanV1, LOCAL_WEIGHT_SOURCE,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_root, same_text, same_u32, write_exclusive,
};

/// Presence and roots for the four pinned micro-fixture artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallObservation {
    pub model_image_present: bool,
    pub weights_present: bool,
    pub tokenizer_present: bool,
    pub template_present: bool,
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub tokenizer_root: DigestHex,
    pub template_root: DigestHex,
    pub publisher: String,
    pub license_id: String,
    pub source_provider: String,
    pub artifact_count: u32,
    pub engine_build_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the installation report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroInstall {
    pub plan: PlacementPlanV1,
    pub observed: InstallObservation,
    pub report: InstallReportV1,
}

/// Record that every pinned artifact is present. No file is opened.
pub fn measure_hub_install(
    plan: &PlacementPlanV1,
    observed: &InstallObservation,
) -> Result<MicroInstall, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_hub_install(&produced)?;
    Ok(produced)
}

/// Recompute the installation report from the stored plan and observation.
pub fn verify_hub_install(measurement: &MicroInstall) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "model image root does not match",
        &report.model_image_root,
        &observed.model_image_root,
    )?;
    same_root(
        "artifact root does not match",
        &report.artifact_root,
        &observed.artifact_root,
    )?;
    same_root(
        "tokenizer root does not match",
        &report.tokenizer_root,
        &observed.tokenizer_root,
    )?;
    same_root(
        "template root does not match",
        &report.template_root,
        &observed.template_root,
    )?;
    same_text(
        "publisher does not match",
        &report.publisher,
        &observed.publisher,
    )?;
    same_text(
        "license id does not match",
        &report.license_id,
        &observed.license_id,
    )?;
    same_text(
        "source provider does not match",
        &report.source_provider,
        &observed.source_provider,
    )?;
    same_u32(
        "artifact count does not match",
        report.artifact_count,
        observed.artifact_count,
    )?;
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
        "install concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "install run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "install warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "install request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "install validation did not match",
        ));
    }
    Ok(())
}

/// Write the installation report. Weight bytes are not written.
pub fn write_install_report(
    base: &Path,
    measurement: &MicroInstall,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_hub_install(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "install",
    )
}

fn missing(present: bool, message: &str) -> Result<(), InferFailure> {
    if present {
        Ok(())
    } else {
        Err(fail(ErrorCode::ModelArtifactMissing, message))
    }
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &InstallObservation,
) -> Result<MicroInstall, InferFailure> {
    accept_micro_plan(plan, "install")?;
    accept_cold_single(
        "install",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.source_provider != LOCAL_WEIGHT_SOURCE {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "the hub install does not download weights",
        ));
    }
    missing(
        observed.model_image_present,
        "the hub install is missing the model image",
    )?;
    missing(
        observed.weights_present,
        "the hub install is missing the weight artifact",
    )?;
    missing(
        observed.tokenizer_present,
        "the hub install is missing the tokenizer",
    )?;
    missing(
        observed.template_present,
        "the hub install is missing the template",
    )?;
    let report = InstallReportV1 {
        model_image_root: observed.model_image_root.clone(),
        artifact_root: observed.artifact_root.clone(),
        tokenizer_root: observed.tokenizer_root.clone(),
        template_root: observed.template_root.clone(),
        publisher: observed.publisher.clone(),
        license_id: observed.license_id.clone(),
        source_provider: observed.source_provider.clone(),
        artifact_count: observed.artifact_count,
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        validation_result: "verified".into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroInstall {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
