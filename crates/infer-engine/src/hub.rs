//! Hub metadata for one cold micro fixture.
//!
//! The record names the `.kmodel` root, the weight artifact root, the license
//! id, and the conformance receipt. It does not store weight bytes and it does
//! not download them. The layout is specified in `spec/KIP-INFER-0046-hub-record.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, HubReportV1, InferFailure, ModelConformanceReceiptV1,
    PlacementPlanV1, MICRO_ADAPTER,
};

use crate::report_io::{accept_cold_single, accept_micro_plan, write_exclusive};

/// Publisher metadata from one cold micro-fixture record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubObservation {
    pub publisher: String,
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub adapter_id: String,
    pub quantization: String,
    pub license_id: String,
    pub source_provider: String,
    pub benchmark_root: DigestHex,
    pub distribution: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation, its conformance receipt, and the Hub record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroHub {
    pub plan: PlacementPlanV1,
    pub observed: HubObservation,
    pub conformance: ModelConformanceReceiptV1,
    pub report: HubReportV1,
}

/// Record the Hub metadata. No weight file is opened and no file is created.
pub fn measure_hub_record(
    plan: &PlacementPlanV1,
    observed: &HubObservation,
    conformance: &ModelConformanceReceiptV1,
) -> Result<MicroHub, InferFailure> {
    let produced = assemble(plan, observed, conformance)?;
    verify_hub_record(&produced)?;
    Ok(produced)
}

/// Recompute the Hub record from the stored plan, observation, and receipt.
pub fn verify_hub_record(measurement: &MicroHub) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_text(
        "publisher does not match",
        &report.publisher,
        &observed.publisher,
    )?;
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
        "adapter id does not match",
        &report.adapter_id,
        &observed.adapter_id,
    )?;
    same_text(
        "quantization does not match",
        &report.quantization,
        &observed.quantization,
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
    same_root(
        "conformance root does not match",
        &report.conformance_root,
        &measurement.conformance.root()?,
    )?;
    same_root(
        "benchmark root does not match",
        &report.benchmark_root,
        &observed.benchmark_root,
    )?;
    same_text(
        "support level does not match",
        &report.support_level,
        &measurement.conformance.support_level,
    )?;
    if report.greedy_token_parity != measurement.conformance.greedy_token_parity {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "greedy token parity does not match",
        ));
    }
    same_text(
        "distribution does not match",
        &report.distribution,
        &observed.distribution,
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
        "hub concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "hub run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "hub warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "hub request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(
        &measurement.plan,
        &measurement.observed,
        &measurement.conformance,
    )?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "hub validation did not match",
        ));
    }
    Ok(())
}

/// Write the Hub record. The conformance receipt and the weights are not written.
pub fn write_hub_report(
    base: &Path,
    measurement: &MicroHub,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_hub_record(measurement)?;
    write_exclusive(base, receipt_path, &measurement.report.to_bytes()?, "hub")
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &HubObservation,
    conformance: &ModelConformanceReceiptV1,
) -> Result<MicroHub, InferFailure> {
    accept_micro_plan(plan, "hub")?;
    accept_cold_single(
        "hub",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.adapter_id != MICRO_ADAPTER {
        return Err(fail(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "architecture adapter {} is not compiled in",
                observed.adapter_id
            ),
        ));
    }
    if observed.quantization != "f32" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "the hub record is the micro fixture",
        ));
    }
    if observed.source_provider != "local" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "the hub record does not download weights",
        ));
    }
    conformance.validate()?;
    if conformance.adapter_id != observed.adapter_id {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "adapter id does not match",
        ));
    }
    if conformance.engine_build_root != observed.engine_build_root {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "engine build root does not match",
        ));
    }
    if observed.benchmark_root == conformance.root()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "the hub record repeats a receipt",
        ));
    }
    let mut report = HubReportV1 {
        publisher: observed.publisher.clone(),
        model_image_root: observed.model_image_root.clone(),
        artifact_root: observed.artifact_root.clone(),
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        adapter_id: observed.adapter_id.clone(),
        quantization: observed.quantization.clone(),
        license_id: observed.license_id.clone(),
        source_provider: observed.source_provider.clone(),
        conformance_root: conformance.root()?,
        benchmark_root: observed.benchmark_root.clone(),
        support_level: conformance.support_level.clone(),
        greedy_token_parity: conformance.greedy_token_parity,
        distribution: observed.distribution.clone(),
        native_supported: false,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        validation_result: "recorded".into(),
        extensions: BTreeMap::new(),
    };
    report.native_supported = report.computed_native();
    report.validate()?;
    Ok(MicroHub {
        plan: plan.clone(),
        observed: observed.clone(),
        conformance: conformance.clone(),
        report,
    })
}

fn same_root(message: &str, left: &DigestHex, right: &DigestHex) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn same_text(message: &str, left: &str, right: &str) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn same_u32(message: &str, left: u32, right: u32) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}
