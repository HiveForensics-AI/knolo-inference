//! Unsupported quantization for one cold micro fixture.
//!
//! The report stores the reason the weight was refused. This measurement
//! does not open a weight file and does not dequant a payload. The layout is
//! specified in `spec/KIP-INFER-0092-unsupported-quantization.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, QuantizationReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied unsupported quantization for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizationObservation {
    pub engine_build_root: DigestHex,
    pub artifact_root: DigestHex,
    pub reason: String,
    pub code: String,
    pub retryable: bool,
    pub weights_opened: bool,
    pub payload_read: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the quantization report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroQuantization {
    pub plan: PlacementPlanV1,
    pub observed: QuantizationObservation,
    pub report: QuantizationReportV1,
}

/// Record the unsupported quantization. The weights are not opened.
pub fn measure_quantization(
    plan: &PlacementPlanV1,
    observed: &QuantizationObservation,
) -> Result<MicroQuantization, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_quantization(&produced)?;
    Ok(produced)
}

/// Recompute the quantization report from the stored plan and observation.
pub fn verify_quantization(measurement: &MicroQuantization) -> Result<(), InferFailure> {
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
    same_bool(
        "weights opened does not match",
        report.weights_opened,
        observed.weights_opened,
    )?;
    same_bool(
        "payload read does not match",
        report.payload_read,
        observed.payload_read,
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
        "quantization concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "quantization run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "quantization warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "quantization request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "quantization validation did not match",
        ));
    }
    Ok(())
}

/// Write the quantization report. The weights are not opened.
pub fn write_quantization_report(
    base: &Path,
    measurement: &MicroQuantization,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_quantization(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "quantization",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &QuantizationObservation,
) -> Result<MicroQuantization, InferFailure> {
    accept_micro_plan(plan, "quantization")?;
    accept_cold_single(
        "quantization",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = QuantizationReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        artifact_root: observed.artifact_root.clone(),
        reason: observed.reason.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        weights_opened: observed.weights_opened,
        payload_read: observed.payload_read,
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
    Ok(MicroQuantization {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
