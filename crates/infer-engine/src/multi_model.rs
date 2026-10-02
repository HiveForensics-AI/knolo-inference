//! One second model the worker did not load for a cold micro fixture.
//!
//! The report stores the reason and the device count. This measurement does
//! not load a model. The layout is specified in `spec/KIP-INFER-0114-multi-model.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, MultiModelReportV1, PlacementPlanV1, MAX_DEVICE_COUNT,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied multi-model record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiModelObservation {
    pub engine_build_root: DigestHex,
    pub resident_root: DigestHex,
    pub incoming_root: DigestHex,
    pub reason: String,
    pub device_count: u32,
    pub models_loaded: u32,
    pub code: String,
    pub retryable: bool,
    pub second_loaded: bool,
    pub resident_replaced: bool,
    pub tensor_parallel: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the multi-model report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroMultiModel {
    pub plan: PlacementPlanV1,
    pub observed: MultiModelObservation,
    pub report: MultiModelReportV1,
}

/// Record the multi-model refusal. A second model is not loaded.
pub fn measure_multi_model(
    plan: &PlacementPlanV1,
    observed: &MultiModelObservation,
) -> Result<MicroMultiModel, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_multi_model(&produced)?;
    Ok(produced)
}

/// Recompute the multi-model report from the stored plan and observation.
pub fn verify_multi_model(measurement: &MicroMultiModel) -> Result<(), InferFailure> {
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
        "resident root does not match",
        &report.resident_root,
        &observed.resident_root,
    )?;
    same_root(
        "incoming root does not match",
        &report.incoming_root,
        &observed.incoming_root,
    )?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_u32(
        "device count does not match",
        report.device_count,
        observed.device_count,
    )?;
    same_u32(
        "models loaded do not match",
        report.models_loaded,
        observed.models_loaded,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "second loaded does not match",
        report.second_loaded,
        observed.second_loaded,
    )?;
    same_bool(
        "resident replaced does not match",
        report.resident_replaced,
        observed.resident_replaced,
    )?;
    same_bool(
        "tensor parallel does not match",
        report.tensor_parallel,
        observed.tensor_parallel,
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
        "multi-model concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "multi-model run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "multi-model warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "multi-model request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "multi-model validation did not match",
        ));
    }
    Ok(())
}

/// Write the multi-model report. A second model is not loaded.
pub fn write_multi_model_report(
    base: &Path,
    measurement: &MicroMultiModel,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_multi_model(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "multi-model",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &MultiModelObservation,
) -> Result<MicroMultiModel, InferFailure> {
    accept_micro_plan(plan, "multi-model")?;
    accept_cold_single(
        "multi-model",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.device_count > MAX_DEVICE_COUNT {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "device count exceeds the record cap",
        ));
    }
    let report = MultiModelReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        resident_root: observed.resident_root.clone(),
        incoming_root: observed.incoming_root.clone(),
        reason: observed.reason.clone(),
        device_count: observed.device_count,
        models_loaded: observed.models_loaded,
        code: observed.code.clone(),
        retryable: observed.retryable,
        second_loaded: observed.second_loaded,
        resident_replaced: observed.resident_replaced,
        tensor_parallel: observed.tensor_parallel,
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
    Ok(MicroMultiModel {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
