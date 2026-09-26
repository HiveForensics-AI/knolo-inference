//! One secondary-slot service the worker did not start for a cold micro fixture.
//!
//! The report stores the reason and the slot name. This measurement does not
//! open a device and does not run embeddings. The layout is specified in
//! `spec/KIP-INFER-0115-secondary-service.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, SecondaryReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, write_exclusive,
};

/// Caller-supplied secondary-service record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecondaryObservation {
    pub engine_build_root: DigestHex,
    pub primary_root: DigestHex,
    pub secondary_root: DigestHex,
    pub reason: String,
    pub secondary_slot: String,
    pub code: String,
    pub retryable: bool,
    pub model_named: bool,
    pub embeddings_requested: bool,
    pub overflow_requested: bool,
    pub service_started: bool,
    pub embeddings_ran: bool,
    pub overflow_placed: bool,
    pub device_opened: bool,
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

/// One micro-fixture observation and the secondary-service report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroSecondary {
    pub plan: PlacementPlanV1,
    pub observed: SecondaryObservation,
    pub report: SecondaryReportV1,
}

/// Record the secondary-service refusal. The service is not started.
pub fn measure_secondary(
    plan: &PlacementPlanV1,
    observed: &SecondaryObservation,
) -> Result<MicroSecondary, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_secondary(&produced)?;
    Ok(produced)
}

/// Recompute the secondary-service report from the stored plan and observation.
pub fn verify_secondary(measurement: &MicroSecondary) -> Result<(), InferFailure> {
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
        "primary root does not match",
        &report.primary_root,
        &observed.primary_root,
    )?;
    same_root(
        "secondary root does not match",
        &report.secondary_root,
        &observed.secondary_root,
    )?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_text(
        "secondary slot does not match",
        &report.secondary_slot,
        &observed.secondary_slot,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "model named does not match",
        report.model_named,
        observed.model_named,
    )?;
    same_bool(
        "embeddings requested does not match",
        report.embeddings_requested,
        observed.embeddings_requested,
    )?;
    same_bool(
        "overflow requested does not match",
        report.overflow_requested,
        observed.overflow_requested,
    )?;
    same_bool(
        "service started does not match",
        report.service_started,
        observed.service_started,
    )?;
    same_bool(
        "embeddings ran does not match",
        report.embeddings_ran,
        observed.embeddings_ran,
    )?;
    same_bool(
        "overflow placed does not match",
        report.overflow_placed,
        observed.overflow_placed,
    )?;
    same_bool(
        "device opened does not match",
        report.device_opened,
        observed.device_opened,
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
    same_u32_pair(report, observed)?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "secondary validation did not match",
        ));
    }
    Ok(())
}

fn same_u32_pair(
    report: &SecondaryReportV1,
    observed: &SecondaryObservation,
) -> Result<(), InferFailure> {
    use crate::report_io::same_u32;
    same_u32(
        "secondary concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "secondary run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    crate::report_io::same_text(
        "secondary warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "secondary request count does not match",
        report.request_count,
        observed.request_count,
    )
}

/// Write the secondary-service report. The service is not started.
pub fn write_secondary_report(
    base: &Path,
    measurement: &MicroSecondary,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_secondary(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "secondary",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &SecondaryObservation,
) -> Result<MicroSecondary, InferFailure> {
    accept_micro_plan(plan, "secondary")?;
    accept_cold_single(
        "secondary",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = SecondaryReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        primary_root: observed.primary_root.clone(),
        secondary_root: observed.secondary_root.clone(),
        reason: observed.reason.clone(),
        secondary_slot: observed.secondary_slot.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        model_named: observed.model_named,
        embeddings_requested: observed.embeddings_requested,
        overflow_requested: observed.overflow_requested,
        service_started: observed.service_started,
        embeddings_ran: observed.embeddings_ran,
        overflow_placed: observed.overflow_placed,
        device_opened: observed.device_opened,
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
    Ok(MicroSecondary {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
