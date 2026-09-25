//! Unsupported kernel for one cold micro fixture.
//!
//! The report stores that the backend is not selected. This measurement
//! does not open a device. The layout is specified in
//! `spec/KIP-INFER-0093-unsupported-kernel.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, KernelReportV1, PlacementPlanV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied unsupported kernel for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub code: String,
    pub retryable: bool,
    pub cuda_requested: bool,
    pub kernel_selected: bool,
    pub device_opened: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the kernel report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroKernel {
    pub plan: PlacementPlanV1,
    pub observed: KernelObservation,
    pub report: KernelReportV1,
}

/// Record the unsupported kernel. The device is not opened.
pub fn measure_kernel(
    plan: &PlacementPlanV1,
    observed: &KernelObservation,
) -> Result<MicroKernel, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_kernel(&produced)?;
    Ok(produced)
}

/// Recompute the kernel report from the stored plan and observation.
pub fn verify_kernel(measurement: &MicroKernel) -> Result<(), InferFailure> {
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
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "cuda requested does not match",
        report.cuda_requested,
        observed.cuda_requested,
    )?;
    same_bool(
        "kernel selected does not match",
        report.kernel_selected,
        observed.kernel_selected,
    )?;
    same_bool(
        "device opened does not match",
        report.device_opened,
        observed.device_opened,
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
        "kernel concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "kernel run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "kernel warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "kernel request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "kernel validation did not match",
        ));
    }
    Ok(())
}

/// Write the kernel report. The device is not opened.
pub fn write_kernel_report(
    base: &Path,
    measurement: &MicroKernel,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_kernel(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "kernel",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &KernelObservation,
) -> Result<MicroKernel, InferFailure> {
    accept_micro_plan(plan, "kernel")?;
    accept_cold_single(
        "kernel",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = KernelReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        cuda_requested: observed.cuda_requested,
        kernel_selected: observed.kernel_selected,
        device_opened: observed.device_opened,
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
    Ok(MicroKernel {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
