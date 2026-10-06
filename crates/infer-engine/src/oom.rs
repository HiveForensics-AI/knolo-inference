//! CUDA out-of-memory record for one cold micro fixture.
//!
//! The report stores the device and the byte counts. This measurement does
//! not allocate device memory and does not fall back to CPU. The layout is
//! specified in `spec/KIP-INFER-0077-cuda-oom.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, OomReportV1, PlacementPlanV1, MAX_PEAK_BYTES,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32, same_u64,
    write_exclusive,
};

/// Caller-supplied CUDA out-of-memory result for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OomObservation {
    pub engine_build_root: DigestHex,
    pub free_bytes: u64,
    pub needed_bytes: u64,
    pub code: String,
    pub retryable: bool,
    pub receipt_stored: bool,
    pub listener_up: bool,
    pub supervisor_exited: bool,
    pub cpu_fallback: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the CUDA out-of-memory report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroOom {
    pub plan: PlacementPlanV1,
    pub observed: OomObservation,
    pub report: OomReportV1,
}

/// Record a CUDA out-of-memory result. Device memory is not allocated.
pub fn measure_cuda_oom(
    plan: &PlacementPlanV1,
    observed: &OomObservation,
) -> Result<MicroOom, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_cuda_oom(&produced)?;
    Ok(produced)
}

/// Recompute the CUDA out-of-memory report from the stored plan and observation.
pub fn verify_cuda_oom(measurement: &MicroOom) -> Result<(), InferFailure> {
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
    let device = measurement
        .plan
        .devices
        .first()
        .map(String::as_str)
        .unwrap_or("");
    same_text("device does not match", &report.device, device)?;
    same_u64(
        "free bytes do not match",
        report.free_bytes,
        observed.free_bytes,
    )?;
    same_u64(
        "needed bytes do not match",
        report.needed_bytes,
        observed.needed_bytes,
    )?;
    same_text("oom code does not match", &report.code, &observed.code)?;
    same_bool(
        "oom retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "receipt stored does not match",
        report.receipt_stored,
        observed.receipt_stored,
    )?;
    same_bool(
        "listener up does not match",
        report.listener_up,
        observed.listener_up,
    )?;
    same_bool(
        "supervisor exited does not match",
        report.supervisor_exited,
        observed.supervisor_exited,
    )?;
    same_bool(
        "cpu fallback does not match",
        report.cpu_fallback,
        observed.cpu_fallback,
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
        "oom concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "oom run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "oom warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "oom request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "oom validation did not match",
        ));
    }
    Ok(())
}

/// Write the CUDA out-of-memory report. Device memory is not allocated.
pub fn write_oom_report(
    base: &Path,
    measurement: &MicroOom,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_cuda_oom(measurement)?;
    write_exclusive(base, receipt_path, &measurement.report.to_bytes()?, "oom")
}

fn assemble(plan: &PlacementPlanV1, observed: &OomObservation) -> Result<MicroOom, InferFailure> {
    accept_micro_plan(plan, "oom")?;
    if plan.devices.len() != 1 || plan.devices[0] != "slot-0" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "a cuda oom record names slot-0",
        ));
    }
    accept_cold_single(
        "oom",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.needed_bytes > MAX_PEAK_BYTES {
        return Err(fail(
            ErrorCode::InsufficientMemory,
            "needed bytes exceed 64 MiB",
        ));
    }
    let report = OomReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        device: "slot-0".into(),
        free_bytes: observed.free_bytes,
        needed_bytes: observed.needed_bytes,
        code: observed.code.clone(),
        retryable: observed.retryable,
        receipt_stored: observed.receipt_stored,
        listener_up: observed.listener_up,
        supervisor_exited: observed.supervisor_exited,
        cpu_fallback: observed.cpu_fallback,
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
    Ok(MicroOom {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
