//! CUDA fault record for one cold micro fixture.
//!
//! The report stores the fault class. This measurement does not query a
//! device, capture a graph, or fall back to CPU. The layout is specified in
//! `spec/KIP-INFER-0078-cuda-fault.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, FaultReportV1, InferFailure, PlacementPlanV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied CUDA fault for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultObservation {
    pub engine_build_root: DigestHex,
    pub fault_class: String,
    pub code: String,
    pub retryable: bool,
    pub receipt_stored: bool,
    pub listener_up: bool,
    pub supervisor_exited: bool,
    pub cpu_fallback: bool,
    pub graph_captured: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the CUDA fault report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroFault {
    pub plan: PlacementPlanV1,
    pub observed: FaultObservation,
    pub report: FaultReportV1,
}

/// Record a CUDA fault. The device is not queried.
pub fn measure_cuda_fault(
    plan: &PlacementPlanV1,
    observed: &FaultObservation,
) -> Result<MicroFault, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_cuda_fault(&produced)?;
    Ok(produced)
}

/// Recompute the CUDA fault report from the stored plan and observation.
pub fn verify_cuda_fault(measurement: &MicroFault) -> Result<(), InferFailure> {
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
    same_text(
        "fault class does not match",
        &report.fault_class,
        &observed.fault_class,
    )?;
    same_text("fault code does not match", &report.code, &observed.code)?;
    same_bool(
        "fault retryable does not match",
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
    same_bool(
        "graph captured does not match",
        report.graph_captured,
        observed.graph_captured,
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
        "fault concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "fault run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "fault warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "fault request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "fault validation did not match",
        ));
    }
    Ok(())
}

/// Write the CUDA fault report. The device is not queried.
pub fn write_fault_report(
    base: &Path,
    measurement: &MicroFault,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_cuda_fault(measurement)?;
    write_exclusive(base, receipt_path, &measurement.report.to_bytes()?, "fault")
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &FaultObservation,
) -> Result<MicroFault, InferFailure> {
    accept_micro_plan(plan, "fault")?;
    if plan.devices.len() != 1 || plan.devices[0] != "slot-0" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "a cuda fault record names slot-0",
        ));
    }
    accept_cold_single(
        "fault",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = FaultReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        device: "slot-0".into(),
        fault_class: observed.fault_class.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        receipt_stored: observed.receipt_stored,
        listener_up: observed.listener_up,
        supervisor_exited: observed.supervisor_exited,
        cpu_fallback: observed.cpu_fallback,
        graph_captured: observed.graph_captured,
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
    Ok(MicroFault {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
