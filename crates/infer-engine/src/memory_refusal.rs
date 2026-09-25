//! Insufficient memory for one cold micro fixture.
//!
//! The report stores the byte counts. This measurement does not allocate a
//! page and does not read a weight file. The layout is specified in
//! `spec/KIP-INFER-0095-insufficient-memory.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, MemoryRefusalReportV1, PlacementPlanV1,
    MAX_PEAK_BYTES,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32, same_u64,
    write_exclusive,
};

/// Caller-supplied insufficient memory for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRefusalObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub free_bytes: u64,
    pub needed_bytes: u64,
    pub code: String,
    pub retryable: bool,
    pub resident_full: bool,
    pub queue_held: bool,
    pub allocated: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub listener_up: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the memory-refusal report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroMemoryRefusal {
    pub plan: PlacementPlanV1,
    pub observed: MemoryRefusalObservation,
    pub report: MemoryRefusalReportV1,
}

/// Record the insufficient memory. No page is allocated.
pub fn measure_memory_refusal(
    plan: &PlacementPlanV1,
    observed: &MemoryRefusalObservation,
) -> Result<MicroMemoryRefusal, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_memory_refusal(&produced)?;
    Ok(produced)
}

/// Recompute the memory-refusal report from the stored plan and observation.
pub fn verify_memory_refusal(measurement: &MicroMemoryRefusal) -> Result<(), InferFailure> {
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
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "resident full does not match",
        report.resident_full,
        observed.resident_full,
    )?;
    same_bool(
        "queue held does not match",
        report.queue_held,
        observed.queue_held,
    )?;
    same_bool(
        "allocated does not match",
        report.allocated,
        observed.allocated,
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
    same_bool(
        "listener up does not match",
        report.listener_up,
        observed.listener_up,
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
        "memory-refusal concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "memory-refusal run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "memory-refusal warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "memory-refusal request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "memory-refusal validation did not match",
        ));
    }
    Ok(())
}

/// Write the memory-refusal report. No page is allocated.
pub fn write_memory_refusal_report(
    base: &Path,
    measurement: &MicroMemoryRefusal,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_memory_refusal(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "memory-refusal",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &MemoryRefusalObservation,
) -> Result<MicroMemoryRefusal, InferFailure> {
    accept_micro_plan(plan, "memory-refusal")?;
    accept_cold_single(
        "memory-refusal",
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
    let report = MemoryRefusalReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        free_bytes: observed.free_bytes,
        needed_bytes: observed.needed_bytes,
        code: observed.code.clone(),
        retryable: observed.retryable,
        resident_full: observed.resident_full,
        queue_held: observed.queue_held,
        allocated: observed.allocated,
        forward_ran: observed.forward_ran,
        receipt_stored: observed.receipt_stored,
        listener_up: observed.listener_up,
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
    Ok(MicroMemoryRefusal {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
