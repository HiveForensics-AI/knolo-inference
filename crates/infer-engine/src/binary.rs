//! Binary hash inventory for one cold micro fixture.
//!
//! The report stores the supervisor and worker hashes. This measurement does
//! not open either binary. The layout is specified in
//! `spec/KIP-INFER-0053-binary-inventory.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, BinaryReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, MAX_BINARY_BYTES,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_root, same_text, same_u32, same_u64,
    write_exclusive,
};

/// Caller-supplied hashes for the two release binaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryObservation {
    pub engine_build_root: DigestHex,
    pub supervisor_name: String,
    pub supervisor_hash: DigestHex,
    pub supervisor_bytes: u64,
    pub worker_name: String,
    pub worker_hash: DigestHex,
    pub worker_bytes: u64,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the binary report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroBinary {
    pub plan: PlacementPlanV1,
    pub observed: BinaryObservation,
    pub report: BinaryReportV1,
}

/// Record the binary hashes. Neither file is opened.
pub fn measure_binary_inventory(
    plan: &PlacementPlanV1,
    observed: &BinaryObservation,
) -> Result<MicroBinary, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_binary_inventory(&produced)?;
    Ok(produced)
}

/// Recompute the binary report from the stored plan and observation.
pub fn verify_binary_inventory(measurement: &MicroBinary) -> Result<(), InferFailure> {
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
    same_text(
        "supervisor name does not match",
        &report.supervisor_name,
        &observed.supervisor_name,
    )?;
    same_root(
        "supervisor hash does not match",
        &report.supervisor_hash,
        &observed.supervisor_hash,
    )?;
    same_u64(
        "supervisor bytes do not match",
        report.supervisor_bytes,
        observed.supervisor_bytes,
    )?;
    same_text(
        "worker name does not match",
        &report.worker_name,
        &observed.worker_name,
    )?;
    same_root(
        "worker hash does not match",
        &report.worker_hash,
        &observed.worker_hash,
    )?;
    same_u64(
        "worker bytes do not match",
        report.worker_bytes,
        observed.worker_bytes,
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
        "binary concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "binary run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "binary warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "binary request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "binary validation did not match",
        ));
    }
    Ok(())
}

/// Write the binary report. The binaries are not written.
pub fn write_binary_report(
    base: &Path,
    measurement: &MicroBinary,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_binary_inventory(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "binary",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &BinaryObservation,
) -> Result<MicroBinary, InferFailure> {
    accept_micro_plan(plan, "binary")?;
    accept_cold_single(
        "binary",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.supervisor_bytes > MAX_BINARY_BYTES {
        return Err(fail(
            ErrorCode::InsufficientMemory,
            "the supervisor binary exceeds 64 MiB",
        ));
    }
    if observed.worker_bytes > MAX_BINARY_BYTES {
        return Err(fail(
            ErrorCode::InsufficientMemory,
            "the worker binary exceeds 64 MiB",
        ));
    }
    let report = BinaryReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        supervisor_name: observed.supervisor_name.clone(),
        supervisor_hash: observed.supervisor_hash.clone(),
        supervisor_bytes: observed.supervisor_bytes,
        worker_name: observed.worker_name.clone(),
        worker_hash: observed.worker_hash.clone(),
        worker_bytes: observed.worker_bytes,
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
    Ok(MicroBinary {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
