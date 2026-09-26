//! Disk-full record for one cold micro fixture.
//!
//! The report stores the store and the missing byte count. This measurement
//! does not fill a disk and does not write the target file. The layout is
//! specified in `spec/KIP-INFER-0069-disk-full.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, DiskReportV1, ErrorCode, InferFailure, PlacementPlanV1, MAX_DISK_NEEDED_BYTES,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32, same_u64,
    write_exclusive,
};

/// Caller-supplied disk-full observation for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskObservation {
    pub engine_build_root: DigestHex,
    pub request_id: String,
    pub store: String,
    pub free_bytes: u64,
    pub needed_bytes: u64,
    pub file_written: bool,
    pub space_reclaimed: bool,
    pub listener_up: bool,
    pub code: String,
    pub retryable: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the disk-full report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroDisk {
    pub plan: PlacementPlanV1,
    pub observed: DiskObservation,
    pub report: DiskReportV1,
}

/// Record the disk-full failure. The target file is not written.
pub fn measure_disk(
    plan: &PlacementPlanV1,
    observed: &DiskObservation,
) -> Result<MicroDisk, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_disk(&produced)?;
    Ok(produced)
}

/// Recompute the disk-full report from the stored plan and observation.
pub fn verify_disk(measurement: &MicroDisk) -> Result<(), InferFailure> {
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
        "request id does not match",
        &report.request_id,
        &observed.request_id,
    )?;
    same_text("store does not match", &report.store, &observed.store)?;
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
    same_bool(
        "file written does not match",
        report.file_written,
        observed.file_written,
    )?;
    same_bool(
        "space reclaimed does not match",
        report.space_reclaimed,
        observed.space_reclaimed,
    )?;
    same_bool(
        "listener up does not match",
        report.listener_up,
        observed.listener_up,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
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
        "disk concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "disk run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "disk warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "disk request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "disk validation did not match",
        ));
    }
    Ok(())
}

/// Write the disk-full report. The target file is not written.
pub fn write_disk_report(
    base: &Path,
    measurement: &MicroDisk,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_disk(measurement)?;
    write_exclusive(base, receipt_path, &measurement.report.to_bytes()?, "disk")
}

fn assemble(plan: &PlacementPlanV1, observed: &DiskObservation) -> Result<MicroDisk, InferFailure> {
    accept_micro_plan(plan, "disk")?;
    accept_cold_single(
        "disk",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.needed_bytes > MAX_DISK_NEEDED_BYTES {
        return Err(fail(
            ErrorCode::InsufficientMemory,
            "needed bytes exceed 1 MiB",
        ));
    }
    let report = DiskReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        request_id: observed.request_id.clone(),
        store: observed.store.clone(),
        free_bytes: observed.free_bytes,
        needed_bytes: observed.needed_bytes,
        file_written: observed.file_written,
        space_reclaimed: observed.space_reclaimed,
        listener_up: observed.listener_up,
        code: observed.code.clone(),
        retryable: observed.retryable,
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
    Ok(MicroDisk {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
