//! Duplicate request-id record for one cold micro fixture.
//!
//! The report stores the occupant root and the refusal. This measurement
//! does not admit the second copy. The layout is specified in
//! `spec/KIP-INFER-0073-duplicate-request.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, DuplicateReportV1, ErrorCode, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied duplicate request for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplicateObservation {
    pub engine_build_root: DigestHex,
    pub request_id: String,
    pub occupant_root: DigestHex,
    pub code: String,
    pub retryable: bool,
    pub duplicate_started: bool,
    pub occupant_kept: bool,
    pub second_journal: bool,
    pub listener_up: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the duplicate-request report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroDuplicate {
    pub plan: PlacementPlanV1,
    pub observed: DuplicateObservation,
    pub report: DuplicateReportV1,
}

/// Record the duplicate request. The second copy is not admitted.
pub fn measure_duplicate(
    plan: &PlacementPlanV1,
    observed: &DuplicateObservation,
) -> Result<MicroDuplicate, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_duplicate(&produced)?;
    Ok(produced)
}

/// Recompute the duplicate-request report from the stored plan and observation.
pub fn verify_duplicate(measurement: &MicroDuplicate) -> Result<(), InferFailure> {
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
    same_root(
        "occupant root does not match",
        &report.occupant_root,
        &observed.occupant_root,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "duplicate started does not match",
        report.duplicate_started,
        observed.duplicate_started,
    )?;
    same_bool(
        "occupant kept does not match",
        report.occupant_kept,
        observed.occupant_kept,
    )?;
    same_bool(
        "second journal does not match",
        report.second_journal,
        observed.second_journal,
    )?;
    same_bool(
        "listener up does not match",
        report.listener_up,
        observed.listener_up,
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
        "duplicate concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "duplicate run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "duplicate warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "duplicate request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "duplicate validation did not match",
        ));
    }
    Ok(())
}

/// Write the duplicate-request report. The second copy is not admitted.
pub fn write_duplicate_report(
    base: &Path,
    measurement: &MicroDuplicate,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_duplicate(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "duplicate",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &DuplicateObservation,
) -> Result<MicroDuplicate, InferFailure> {
    accept_micro_plan(plan, "duplicate")?;
    accept_cold_single(
        "duplicate",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = DuplicateReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        request_id: observed.request_id.clone(),
        occupant_root: observed.occupant_root.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        duplicate_started: observed.duplicate_started,
        occupant_kept: observed.occupant_kept,
        second_journal: observed.second_journal,
        listener_up: observed.listener_up,
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
    Ok(MicroDuplicate {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
