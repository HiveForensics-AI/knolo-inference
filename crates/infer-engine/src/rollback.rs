//! Rollback record for one cold micro fixture.
//!
//! The report stores the previous pin and the incoming pin. This measurement
//! does not rewrite a lockfile. The layout is specified in
//! `spec/KIP-INFER-0065-rollback.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, RollbackReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied rollback for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RollbackObservation {
    pub engine_build_root: DigestHex,
    pub previous_image_root: DigestHex,
    pub incoming_image_root: DigestHex,
    pub previous_artifact_root: DigestHex,
    pub incoming_artifact_root: DigestHex,
    pub reason: String,
    pub lockfile_mutated: bool,
    pub core_lock_touched: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the rollback report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroRollback {
    pub plan: PlacementPlanV1,
    pub observed: RollbackObservation,
    pub report: RollbackReportV1,
}

/// Record the rollback. The lockfile is not rewritten.
pub fn measure_rollback(
    plan: &PlacementPlanV1,
    observed: &RollbackObservation,
) -> Result<MicroRollback, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_rollback(&produced)?;
    Ok(produced)
}

/// Recompute the rollback report from the stored plan and observation.
pub fn verify_rollback(measurement: &MicroRollback) -> Result<(), InferFailure> {
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
        "previous image root does not match",
        &report.previous_image_root,
        &observed.previous_image_root,
    )?;
    same_root(
        "incoming image root does not match",
        &report.incoming_image_root,
        &observed.incoming_image_root,
    )?;
    same_root(
        "previous artifact root does not match",
        &report.previous_artifact_root,
        &observed.previous_artifact_root,
    )?;
    same_root(
        "incoming artifact root does not match",
        &report.incoming_artifact_root,
        &observed.incoming_artifact_root,
    )?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_bool(
        "lockfile mutated does not match",
        report.lockfile_mutated,
        observed.lockfile_mutated,
    )?;
    same_bool(
        "core lock touched does not match",
        report.core_lock_touched,
        observed.core_lock_touched,
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
        "rollback concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "rollback run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "rollback warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "rollback request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "rollback validation did not match",
        ));
    }
    Ok(())
}

/// Write the rollback report. The lockfile is not rewritten.
pub fn write_rollback_report(
    base: &Path,
    measurement: &MicroRollback,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_rollback(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "rollback",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &RollbackObservation,
) -> Result<MicroRollback, InferFailure> {
    accept_micro_plan(plan, "rollback")?;
    accept_cold_single(
        "rollback",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = RollbackReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        previous_image_root: observed.previous_image_root.clone(),
        incoming_image_root: observed.incoming_image_root.clone(),
        previous_artifact_root: observed.previous_artifact_root.clone(),
        incoming_artifact_root: observed.incoming_artifact_root.clone(),
        reason: observed.reason.clone(),
        lockfile_mutated: observed.lockfile_mutated,
        core_lock_touched: observed.core_lock_touched,
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
    Ok(MicroRollback {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
