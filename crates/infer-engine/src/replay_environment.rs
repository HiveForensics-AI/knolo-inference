//! Replay environment mismatch for one cold micro fixture.
//!
//! The report stores which receipt field differed. This measurement does not
//! open a receipt and does not run the forward. The layout is specified in
//! `spec/KIP-INFER-0082-replay-environment.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, ReplayEnvironmentReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied environment mismatch for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayEnvironmentObservation {
    pub engine_build_root: DigestHex,
    pub receipt_root: DigestHex,
    pub mismatched_field: String,
    pub code: String,
    pub retryable: bool,
    pub check_stored: bool,
    pub forward_ran: bool,
    pub output_compared: bool,
    pub assurance: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the replay-environment report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroReplayEnvironment {
    pub plan: PlacementPlanV1,
    pub observed: ReplayEnvironmentObservation,
    pub report: ReplayEnvironmentReportV1,
}

/// Record the environment mismatch. The forward is not run.
pub fn measure_replay_environment(
    plan: &PlacementPlanV1,
    observed: &ReplayEnvironmentObservation,
) -> Result<MicroReplayEnvironment, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_replay_environment(&produced)?;
    Ok(produced)
}

/// Recompute the replay-environment report from the stored plan and observation.
pub fn verify_replay_environment(measurement: &MicroReplayEnvironment) -> Result<(), InferFailure> {
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
        "receipt root does not match",
        &report.receipt_root,
        &observed.receipt_root,
    )?;
    same_text(
        "mismatched field does not match",
        &report.mismatched_field,
        &observed.mismatched_field,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "check stored does not match",
        report.check_stored,
        observed.check_stored,
    )?;
    same_bool(
        "forward ran does not match",
        report.forward_ran,
        observed.forward_ran,
    )?;
    same_bool(
        "output compared does not match",
        report.output_compared,
        observed.output_compared,
    )?;
    same_text(
        "assurance does not match",
        &report.assurance,
        &observed.assurance,
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
        "replay-environment concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "replay-environment run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "replay-environment warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "replay-environment request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "replay-environment validation did not match",
        ));
    }
    Ok(())
}

/// Write the replay-environment report. The receipt is not opened.
pub fn write_replay_environment_report(
    base: &Path,
    measurement: &MicroReplayEnvironment,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_replay_environment(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "replay-environment",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ReplayEnvironmentObservation,
) -> Result<MicroReplayEnvironment, InferFailure> {
    accept_micro_plan(plan, "replay-environment")?;
    accept_cold_single(
        "replay-environment",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = ReplayEnvironmentReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        receipt_root: observed.receipt_root.clone(),
        mismatched_field: observed.mismatched_field.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        check_stored: observed.check_stored,
        forward_ran: observed.forward_ran,
        output_compared: observed.output_compared,
        assurance: observed.assurance.clone(),
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
    Ok(MicroReplayEnvironment {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
