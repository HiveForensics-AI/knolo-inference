//! Replay output mismatch for one cold micro fixture.
//!
//! The report stores that the candidate output root differed. This measurement
//! does not run the forward and does not store token ids. The layout is
//! specified in `spec/KIP-INFER-0083-replay-output.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, ReplayOutputReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied output mismatch for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayOutputObservation {
    pub engine_build_root: DigestHex,
    pub receipt_root: DigestHex,
    pub candidate_output_root: DigestHex,
    pub code: String,
    pub retryable: bool,
    pub check_stored: bool,
    pub forward_ran: bool,
    pub environment_matched: bool,
    pub assurance: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the replay-output report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroReplayOutput {
    pub plan: PlacementPlanV1,
    pub observed: ReplayOutputObservation,
    pub report: ReplayOutputReportV1,
}

/// Record the output mismatch. The forward is not run.
pub fn measure_replay_output(
    plan: &PlacementPlanV1,
    observed: &ReplayOutputObservation,
) -> Result<MicroReplayOutput, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_replay_output(&produced)?;
    Ok(produced)
}

/// Recompute the replay-output report from the stored plan and observation.
pub fn verify_replay_output(measurement: &MicroReplayOutput) -> Result<(), InferFailure> {
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
    same_root(
        "candidate output root does not match",
        &report.candidate_output_root,
        &observed.candidate_output_root,
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
        "environment matched does not match",
        report.environment_matched,
        observed.environment_matched,
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
        "replay-output concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "replay-output run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "replay-output warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "replay-output request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "replay-output validation did not match",
        ));
    }
    Ok(())
}

/// Write the replay-output report. Token ids are not written.
pub fn write_replay_output_report(
    base: &Path,
    measurement: &MicroReplayOutput,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_replay_output(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "replay-output",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ReplayOutputObservation,
) -> Result<MicroReplayOutput, InferFailure> {
    accept_micro_plan(plan, "replay-output")?;
    accept_cold_single(
        "replay-output",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = ReplayOutputReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        receipt_root: observed.receipt_root.clone(),
        candidate_output_root: observed.candidate_output_root.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        check_stored: observed.check_stored,
        forward_ran: observed.forward_ran,
        environment_matched: observed.environment_matched,
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
    Ok(MicroReplayOutput {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
