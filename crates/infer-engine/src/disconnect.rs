//! Client disconnect record for one cold micro fixture.
//!
//! The report stores the stage and the token counts. This measurement does
//! not close a socket. The layout is specified in
//! `spec/KIP-INFER-0066-disconnect.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, DisconnectReportV1, ErrorCode, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied disconnect for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisconnectObservation {
    pub engine_build_root: DigestHex,
    pub request_id: String,
    pub stage: String,
    pub outcome: String,
    pub code: String,
    pub listener_up: bool,
    pub socket_closed: bool,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the disconnect report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroDisconnect {
    pub plan: PlacementPlanV1,
    pub observed: DisconnectObservation,
    pub report: DisconnectReportV1,
}

/// Record the disconnect. The socket is not closed.
pub fn measure_disconnect(
    plan: &PlacementPlanV1,
    observed: &DisconnectObservation,
) -> Result<MicroDisconnect, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_disconnect(&produced)?;
    Ok(produced)
}

/// Recompute the disconnect report from the stored plan and observation.
pub fn verify_disconnect(measurement: &MicroDisconnect) -> Result<(), InferFailure> {
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
    same_text("stage does not match", &report.stage, &observed.stage)?;
    same_text("outcome does not match", &report.outcome, &observed.outcome)?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "listener up does not match",
        report.listener_up,
        observed.listener_up,
    )?;
    same_bool(
        "socket closed does not match",
        report.socket_closed,
        observed.socket_closed,
    )?;
    same_u32(
        "prompt tokens do not match",
        report.prompt_tokens,
        observed.prompt_tokens,
    )?;
    same_u32(
        "output tokens do not match",
        report.output_tokens,
        observed.output_tokens,
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
        "disconnect concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "disconnect run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "disconnect warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "disconnect request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "disconnect validation did not match",
        ));
    }
    Ok(())
}

/// Write the disconnect report. The socket is not closed.
pub fn write_disconnect_report(
    base: &Path,
    measurement: &MicroDisconnect,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_disconnect(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "disconnect",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &DisconnectObservation,
) -> Result<MicroDisconnect, InferFailure> {
    accept_micro_plan(plan, "disconnect")?;
    accept_cold_single(
        "disconnect",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = DisconnectReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        request_id: observed.request_id.clone(),
        stage: observed.stage.clone(),
        outcome: observed.outcome.clone(),
        code: observed.code.clone(),
        listener_up: observed.listener_up,
        socket_closed: observed.socket_closed,
        prompt_tokens: observed.prompt_tokens,
        output_tokens: observed.output_tokens,
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
    Ok(MicroDisconnect {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
