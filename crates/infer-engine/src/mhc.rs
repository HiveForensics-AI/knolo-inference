//! One mHC connection the engine did not apply for a cold micro fixture.
//!
//! The report stores the reason and the requested stream count. This
//! measurement does not apply a connection.
//!
//! The layout is specified in `spec/KIP-INFER-0129-mhc.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, MhcReportV1, PlacementPlanV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied mhc record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MhcObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub requested_streams: u32,
    pub applied_streams: u32,
    pub code: String,
    pub retryable: bool,
    pub connected: bool,
    pub manifold_applied: bool,
    pub residual_mapped: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the mhc report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroMhc {
    pub plan: PlacementPlanV1,
    pub observed: MhcObservation,
    pub report: MhcReportV1,
}

/// Record the mhc measurement.
pub fn measure_mhc(
    plan: &PlacementPlanV1,
    observed: &MhcObservation,
) -> Result<MicroMhc, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_mhc(&produced)?;
    Ok(produced)
}

/// Recompute the mhc report from the stored plan and observation.
pub fn verify_mhc(measurement: &MicroMhc) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "engine build root does not match",
        &report.engine_build_root,
        &observed.engine_build_root,
    )?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_u32(
        "requested streams do not match",
        report.requested_streams,
        observed.requested_streams,
    )?;
    same_u32(
        "applied streams do not match",
        report.applied_streams,
        observed.applied_streams,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "connected does not match",
        report.connected,
        observed.connected,
    )?;
    same_bool(
        "manifold applied does not match",
        report.manifold_applied,
        observed.manifold_applied,
    )?;
    same_bool(
        "residual mapped does not match",
        report.residual_mapped,
        observed.residual_mapped,
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
        "mhc concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "mhc run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "mhc warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "mhc request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    same_root(
        "placement root does not match",
        &report.placement_root,
        &measurement.plan.root()?,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "mhc validation did not match",
        ));
    }
    Ok(())
}

/// Write the mhc report.
pub fn write_mhc_report(
    base: &Path,
    measurement: &MicroMhc,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_mhc(measurement)?;
    write_exclusive(base, receipt_path, &measurement.report.to_bytes()?, "mhc")
}

fn assemble(plan: &PlacementPlanV1, observed: &MhcObservation) -> Result<MicroMhc, InferFailure> {
    accept_micro_plan(plan, "mhc")?;
    accept_cold_single(
        "mhc",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = MhcReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        requested_streams: observed.requested_streams,
        applied_streams: observed.applied_streams,
        code: observed.code.clone(),
        retryable: observed.retryable,
        connected: observed.connected,
        manifold_applied: observed.manifold_applied,
        residual_mapped: observed.residual_mapped,
        forward_ran: observed.forward_ran,
        receipt_stored: observed.receipt_stored,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        validation_result: ("recorded").into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroMhc {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
