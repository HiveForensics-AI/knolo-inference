//! Throughput mode the reference refused for one cold micro fixture.
//!
//! The report stores the refusal. This measurement does not select the
//! backend and does not run the forward. The layout is specified in
//! `spec/KIP-INFER-0104-backend-not-allowed.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, BackendReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied backend refusal for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendObservation {
    pub engine_build_root: DigestHex,
    pub surface: String,
    pub requested_mode: String,
    pub code: String,
    pub retryable: bool,
    pub backend_selected: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the backend report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroBackend {
    pub plan: PlacementPlanV1,
    pub observed: BackendObservation,
    pub report: BackendReportV1,
}

/// Record the refused backend. The backend is not selected.
pub fn measure_backend(
    plan: &PlacementPlanV1,
    observed: &BackendObservation,
) -> Result<MicroBackend, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_backend(&produced)?;
    Ok(produced)
}

/// Recompute the backend report from the stored plan and observation.
pub fn verify_backend(measurement: &MicroBackend) -> Result<(), InferFailure> {
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
    same_text("surface does not match", &report.surface, &observed.surface)?;
    same_text(
        "requested mode does not match",
        &report.requested_mode,
        &observed.requested_mode,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "backend selected does not match",
        report.backend_selected,
        observed.backend_selected,
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
        "backend concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "backend run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "backend warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "backend request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "backend validation did not match",
        ));
    }
    Ok(())
}

/// Write the backend report. The backend is not selected.
pub fn write_backend_report(
    base: &Path,
    measurement: &MicroBackend,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_backend(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "backend",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &BackendObservation,
) -> Result<MicroBackend, InferFailure> {
    accept_micro_plan(plan, "backend")?;
    accept_cold_single(
        "backend",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = BackendReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        surface: observed.surface.clone(),
        requested_mode: observed.requested_mode.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        backend_selected: observed.backend_selected,
        forward_ran: observed.forward_ran,
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
    Ok(MicroBackend {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
