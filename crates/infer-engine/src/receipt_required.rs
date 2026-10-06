//! Receipt the verifier could not use for one cold micro fixture.
//!
//! The report stores the refusal. This measurement does not read a receipt
//! file and does not open a journal. The layout is specified in
//! `spec/KIP-INFER-0103-receipt-required.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, ReceiptRequiredReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied missing receipt for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptRequiredObservation {
    pub engine_build_root: DigestHex,
    pub receipt_root: DigestHex,
    pub reason: String,
    pub code: String,
    pub retryable: bool,
    pub http_status: u32,
    pub receipt_read: bool,
    pub request_bound: bool,
    pub journal_opened: bool,
    pub event_count: u32,
    pub listener_up: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the receipt-required report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroReceiptRequired {
    pub plan: PlacementPlanV1,
    pub observed: ReceiptRequiredObservation,
    pub report: ReceiptRequiredReportV1,
}

/// Record the missing receipt. The file is not read.
pub fn measure_receipt_required(
    plan: &PlacementPlanV1,
    observed: &ReceiptRequiredObservation,
) -> Result<MicroReceiptRequired, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_receipt_required(&produced)?;
    Ok(produced)
}

/// Recompute the receipt-required report from the stored plan and observation.
pub fn verify_receipt_required(measurement: &MicroReceiptRequired) -> Result<(), InferFailure> {
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
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_u32(
        "http status does not match",
        report.http_status,
        observed.http_status,
    )?;
    same_bool(
        "receipt read does not match",
        report.receipt_read,
        observed.receipt_read,
    )?;
    same_bool(
        "request bound does not match",
        report.request_bound,
        observed.request_bound,
    )?;
    same_bool(
        "journal opened does not match",
        report.journal_opened,
        observed.journal_opened,
    )?;
    same_u32(
        "event count does not match",
        report.event_count,
        observed.event_count,
    )?;
    same_bool(
        "listener up does not match",
        report.listener_up,
        observed.listener_up,
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
        "receipt-required concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "receipt-required run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "receipt-required warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "receipt-required request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "receipt-required validation did not match",
        ));
    }
    Ok(())
}

/// Write the receipt-required report. The file is not read.
pub fn write_receipt_required_report(
    base: &Path,
    measurement: &MicroReceiptRequired,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_receipt_required(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "receipt-required",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ReceiptRequiredObservation,
) -> Result<MicroReceiptRequired, InferFailure> {
    accept_micro_plan(plan, "receipt-required")?;
    accept_cold_single(
        "receipt-required",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = ReceiptRequiredReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        receipt_root: observed.receipt_root.clone(),
        reason: observed.reason.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        http_status: observed.http_status,
        receipt_read: observed.receipt_read,
        request_bound: observed.request_bound,
        journal_opened: observed.journal_opened,
        event_count: observed.event_count,
        listener_up: observed.listener_up,
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
    Ok(MicroReceiptRequired {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
