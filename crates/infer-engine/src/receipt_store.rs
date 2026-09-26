//! Receipt-store failure for one cold micro fixture.
//!
//! The report stores the stable code and the journal event. This measurement
//! does not write a receipt file. The layout is specified in
//! `spec/KIP-INFER-0067-receipt-store.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, ReceiptStoreReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied receipt-store failure for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptStoreObservation {
    pub engine_build_root: DigestHex,
    pub request_id: String,
    pub code: String,
    pub retryable: bool,
    pub receipt_stored: bool,
    pub journal_event: String,
    pub partial_receipt: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the receipt-store report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroReceiptStore {
    pub plan: PlacementPlanV1,
    pub observed: ReceiptStoreObservation,
    pub report: ReceiptStoreReportV1,
}

/// Record the receipt-store failure. The receipt file is not written.
pub fn measure_receipt_store(
    plan: &PlacementPlanV1,
    observed: &ReceiptStoreObservation,
) -> Result<MicroReceiptStore, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_receipt_store(&produced)?;
    Ok(produced)
}

/// Recompute the receipt-store report from the stored plan and observation.
pub fn verify_receipt_store(measurement: &MicroReceiptStore) -> Result<(), InferFailure> {
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
        "journal event does not match",
        &report.journal_event,
        &observed.journal_event,
    )?;
    same_text(
        "partial receipt does not match",
        &report.partial_receipt,
        &observed.partial_receipt,
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
        "receipt store concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "receipt store run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "receipt store warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "receipt store request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "receipt store validation did not match",
        ));
    }
    Ok(())
}

/// Write the receipt-store report. The receipt file is not written.
pub fn write_receipt_store_report(
    base: &Path,
    measurement: &MicroReceiptStore,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_receipt_store(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "receipt store",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ReceiptStoreObservation,
) -> Result<MicroReceiptStore, InferFailure> {
    accept_micro_plan(plan, "receipt store")?;
    accept_cold_single(
        "receipt store",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = ReceiptStoreReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        request_id: observed.request_id.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        receipt_stored: observed.receipt_stored,
        journal_event: observed.journal_event.clone(),
        partial_receipt: observed.partial_receipt.clone(),
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
    Ok(MicroReceiptStore {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
