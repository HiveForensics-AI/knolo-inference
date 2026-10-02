//! Digest mismatch for one cold micro fixture.
//!
//! The report stores that a weight file's size or digest did not match. This
//! measurement does not open a weight file. The layout is specified in
//! `spec/KIP-INFER-0087-digest-mismatch.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, DigestMismatchReportV1, ErrorCode, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied digest mismatch for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DigestMismatchObservation {
    pub engine_build_root: DigestHex,
    pub artifact_root: DigestHex,
    pub mismatch: String,
    pub code: String,
    pub retryable: bool,
    pub header_parsed: bool,
    pub body_read: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the digest-mismatch report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroDigestMismatch {
    pub plan: PlacementPlanV1,
    pub observed: DigestMismatchObservation,
    pub report: DigestMismatchReportV1,
}

/// Record the digest mismatch. The weight file is not opened.
pub fn measure_digest_mismatch(
    plan: &PlacementPlanV1,
    observed: &DigestMismatchObservation,
) -> Result<MicroDigestMismatch, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_digest_mismatch(&produced)?;
    Ok(produced)
}

/// Recompute the digest-mismatch report from the stored plan and observation.
pub fn verify_digest_mismatch(measurement: &MicroDigestMismatch) -> Result<(), InferFailure> {
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
        "artifact root does not match",
        &report.artifact_root,
        &observed.artifact_root,
    )?;
    same_text(
        "mismatch does not match",
        &report.mismatch,
        &observed.mismatch,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "header parsed does not match",
        report.header_parsed,
        observed.header_parsed,
    )?;
    same_bool(
        "body read does not match",
        report.body_read,
        observed.body_read,
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
    same_cold(report, observed)?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "digest-mismatch validation did not match",
        ));
    }
    Ok(())
}

/// Write the digest-mismatch report. The weight file is not opened.
pub fn write_digest_mismatch_report(
    base: &Path,
    measurement: &MicroDigestMismatch,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_digest_mismatch(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "digest-mismatch",
    )
}

fn same_cold(
    report: &DigestMismatchReportV1,
    observed: &DigestMismatchObservation,
) -> Result<(), InferFailure> {
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
        "digest-mismatch concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "digest-mismatch run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "digest-mismatch warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "digest-mismatch request count does not match",
        report.request_count,
        observed.request_count,
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &DigestMismatchObservation,
) -> Result<MicroDigestMismatch, InferFailure> {
    accept_micro_plan(plan, "digest-mismatch")?;
    accept_cold_single(
        "digest-mismatch",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = DigestMismatchReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        artifact_root: observed.artifact_root.clone(),
        mismatch: observed.mismatch.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        header_parsed: observed.header_parsed,
        body_read: observed.body_read,
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
    Ok(MicroDigestMismatch {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
