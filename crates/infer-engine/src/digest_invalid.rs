//! Digest text or domain the parser refused for one cold micro fixture.
//!
//! The report stores the refusal. This measurement does not hash a payload
//! and does not open a receipt file. The layout is specified in
//! `spec/KIP-INFER-0105-digest-invalid.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, DigestInvalidReportV1, ErrorCode, InferFailure, PlacementPlanV1,
    MAX_DIGEST_DOMAIN_BYTES, MAX_DIGEST_RECORD_HEX,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied invalid digest for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DigestInvalidObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub hex_length: u32,
    pub domain: String,
    pub code: String,
    pub retryable: bool,
    pub prefix_accepted: bool,
    pub length_accepted: bool,
    pub alphabet_accepted: bool,
    pub domain_accepted: bool,
    pub hashed: bool,
    pub file_opened: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the digest-invalid report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroDigestInvalid {
    pub plan: PlacementPlanV1,
    pub observed: DigestInvalidObservation,
    pub report: DigestInvalidReportV1,
}

/// Record the invalid digest. The payload is not hashed.
pub fn measure_digest_invalid(
    plan: &PlacementPlanV1,
    observed: &DigestInvalidObservation,
) -> Result<MicroDigestInvalid, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_digest_invalid(&produced)?;
    Ok(produced)
}

/// Recompute the digest-invalid report from the stored plan and observation.
pub fn verify_digest_invalid(measurement: &MicroDigestInvalid) -> Result<(), InferFailure> {
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
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_u32(
        "hex length does not match",
        report.hex_length,
        observed.hex_length,
    )?;
    same_text("domain does not match", &report.domain, &observed.domain)?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "prefix accepted does not match",
        report.prefix_accepted,
        observed.prefix_accepted,
    )?;
    same_bool(
        "length accepted does not match",
        report.length_accepted,
        observed.length_accepted,
    )?;
    same_bool(
        "alphabet accepted does not match",
        report.alphabet_accepted,
        observed.alphabet_accepted,
    )?;
    same_bool(
        "domain accepted does not match",
        report.domain_accepted,
        observed.domain_accepted,
    )?;
    same_bool("hashed does not match", report.hashed, observed.hashed)?;
    same_bool(
        "file opened does not match",
        report.file_opened,
        observed.file_opened,
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
        "digest-invalid concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "digest-invalid run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "digest-invalid warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "digest-invalid request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "digest-invalid validation did not match",
        ));
    }
    Ok(())
}

/// Write the digest-invalid report. The payload is not hashed.
pub fn write_digest_invalid_report(
    base: &Path,
    measurement: &MicroDigestInvalid,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_digest_invalid(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "digest-invalid",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &DigestInvalidObservation,
) -> Result<MicroDigestInvalid, InferFailure> {
    accept_micro_plan(plan, "digest-invalid")?;
    accept_cold_single(
        "digest-invalid",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.reason == "length" && observed.hex_length > MAX_DIGEST_RECORD_HEX {
        return Err(fail(
            ErrorCode::DigestInvalid,
            "digest hex exceeds the record cap",
        ));
    }
    if observed.reason == "domain" && observed.domain.len() > MAX_DIGEST_DOMAIN_BYTES as usize {
        return Err(fail(
            ErrorCode::DigestInvalid,
            "domain exceeds the record cap",
        ));
    }
    let report = DigestInvalidReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        hex_length: observed.hex_length,
        domain: observed.domain.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        prefix_accepted: observed.prefix_accepted,
        length_accepted: observed.length_accepted,
        alphabet_accepted: observed.alphabet_accepted,
        domain_accepted: observed.domain_accepted,
        hashed: observed.hashed,
        file_opened: observed.file_opened,
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
    Ok(MicroDigestInvalid {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
