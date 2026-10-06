//! One canonical-CBOR header the decoder refused for a cold micro fixture.
//!
//! The report stores the major type and the additional information. This
//! measurement does not decode the document. The layout is specified in
//! `spec/KIP-INFER-0107-canonical-cbor.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, CanonicalCborReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1,
    MAX_CBOR_ADDITIONAL, MAX_CBOR_MAJOR, MAX_MAP_ADDITIONAL,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied canonical refusal for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalCborObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub major: u32,
    pub additional_info: u32,
    pub code: String,
    pub retryable: bool,
    pub argument_read: bool,
    pub value_accepted: bool,
    pub keys_ordered: bool,
    pub reencoded: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the canonical-CBOR report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroCanonicalCbor {
    pub plan: PlacementPlanV1,
    pub observed: CanonicalCborObservation,
    pub report: CanonicalCborReportV1,
}

/// Record the canonical refusal. The document is not decoded.
pub fn measure_canonical_cbor(
    plan: &PlacementPlanV1,
    observed: &CanonicalCborObservation,
) -> Result<MicroCanonicalCbor, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_canonical_cbor(&produced)?;
    Ok(produced)
}

/// Recompute the canonical-CBOR report from the stored plan and observation.
pub fn verify_canonical_cbor(measurement: &MicroCanonicalCbor) -> Result<(), InferFailure> {
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
    same_u32("major does not match", report.major, observed.major)?;
    same_u32(
        "additional information does not match",
        report.additional_info,
        observed.additional_info,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "argument read does not match",
        report.argument_read,
        observed.argument_read,
    )?;
    same_bool(
        "value accepted does not match",
        report.value_accepted,
        observed.value_accepted,
    )?;
    same_bool(
        "keys ordered does not match",
        report.keys_ordered,
        observed.keys_ordered,
    )?;
    same_bool(
        "reencoded does not match",
        report.reencoded,
        observed.reencoded,
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
        "canonical-cbor concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "canonical-cbor run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "canonical-cbor warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "canonical-cbor request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "canonical-cbor validation did not match",
        ));
    }
    Ok(())
}

/// Write the canonical-CBOR report. The document is not decoded.
pub fn write_canonical_cbor_report(
    base: &Path,
    measurement: &MicroCanonicalCbor,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_canonical_cbor(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "canonical-cbor",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &CanonicalCborObservation,
) -> Result<MicroCanonicalCbor, InferFailure> {
    accept_micro_plan(plan, "canonical-cbor")?;
    accept_cold_single(
        "canonical-cbor",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.major > MAX_CBOR_MAJOR {
        return Err(fail(
            ErrorCode::CanonicalCborInvalid,
            "CBOR major exceeds the record cap",
        ));
    }
    if observed.additional_info > MAX_CBOR_ADDITIONAL {
        return Err(fail(
            ErrorCode::CanonicalCborInvalid,
            "additional information exceeds the record cap",
        ));
    }
    if observed.reason == "order" && observed.additional_info > MAX_MAP_ADDITIONAL {
        return Err(fail(
            ErrorCode::CanonicalCborInvalid,
            "map length exceeds the record cap",
        ));
    }
    let report = CanonicalCborReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        major: observed.major,
        additional_info: observed.additional_info,
        code: observed.code.clone(),
        retryable: observed.retryable,
        argument_read: observed.argument_read,
        value_accepted: observed.value_accepted,
        keys_ordered: observed.keys_ordered,
        reencoded: observed.reencoded,
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
    Ok(MicroCanonicalCbor {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
