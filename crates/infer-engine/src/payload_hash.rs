//! Host-supplied payload hash for one cold micro fixture.
//!
//! The report stores the status and the byte counts. This measurement does
//! not hash a payload and does not bind a digest.
//!
//! The layout is specified in `spec/KIP-INFER-0126-payload-hash.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, PayloadReportV1, PlacementPlanV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied payload record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadObservation {
    pub engine_build_root: DigestHex,
    pub payload_root: DigestHex,
    pub domain_root: DigestHex,
    pub hash_status: String,
    pub payload_hashed: bool,
    pub digest_bound: bool,
    pub public_key_bytes: u32,
    pub signature_bytes: u32,
    pub scalar_bytes: u32,
    pub key_material_present: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the payload report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroPayload {
    pub plan: PlacementPlanV1,
    pub observed: PayloadObservation,
    pub report: PayloadReportV1,
}

/// Record the payload measurement.
pub fn measure_payload(
    plan: &PlacementPlanV1,
    observed: &PayloadObservation,
) -> Result<MicroPayload, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_payload(&produced)?;
    Ok(produced)
}

/// Recompute the payload report from the stored plan and observation.
pub fn verify_payload(measurement: &MicroPayload) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "engine build root does not match",
        &report.engine_build_root,
        &observed.engine_build_root,
    )?;
    same_root(
        "payload root does not match",
        &report.payload_root,
        &observed.payload_root,
    )?;
    same_root(
        "domain root does not match",
        &report.domain_root,
        &observed.domain_root,
    )?;
    same_text(
        "hash status does not match",
        &report.hash_status,
        &observed.hash_status,
    )?;
    same_bool(
        "payload hashed does not match",
        report.payload_hashed,
        observed.payload_hashed,
    )?;
    same_bool(
        "digest bound does not match",
        report.digest_bound,
        observed.digest_bound,
    )?;
    same_u32(
        "public key bytes do not match",
        report.public_key_bytes,
        observed.public_key_bytes,
    )?;
    same_u32(
        "signature bytes do not match",
        report.signature_bytes,
        observed.signature_bytes,
    )?;
    same_u32(
        "scalar bytes do not match",
        report.scalar_bytes,
        observed.scalar_bytes,
    )?;
    same_bool(
        "key material present does not match",
        report.key_material_present,
        observed.key_material_present,
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
        "payload concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "payload run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "payload warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "payload request count does not match",
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
            "payload validation did not match",
        ));
    }
    Ok(())
}

/// Write the payload report.
pub fn write_payload_report(
    base: &Path,
    measurement: &MicroPayload,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_payload(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "payload",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &PayloadObservation,
) -> Result<MicroPayload, InferFailure> {
    accept_micro_plan(plan, "payload")?;
    accept_cold_single(
        "payload",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = PayloadReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        payload_root: observed.payload_root.clone(),
        domain_root: observed.domain_root.clone(),
        hash_status: observed.hash_status.clone(),
        payload_hashed: observed.payload_hashed,
        digest_bound: observed.digest_bound,
        public_key_bytes: observed.public_key_bytes,
        signature_bytes: observed.signature_bytes,
        scalar_bytes: observed.scalar_bytes,
        key_material_present: observed.key_material_present,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        validation_result: (if observed.hash_status == "hashed" {
            "verified"
        } else {
            "recorded"
        })
        .into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroPayload {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
