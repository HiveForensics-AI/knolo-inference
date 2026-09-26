//! Host-supplied Ed25519 domain separation for one cold micro fixture.
//!
//! The report stores the status and the byte counts. This measurement does
//! not hash a payload and does not separate a domain. The layout is specified
//! in `spec/KIP-INFER-0116-domain-separation.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, DomainReportV1, ErrorCode, InferFailure, PlacementPlanV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied domain record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainObservation {
    pub engine_build_root: DigestHex,
    pub domain_root: DigestHex,
    pub message_root: DigestHex,
    pub separate_status: String,
    pub domain_separated: bool,
    pub payload_hashed: bool,
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

/// One micro-fixture observation and the domain report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroDomain {
    pub plan: PlacementPlanV1,
    pub observed: DomainObservation,
    pub report: DomainReportV1,
}

/// Record the domain measurement.
pub fn measure_domain(
    plan: &PlacementPlanV1,
    observed: &DomainObservation,
) -> Result<MicroDomain, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_domain(&produced)?;
    Ok(produced)
}

/// Recompute the domain report from the stored plan and observation.
pub fn verify_domain(measurement: &MicroDomain) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "engine build root does not match",
        &report.engine_build_root,
        &observed.engine_build_root,
    )?;
    same_root(
        "domain root does not match",
        &report.domain_root,
        &observed.domain_root,
    )?;
    same_root(
        "message root does not match",
        &report.message_root,
        &observed.message_root,
    )?;
    same_text(
        "separate status does not match",
        &report.separate_status,
        &observed.separate_status,
    )?;
    same_bool(
        "domain separated does not match",
        report.domain_separated,
        observed.domain_separated,
    )?;
    same_bool(
        "payload hashed does not match",
        report.payload_hashed,
        observed.payload_hashed,
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
        "domain concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "domain run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "domain warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "domain request count does not match",
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
            "domain validation did not match",
        ));
    }
    Ok(())
}

/// Write the domain report.
pub fn write_domain_report(
    base: &Path,
    measurement: &MicroDomain,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_domain(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "domain",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &DomainObservation,
) -> Result<MicroDomain, InferFailure> {
    accept_micro_plan(plan, "domain")?;
    accept_cold_single(
        "domain",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = DomainReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        domain_root: observed.domain_root.clone(),
        message_root: observed.message_root.clone(),
        separate_status: observed.separate_status.clone(),
        domain_separated: observed.domain_separated,
        payload_hashed: observed.payload_hashed,
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
        validation_result: (if observed.separate_status == "separated" {
            "verified"
        } else {
            "recorded"
        })
        .into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroDomain {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
