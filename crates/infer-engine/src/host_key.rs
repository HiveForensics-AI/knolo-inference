//! Host-key check for one cold micro fixture.
//!
//! The report stores a host-supplied match. This measurement does not read
//! key bytes and does not evaluate Ed25519. The layout is specified in
//! `spec/KIP-INFER-0056-host-key.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, HostKeyReportV1, InferFailure, PlacementPlanV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied host-key match for one release.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostKeyObservation {
    pub engine_build_root: DigestHex,
    pub release_root: DigestHex,
    pub host_key_root: DigestHex,
    pub signature_status: String,
    pub key_id: String,
    pub signature_bytes: u32,
    pub key_verified: bool,
    pub key_material_present: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the host-key report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroHostKey {
    pub plan: PlacementPlanV1,
    pub observed: HostKeyObservation,
    pub report: HostKeyReportV1,
}

/// Record the host-key match. The key bytes are not read.
pub fn measure_host_key(
    plan: &PlacementPlanV1,
    observed: &HostKeyObservation,
) -> Result<MicroHostKey, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_host_key(&produced)?;
    Ok(produced)
}

/// Recompute the host-key report from the stored plan and observation.
pub fn verify_host_key(measurement: &MicroHostKey) -> Result<(), InferFailure> {
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
        "release root does not match",
        &report.release_root,
        &observed.release_root,
    )?;
    same_root(
        "host key root does not match",
        &report.host_key_root,
        &observed.host_key_root,
    )?;
    same_text(
        "signature status does not match",
        &report.signature_status,
        &observed.signature_status,
    )?;
    same_text("key id does not match", &report.key_id, &observed.key_id)?;
    same_u32(
        "signature bytes do not match",
        report.signature_bytes,
        observed.signature_bytes,
    )?;
    same_bool(
        "key verified does not match",
        report.key_verified,
        observed.key_verified,
    )?;
    same_bool(
        "key material does not match",
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
        "host key concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "host key run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "host key warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "host key request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "host key validation did not match",
        ));
    }
    Ok(())
}

/// Write the host-key report. The key bytes are not written.
pub fn write_host_key_report(
    base: &Path,
    measurement: &MicroHostKey,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_host_key(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "host key",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &HostKeyObservation,
) -> Result<MicroHostKey, InferFailure> {
    accept_micro_plan(plan, "host key")?;
    accept_cold_single(
        "host key",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let validation_result = if observed.signature_status == "matched" {
        "verified"
    } else {
        "recorded"
    };
    let report = HostKeyReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        release_root: observed.release_root.clone(),
        host_key_root: observed.host_key_root.clone(),
        signature_status: observed.signature_status.clone(),
        key_id: observed.key_id.clone(),
        signature_bytes: observed.signature_bytes,
        key_verified: observed.key_verified,
        key_material_present: observed.key_material_present,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        validation_result: validation_result.into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroHostKey {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
