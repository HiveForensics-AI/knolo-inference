//! Host-supplied digest binding for one cold micro fixture.
//!
//! The report stores the status and the byte counts. This measurement does
//! not bind a digest and does not bind evidence.
//!
//! The layout is specified in `spec/KIP-INFER-0131-digest-binding.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, BindingReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied binding record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingObservation {
    pub engine_build_root: DigestHex,
    pub digest_root: DigestHex,
    pub evidence_root: DigestHex,
    pub bind_status: String,
    pub digest_bound: bool,
    pub evidence_bound: bool,
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

/// One micro-fixture observation and the binding report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroBinding {
    pub plan: PlacementPlanV1,
    pub observed: BindingObservation,
    pub report: BindingReportV1,
}

/// Record the binding measurement.
pub fn measure_binding(
    plan: &PlacementPlanV1,
    observed: &BindingObservation,
) -> Result<MicroBinding, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_binding(&produced)?;
    Ok(produced)
}

/// Recompute the binding report from the stored plan and observation.
pub fn verify_binding(measurement: &MicroBinding) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "engine build root does not match",
        &report.engine_build_root,
        &observed.engine_build_root,
    )?;
    same_root(
        "digest root does not match",
        &report.digest_root,
        &observed.digest_root,
    )?;
    same_root(
        "evidence root does not match",
        &report.evidence_root,
        &observed.evidence_root,
    )?;
    same_text(
        "bind status does not match",
        &report.bind_status,
        &observed.bind_status,
    )?;
    same_bool(
        "digest bound does not match",
        report.digest_bound,
        observed.digest_bound,
    )?;
    same_bool(
        "evidence bound does not match",
        report.evidence_bound,
        observed.evidence_bound,
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
        "binding concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "binding run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "binding warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "binding request count does not match",
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
            "binding validation did not match",
        ));
    }
    Ok(())
}

/// Write the binding report.
pub fn write_binding_report(
    base: &Path,
    measurement: &MicroBinding,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_binding(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "binding",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &BindingObservation,
) -> Result<MicroBinding, InferFailure> {
    accept_micro_plan(plan, "binding")?;
    accept_cold_single(
        "binding",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = BindingReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        digest_root: observed.digest_root.clone(),
        evidence_root: observed.evidence_root.clone(),
        bind_status: observed.bind_status.clone(),
        digest_bound: observed.digest_bound,
        evidence_bound: observed.evidence_bound,
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
        placement_root: plan.root()?,
        validation_result: (if observed.bind_status == "bound" {
            "verified"
        } else {
            "recorded"
        })
        .into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroBinding {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
