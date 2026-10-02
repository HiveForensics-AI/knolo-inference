//! Host-supplied Ed25519 challenge hash for one cold micro fixture.
//!
//! The report stores the status and the byte counts. This measurement does
//! not read key bytes and does not reduce the scalar. The layout is
//! specified in `spec/KIP-INFER-0081-challenge-hash.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, ChallengeReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied challenge hash for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChallengeObservation {
    pub engine_build_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub challenge_status: String,
    pub challenge_hashed: bool,
    pub scalar_reduced: bool,
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

/// One micro-fixture observation and the challenge-hash report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroChallenge {
    pub plan: PlacementPlanV1,
    pub observed: ChallengeObservation,
    pub report: ChallengeReportV1,
}

/// Record the challenge hash. The challenge is not hashed.
pub fn measure_challenge(
    plan: &PlacementPlanV1,
    observed: &ChallengeObservation,
) -> Result<MicroChallenge, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_challenge(&produced)?;
    Ok(produced)
}

/// Recompute the challenge-hash report from the stored plan and observation.
pub fn verify_challenge(measurement: &MicroChallenge) -> Result<(), InferFailure> {
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
        "message root does not match",
        &report.message_root,
        &observed.message_root,
    )?;
    same_text(
        "challenge status does not match",
        &report.challenge_status,
        &observed.challenge_status,
    )?;
    same_bool(
        "challenge hashed does not match",
        report.challenge_hashed,
        observed.challenge_hashed,
    )?;
    same_bool(
        "scalar reduced does not match",
        report.scalar_reduced,
        observed.scalar_reduced,
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
        "challenge concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "challenge run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "challenge warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "challenge request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "challenge validation did not match",
        ));
    }
    Ok(())
}

/// Write the challenge-hash report. The challenge is not hashed.
pub fn write_challenge_report(
    base: &Path,
    measurement: &MicroChallenge,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_challenge(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "challenge",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ChallengeObservation,
) -> Result<MicroChallenge, InferFailure> {
    accept_micro_plan(plan, "challenge")?;
    accept_cold_single(
        "challenge",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let validation_result = if observed.challenge_status == "hashed" {
        "verified"
    } else {
        "recorded"
    };
    let report = ChallengeReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        release_root: observed.release_root.clone(),
        message_root: observed.message_root.clone(),
        challenge_status: observed.challenge_status.clone(),
        challenge_hashed: observed.challenge_hashed,
        scalar_reduced: observed.scalar_reduced,
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
        validation_result: validation_result.into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroChallenge {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
