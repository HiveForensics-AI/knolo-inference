//! One vision projector the engine did not apply for a cold micro fixture.
//!
//! The report stores the reason and the count. This measurement does
//! not run the feature.
//!
//! The layout is specified in `spec/KIP-INFER-0135-vision-projector.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, VisionReportV1, MAX_VISION_PATCHES,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied vision record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub requested_patches: u32,
    pub projected_tokens: u32,
    pub code: String,
    pub retryable: bool,
    pub projector_applied: bool,
    pub patches_opened: bool,
    pub embedded: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the vision report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroVision {
    pub plan: PlacementPlanV1,
    pub observed: VisionObservation,
    pub report: VisionReportV1,
}

/// Record the vision measurement.
pub fn measure_vision(
    plan: &PlacementPlanV1,
    observed: &VisionObservation,
) -> Result<MicroVision, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_vision(&produced)?;
    Ok(produced)
}

/// Recompute the vision report from the stored plan and observation.
pub fn verify_vision(measurement: &MicroVision) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "engine build root does not match",
        &report.engine_build_root,
        &observed.engine_build_root,
    )?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_u32(
        "requested patches do not match",
        report.requested_patches,
        observed.requested_patches,
    )?;
    same_u32(
        "projected tokens do not match",
        report.projected_tokens,
        observed.projected_tokens,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "projector applied does not match",
        report.projector_applied,
        observed.projector_applied,
    )?;
    same_bool(
        "patches opened does not match",
        report.patches_opened,
        observed.patches_opened,
    )?;
    same_bool(
        "embedded does not match",
        report.embedded,
        observed.embedded,
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
        "vision concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "vision run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "vision warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "vision request count does not match",
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
            "vision validation did not match",
        ));
    }
    Ok(())
}

/// Write the vision report.
pub fn write_vision_report(
    base: &Path,
    measurement: &MicroVision,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_vision(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "vision",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &VisionObservation,
) -> Result<MicroVision, InferFailure> {
    accept_micro_plan(plan, "vision")?;
    accept_cold_single(
        "vision",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.requested_patches > MAX_VISION_PATCHES {
        return Err(fail(
            ErrorCode::ContextLimitExceeded,
            "a vision patch exceeds the context",
        ));
    }
    let report = VisionReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        reason: observed.reason.clone(),
        requested_patches: observed.requested_patches,
        projected_tokens: observed.projected_tokens,
        code: observed.code.clone(),
        retryable: observed.retryable,
        projector_applied: observed.projector_applied,
        patches_opened: observed.patches_opened,
        embedded: observed.embedded,
        forward_ran: observed.forward_ran,
        receipt_stored: observed.receipt_stored,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        placement_root: plan.root()?,
        validation_result: ("recorded").into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroVision {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
