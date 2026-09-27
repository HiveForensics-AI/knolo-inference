//! One multimodal input the engine did not apply for a cold micro fixture.
//!
//! The report stores the reason and the count. This measurement does
//! not run the feature.
//!
//! The layout is specified in `spec/KIP-INFER-0132-multimodal-input.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, MultimodalReportV1, PlacementPlanV1,
    MAX_MULTIMODAL_PARTS,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied multimodal record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultimodalObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub requested_parts: u32,
    pub accepted_parts: u32,
    pub code: String,
    pub retryable: bool,
    pub media_opened: bool,
    pub projected: bool,
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

/// One micro-fixture observation and the multimodal report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroMultimodal {
    pub plan: PlacementPlanV1,
    pub observed: MultimodalObservation,
    pub report: MultimodalReportV1,
}

/// Record the multimodal measurement.
pub fn measure_multimodal(
    plan: &PlacementPlanV1,
    observed: &MultimodalObservation,
) -> Result<MicroMultimodal, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_multimodal(&produced)?;
    Ok(produced)
}

/// Recompute the multimodal report from the stored plan and observation.
pub fn verify_multimodal(measurement: &MicroMultimodal) -> Result<(), InferFailure> {
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
        "requested parts do not match",
        report.requested_parts,
        observed.requested_parts,
    )?;
    same_u32(
        "accepted parts do not match",
        report.accepted_parts,
        observed.accepted_parts,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "media opened does not match",
        report.media_opened,
        observed.media_opened,
    )?;
    same_bool(
        "projected does not match",
        report.projected,
        observed.projected,
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
        "multimodal concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "multimodal run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "multimodal warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "multimodal request count does not match",
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
            "multimodal validation did not match",
        ));
    }
    Ok(())
}

/// Write the multimodal report.
pub fn write_multimodal_report(
    base: &Path,
    measurement: &MicroMultimodal,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_multimodal(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "multimodal",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &MultimodalObservation,
) -> Result<MicroMultimodal, InferFailure> {
    accept_micro_plan(plan, "multimodal")?;
    accept_cold_single(
        "multimodal",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.requested_parts > MAX_MULTIMODAL_PARTS {
        return Err(fail(
            ErrorCode::ContextLimitExceeded,
            "a multimodal part exceeds the context",
        ));
    }
    let report = MultimodalReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        reason: observed.reason.clone(),
        requested_parts: observed.requested_parts,
        accepted_parts: observed.accepted_parts,
        code: observed.code.clone(),
        retryable: observed.retryable,
        media_opened: observed.media_opened,
        projected: observed.projected,
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
    Ok(MicroMultimodal {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
