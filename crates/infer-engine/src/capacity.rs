//! One expert-capacity request the router did not apply for a cold micro fixture.
//!
//! The report stores the reason and the requested token count. This
//! measurement does not drop a token and does not balance experts. The layout
//! is specified in `spec/KIP-INFER-0125-expert-capacity.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, CapacityReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1,
    MAX_CAPACITY_TOKENS,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied capacity record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapacityObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub requested_tokens: u32,
    pub capacity_tokens: u32,
    pub code: String,
    pub retryable: bool,
    pub overflowed: bool,
    pub dropped: bool,
    pub balanced: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the capacity report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroCapacity {
    pub plan: PlacementPlanV1,
    pub observed: CapacityObservation,
    pub report: CapacityReportV1,
}

/// Record the capacity measurement.
pub fn measure_capacity(
    plan: &PlacementPlanV1,
    observed: &CapacityObservation,
) -> Result<MicroCapacity, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_capacity(&produced)?;
    Ok(produced)
}

/// Recompute the capacity report from the stored plan and observation.
pub fn verify_capacity(measurement: &MicroCapacity) -> Result<(), InferFailure> {
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
        "requested tokens do not match",
        report.requested_tokens,
        observed.requested_tokens,
    )?;
    same_u32(
        "capacity tokens do not match",
        report.capacity_tokens,
        observed.capacity_tokens,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "overflowed does not match",
        report.overflowed,
        observed.overflowed,
    )?;
    same_bool("dropped does not match", report.dropped, observed.dropped)?;
    same_bool(
        "balanced does not match",
        report.balanced,
        observed.balanced,
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
        "capacity concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "capacity run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "capacity warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "capacity request count does not match",
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
            "capacity validation did not match",
        ));
    }
    Ok(())
}

/// Write the capacity report.
pub fn write_capacity_report(
    base: &Path,
    measurement: &MicroCapacity,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_capacity(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "capacity",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &CapacityObservation,
) -> Result<MicroCapacity, InferFailure> {
    accept_micro_plan(plan, "capacity")?;
    accept_cold_single(
        "capacity",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.requested_tokens > MAX_CAPACITY_TOKENS {
        return Err(fail(
            ErrorCode::ContextLimitExceeded,
            "an expert capacity request exceeds the context",
        ));
    }

    let report = CapacityReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        requested_tokens: observed.requested_tokens,
        capacity_tokens: observed.capacity_tokens,
        code: observed.code.clone(),
        retryable: observed.retryable,
        overflowed: observed.overflowed,
        dropped: observed.dropped,
        balanced: observed.balanced,
        forward_ran: observed.forward_ran,
        receipt_stored: observed.receipt_stored,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        validation_result: ("recorded").into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroCapacity {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
