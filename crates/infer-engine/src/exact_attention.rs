//! One exact attention the engine did not apply for a cold micro fixture.
//!
//! The report stores the reason and the count. This measurement does
//! not run the feature.
//!
//! The layout is specified in `spec/KIP-INFER-0137-exact-attention.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, ExactAttentionReportV1, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied exact-attention record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactAttentionObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub requested_tiles: u32,
    pub selected_tiles: u32,
    pub code: String,
    pub retryable: bool,
    pub flash_selected: bool,
    pub tiled: bool,
    pub split: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the exact-attention report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroExactAttention {
    pub plan: PlacementPlanV1,
    pub observed: ExactAttentionObservation,
    pub report: ExactAttentionReportV1,
}

/// Record the exact-attention measurement.
pub fn measure_exact_attention(
    plan: &PlacementPlanV1,
    observed: &ExactAttentionObservation,
) -> Result<MicroExactAttention, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_exact_attention(&produced)?;
    Ok(produced)
}

/// Recompute the exact-attention report from the stored plan and observation.
pub fn verify_exact_attention(measurement: &MicroExactAttention) -> Result<(), InferFailure> {
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
        "requested tiles do not match",
        report.requested_tiles,
        observed.requested_tiles,
    )?;
    same_u32(
        "selected tiles do not match",
        report.selected_tiles,
        observed.selected_tiles,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "flash selected does not match",
        report.flash_selected,
        observed.flash_selected,
    )?;
    same_bool("tiled does not match", report.tiled, observed.tiled)?;
    same_bool("split does not match", report.split, observed.split)?;
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
        "exact-attention concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "exact-attention run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "exact-attention warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "exact-attention request count does not match",
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
            "exact-attention validation did not match",
        ));
    }
    Ok(())
}

/// Write the exact-attention report.
pub fn write_exact_attention_report(
    base: &Path,
    measurement: &MicroExactAttention,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_exact_attention(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "exact-attention",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ExactAttentionObservation,
) -> Result<MicroExactAttention, InferFailure> {
    accept_micro_plan(plan, "exact-attention")?;
    accept_cold_single(
        "exact-attention",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = ExactAttentionReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        reason: observed.reason.clone(),
        requested_tiles: observed.requested_tiles,
        selected_tiles: observed.selected_tiles,
        code: observed.code.clone(),
        retryable: observed.retryable,
        flash_selected: observed.flash_selected,
        tiled: observed.tiled,
        split: observed.split,
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
    Ok(MicroExactAttention {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
