//! One mixture-of-experts request the router did not run for a cold micro fixture.
//!
//! The report stores the reason and the requested expert count. This
//! measurement does not route an expert. The layout is specified in
//! `spec/KIP-INFER-0118-mixture-of-experts.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, MoeReportV1, PlacementPlanV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied moe record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub requested_experts: u32,
    pub selected_experts: u32,
    pub code: String,
    pub retryable: bool,
    pub routed: bool,
    pub shared_used: bool,
    pub tie_broken: bool,
    pub grouped: bool,
    pub expert_placed: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the moe report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroMoe {
    pub plan: PlacementPlanV1,
    pub observed: MoeObservation,
    pub report: MoeReportV1,
}

/// Record the moe measurement.
pub fn measure_moe(
    plan: &PlacementPlanV1,
    observed: &MoeObservation,
) -> Result<MicroMoe, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_moe(&produced)?;
    Ok(produced)
}

/// Recompute the moe report from the stored plan and observation.
pub fn verify_moe(measurement: &MicroMoe) -> Result<(), InferFailure> {
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
        "requested experts do not match",
        report.requested_experts,
        observed.requested_experts,
    )?;
    same_u32(
        "selected experts do not match",
        report.selected_experts,
        observed.selected_experts,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool("routed does not match", report.routed, observed.routed)?;
    same_bool(
        "shared used does not match",
        report.shared_used,
        observed.shared_used,
    )?;
    same_bool(
        "tie broken does not match",
        report.tie_broken,
        observed.tie_broken,
    )?;
    same_bool("grouped does not match", report.grouped, observed.grouped)?;
    same_bool(
        "expert placed does not match",
        report.expert_placed,
        observed.expert_placed,
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
        "moe concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "moe run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "moe warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "moe request count does not match",
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
            "moe validation did not match",
        ));
    }
    Ok(())
}

/// Write the moe report.
pub fn write_moe_report(
    base: &Path,
    measurement: &MicroMoe,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_moe(measurement)?;
    write_exclusive(base, receipt_path, &measurement.report.to_bytes()?, "moe")
}

fn assemble(plan: &PlacementPlanV1, observed: &MoeObservation) -> Result<MicroMoe, InferFailure> {
    accept_micro_plan(plan, "moe")?;
    accept_cold_single(
        "moe",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = MoeReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        requested_experts: observed.requested_experts,
        selected_experts: observed.selected_experts,
        code: observed.code.clone(),
        retryable: observed.retryable,
        routed: observed.routed,
        shared_used: observed.shared_used,
        tie_broken: observed.tie_broken,
        grouped: observed.grouped,
        expert_placed: observed.expert_placed,
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
    Ok(MicroMoe {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
