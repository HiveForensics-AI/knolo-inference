//! One tool call the engine did not execute for a cold micro fixture.
//!
//! The report stores the reason and the name length. This measurement does
//! not call Agents and does not execute a tool. The layout is specified in
//! `spec/KIP-INFER-0110-tool-refusal.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, ToolRefusalReportV1,
    MAX_TOOL_NAME_BYTES,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied tool refusal for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRefusalObservation {
    pub engine_build_root: DigestHex,
    pub object_root: DigestHex,
    pub reason: String,
    pub name_bytes: u32,
    pub code: String,
    pub retryable: bool,
    pub name_accepted: bool,
    pub object_generated: bool,
    pub tool_executed: bool,
    pub authority_checked: bool,
    pub budget_checked: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the tool-refusal report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroToolRefusal {
    pub plan: PlacementPlanV1,
    pub observed: ToolRefusalObservation,
    pub report: ToolRefusalReportV1,
}

/// Record the tool refusal. The tool is not called.
pub fn measure_tool_refusal(
    plan: &PlacementPlanV1,
    observed: &ToolRefusalObservation,
) -> Result<MicroToolRefusal, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_tool_refusal(&produced)?;
    Ok(produced)
}

/// Recompute the tool-refusal report from the stored plan and observation.
pub fn verify_tool_refusal(measurement: &MicroToolRefusal) -> Result<(), InferFailure> {
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
        "object root does not match",
        &report.object_root,
        &observed.object_root,
    )?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_u32(
        "name bytes do not match",
        report.name_bytes,
        observed.name_bytes,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "name accepted does not match",
        report.name_accepted,
        observed.name_accepted,
    )?;
    same_bool(
        "object generated does not match",
        report.object_generated,
        observed.object_generated,
    )?;
    same_bool(
        "tool executed does not match",
        report.tool_executed,
        observed.tool_executed,
    )?;
    same_bool(
        "authority checked does not match",
        report.authority_checked,
        observed.authority_checked,
    )?;
    same_bool(
        "budget checked does not match",
        report.budget_checked,
        observed.budget_checked,
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
        "tool-refusal concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "tool-refusal run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "tool-refusal warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "tool-refusal request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "tool-refusal validation did not match",
        ));
    }
    Ok(())
}

/// Write the tool-refusal report. The tool is not called.
pub fn write_tool_refusal_report(
    base: &Path,
    measurement: &MicroToolRefusal,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_tool_refusal(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "tool-refusal",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ToolRefusalObservation,
) -> Result<MicroToolRefusal, InferFailure> {
    accept_micro_plan(plan, "tool-refusal")?;
    accept_cold_single(
        "tool-refusal",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if matches!(observed.reason.as_str(), "object" | "execute")
        && observed.name_bytes > MAX_TOOL_NAME_BYTES
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "tool name exceeds the record cap",
        ));
    }
    let report = ToolRefusalReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        object_root: observed.object_root.clone(),
        reason: observed.reason.clone(),
        name_bytes: observed.name_bytes,
        code: observed.code.clone(),
        retryable: observed.retryable,
        name_accepted: observed.name_accepted,
        object_generated: observed.object_generated,
        tool_executed: observed.tool_executed,
        authority_checked: observed.authority_checked,
        budget_checked: observed.budget_checked,
        forward_ran: observed.forward_ran,
        receipt_stored: observed.receipt_stored,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        validation_result: "recorded".into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroToolRefusal {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
