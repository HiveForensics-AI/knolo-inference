//! One grouped expert kernel that was not selected for a cold micro fixture.
//!
//! The report stores the reason. This measurement does not select a kernel.
//! The layout is specified in `spec/KIP-INFER-0120-grouped-kernel.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, GroupedKernelReportV1, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied grouped-kernel record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedKernelObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub group_count: u32,
    pub code: String,
    pub retryable: bool,
    pub kernel_selected: bool,
    pub grouped: bool,
    pub routing_ran: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the grouped-kernel report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroGroupedKernel {
    pub plan: PlacementPlanV1,
    pub observed: GroupedKernelObservation,
    pub report: GroupedKernelReportV1,
}

/// Record the grouped-kernel measurement.
pub fn measure_grouped_kernel(
    plan: &PlacementPlanV1,
    observed: &GroupedKernelObservation,
) -> Result<MicroGroupedKernel, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_grouped_kernel(&produced)?;
    Ok(produced)
}

/// Recompute the grouped-kernel report from the stored plan and observation.
pub fn verify_grouped_kernel(measurement: &MicroGroupedKernel) -> Result<(), InferFailure> {
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
        "group count does not match",
        report.group_count,
        observed.group_count,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "kernel selected does not match",
        report.kernel_selected,
        observed.kernel_selected,
    )?;
    same_bool("grouped does not match", report.grouped, observed.grouped)?;
    same_bool(
        "routing ran does not match",
        report.routing_ran,
        observed.routing_ran,
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
        "grouped-kernel concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "grouped-kernel run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "grouped-kernel warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "grouped-kernel request count does not match",
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
            "grouped-kernel validation did not match",
        ));
    }
    Ok(())
}

/// Write the grouped-kernel report.
pub fn write_grouped_kernel_report(
    base: &Path,
    measurement: &MicroGroupedKernel,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_grouped_kernel(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "grouped-kernel",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &GroupedKernelObservation,
) -> Result<MicroGroupedKernel, InferFailure> {
    accept_micro_plan(plan, "grouped-kernel")?;
    accept_cold_single(
        "grouped-kernel",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = GroupedKernelReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        group_count: observed.group_count,
        code: observed.code.clone(),
        retryable: observed.retryable,
        kernel_selected: observed.kernel_selected,
        grouped: observed.grouped,
        routing_ran: observed.routing_ran,
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
    Ok(MicroGroupedKernel {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
