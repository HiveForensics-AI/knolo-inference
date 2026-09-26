//! One CUDA graph constraint that was not captured for a cold micro fixture.
//!
//! The report stores the reason and zero capture counts. This measurement
//! does not capture a graph. The layout is specified in
//! `spec/KIP-INFER-0113-cuda-graph.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, GraphReportV1, InferFailure, PlacementPlanV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied CUDA graph record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub code: String,
    pub retryable: bool,
    pub graph_captured: bool,
    pub captured_nodes: u32,
    pub workspace_bytes: u32,
    pub shape_buckets: u32,
    pub kernel_repeated: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the CUDA graph report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroGraph {
    pub plan: PlacementPlanV1,
    pub observed: GraphObservation,
    pub report: GraphReportV1,
}

/// Record the CUDA graph constraint. The graph is not captured.
pub fn measure_cuda_graph(
    plan: &PlacementPlanV1,
    observed: &GraphObservation,
) -> Result<MicroGraph, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_cuda_graph(&produced)?;
    Ok(produced)
}

/// Recompute the CUDA graph report from the stored plan and observation.
pub fn verify_cuda_graph(measurement: &MicroGraph) -> Result<(), InferFailure> {
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
    let device = measurement
        .plan
        .devices
        .first()
        .map(String::as_str)
        .unwrap_or("");
    same_text("device does not match", &report.device, device)?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "graph captured does not match",
        report.graph_captured,
        observed.graph_captured,
    )?;
    same_u32(
        "captured nodes do not match",
        report.captured_nodes,
        observed.captured_nodes,
    )?;
    same_u32(
        "workspace bytes do not match",
        report.workspace_bytes,
        observed.workspace_bytes,
    )?;
    same_u32(
        "shape buckets do not match",
        report.shape_buckets,
        observed.shape_buckets,
    )?;
    same_bool(
        "kernel repeated does not match",
        report.kernel_repeated,
        observed.kernel_repeated,
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
        "graph concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "graph run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "graph warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "graph request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "graph validation did not match",
        ));
    }
    Ok(())
}

/// Write the CUDA graph report. The graph is not captured.
pub fn write_cuda_graph_report(
    base: &Path,
    measurement: &MicroGraph,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_cuda_graph(measurement)?;
    write_exclusive(base, receipt_path, &measurement.report.to_bytes()?, "graph")
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &GraphObservation,
) -> Result<MicroGraph, InferFailure> {
    accept_micro_plan(plan, "graph")?;
    if plan.devices.len() != 1 || plan.devices[0] != "slot-0" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "a cuda graph record names slot-0",
        ));
    }
    accept_cold_single(
        "graph",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = GraphReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        device: plan.devices[0].clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        graph_captured: observed.graph_captured,
        captured_nodes: observed.captured_nodes,
        workspace_bytes: observed.workspace_bytes,
        shape_buckets: observed.shape_buckets,
        kernel_repeated: observed.kernel_repeated,
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
    Ok(MicroGraph {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
