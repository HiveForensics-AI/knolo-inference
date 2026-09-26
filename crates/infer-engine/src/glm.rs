//! One GLM-4.7 adapter that is not compiled in for a cold micro fixture.
//!
//! The report stores the reason. This measurement does not open a weight
//! file. The layout is specified in `spec/KIP-INFER-0122-glm-adapter.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, GlmReportV1, InferFailure, PlacementPlanV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied glm record for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlmObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub rejected_adapter: String,
    pub code: String,
    pub retryable: bool,
    pub weights_opened: bool,
    pub inventory_read: bool,
    pub recipe_blessed: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the glm report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroGlm {
    pub plan: PlacementPlanV1,
    pub observed: GlmObservation,
    pub report: GlmReportV1,
}

/// Record the glm measurement.
pub fn measure_glm(
    plan: &PlacementPlanV1,
    observed: &GlmObservation,
) -> Result<MicroGlm, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_glm(&produced)?;
    Ok(produced)
}

/// Recompute the glm report from the stored plan and observation.
pub fn verify_glm(measurement: &MicroGlm) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "engine build root does not match",
        &report.engine_build_root,
        &observed.engine_build_root,
    )?;
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_text(
        "rejected adapter does not match",
        &report.rejected_adapter,
        &observed.rejected_adapter,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "weights opened does not match",
        report.weights_opened,
        observed.weights_opened,
    )?;
    same_bool(
        "inventory read does not match",
        report.inventory_read,
        observed.inventory_read,
    )?;
    same_bool(
        "recipe blessed does not match",
        report.recipe_blessed,
        observed.recipe_blessed,
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
        "glm concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "glm run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "glm warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "glm request count does not match",
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
            "glm validation did not match",
        ));
    }
    Ok(())
}

/// Write the glm report.
pub fn write_glm_report(
    base: &Path,
    measurement: &MicroGlm,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_glm(measurement)?;
    write_exclusive(base, receipt_path, &measurement.report.to_bytes()?, "glm")
}

fn assemble(plan: &PlacementPlanV1, observed: &GlmObservation) -> Result<MicroGlm, InferFailure> {
    accept_micro_plan(plan, "glm")?;
    accept_cold_single(
        "glm",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = GlmReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        rejected_adapter: observed.rejected_adapter.clone(),
        code: observed.code.clone(),
        retryable: observed.retryable,
        weights_opened: observed.weights_opened,
        inventory_read: observed.inventory_read,
        recipe_blessed: observed.recipe_blessed,
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
    Ok(MicroGlm {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
