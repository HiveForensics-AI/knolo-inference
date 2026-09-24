//! Recipe support mark for one cold run of the micro fixture.
//!
//! The measurement does not bless a recipe that lacks greedy token parity,
//! and a blessed mark requires four distinct receipt roots. It does not open
//! a model file. The layout is specified in `spec/KIP-INFER-0043-recipe-status.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, ModelConformanceReceiptV1, PlacementPlanV1,
    RecipeReportV1, RECIPE_ADAPTER,
};

use crate::report_io::{accept_cold_single, accept_micro_plan, write_exclusive};

/// Roots and the support mark from one completed cold run of the micro fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecipeObservation {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub adapter_id: String,
    pub security_root: DigestHex,
    pub stability_root: DigestHex,
    pub benchmark_root: DigestHex,
    pub support_level: String,
}

/// One micro-fixture observation, its conformance receipt, and the recipe mark.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroRecipe {
    pub plan: PlacementPlanV1,
    pub observed: RecipeObservation,
    pub conformance: ModelConformanceReceiptV1,
    pub report: RecipeReportV1,
}

/// Record the recipe mark. No file is created and no model file is opened.
pub fn measure_recipe_status(
    plan: &PlacementPlanV1,
    observed: &RecipeObservation,
    conformance: &ModelConformanceReceiptV1,
) -> Result<MicroRecipe, InferFailure> {
    let produced = assemble(plan, observed, conformance)?;
    verify_recipe_status(&produced)?;
    Ok(produced)
}

/// Recompute the recipe report from the stored plan, observation, and receipt.
pub fn verify_recipe_status(measurement: &MicroRecipe) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "model image root does not match",
        &report.model_image_root,
        &observed.model_image_root,
    )?;
    same_root(
        "artifact root does not match",
        &report.artifact_root,
        &observed.artifact_root,
    )?;
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
    same_text(
        "adapter id does not match",
        &report.adapter_id,
        &observed.adapter_id,
    )?;
    same_root(
        "conformance root does not match",
        &report.conformance_root,
        &measurement.conformance.root()?,
    )?;
    same_root(
        "security root does not match",
        &report.security_root,
        &observed.security_root,
    )?;
    same_root(
        "stability root does not match",
        &report.stability_root,
        &observed.stability_root,
    )?;
    same_root(
        "benchmark root does not match",
        &report.benchmark_root,
        &observed.benchmark_root,
    )?;
    same_text(
        "support level does not match",
        &report.support_level,
        &observed.support_level,
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
        "recipe concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "recipe run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "recipe warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "recipe request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(
        &measurement.plan,
        &measurement.observed,
        &measurement.conformance,
    )?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "recipe validation did not match",
        ));
    }
    Ok(())
}

/// Write the recipe report. The conformance receipt and the plan are not written.
pub fn write_recipe_report(
    base: &Path,
    measurement: &MicroRecipe,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_recipe_status(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "recipe",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &RecipeObservation,
    conformance: &ModelConformanceReceiptV1,
) -> Result<MicroRecipe, InferFailure> {
    accept_micro_plan(plan, "recipe")?;
    accept_cold_single(
        "recipe",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.adapter_id != RECIPE_ADAPTER {
        return Err(fail(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "architecture adapter {} is not compiled in",
                observed.adapter_id
            ),
        ));
    }
    if observed.support_level != "experimental"
        && observed.support_level != "conformant"
        && observed.support_level != "blessed"
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "field supportLevel has an unsupported value",
        ));
    }
    if observed.support_level == "blessed" && !conformance.greedy_token_parity {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "fast execution is not blessed",
        ));
    }
    if observed.support_level == "conformant" && !conformance.greedy_token_parity {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "conformant support requires greedy token parity",
        ));
    }
    conformance.validate()?;
    if conformance.adapter_id != observed.adapter_id {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "adapter id does not match",
        ));
    }
    if conformance.engine_build_root != observed.engine_build_root {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "engine build root does not match",
        ));
    }
    if conformance.support_level != observed.support_level {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "support level does not match",
        ));
    }
    let report = RecipeReportV1 {
        model_image_root: observed.model_image_root.clone(),
        artifact_root: observed.artifact_root.clone(),
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        adapter_id: observed.adapter_id.clone(),
        conformance_root: conformance.root()?,
        security_root: observed.security_root.clone(),
        stability_root: observed.stability_root.clone(),
        benchmark_root: observed.benchmark_root.clone(),
        support_level: observed.support_level.clone(),
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
    Ok(MicroRecipe {
        plan: plan.clone(),
        observed: observed.clone(),
        conformance: conformance.clone(),
        report,
    })
}

fn same_root(message: &str, left: &DigestHex, right: &DigestHex) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn same_text(message: &str, left: &str, right: &str) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn same_u32(message: &str, left: u32, right: u32) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}
