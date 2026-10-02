//! Reproducible-build record for one cold micro fixture.
//!
//! The report stores the lock root and the instruction source. This
//! measurement does not run Cargo. The layout is specified in
//! `spec/KIP-INFER-0054-reproducible-build.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, ReproducibleReportV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_root, same_text, same_u32, write_exclusive,
};

/// Caller-supplied pin for the build that would run the fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReproducibleObservation {
    pub engine_build_root: DigestHex,
    pub lock_root: DigestHex,
    pub source_root: DigestHex,
    pub feature_set: String,
    pub build_profile: String,
    pub instruction_count: u32,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the reproducible-build report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroReproducible {
    pub plan: PlacementPlanV1,
    pub observed: ReproducibleObservation,
    pub report: ReproducibleReportV1,
}

/// Record the build pin. Cargo is not run.
pub fn measure_reproducible_build(
    plan: &PlacementPlanV1,
    observed: &ReproducibleObservation,
) -> Result<MicroReproducible, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_reproducible_build(&produced)?;
    Ok(produced)
}

/// Recompute the reproducible report from the stored plan and observation.
pub fn verify_reproducible_build(measurement: &MicroReproducible) -> Result<(), InferFailure> {
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
        "lock root does not match",
        &report.lock_root,
        &observed.lock_root,
    )?;
    same_root(
        "source root does not match",
        &report.source_root,
        &observed.source_root,
    )?;
    same_text(
        "feature set does not match",
        &report.feature_set,
        &observed.feature_set,
    )?;
    same_text(
        "build profile does not match",
        &report.build_profile,
        &observed.build_profile,
    )?;
    same_u32(
        "instruction count does not match",
        report.instruction_count,
        observed.instruction_count,
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
        "reproducible concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "reproducible run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "reproducible warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "reproducible request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "reproducible validation did not match",
        ));
    }
    Ok(())
}

/// Write the reproducible report. The instructions are not written.
pub fn write_reproducible_report(
    base: &Path,
    measurement: &MicroReproducible,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_reproducible_build(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "reproducible",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ReproducibleObservation,
) -> Result<MicroReproducible, InferFailure> {
    accept_micro_plan(plan, "reproducible")?;
    accept_cold_single(
        "reproducible",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = ReproducibleReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        lock_root: observed.lock_root.clone(),
        source_root: observed.source_root.clone(),
        feature_set: observed.feature_set.clone(),
        build_profile: observed.build_profile.clone(),
        instruction_count: observed.instruction_count,
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
    Ok(MicroReproducible {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
