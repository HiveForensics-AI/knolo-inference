//! Worker sandbox profile for one cold micro fixture.
//!
//! The report stores the profile. This measurement does not apply it. The
//! layout is specified in `spec/KIP-INFER-0058-sandbox-profile.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, PlacementPlanV1, SandboxReportV1, MAX_SANDBOX_MEMORY,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_root, same_text, same_u32, same_u64,
    write_exclusive,
};

/// Caller-supplied worker profile for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxObservation {
    pub engine_build_root: DigestHex,
    pub user_class: String,
    pub network: String,
    pub model_cas: String,
    pub scratch: String,
    pub seccomp: String,
    pub memory_limit_bytes: u64,
    pub process_group: String,
    pub parent_death: String,
    pub shared_memory_bytes: u64,
    pub arguments: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the sandbox report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroSandbox {
    pub plan: PlacementPlanV1,
    pub observed: SandboxObservation,
    pub report: SandboxReportV1,
}

/// Record the worker profile. The profile is not applied.
pub fn measure_sandbox_profile(
    plan: &PlacementPlanV1,
    observed: &SandboxObservation,
) -> Result<MicroSandbox, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_sandbox_profile(&produced)?;
    Ok(produced)
}

/// Recompute the sandbox report from the stored plan and observation.
pub fn verify_sandbox_profile(measurement: &MicroSandbox) -> Result<(), InferFailure> {
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
    same_text(
        "user class does not match",
        &report.user_class,
        &observed.user_class,
    )?;
    same_text("network does not match", &report.network, &observed.network)?;
    same_text(
        "model cas does not match",
        &report.model_cas,
        &observed.model_cas,
    )?;
    same_text("scratch does not match", &report.scratch, &observed.scratch)?;
    same_text("seccomp does not match", &report.seccomp, &observed.seccomp)?;
    same_u64(
        "memory limit does not match",
        report.memory_limit_bytes,
        observed.memory_limit_bytes,
    )?;
    same_text(
        "process group does not match",
        &report.process_group,
        &observed.process_group,
    )?;
    same_text(
        "parent death does not match",
        &report.parent_death,
        &observed.parent_death,
    )?;
    same_u64(
        "shared memory does not match",
        report.shared_memory_bytes,
        observed.shared_memory_bytes,
    )?;
    same_text(
        "arguments do not match",
        &report.arguments,
        &observed.arguments,
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
        "sandbox concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "sandbox run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "sandbox warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "sandbox request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "sandbox validation did not match",
        ));
    }
    Ok(())
}

/// Write the sandbox report. The profile is not applied.
pub fn write_sandbox_report(
    base: &Path,
    measurement: &MicroSandbox,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_sandbox_profile(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "sandbox",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &SandboxObservation,
) -> Result<MicroSandbox, InferFailure> {
    accept_micro_plan(plan, "sandbox")?;
    accept_cold_single(
        "sandbox",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if observed.memory_limit_bytes > MAX_SANDBOX_MEMORY {
        return Err(fail(
            ErrorCode::InsufficientMemory,
            "worker memory exceeds 64 MiB",
        ));
    }
    let report = SandboxReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        user_class: observed.user_class.clone(),
        network: observed.network.clone(),
        model_cas: observed.model_cas.clone(),
        scratch: observed.scratch.clone(),
        seccomp: observed.seccomp.clone(),
        memory_limit_bytes: observed.memory_limit_bytes,
        process_group: observed.process_group.clone(),
        parent_death: observed.parent_death.clone(),
        shared_memory_bytes: observed.shared_memory_bytes,
        arguments: observed.arguments.clone(),
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
    Ok(MicroSandbox {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
