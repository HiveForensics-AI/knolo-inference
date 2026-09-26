//! Agent host-effect policy for one cold run of the micro fixture.
//!
//! A missing final receipt is not authorization to act. An unverified
//! compatibility backend is denied. This measurement does not call Agents
//! and does not execute a tool. The layout is specified in
//! `spec/KIP-INFER-0045-agent-effect.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, AgentEffectReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1,
};

use crate::report_io::{accept_cold_single, accept_micro_plan, write_exclusive};

/// Policy inputs from one cold micro-fixture request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentEffectObservation {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub model_runtime_root: DigestHex,
    pub required_model_runtime_root: DigestHex,
    pub required_artifact_root: DigestHex,
    pub knowledge_image_root: DigestHex,
    pub required_knowledge_image_root: DigestHex,
    pub backend: String,
    pub assurance: String,
    pub required_assurance: String,
    pub execution_mode: String,
    pub allowed_execution_modes: Vec<String>,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub max_prompt_tokens: u32,
    pub max_output_tokens: u32,
    pub receipt_present: bool,
}

/// One micro-fixture observation and the host-effect decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroAgentEffect {
    pub plan: PlacementPlanV1,
    pub observed: AgentEffectObservation,
    pub report: AgentEffectReportV1,
}

/// Record the policy decision. No tool runs and no file is created.
pub fn measure_agent_effect(
    plan: &PlacementPlanV1,
    observed: &AgentEffectObservation,
) -> Result<MicroAgentEffect, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_agent_effect(&produced)?;
    Ok(produced)
}

/// Recompute the host-effect report from the stored plan and observation.
pub fn verify_agent_effect(measurement: &MicroAgentEffect) -> Result<(), InferFailure> {
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
    same_root(
        "model runtime root does not match",
        &report.model_runtime_root,
        &observed.model_runtime_root,
    )?;
    same_root(
        "required model runtime root does not match",
        &report.required_model_runtime_root,
        &observed.required_model_runtime_root,
    )?;
    same_root(
        "required artifact root does not match",
        &report.required_artifact_root,
        &observed.required_artifact_root,
    )?;
    same_root(
        "knowledge image root does not match",
        &report.knowledge_image_root,
        &observed.knowledge_image_root,
    )?;
    same_root(
        "required knowledge image root does not match",
        &report.required_knowledge_image_root,
        &observed.required_knowledge_image_root,
    )?;
    same_text("backend does not match", &report.backend, &observed.backend)?;
    same_text(
        "assurance does not match",
        &report.assurance,
        &observed.assurance,
    )?;
    same_text(
        "required assurance does not match",
        &report.required_assurance,
        &observed.required_assurance,
    )?;
    same_text(
        "execution mode does not match",
        &report.execution_mode,
        &observed.execution_mode,
    )?;
    if report.allowed_execution_modes != observed.allowed_execution_modes {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "allowed execution modes do not match",
        ));
    }
    same_text(
        "cache policy does not match",
        &report.cache_policy,
        &observed.cache_policy,
    )?;
    same_u32(
        "agent effect concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "agent effect run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "agent effect warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "agent effect request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    same_u32(
        "prompt token count does not match",
        report.prompt_tokens,
        observed.prompt_tokens,
    )?;
    same_u32(
        "output token count does not match",
        report.output_tokens,
        observed.output_tokens,
    )?;
    same_u32(
        "max prompt tokens do not match",
        report.max_prompt_tokens,
        observed.max_prompt_tokens,
    )?;
    same_u32(
        "max output tokens do not match",
        report.max_output_tokens,
        observed.max_output_tokens,
    )?;
    if report.receipt_present != observed.receipt_present {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "receipt presence does not match",
        ));
    }
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "agent effect validation did not match",
        ));
    }
    Ok(())
}

/// Write the host-effect report. Token ids are not written.
pub fn write_agent_effect_report(
    base: &Path,
    measurement: &MicroAgentEffect,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_agent_effect(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "agent effect",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &AgentEffectObservation,
) -> Result<MicroAgentEffect, InferFailure> {
    accept_micro_plan(plan, "agent effect")?;
    accept_cold_single(
        "agent effect",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let mut report = AgentEffectReportV1 {
        model_image_root: observed.model_image_root.clone(),
        artifact_root: observed.artifact_root.clone(),
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        model_runtime_root: observed.model_runtime_root.clone(),
        required_model_runtime_root: observed.required_model_runtime_root.clone(),
        required_artifact_root: observed.required_artifact_root.clone(),
        knowledge_image_root: observed.knowledge_image_root.clone(),
        required_knowledge_image_root: observed.required_knowledge_image_root.clone(),
        backend: observed.backend.clone(),
        assurance: observed.assurance.clone(),
        required_assurance: observed.required_assurance.clone(),
        execution_mode: observed.execution_mode.clone(),
        allowed_execution_modes: observed.allowed_execution_modes.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        prompt_tokens: observed.prompt_tokens,
        output_tokens: observed.output_tokens,
        max_prompt_tokens: observed.max_prompt_tokens,
        max_output_tokens: observed.max_output_tokens,
        receipt_present: observed.receipt_present,
        decision: "deny".into(),
        reason: "stream-not-authorization".into(),
        validation_result: "recorded".into(),
        extensions: BTreeMap::new(),
    };
    let (decision, reason) = report.computed_decision();
    report.decision = decision.to_string();
    report.reason = reason.to_string();
    report.validate()?;
    Ok(MicroAgentEffect {
        plan: plan.clone(),
        observed: observed.clone(),
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
