//! Receipt chain for one cold micro fixture.
//!
//! The chain stores the five links from a Knowledge Image to an agent effect.
//! This measurement does not open a `.knolo` image and does not run a model.
//! The layout is specified in `spec/KIP-INFER-0048-receipt-chain.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, ChainReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_root, same_text, same_u32, write_exclusive,
};

/// Roots for one ordered micro-fixture receipt chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainObservation {
    pub model_runtime_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub kernel_bundle_root: DigestHex,
    pub prompt_token_root: DigestHex,
    pub knowledge_image_root: DigestHex,
    pub knowledge_commit_root: DigestHex,
    pub query_receipt_root: DigestHex,
    pub query_receipt_count: u32,
    pub reflex_receipt_root: DigestHex,
    pub reflex_receipt_count: u32,
    pub receipt_root: DigestHex,
    pub effect_root: DigestHex,
    pub output_token_root: DigestHex,
    pub output_text_root: DigestHex,
    pub prompt_token_count: u32,
    pub output_token_count: u32,
    pub finish_reason: String,
    pub assurance: String,
    pub link_count: u32,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the receipt chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroChain {
    pub plan: PlacementPlanV1,
    pub observed: ChainObservation,
    pub report: ChainReportV1,
}

/// Record the receipt chain. No knowledge image is opened and no file is created.
pub fn measure_receipt_chain(
    plan: &PlacementPlanV1,
    observed: &ChainObservation,
) -> Result<MicroChain, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_receipt_chain(&produced)?;
    Ok(produced)
}

/// Recompute the receipt chain from the stored plan and observation.
pub fn verify_receipt_chain(measurement: &MicroChain) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "model runtime root does not match",
        &report.model_runtime_root,
        &observed.model_runtime_root,
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
        "kernel bundle root does not match",
        &report.kernel_bundle_root,
        &observed.kernel_bundle_root,
    )?;
    same_root(
        "placement root does not match",
        &report.placement_root,
        &measurement.plan.root()?,
    )?;
    same_root(
        "prompt token root does not match",
        &report.prompt_token_root,
        &observed.prompt_token_root,
    )?;
    same_root(
        "knowledge image root does not match",
        &report.knowledge_image_root,
        &observed.knowledge_image_root,
    )?;
    same_root(
        "knowledge commit root does not match",
        &report.knowledge_commit_root,
        &observed.knowledge_commit_root,
    )?;
    same_root(
        "query receipt root does not match",
        &report.query_receipt_root,
        &observed.query_receipt_root,
    )?;
    same_u32(
        "query receipt count does not match",
        report.query_receipt_count,
        observed.query_receipt_count,
    )?;
    same_root(
        "reflex receipt root does not match",
        &report.reflex_receipt_root,
        &observed.reflex_receipt_root,
    )?;
    same_u32(
        "reflex receipt count does not match",
        report.reflex_receipt_count,
        observed.reflex_receipt_count,
    )?;
    same_root(
        "receipt root does not match",
        &report.receipt_root,
        &observed.receipt_root,
    )?;
    same_root(
        "effect root does not match",
        &report.effect_root,
        &observed.effect_root,
    )?;
    same_root(
        "output token root does not match",
        &report.output_token_root,
        &observed.output_token_root,
    )?;
    same_root(
        "output text root does not match",
        &report.output_text_root,
        &observed.output_text_root,
    )?;
    same_u32(
        "prompt token count does not match",
        report.prompt_token_count,
        observed.prompt_token_count,
    )?;
    same_u32(
        "output token count does not match",
        report.output_token_count,
        observed.output_token_count,
    )?;
    same_text(
        "finish reason does not match",
        &report.finish_reason,
        &observed.finish_reason,
    )?;
    same_text(
        "assurance does not match",
        &report.assurance,
        &observed.assurance,
    )?;
    same_u32(
        "link count does not match",
        report.link_count,
        observed.link_count,
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
        "chain concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "chain run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "chain warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "chain request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "chain validation did not match",
        ));
    }
    Ok(())
}

/// Write the receipt chain. The inference receipt is not written.
pub fn write_chain_report(
    base: &Path,
    measurement: &MicroChain,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_receipt_chain(measurement)?;
    write_exclusive(base, receipt_path, &measurement.report.to_bytes()?, "chain")
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ChainObservation,
) -> Result<MicroChain, InferFailure> {
    accept_micro_plan(plan, "chain")?;
    accept_cold_single(
        "chain",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = ChainReportV1 {
        model_runtime_root: observed.model_runtime_root.clone(),
        artifact_root: observed.artifact_root.clone(),
        engine_build_root: observed.engine_build_root.clone(),
        kernel_bundle_root: observed.kernel_bundle_root.clone(),
        placement_root: plan.root()?,
        prompt_token_root: observed.prompt_token_root.clone(),
        knowledge_image_root: observed.knowledge_image_root.clone(),
        knowledge_commit_root: observed.knowledge_commit_root.clone(),
        query_receipt_root: observed.query_receipt_root.clone(),
        query_receipt_count: observed.query_receipt_count,
        reflex_receipt_root: observed.reflex_receipt_root.clone(),
        reflex_receipt_count: observed.reflex_receipt_count,
        receipt_root: observed.receipt_root.clone(),
        effect_root: observed.effect_root.clone(),
        output_token_root: observed.output_token_root.clone(),
        output_text_root: observed.output_text_root.clone(),
        prompt_token_count: observed.prompt_token_count,
        output_token_count: observed.output_token_count,
        finish_reason: observed.finish_reason.clone(),
        assurance: observed.assurance.clone(),
        link_count: observed.link_count,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        validation_result: "verified".into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroChain {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
