//! Evidence-to-output check for one cold micro fixture.
//!
//! The output roots must match the chain's evidence. This measurement does not
//! open a `.knolo` image and does not run a model. The layout is specified in
//! `spec/KIP-INFER-0049-evidence-output.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, EvidenceOutputReportV1, InferFailure, PlacementPlanV1,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_root, same_text, same_u32, write_exclusive,
};

/// Chain evidence and the output roots claimed for the same request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceObservation {
    pub evidence_root: DigestHex,
    pub bound_evidence_root: DigestHex,
    pub knowledge_image_root: DigestHex,
    pub bound_knowledge_image_root: DigestHex,
    pub query_receipt_root: DigestHex,
    pub bound_query_receipt_root: DigestHex,
    pub reflex_receipt_root: DigestHex,
    pub bound_reflex_receipt_root: DigestHex,
    pub output_token_root: DigestHex,
    pub bound_output_token_root: DigestHex,
    pub output_text_root: DigestHex,
    pub bound_output_text_root: DigestHex,
    pub receipt_root: DigestHex,
    pub chain_root: DigestHex,
    pub prompt_token_count: u32,
    pub output_token_count: u32,
    pub finish_reason: String,
    pub assurance: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the evidence-to-output report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroEvidenceOutput {
    pub plan: PlacementPlanV1,
    pub observed: EvidenceObservation,
    pub report: EvidenceOutputReportV1,
}

/// Record that the output is bound to the chain's evidence. No file is created.
pub fn measure_evidence_output(
    plan: &PlacementPlanV1,
    observed: &EvidenceObservation,
) -> Result<MicroEvidenceOutput, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_evidence_output(&produced)?;
    Ok(produced)
}

/// Recompute the evidence-to-output report from the stored plan and observation.
pub fn verify_evidence_output(measurement: &MicroEvidenceOutput) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "evidence root does not match",
        &report.evidence_root,
        &observed.evidence_root,
    )?;
    same_root(
        "knowledge image root does not match",
        &report.knowledge_image_root,
        &observed.knowledge_image_root,
    )?;
    same_root(
        "query receipt root does not match",
        &report.query_receipt_root,
        &observed.query_receipt_root,
    )?;
    same_root(
        "reflex receipt root does not match",
        &report.reflex_receipt_root,
        &observed.reflex_receipt_root,
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
    same_root(
        "receipt root does not match",
        &report.receipt_root,
        &observed.receipt_root,
    )?;
    same_root(
        "chain root does not match",
        &report.chain_root,
        &observed.chain_root,
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
        "evidence concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "evidence run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "evidence warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "evidence request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "evidence output validation did not match",
        ));
    }
    Ok(())
}

/// Write the evidence-to-output report. The inference receipt is not written.
pub fn write_evidence_output_report(
    base: &Path,
    measurement: &MicroEvidenceOutput,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_evidence_output(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "evidence",
    )
}

fn bind(message: &str, left: &DigestHex, right: &DigestHex) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &EvidenceObservation,
) -> Result<MicroEvidenceOutput, InferFailure> {
    accept_micro_plan(plan, "evidence")?;
    accept_cold_single(
        "evidence",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    bind(
        "the evidence root does not bind the output",
        &observed.evidence_root,
        &observed.bound_evidence_root,
    )?;
    bind(
        "the knowledge image does not bind the output",
        &observed.knowledge_image_root,
        &observed.bound_knowledge_image_root,
    )?;
    bind(
        "the query receipt does not bind the output",
        &observed.query_receipt_root,
        &observed.bound_query_receipt_root,
    )?;
    bind(
        "the reflex receipt does not bind the output",
        &observed.reflex_receipt_root,
        &observed.bound_reflex_receipt_root,
    )?;
    bind(
        "the output token root does not bind the receipt",
        &observed.output_token_root,
        &observed.bound_output_token_root,
    )?;
    bind(
        "the output text root does not bind the receipt",
        &observed.output_text_root,
        &observed.bound_output_text_root,
    )?;
    let report = EvidenceOutputReportV1 {
        evidence_root: observed.evidence_root.clone(),
        knowledge_image_root: observed.knowledge_image_root.clone(),
        query_receipt_root: observed.query_receipt_root.clone(),
        reflex_receipt_root: observed.reflex_receipt_root.clone(),
        receipt_root: observed.receipt_root.clone(),
        chain_root: observed.chain_root.clone(),
        output_token_root: observed.output_token_root.clone(),
        output_text_root: observed.output_text_root.clone(),
        prompt_token_count: observed.prompt_token_count,
        output_token_count: observed.output_token_count,
        finish_reason: observed.finish_reason.clone(),
        assurance: observed.assurance.clone(),
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
    Ok(MicroEvidenceOutput {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
