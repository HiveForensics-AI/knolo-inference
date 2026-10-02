//! Core and Reflex binding for one cold run of the micro fixture.
//!
//! The host supplies the Knowledge Image root and the receipt ids. This
//! measurement does not open a `.knolo` image. The layout is specified in
//! `spec/KIP-INFER-0044-evidence-composition.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, CompositionReportV1, DigestHex, ErrorCode, EvidenceBindingV1, InferFailure,
    PlacementPlanV1,
};

use crate::report_io::{accept_cold_single, accept_micro_plan, write_exclusive};

/// Roots and counts from one host-supplied binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositionObservation {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub knowledge_image_root: DigestHex,
    pub knowledge_commit_root: DigestHex,
    pub context_root: DigestHex,
    pub query_receipt_count: u32,
    pub reflex_receipt_count: u32,
    pub ordered_evidence_count: u32,
}

/// One micro-fixture observation, the host binding, and the composition report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroComposition {
    pub plan: PlacementPlanV1,
    pub observed: CompositionObservation,
    pub binding: EvidenceBindingV1,
    pub report: CompositionReportV1,
}

/// Record the binding. No image is opened and no file is created.
pub fn measure_evidence_composition(
    plan: &PlacementPlanV1,
    observed: &CompositionObservation,
    binding: &EvidenceBindingV1,
) -> Result<MicroComposition, InferFailure> {
    let produced = assemble(plan, observed, binding)?;
    verify_evidence_composition(&produced)?;
    Ok(produced)
}

/// Recompute the composition report from the stored plan, observation, and binding.
pub fn verify_evidence_composition(measurement: &MicroComposition) -> Result<(), InferFailure> {
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
        "evidence root does not match",
        &report.evidence_root,
        &measurement.binding.root()?,
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
        "context root does not match",
        &report.context_root,
        &observed.context_root,
    )?;
    same_u32(
        "query receipt count does not match",
        report.query_receipt_count,
        observed.query_receipt_count,
    )?;
    same_u32(
        "reflex receipt count does not match",
        report.reflex_receipt_count,
        observed.reflex_receipt_count,
    )?;
    same_u32(
        "ordered evidence count does not match",
        report.ordered_evidence_count,
        observed.ordered_evidence_count,
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
        "composition concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "composition run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "composition warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "composition request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(
        &measurement.plan,
        &measurement.observed,
        &measurement.binding,
    )?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "composition validation did not match",
        ));
    }
    Ok(())
}

/// Write the composition report. The binding is not written.
pub fn write_composition_report(
    base: &Path,
    measurement: &MicroComposition,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_evidence_composition(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "composition",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &CompositionObservation,
    binding: &EvidenceBindingV1,
) -> Result<MicroComposition, InferFailure> {
    accept_micro_plan(plan, "composition")?;
    accept_cold_single(
        "composition",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    binding.validate()?;
    let image = binding.knowledge_image_root.as_ref().ok_or_else(|| {
        fail(
            ErrorCode::ContractInvalid,
            "the composition has no knowledge image",
        )
    })?;
    let commit = binding.knowledge_commit_root.as_ref().ok_or_else(|| {
        fail(
            ErrorCode::ContractInvalid,
            "the composition has no knowledge commit",
        )
    })?;
    if image == commit {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "the knowledge commit repeats the image",
        ));
    }
    if binding.query_receipt_ids.is_empty() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "the composition has no query receipt",
        ));
    }
    if binding.reflex_receipt_ids.is_empty() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "the composition has no reflex receipt",
        ));
    }
    if binding.ordered_evidence_ids.is_empty() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "the composition has no evidence id",
        ));
    }
    same_root(
        "knowledge image root does not match",
        image,
        &observed.knowledge_image_root,
    )?;
    same_root(
        "knowledge commit root does not match",
        commit,
        &observed.knowledge_commit_root,
    )?;
    same_root(
        "context root does not match",
        &binding.context_root,
        &observed.context_root,
    )?;
    let query_count = list_count(binding.query_receipt_ids.len())?;
    let reflex_count = list_count(binding.reflex_receipt_ids.len())?;
    let evidence_count = list_count(binding.ordered_evidence_ids.len())?;
    same_u32(
        "query receipt count does not match",
        query_count,
        observed.query_receipt_count,
    )?;
    same_u32(
        "reflex receipt count does not match",
        reflex_count,
        observed.reflex_receipt_count,
    )?;
    same_u32(
        "ordered evidence count does not match",
        evidence_count,
        observed.ordered_evidence_count,
    )?;
    let report = CompositionReportV1 {
        model_image_root: observed.model_image_root.clone(),
        artifact_root: observed.artifact_root.clone(),
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        evidence_root: binding.root()?,
        knowledge_image_root: image.clone(),
        knowledge_commit_root: commit.clone(),
        context_root: binding.context_root.clone(),
        query_receipt_count: query_count,
        reflex_receipt_count: reflex_count,
        ordered_evidence_count: evidence_count,
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
    Ok(MicroComposition {
        plan: plan.clone(),
        observed: observed.clone(),
        binding: binding.clone(),
        report,
    })
}

fn list_count(len: usize) -> Result<u32, InferFailure> {
    u32::try_from(len).map_err(|_| fail(ErrorCode::ContractInvalid, "evidence list is too large"))
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
