//! Pinned reference comparisons for one cold run of the micro fixture.
//!
//! The measurement does not spawn llama.cpp, vLLM, or mistral.rs. It records
//! that the caller pinned the same artifact, quantization, tokenizer,
//! template, sampler, hardware, and prompt and output distributions. The
//! layouts are specified in `spec/KIP-INFER-0040-llama-comparison.md`,
//! `spec/KIP-INFER-0041-vllm-comparison.md`, and
//! `spec/KIP-INFER-0042-mistral-comparison.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, LlamaReportV1, MistralReportV1, PlacementPlanV1,
    VllmReportV1,
};

use crate::report_io::{accept_cold_single, accept_micro_plan, write_exclusive};

/// Roots from one completed cold comparison of the micro fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceObservation {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub quantization: String,
    pub context_tokens: u32,
    pub tokenizer_root: DigestHex,
    pub template_root: DigestHex,
    pub sampler_root: DigestHex,
    pub hardware_probe_root: DigestHex,
    pub prompt_distribution_root: DigestHex,
    pub output_distribution_root: DigestHex,
    pub reference_runtime: String,
    pub reference_family: String,
    pub reference_build_root: DigestHex,
    pub reference_artifact_root: DigestHex,
    pub reference_quantization: String,
    pub reference_tokenizer_root: DigestHex,
    pub reference_template_root: DigestHex,
    pub reference_sampler_root: DigestHex,
    pub reference_hardware_root: DigestHex,
    pub reference_prompt_distribution_root: DigestHex,
    pub reference_output_distribution_root: DigestHex,
    pub reference_verification_class: String,
}

/// One micro-fixture observation and the llama.cpp comparison it records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroLlama {
    pub plan: PlacementPlanV1,
    pub observed: ReferenceObservation,
    pub report: LlamaReportV1,
}

/// One micro-fixture observation and the vLLM comparison it records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroVllm {
    pub plan: PlacementPlanV1,
    pub observed: ReferenceObservation,
    pub report: VllmReportV1,
}

/// One micro-fixture observation and the mistral.rs comparison it records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroMistral {
    pub plan: PlacementPlanV1,
    pub observed: ReferenceObservation,
    pub report: MistralReportV1,
}

/// Record a llama.cpp comparison. No process is spawned and no file is created.
pub fn measure_llama_comparison(
    plan: &PlacementPlanV1,
    observed: &ReferenceObservation,
) -> Result<MicroLlama, InferFailure> {
    let produced = assemble_llama(plan, observed)?;
    verify_llama_comparison(&produced)?;
    Ok(produced)
}

/// Recompute the llama.cpp report from the stored plan and observation.
pub fn verify_llama_comparison(measurement: &MicroLlama) -> Result<(), InferFailure> {
    agree_llama(measurement)?;
    let recomputed = assemble_llama(&measurement.plan, &measurement.observed)?;
    same_bytes(
        "llama validation did not match",
        &recomputed.report.to_bytes()?,
        &measurement.report.to_bytes()?,
    )
}

/// Write the llama.cpp report. The plan and the reference runtime are not executed.
pub fn write_llama_report(
    base: &Path,
    measurement: &MicroLlama,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_llama_comparison(measurement)?;
    write_exclusive(base, receipt_path, &measurement.report.to_bytes()?, "llama")
}

/// Record a vLLM comparison. No process is spawned and no file is created.
pub fn measure_vllm_comparison(
    plan: &PlacementPlanV1,
    observed: &ReferenceObservation,
) -> Result<MicroVllm, InferFailure> {
    let produced = assemble_vllm(plan, observed)?;
    verify_vllm_comparison(&produced)?;
    Ok(produced)
}

/// Recompute the vLLM report from the stored plan and observation.
pub fn verify_vllm_comparison(measurement: &MicroVllm) -> Result<(), InferFailure> {
    agree_vllm(measurement)?;
    let recomputed = assemble_vllm(&measurement.plan, &measurement.observed)?;
    same_bytes(
        "vllm validation did not match",
        &recomputed.report.to_bytes()?,
        &measurement.report.to_bytes()?,
    )
}

/// Write the vLLM report. The plan and the reference runtime are not executed.
pub fn write_vllm_report(
    base: &Path,
    measurement: &MicroVllm,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_vllm_comparison(measurement)?;
    write_exclusive(base, receipt_path, &measurement.report.to_bytes()?, "vllm")
}

/// Record a mistral.rs comparison. No process is spawned and no file is created.
pub fn measure_mistral_comparison(
    plan: &PlacementPlanV1,
    observed: &ReferenceObservation,
) -> Result<MicroMistral, InferFailure> {
    let produced = assemble_mistral(plan, observed)?;
    verify_mistral_comparison(&produced)?;
    Ok(produced)
}

/// Recompute the mistral.rs report from the stored plan and observation.
pub fn verify_mistral_comparison(measurement: &MicroMistral) -> Result<(), InferFailure> {
    agree_mistral(measurement)?;
    let recomputed = assemble_mistral(&measurement.plan, &measurement.observed)?;
    same_bytes(
        "mistral validation did not match",
        &recomputed.report.to_bytes()?,
        &measurement.report.to_bytes()?,
    )
}

/// Write the mistral.rs report. The plan and the reference runtime are not executed.
pub fn write_mistral_report(
    base: &Path,
    measurement: &MicroMistral,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_mistral_comparison(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "mistral",
    )
}

fn assemble_llama(
    plan: &PlacementPlanV1,
    observed: &ReferenceObservation,
) -> Result<MicroLlama, InferFailure> {
    gate(plan, observed, "llama")?;
    let report = llama_report(plan, observed)?;
    report.validate()?;
    Ok(MicroLlama {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}

fn assemble_vllm(
    plan: &PlacementPlanV1,
    observed: &ReferenceObservation,
) -> Result<MicroVllm, InferFailure> {
    gate(plan, observed, "vllm")?;
    let report = vllm_report(plan, observed)?;
    report.validate()?;
    Ok(MicroVllm {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}

fn assemble_mistral(
    plan: &PlacementPlanV1,
    observed: &ReferenceObservation,
) -> Result<MicroMistral, InferFailure> {
    gate(plan, observed, "mistral")?;
    let report = mistral_report(plan, observed)?;
    report.validate()?;
    Ok(MicroMistral {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}

fn gate(
    plan: &PlacementPlanV1,
    observed: &ReferenceObservation,
    noun: &str,
) -> Result<(), InferFailure> {
    accept_micro_plan(plan, noun)?;
    accept_cold_single(
        noun,
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )
}

fn llama_report(
    plan: &PlacementPlanV1,
    observed: &ReferenceObservation,
) -> Result<LlamaReportV1, InferFailure> {
    Ok(LlamaReportV1 {
        model_image_root: observed.model_image_root.clone(),
        artifact_root: observed.artifact_root.clone(),
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        quantization: observed.quantization.clone(),
        context_tokens: observed.context_tokens,
        tokenizer_root: observed.tokenizer_root.clone(),
        template_root: observed.template_root.clone(),
        sampler_root: observed.sampler_root.clone(),
        hardware_probe_root: observed.hardware_probe_root.clone(),
        prompt_distribution_root: observed.prompt_distribution_root.clone(),
        output_distribution_root: observed.output_distribution_root.clone(),
        reference_runtime: observed.reference_runtime.clone(),
        reference_family: observed.reference_family.clone(),
        reference_build_root: observed.reference_build_root.clone(),
        reference_artifact_root: observed.reference_artifact_root.clone(),
        reference_quantization: observed.reference_quantization.clone(),
        reference_tokenizer_root: observed.reference_tokenizer_root.clone(),
        reference_template_root: observed.reference_template_root.clone(),
        reference_sampler_root: observed.reference_sampler_root.clone(),
        reference_hardware_root: observed.reference_hardware_root.clone(),
        reference_prompt_distribution_root: observed.reference_prompt_distribution_root.clone(),
        reference_output_distribution_root: observed.reference_output_distribution_root.clone(),
        reference_verification_class: observed.reference_verification_class.clone(),
        validation_result: "recorded".into(),
        extensions: BTreeMap::new(),
    })
}

fn vllm_report(
    plan: &PlacementPlanV1,
    observed: &ReferenceObservation,
) -> Result<VllmReportV1, InferFailure> {
    Ok(VllmReportV1 {
        model_image_root: observed.model_image_root.clone(),
        artifact_root: observed.artifact_root.clone(),
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        quantization: observed.quantization.clone(),
        context_tokens: observed.context_tokens,
        tokenizer_root: observed.tokenizer_root.clone(),
        template_root: observed.template_root.clone(),
        sampler_root: observed.sampler_root.clone(),
        hardware_probe_root: observed.hardware_probe_root.clone(),
        prompt_distribution_root: observed.prompt_distribution_root.clone(),
        output_distribution_root: observed.output_distribution_root.clone(),
        reference_runtime: observed.reference_runtime.clone(),
        reference_family: observed.reference_family.clone(),
        reference_build_root: observed.reference_build_root.clone(),
        reference_artifact_root: observed.reference_artifact_root.clone(),
        reference_quantization: observed.reference_quantization.clone(),
        reference_tokenizer_root: observed.reference_tokenizer_root.clone(),
        reference_template_root: observed.reference_template_root.clone(),
        reference_sampler_root: observed.reference_sampler_root.clone(),
        reference_hardware_root: observed.reference_hardware_root.clone(),
        reference_prompt_distribution_root: observed.reference_prompt_distribution_root.clone(),
        reference_output_distribution_root: observed.reference_output_distribution_root.clone(),
        reference_verification_class: observed.reference_verification_class.clone(),
        validation_result: "recorded".into(),
        extensions: BTreeMap::new(),
    })
}

fn mistral_report(
    plan: &PlacementPlanV1,
    observed: &ReferenceObservation,
) -> Result<MistralReportV1, InferFailure> {
    Ok(MistralReportV1 {
        model_image_root: observed.model_image_root.clone(),
        artifact_root: observed.artifact_root.clone(),
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        quantization: observed.quantization.clone(),
        context_tokens: observed.context_tokens,
        tokenizer_root: observed.tokenizer_root.clone(),
        template_root: observed.template_root.clone(),
        sampler_root: observed.sampler_root.clone(),
        hardware_probe_root: observed.hardware_probe_root.clone(),
        prompt_distribution_root: observed.prompt_distribution_root.clone(),
        output_distribution_root: observed.output_distribution_root.clone(),
        reference_runtime: observed.reference_runtime.clone(),
        reference_family: observed.reference_family.clone(),
        reference_build_root: observed.reference_build_root.clone(),
        reference_artifact_root: observed.reference_artifact_root.clone(),
        reference_quantization: observed.reference_quantization.clone(),
        reference_tokenizer_root: observed.reference_tokenizer_root.clone(),
        reference_template_root: observed.reference_template_root.clone(),
        reference_sampler_root: observed.reference_sampler_root.clone(),
        reference_hardware_root: observed.reference_hardware_root.clone(),
        reference_prompt_distribution_root: observed.reference_prompt_distribution_root.clone(),
        reference_output_distribution_root: observed.reference_output_distribution_root.clone(),
        reference_verification_class: observed.reference_verification_class.clone(),
        validation_result: "recorded".into(),
        extensions: BTreeMap::new(),
    })
}

fn agree_llama(measurement: &MicroLlama) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    agree_common(
        "llama",
        &report.model_image_root,
        &report.artifact_root,
        &report.engine_build_root,
        &report.placement_root,
        &report.execution_mode,
        &report.cache_policy,
        report.concurrency,
        report.run_count,
        &report.warm_state,
        report.request_count,
        &report.quantization,
        report.context_tokens,
        &report.tokenizer_root,
        &report.template_root,
        &report.sampler_root,
        &report.hardware_probe_root,
        &report.prompt_distribution_root,
        &report.output_distribution_root,
        &report.reference_runtime,
        &report.reference_family,
        &report.reference_build_root,
        &report.reference_artifact_root,
        &report.reference_quantization,
        &report.reference_tokenizer_root,
        &report.reference_template_root,
        &report.reference_sampler_root,
        &report.reference_hardware_root,
        &report.reference_prompt_distribution_root,
        &report.reference_output_distribution_root,
        &report.reference_verification_class,
        &measurement.plan,
        observed,
    )
}

fn agree_vllm(measurement: &MicroVllm) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    agree_common(
        "vllm",
        &report.model_image_root,
        &report.artifact_root,
        &report.engine_build_root,
        &report.placement_root,
        &report.execution_mode,
        &report.cache_policy,
        report.concurrency,
        report.run_count,
        &report.warm_state,
        report.request_count,
        &report.quantization,
        report.context_tokens,
        &report.tokenizer_root,
        &report.template_root,
        &report.sampler_root,
        &report.hardware_probe_root,
        &report.prompt_distribution_root,
        &report.output_distribution_root,
        &report.reference_runtime,
        &report.reference_family,
        &report.reference_build_root,
        &report.reference_artifact_root,
        &report.reference_quantization,
        &report.reference_tokenizer_root,
        &report.reference_template_root,
        &report.reference_sampler_root,
        &report.reference_hardware_root,
        &report.reference_prompt_distribution_root,
        &report.reference_output_distribution_root,
        &report.reference_verification_class,
        &measurement.plan,
        observed,
    )
}

fn agree_mistral(measurement: &MicroMistral) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    agree_common(
        "mistral",
        &report.model_image_root,
        &report.artifact_root,
        &report.engine_build_root,
        &report.placement_root,
        &report.execution_mode,
        &report.cache_policy,
        report.concurrency,
        report.run_count,
        &report.warm_state,
        report.request_count,
        &report.quantization,
        report.context_tokens,
        &report.tokenizer_root,
        &report.template_root,
        &report.sampler_root,
        &report.hardware_probe_root,
        &report.prompt_distribution_root,
        &report.output_distribution_root,
        &report.reference_runtime,
        &report.reference_family,
        &report.reference_build_root,
        &report.reference_artifact_root,
        &report.reference_quantization,
        &report.reference_tokenizer_root,
        &report.reference_template_root,
        &report.reference_sampler_root,
        &report.reference_hardware_root,
        &report.reference_prompt_distribution_root,
        &report.reference_output_distribution_root,
        &report.reference_verification_class,
        &measurement.plan,
        observed,
    )
}

#[allow(clippy::too_many_arguments)]
fn agree_common(
    noun: &str,
    model_image_root: &DigestHex,
    artifact_root: &DigestHex,
    engine_build_root: &DigestHex,
    placement_root: &DigestHex,
    execution_mode: &str,
    cache_policy: &str,
    concurrency: u32,
    run_count: u32,
    warm_state: &str,
    request_count: u32,
    quantization: &str,
    context_tokens: u32,
    tokenizer_root: &DigestHex,
    template_root: &DigestHex,
    sampler_root: &DigestHex,
    hardware_probe_root: &DigestHex,
    prompt_distribution_root: &DigestHex,
    output_distribution_root: &DigestHex,
    reference_runtime: &str,
    reference_family: &str,
    reference_build_root: &DigestHex,
    reference_artifact_root: &DigestHex,
    reference_quantization: &str,
    reference_tokenizer_root: &DigestHex,
    reference_template_root: &DigestHex,
    reference_sampler_root: &DigestHex,
    reference_hardware_root: &DigestHex,
    reference_prompt_distribution_root: &DigestHex,
    reference_output_distribution_root: &DigestHex,
    reference_verification_class: &str,
    plan: &PlacementPlanV1,
    observed: &ReferenceObservation,
) -> Result<(), InferFailure> {
    same_root(
        "model image root does not match",
        model_image_root,
        &observed.model_image_root,
    )?;
    same_root(
        "artifact root does not match",
        artifact_root,
        &observed.artifact_root,
    )?;
    same_root(
        "engine build root does not match",
        engine_build_root,
        &observed.engine_build_root,
    )?;
    same_root(
        "placement root does not match",
        placement_root,
        &plan.root()?,
    )?;
    same_text(
        "execution mode does not match",
        execution_mode,
        &observed.execution_mode,
    )?;
    same_text(
        "cache policy does not match",
        cache_policy,
        &observed.cache_policy,
    )?;
    same_u32(
        &format!("{noun} concurrency does not match"),
        concurrency,
        observed.concurrency,
    )?;
    same_u32(
        &format!("{noun} run count does not match"),
        run_count,
        observed.run_count,
    )?;
    same_text(
        &format!("{noun} warm state does not match"),
        warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        &format!("{noun} request count does not match"),
        request_count,
        observed.request_count,
    )?;
    same_text(
        "quantization does not match",
        quantization,
        &observed.quantization,
    )?;
    same_u32(
        "context length does not match",
        context_tokens,
        observed.context_tokens,
    )?;
    same_root(
        "tokenizer root does not match",
        tokenizer_root,
        &observed.tokenizer_root,
    )?;
    same_root(
        "template root does not match",
        template_root,
        &observed.template_root,
    )?;
    same_root(
        "sampler root does not match",
        sampler_root,
        &observed.sampler_root,
    )?;
    same_root(
        "hardware probe root does not match",
        hardware_probe_root,
        &observed.hardware_probe_root,
    )?;
    same_root(
        "prompt distribution root does not match",
        prompt_distribution_root,
        &observed.prompt_distribution_root,
    )?;
    same_root(
        "output distribution root does not match",
        output_distribution_root,
        &observed.output_distribution_root,
    )?;
    same_text(
        "reference runtime does not match",
        reference_runtime,
        &observed.reference_runtime,
    )?;
    same_text(
        "reference family does not match",
        reference_family,
        &observed.reference_family,
    )?;
    same_root(
        "reference build root does not match",
        reference_build_root,
        &observed.reference_build_root,
    )?;
    same_root(
        "reference artifact root does not match",
        reference_artifact_root,
        &observed.reference_artifact_root,
    )?;
    same_text(
        "reference quantization does not match",
        reference_quantization,
        &observed.reference_quantization,
    )?;
    same_root(
        "reference tokenizer root does not match",
        reference_tokenizer_root,
        &observed.reference_tokenizer_root,
    )?;
    same_root(
        "reference template root does not match",
        reference_template_root,
        &observed.reference_template_root,
    )?;
    same_root(
        "reference sampler root does not match",
        reference_sampler_root,
        &observed.reference_sampler_root,
    )?;
    same_root(
        "reference hardware root does not match",
        reference_hardware_root,
        &observed.reference_hardware_root,
    )?;
    same_root(
        "reference prompt distribution root does not match",
        reference_prompt_distribution_root,
        &observed.reference_prompt_distribution_root,
    )?;
    same_root(
        "reference output distribution root does not match",
        reference_output_distribution_root,
        &observed.reference_output_distribution_root,
    )?;
    same_text(
        "reference verification class does not match",
        reference_verification_class,
        &observed.reference_verification_class,
    )?;
    Ok(())
}

fn same_bytes(message: &str, recomputed: &[u8], stored: &[u8]) -> Result<(), InferFailure> {
    if recomputed != stored {
        Err(fail(ErrorCode::ContractInvalid, message))
    } else {
        Ok(())
    }
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
