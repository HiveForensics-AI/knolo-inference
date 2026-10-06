//! Reference comparisons and the micro-fixture recipe mark.
//!
//! The three comparisons record a pinned external runtime against the same
//! micro-fixture artifact. They do not spawn that runtime. The recipe mark
//! records `experimental`, `conformant`, or `blessed`. The layouts are
//! specified in `spec/KIP-INFER-0040-llama-comparison.md`,
//! `spec/KIP-INFER-0041-vllm-comparison.md`,
//! `spec/KIP-INFER-0042-mistral-comparison.md`, and
//! `spec/KIP-INFER-0043-recipe-status.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    bounded_text, cbor_digest, cbor_text, cbor_u32, expect_kind_version, one_of, Fields,
};

use super::common::{put_extensions, Builder};
use super::lifecycle::cold_single;

pub const LLAMA_KIND: &str = "knolo.infer.llama-report";
pub const VLLM_KIND: &str = "knolo.infer.vllm-report";
pub const MISTRAL_KIND: &str = "knolo.infer.mistral-report";
pub const RECIPE_KIND: &str = "knolo.infer.recipe-report";

pub const MICRO_COMPARISON_CONTEXT: u32 = 16;
pub const MICRO_COMPARISON_QUANTIZATION: &str = "f32";
pub const LLAMA_RUNTIME: &str = "llama.cpp";
pub const LLAMA_FAMILY: &str = "gguf";
pub const VLLM_RUNTIME: &str = "vllm";
pub const VLLM_FAMILY: &str = "safetensors";
pub const MISTRAL_RUNTIME: &str = "mistral.rs";
pub const MISTRAL_FAMILY: &str = "rust";
pub const SIDECAR_ARTIFACT_VERIFIED: &str = "sidecar-artifact-verified";
pub const BACKEND_REPORTED: &str = "backend-reported";
pub const NATIVE_VERIFIED: &str = "native-verified";
pub const RECIPE_ADAPTER: &str = "knolo.micro.v1";

struct ReferenceBody {
    model_image_root: DigestHex,
    artifact_root: DigestHex,
    engine_build_root: DigestHex,
    placement_root: DigestHex,
    execution_mode: String,
    cache_policy: String,
    concurrency: u32,
    run_count: u32,
    warm_state: String,
    request_count: u32,
    quantization: String,
    context_tokens: u32,
    tokenizer_root: DigestHex,
    template_root: DigestHex,
    sampler_root: DigestHex,
    hardware_probe_root: DigestHex,
    prompt_distribution_root: DigestHex,
    output_distribution_root: DigestHex,
    reference_runtime: String,
    reference_family: String,
    reference_build_root: DigestHex,
    reference_artifact_root: DigestHex,
    reference_quantization: String,
    reference_tokenizer_root: DigestHex,
    reference_template_root: DigestHex,
    reference_sampler_root: DigestHex,
    reference_hardware_root: DigestHex,
    reference_prompt_distribution_root: DigestHex,
    reference_output_distribution_root: DigestHex,
    reference_verification_class: String,
    validation_result: String,
    extensions: BTreeMap<String, CborValue>,
}

fn validate_reference(
    noun: &str,
    runtime: &str,
    family: &str,
    class: &str,
    body: &ReferenceBody,
) -> Result<(), InferFailure> {
    one_of("validationResult", &body.validation_result, &["recorded"])?;
    one_of(
        "executionMode",
        &body.execution_mode,
        &["isolated-replay", "pinned"],
    )?;
    cold_single(
        noun,
        &body.cache_policy,
        body.concurrency,
        body.run_count,
        &body.warm_state,
        body.request_count,
    )?;
    if body.quantization != MICRO_COMPARISON_QUANTIZATION
        || body.context_tokens != MICRO_COMPARISON_CONTEXT
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("the {noun} comparison is the micro fixture"),
        ));
    }
    one_of("referenceRuntime", &body.reference_runtime, &[runtime])?;
    if body.reference_family != family {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("{runtime} comparison is a {family} pin"),
        ));
    }
    if body.reference_verification_class == NATIVE_VERIFIED {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "a compatibility backend is not native-verified",
        ));
    }
    one_of(
        "referenceVerificationClass",
        &body.reference_verification_class,
        &[class],
    )?;
    if body.reference_build_root == body.engine_build_root {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "the reference build is the engine build",
        ));
    }
    same_root(
        "artifact root does not match",
        &body.reference_artifact_root,
        &body.artifact_root,
    )?;
    if body.reference_quantization != body.quantization {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "quantization does not match",
        ));
    }
    same_root(
        "tokenizer root does not match",
        &body.reference_tokenizer_root,
        &body.tokenizer_root,
    )?;
    same_root(
        "template root does not match",
        &body.reference_template_root,
        &body.template_root,
    )?;
    same_root(
        "sampler root does not match",
        &body.reference_sampler_root,
        &body.sampler_root,
    )?;
    same_root(
        "hardware probe root does not match",
        &body.reference_hardware_root,
        &body.hardware_probe_root,
    )?;
    same_root(
        "prompt distribution root does not match",
        &body.reference_prompt_distribution_root,
        &body.prompt_distribution_root,
    )?;
    same_root(
        "output distribution root does not match",
        &body.reference_output_distribution_root,
        &body.output_distribution_root,
    )?;
    Ok(())
}

fn write_reference(kind: &str, body: &ReferenceBody) -> Result<CborValue, InferFailure> {
    let mut b = Builder::typed(kind);
    b.put("artifactRoot", cbor_digest(&body.artifact_root));
    b.put("cachePolicy", cbor_text(&body.cache_policy));
    b.put("concurrency", cbor_u32(body.concurrency));
    b.put("contextTokens", cbor_u32(body.context_tokens));
    b.put("engineBuildRoot", cbor_digest(&body.engine_build_root));
    b.put("executionMode", cbor_text(&body.execution_mode));
    put_extensions(&mut b, &body.extensions)?;
    b.put("hardwareProbeRoot", cbor_digest(&body.hardware_probe_root));
    b.put("modelImageRoot", cbor_digest(&body.model_image_root));
    b.put(
        "outputDistributionRoot",
        cbor_digest(&body.output_distribution_root),
    );
    b.put("placementRoot", cbor_digest(&body.placement_root));
    b.put(
        "promptDistributionRoot",
        cbor_digest(&body.prompt_distribution_root),
    );
    b.put("quantization", cbor_text(&body.quantization));
    b.put(
        "referenceArtifactRoot",
        cbor_digest(&body.reference_artifact_root),
    );
    b.put(
        "referenceBuildRoot",
        cbor_digest(&body.reference_build_root),
    );
    b.put("referenceFamily", cbor_text(&body.reference_family));
    b.put(
        "referenceHardwareRoot",
        cbor_digest(&body.reference_hardware_root),
    );
    b.put(
        "referenceOutputDistributionRoot",
        cbor_digest(&body.reference_output_distribution_root),
    );
    b.put(
        "referencePromptDistributionRoot",
        cbor_digest(&body.reference_prompt_distribution_root),
    );
    b.put(
        "referenceQuantization",
        cbor_text(&body.reference_quantization),
    );
    b.put("referenceRuntime", cbor_text(&body.reference_runtime));
    b.put(
        "referenceSamplerRoot",
        cbor_digest(&body.reference_sampler_root),
    );
    b.put(
        "referenceTemplateRoot",
        cbor_digest(&body.reference_template_root),
    );
    b.put(
        "referenceTokenizerRoot",
        cbor_digest(&body.reference_tokenizer_root),
    );
    b.put(
        "referenceVerificationClass",
        cbor_text(&body.reference_verification_class),
    );
    b.put("requestCount", cbor_u32(body.request_count));
    b.put("runCount", cbor_u32(body.run_count));
    b.put("samplerRoot", cbor_digest(&body.sampler_root));
    b.put("templateRoot", cbor_digest(&body.template_root));
    b.put("tokenizerRoot", cbor_digest(&body.tokenizer_root));
    b.put("validationResult", cbor_text(&body.validation_result));
    b.put("warmState", cbor_text(&body.warm_state));
    Ok(b.finish())
}

fn read_reference(fields: &mut Fields) -> Result<ReferenceBody, InferFailure> {
    Ok(ReferenceBody {
        artifact_root: fields.digest("artifactRoot")?,
        cache_policy: fields.text("cachePolicy")?,
        concurrency: fields.u32("concurrency")?,
        context_tokens: fields.u32("contextTokens")?,
        engine_build_root: fields.digest("engineBuildRoot")?,
        execution_mode: fields.text("executionMode")?,
        extensions: fields.extensions()?,
        hardware_probe_root: fields.digest("hardwareProbeRoot")?,
        model_image_root: fields.digest("modelImageRoot")?,
        output_distribution_root: fields.digest("outputDistributionRoot")?,
        placement_root: fields.digest("placementRoot")?,
        prompt_distribution_root: fields.digest("promptDistributionRoot")?,
        quantization: fields.text("quantization")?,
        reference_artifact_root: fields.digest("referenceArtifactRoot")?,
        reference_build_root: fields.digest("referenceBuildRoot")?,
        reference_family: fields.text("referenceFamily")?,
        reference_hardware_root: fields.digest("referenceHardwareRoot")?,
        reference_output_distribution_root: fields.digest("referenceOutputDistributionRoot")?,
        reference_prompt_distribution_root: fields.digest("referencePromptDistributionRoot")?,
        reference_quantization: fields.text("referenceQuantization")?,
        reference_runtime: fields.text("referenceRuntime")?,
        reference_sampler_root: fields.digest("referenceSamplerRoot")?,
        reference_template_root: fields.digest("referenceTemplateRoot")?,
        reference_tokenizer_root: fields.digest("referenceTokenizerRoot")?,
        reference_verification_class: fields.text("referenceVerificationClass")?,
        request_count: fields.u32("requestCount")?,
        run_count: fields.u32("runCount")?,
        sampler_root: fields.digest("samplerRoot")?,
        template_root: fields.digest("templateRoot")?,
        tokenizer_root: fields.digest("tokenizerRoot")?,
        validation_result: fields.text("validationResult")?,
        warm_state: fields.text("warmState")?,
    })
}

fn same_root(message: &str, left: &DigestHex, right: &DigestHex) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

macro_rules! reference_report {
    ($name:ident, $runtime:expr, $family:expr, $class:expr, $noun:literal, $kind:expr, $domain:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $name {
            pub model_image_root: DigestHex,
            pub artifact_root: DigestHex,
            pub engine_build_root: DigestHex,
            pub placement_root: DigestHex,
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
            pub validation_result: String,
            pub extensions: BTreeMap<String, CborValue>,
        }

        impl $name {
            fn body(&self) -> ReferenceBody {
                ReferenceBody {
                    model_image_root: self.model_image_root.clone(),
                    artifact_root: self.artifact_root.clone(),
                    engine_build_root: self.engine_build_root.clone(),
                    placement_root: self.placement_root.clone(),
                    execution_mode: self.execution_mode.clone(),
                    cache_policy: self.cache_policy.clone(),
                    concurrency: self.concurrency,
                    run_count: self.run_count,
                    warm_state: self.warm_state.clone(),
                    request_count: self.request_count,
                    quantization: self.quantization.clone(),
                    context_tokens: self.context_tokens,
                    tokenizer_root: self.tokenizer_root.clone(),
                    template_root: self.template_root.clone(),
                    sampler_root: self.sampler_root.clone(),
                    hardware_probe_root: self.hardware_probe_root.clone(),
                    prompt_distribution_root: self.prompt_distribution_root.clone(),
                    output_distribution_root: self.output_distribution_root.clone(),
                    reference_runtime: self.reference_runtime.clone(),
                    reference_family: self.reference_family.clone(),
                    reference_build_root: self.reference_build_root.clone(),
                    reference_artifact_root: self.reference_artifact_root.clone(),
                    reference_quantization: self.reference_quantization.clone(),
                    reference_tokenizer_root: self.reference_tokenizer_root.clone(),
                    reference_template_root: self.reference_template_root.clone(),
                    reference_sampler_root: self.reference_sampler_root.clone(),
                    reference_hardware_root: self.reference_hardware_root.clone(),
                    reference_prompt_distribution_root: self
                        .reference_prompt_distribution_root
                        .clone(),
                    reference_output_distribution_root: self
                        .reference_output_distribution_root
                        .clone(),
                    reference_verification_class: self.reference_verification_class.clone(),
                    validation_result: self.validation_result.clone(),
                    extensions: self.extensions.clone(),
                }
            }

            pub fn validate(&self) -> Result<(), InferFailure> {
                validate_reference($noun, $runtime, $family, $class, &self.body())
            }

            pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
                self.validate()?;
                write_reference($kind, &self.body())
            }

            pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
                let mut fields = Fields::parse(value)?;
                expect_kind_version(&mut fields, $kind)?;
                let body = read_reference(&mut fields)?;
                fields.finish()?;
                let report = Self {
                    model_image_root: body.model_image_root,
                    artifact_root: body.artifact_root,
                    engine_build_root: body.engine_build_root,
                    placement_root: body.placement_root,
                    execution_mode: body.execution_mode,
                    cache_policy: body.cache_policy,
                    concurrency: body.concurrency,
                    run_count: body.run_count,
                    warm_state: body.warm_state,
                    request_count: body.request_count,
                    quantization: body.quantization,
                    context_tokens: body.context_tokens,
                    tokenizer_root: body.tokenizer_root,
                    template_root: body.template_root,
                    sampler_root: body.sampler_root,
                    hardware_probe_root: body.hardware_probe_root,
                    prompt_distribution_root: body.prompt_distribution_root,
                    output_distribution_root: body.output_distribution_root,
                    reference_runtime: body.reference_runtime,
                    reference_family: body.reference_family,
                    reference_build_root: body.reference_build_root,
                    reference_artifact_root: body.reference_artifact_root,
                    reference_quantization: body.reference_quantization,
                    reference_tokenizer_root: body.reference_tokenizer_root,
                    reference_template_root: body.reference_template_root,
                    reference_sampler_root: body.reference_sampler_root,
                    reference_hardware_root: body.reference_hardware_root,
                    reference_prompt_distribution_root: body.reference_prompt_distribution_root,
                    reference_output_distribution_root: body.reference_output_distribution_root,
                    reference_verification_class: body.reference_verification_class,
                    validation_result: body.validation_result,
                    extensions: body.extensions,
                };
                report.validate()?;
                Ok(report)
            }

            pub fn root(&self) -> Result<DigestHex, InferFailure> {
                digest_value($domain, &self.to_cbor()?)
            }

            pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
                Ok(self.to_cbor()?.to_bytes())
            }
        }
    };
}

reference_report!(
    LlamaReportV1,
    LLAMA_RUNTIME,
    LLAMA_FAMILY,
    SIDECAR_ARTIFACT_VERIFIED,
    "llama",
    LLAMA_KIND,
    "infer-llama"
);
reference_report!(
    VllmReportV1,
    VLLM_RUNTIME,
    VLLM_FAMILY,
    SIDECAR_ARTIFACT_VERIFIED,
    "vllm",
    VLLM_KIND,
    "infer-vllm"
);
reference_report!(
    MistralReportV1,
    MISTRAL_RUNTIME,
    MISTRAL_FAMILY,
    BACKEND_REPORTED,
    "mistral",
    MISTRAL_KIND,
    "infer-mistral"
);

/// Support mark for one cold micro-fixture recipe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecipeReportV1 {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub adapter_id: String,
    pub conformance_root: DigestHex,
    pub security_root: DigestHex,
    pub stability_root: DigestHex,
    pub benchmark_root: DigestHex,
    pub support_level: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl RecipeReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("validationResult", &self.validation_result, &["recorded"])?;
        one_of(
            "executionMode",
            &self.execution_mode,
            &["isolated-replay", "pinned"],
        )?;
        cold_single(
            "recipe",
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
        )?;
        bounded_text("adapterId", &self.adapter_id, 128)?;
        if self.adapter_id != RECIPE_ADAPTER {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the recipe report is the micro fixture",
            ));
        }
        one_of(
            "supportLevel",
            &self.support_level,
            &["experimental", "conformant", "blessed"],
        )?;
        if self.support_level == "blessed" {
            distinct_receipts(&[
                &self.conformance_root,
                &self.security_root,
                &self.stability_root,
                &self.benchmark_root,
            ])?;
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(RECIPE_KIND);
        b.put("adapterId", cbor_text(&self.adapter_id));
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("benchmarkRoot", cbor_digest(&self.benchmark_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("conformanceRoot", cbor_digest(&self.conformance_root));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("securityRoot", cbor_digest(&self.security_root));
        b.put("stabilityRoot", cbor_digest(&self.stability_root));
        b.put("supportLevel", cbor_text(&self.support_level));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, RECIPE_KIND)?;
        let out = Self {
            adapter_id: fields.text("adapterId")?,
            artifact_root: fields.digest("artifactRoot")?,
            benchmark_root: fields.digest("benchmarkRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            conformance_root: fields.digest("conformanceRoot")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            model_image_root: fields.digest("modelImageRoot")?,
            placement_root: fields.digest("placementRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            security_root: fields.digest("securityRoot")?,
            stability_root: fields.digest("stabilityRoot")?,
            support_level: fields.text("supportLevel")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-recipe", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

fn distinct_receipts(roots: &[&DigestHex]) -> Result<(), InferFailure> {
    for (index, root) in roots.iter().enumerate() {
        if roots[..index].contains(root) {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "blessed recipe repeats a receipt",
            ));
        }
    }
    Ok(())
}
