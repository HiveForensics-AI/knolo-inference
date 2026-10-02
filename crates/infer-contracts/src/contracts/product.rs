//! Phase 5 integration reports for one cold micro fixture.
//!
//! The composition record binds a host-supplied Knowledge Image. The agent
//! effect records a policy decision. The Hub record names a `.kmodel` without
//! its weight bytes. The studio view stores receipt roots. None of them opens
//! a product repository. The layouts are specified in
//! `spec/KIP-INFER-0044-evidence-composition.md`,
//! `spec/KIP-INFER-0045-agent-effect.md`,
//! `spec/KIP-INFER-0046-hub-record.md`, and
//! `spec/KIP-INFER-0047-receipt-view.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    bounded_text, cbor_digest, cbor_text, cbor_u32, expect_kind_version, one_of, sorted_unique,
    text_array, Fields,
};

use super::common::{put_extensions, string_field_array, Builder};
use super::lifecycle::cold_single;

pub const COMPOSITION_KIND: &str = "knolo.infer.composition-report";
pub const AGENT_EFFECT_KIND: &str = "knolo.infer.agent-effect-report";
pub const HUB_KIND: &str = "knolo.infer.hub-report";
pub const STUDIO_KIND: &str = "knolo.infer.studio-report";

pub const MICRO_ADAPTER: &str = "knolo.micro.v1";
pub const MICRO_CONTEXT: u32 = 16;
pub const LOCAL_WEIGHT_SOURCE: &str = "local";

const NATIVE_BACKENDS: &[&str] = &["candle-cuda", "reference-f32"];
const AGENT_BACKENDS: &[&str] = &[
    "candle-cuda",
    "llama.cpp",
    "mistral.rs",
    "ollama",
    "reference-f32",
    "vllm",
];
const AGENT_REASONS: &[&str] = &[
    "accepted",
    "artifact-rejected",
    "assurance-rejected",
    "execution-mode-rejected",
    "knowledge-image-rejected",
    "model-runtime-rejected",
    "output-budget-exceeded",
    "prompt-budget-exceeded",
    "stream-not-authorization",
    "unverified-backend",
];

fn recorded(value: &str) -> Result<(), InferFailure> {
    one_of("validationResult", value, &["recorded"])
}

fn execution_mode(value: &str) -> Result<(), InferFailure> {
    one_of("executionMode", value, &["isolated-replay", "pinned"])
}

fn micro_context(noun: &str, prompt: u32, output: u32) -> Result<(), InferFailure> {
    if prompt > MICRO_CONTEXT
        || output > MICRO_CONTEXT
        || prompt.saturating_add(output) > MICRO_CONTEXT
    {
        return Err(fail(
            ErrorCode::ContextLimitExceeded,
            format!("the {noun} is the micro fixture"),
        ));
    }
    Ok(())
}

/// Host-supplied Core and Reflex binding for one cold micro fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositionReportV1 {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub evidence_root: DigestHex,
    pub knowledge_image_root: DigestHex,
    pub knowledge_commit_root: DigestHex,
    pub context_root: DigestHex,
    pub query_receipt_count: u32,
    pub reflex_receipt_count: u32,
    pub ordered_evidence_count: u32,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl CompositionReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        recorded(&self.validation_result)?;
        execution_mode(&self.execution_mode)?;
        cold_single(
            "composition",
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
        )?;
        if self.knowledge_image_root == self.knowledge_commit_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the knowledge commit repeats the image",
            ));
        }
        if self.query_receipt_count == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the composition has no query receipt",
            ));
        }
        if self.reflex_receipt_count == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the composition has no reflex receipt",
            ));
        }
        if self.ordered_evidence_count == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the composition has no evidence id",
            ));
        }
        if self.query_receipt_count > 256
            || self.reflex_receipt_count > 256
            || self.ordered_evidence_count > 4096
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "evidence list is too large",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(COMPOSITION_KIND);
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("contextRoot", cbor_digest(&self.context_root));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("evidenceRoot", cbor_digest(&self.evidence_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put(
            "knowledgeCommitRoot",
            cbor_digest(&self.knowledge_commit_root),
        );
        b.put(
            "knowledgeImageRoot",
            cbor_digest(&self.knowledge_image_root),
        );
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put(
            "orderedEvidenceCount",
            cbor_u32(self.ordered_evidence_count),
        );
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("queryReceiptCount", cbor_u32(self.query_receipt_count));
        b.put("reflexReceiptCount", cbor_u32(self.reflex_receipt_count));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, COMPOSITION_KIND)?;
        let out = Self {
            artifact_root: fields.digest("artifactRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            context_root: fields.digest("contextRoot")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            evidence_root: fields.digest("evidenceRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            knowledge_commit_root: fields.digest("knowledgeCommitRoot")?,
            knowledge_image_root: fields.digest("knowledgeImageRoot")?,
            model_image_root: fields.digest("modelImageRoot")?,
            ordered_evidence_count: fields.u32("orderedEvidenceCount")?,
            placement_root: fields.digest("placementRoot")?,
            query_receipt_count: fields.u32("queryReceiptCount")?,
            reflex_receipt_count: fields.u32("reflexReceiptCount")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-composition", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Policy decision for one cold micro-fixture host effect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentEffectReportV1 {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
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
    pub decision: String,
    pub reason: String,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl AgentEffectReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        recorded(&self.validation_result)?;
        execution_mode(&self.execution_mode)?;
        cold_single(
            "agent effect",
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
        )?;
        one_of("backend", &self.backend, AGENT_BACKENDS)?;
        one_of(
            "assurance",
            &self.assurance,
            &["compatibility", "same_build_replayable"],
        )?;
        one_of(
            "requiredAssurance",
            &self.required_assurance,
            &["compatibility", "same_build_replayable"],
        )?;
        if self.allowed_execution_modes.is_empty() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the agent effect allows no execution mode",
            ));
        }
        if self.allowed_execution_modes.len() > 2 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the agent effect is the micro fixture",
            ));
        }
        sorted_unique("allowedExecutionModes", &self.allowed_execution_modes)?;
        for mode in &self.allowed_execution_modes {
            one_of(
                "allowedExecutionModes",
                mode,
                &["isolated-replay", "pinned"],
            )?;
        }
        micro_context("agent effect", self.prompt_tokens, self.output_tokens)?;
        if !(1..=MICRO_CONTEXT).contains(&self.max_prompt_tokens)
            || !(1..=MICRO_CONTEXT).contains(&self.max_output_tokens)
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the agent effect is the micro fixture",
            ));
        }
        one_of("decision", &self.decision, &["allow", "deny"])?;
        one_of("reason", &self.reason, AGENT_REASONS)?;
        let (decision, reason) = self.expected_decision();
        if self.decision != decision || self.reason != reason {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "agent effect decision does not match",
            ));
        }
        Ok(())
    }

    /// Policy outcome. The stored `decision` and `reason` must equal this pair.
    pub fn computed_decision(&self) -> (&'static str, &'static str) {
        self.expected_decision()
    }

    fn expected_decision(&self) -> (&'static str, &'static str) {
        if !self.receipt_present {
            return ("deny", "stream-not-authorization");
        }
        if !NATIVE_BACKENDS.contains(&self.backend.as_str()) {
            return ("deny", "unverified-backend");
        }
        if self.model_runtime_root != self.required_model_runtime_root {
            return ("deny", "model-runtime-rejected");
        }
        if self.artifact_root != self.required_artifact_root {
            return ("deny", "artifact-rejected");
        }
        if self.knowledge_image_root != self.required_knowledge_image_root {
            return ("deny", "knowledge-image-rejected");
        }
        if !self
            .allowed_execution_modes
            .iter()
            .any(|mode| mode == &self.execution_mode)
        {
            return ("deny", "execution-mode-rejected");
        }
        if self.prompt_tokens > self.max_prompt_tokens {
            return ("deny", "prompt-budget-exceeded");
        }
        if self.output_tokens > self.max_output_tokens {
            return ("deny", "output-budget-exceeded");
        }
        let assurance_ok = self.assurance == self.required_assurance
            || (self.required_assurance == "compatibility"
                && self.assurance == "same_build_replayable");
        if !assurance_ok {
            return ("deny", "assurance-rejected");
        }
        ("allow", "accepted")
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(AGENT_EFFECT_KIND);
        b.put(
            "allowedExecutionModes",
            string_field_array(&self.allowed_execution_modes),
        );
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("assurance", cbor_text(&self.assurance));
        b.put("backend", cbor_text(&self.backend));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("decision", cbor_text(&self.decision));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put(
            "knowledgeImageRoot",
            cbor_digest(&self.knowledge_image_root),
        );
        b.put("maxOutputTokens", cbor_u32(self.max_output_tokens));
        b.put("maxPromptTokens", cbor_u32(self.max_prompt_tokens));
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("modelRuntimeRoot", cbor_digest(&self.model_runtime_root));
        b.put("outputTokens", cbor_u32(self.output_tokens));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("promptTokens", cbor_u32(self.prompt_tokens));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptPresent", CborValue::Bool(self.receipt_present));
        b.put(
            "requiredArtifactRoot",
            cbor_digest(&self.required_artifact_root),
        );
        b.put("requiredAssurance", cbor_text(&self.required_assurance));
        b.put(
            "requiredKnowledgeImageRoot",
            cbor_digest(&self.required_knowledge_image_root),
        );
        b.put(
            "requiredModelRuntimeRoot",
            cbor_digest(&self.required_model_runtime_root),
        );
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, AGENT_EFFECT_KIND)?;
        let out = Self {
            allowed_execution_modes: text_array(
                &fields.array("allowedExecutionModes")?,
                "allowedExecutionModes",
            )?,
            artifact_root: fields.digest("artifactRoot")?,
            assurance: fields.text("assurance")?,
            backend: fields.text("backend")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            decision: fields.text("decision")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            knowledge_image_root: fields.digest("knowledgeImageRoot")?,
            max_output_tokens: fields.u32("maxOutputTokens")?,
            max_prompt_tokens: fields.u32("maxPromptTokens")?,
            model_image_root: fields.digest("modelImageRoot")?,
            model_runtime_root: fields.digest("modelRuntimeRoot")?,
            output_tokens: fields.u32("outputTokens")?,
            placement_root: fields.digest("placementRoot")?,
            prompt_tokens: fields.u32("promptTokens")?,
            reason: fields.text("reason")?,
            receipt_present: fields.bool("receiptPresent")?,
            required_artifact_root: fields.digest("requiredArtifactRoot")?,
            required_assurance: fields.text("requiredAssurance")?,
            required_knowledge_image_root: fields.digest("requiredKnowledgeImageRoot")?,
            required_model_runtime_root: fields.digest("requiredModelRuntimeRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-agent-effect", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Hub metadata for one cold micro fixture. Weight bytes are not a field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubReportV1 {
    pub publisher: String,
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub adapter_id: String,
    pub quantization: String,
    pub license_id: String,
    pub source_provider: String,
    pub conformance_root: DigestHex,
    pub benchmark_root: DigestHex,
    pub support_level: String,
    pub greedy_token_parity: bool,
    pub distribution: String,
    pub native_supported: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl HubReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        recorded(&self.validation_result)?;
        execution_mode(&self.execution_mode)?;
        cold_single(
            "hub",
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
        )?;
        if self.adapter_id != MICRO_ADAPTER || self.quantization != "f32" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the hub record is the micro fixture",
            ));
        }
        if self.source_provider != LOCAL_WEIGHT_SOURCE {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the hub record does not download weights",
            ));
        }
        bounded_text("publisher", &self.publisher, 128)?;
        bounded_text("licenseId", &self.license_id, 128)?;
        one_of(
            "supportLevel",
            &self.support_level,
            &["blessed", "conformant", "experimental"],
        )?;
        one_of("distribution", &self.distribution, &["active", "yanked"])?;
        if self.support_level == "blessed" && !self.greedy_token_parity {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "fast execution is not blessed",
            ));
        }
        if self.support_level == "conformant" && !self.greedy_token_parity {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "conformant support requires greedy token parity",
            ));
        }
        if self.benchmark_root == self.conformance_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the hub record repeats a receipt",
            ));
        }
        if self.native_supported != self.expected_native() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "native support does not match",
            ));
        }
        Ok(())
    }

    /// True only for an active conformant or blessed mark with greedy parity.
    pub fn computed_native(&self) -> bool {
        self.expected_native()
    }

    fn expected_native(&self) -> bool {
        self.distribution == "active"
            && self.greedy_token_parity
            && (self.support_level == "conformant" || self.support_level == "blessed")
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(HUB_KIND);
        b.put("adapterId", cbor_text(&self.adapter_id));
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("benchmarkRoot", cbor_digest(&self.benchmark_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("conformanceRoot", cbor_digest(&self.conformance_root));
        b.put("distribution", cbor_text(&self.distribution));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put(
            "greedyTokenParity",
            CborValue::Bool(self.greedy_token_parity),
        );
        b.put("licenseId", cbor_text(&self.license_id));
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("nativeSupported", CborValue::Bool(self.native_supported));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("publisher", cbor_text(&self.publisher));
        b.put("quantization", cbor_text(&self.quantization));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("sourceProvider", cbor_text(&self.source_provider));
        b.put("supportLevel", cbor_text(&self.support_level));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, HUB_KIND)?;
        let out = Self {
            adapter_id: fields.text("adapterId")?,
            artifact_root: fields.digest("artifactRoot")?,
            benchmark_root: fields.digest("benchmarkRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            conformance_root: fields.digest("conformanceRoot")?,
            distribution: fields.text("distribution")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            greedy_token_parity: fields.bool("greedyTokenParity")?,
            license_id: fields.text("licenseId")?,
            model_image_root: fields.digest("modelImageRoot")?,
            native_supported: fields.bool("nativeSupported")?,
            placement_root: fields.digest("placementRoot")?,
            publisher: fields.text("publisher")?,
            quantization: fields.text("quantization")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            source_provider: fields.text("sourceProvider")?,
            support_level: fields.text("supportLevel")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-hub", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Receipt fields a Studio panel may show. Prompt text and output text are absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StudioReportV1 {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub receipt_root: DigestHex,
    pub prompt_token_root: DigestHex,
    pub output_token_root: DigestHex,
    pub output_text_root: DigestHex,
    pub knowledge_image_root: DigestHex,
    pub evidence_root: DigestHex,
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
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl StudioReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        recorded(&self.validation_result)?;
        execution_mode(&self.execution_mode)?;
        cold_single(
            "studio",
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
        )?;
        if !self.extensions.is_empty() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the studio view extensions are empty",
            ));
        }
        one_of(
            "assurance",
            &self.assurance,
            &["compatibility", "same_build_replayable"],
        )?;
        one_of("finishReason", &self.finish_reason, &["length", "stop"])?;
        if self.prompt_token_count == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the studio view has no prompt",
            ));
        }
        micro_context(
            "studio view",
            self.prompt_token_count,
            self.output_token_count,
        )?;
        if self.finish_reason == "stop" && self.output_token_count == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a stop receipt has no output tokens",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(STUDIO_KIND);
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("assurance", cbor_text(&self.assurance));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("evidenceRoot", cbor_digest(&self.evidence_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("finishReason", cbor_text(&self.finish_reason));
        b.put(
            "knowledgeImageRoot",
            cbor_digest(&self.knowledge_image_root),
        );
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("outputTextRoot", cbor_digest(&self.output_text_root));
        b.put("outputTokenCount", cbor_u32(self.output_token_count));
        b.put("outputTokenRoot", cbor_digest(&self.output_token_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("promptTokenCount", cbor_u32(self.prompt_token_count));
        b.put("promptTokenRoot", cbor_digest(&self.prompt_token_root));
        b.put("receiptRoot", cbor_digest(&self.receipt_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, STUDIO_KIND)?;
        let out = Self {
            artifact_root: fields.digest("artifactRoot")?,
            assurance: fields.text("assurance")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            evidence_root: fields.digest("evidenceRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            finish_reason: fields.text("finishReason")?,
            knowledge_image_root: fields.digest("knowledgeImageRoot")?,
            model_image_root: fields.digest("modelImageRoot")?,
            output_text_root: fields.digest("outputTextRoot")?,
            output_token_count: fields.u32("outputTokenCount")?,
            output_token_root: fields.digest("outputTokenRoot")?,
            placement_root: fields.digest("placementRoot")?,
            prompt_token_count: fields.u32("promptTokenCount")?,
            prompt_token_root: fields.digest("promptTokenRoot")?,
            receipt_root: fields.digest("receiptRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-studio", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
