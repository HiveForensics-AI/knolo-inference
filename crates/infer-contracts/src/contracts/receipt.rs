use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_bytes, digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    bounded_text, cbor_digest, cbor_i64, cbor_text, cbor_u32, cbor_u64, expect_kind_version,
    one_of, sorted_unique, text_array, Fields,
};

use super::common::{
    put_extensions, signatures_from_cbor, signatures_to_cbor, Builder, SignatureV1,
};
use super::prompt::EvidenceBindingV1;

pub const RECEIPT_KIND: &str = "knolo.infer.inference-receipt";
pub const REPLAY_KIND: &str = "knolo.infer.replay-check";
pub const CONFORMANCE_KIND: &str = "knolo.infer.model-conformance";
pub const CONVERSION_KIND: &str = "knolo.infer.conversion-receipt";
pub const PERPLEXITY_KIND: &str = "knolo.infer.perplexity-report";
pub const MEMORY_KIND: &str = "knolo.infer.memory-estimate";
pub const THROUGHPUT_KIND: &str = "knolo.infer.throughput-report";
pub const LATENCY_KIND: &str = "knolo.infer.latency-report";
pub const FUZZ_KIND: &str = "knolo.infer.fuzz-report";
pub const CANCELLATION_KIND: &str = "knolo.infer.cancellation-report";
pub const FINALIZATION_KIND: &str = "knolo.infer.finalization-report";
pub const OVERHEAD_KIND: &str = "knolo.infer.overhead-report";

/// The Phase 4 corruption campaign exercises these nine parsers, in this order.
pub const FUZZ_TARGETS: [&str; 9] = [
    "cbor",
    "kmodel",
    "gguf",
    "safetensors",
    "template",
    "tokenizer",
    "api",
    "ipc",
    "receipt",
];
pub const FUZZ_SEED_COUNT: u32 = 9;
pub const FUZZ_MUTATIONS_PER_SEED: u32 = 6;
pub const FUZZ_MUTATION_COUNT: u32 = FUZZ_SEED_COUNT * FUZZ_MUTATIONS_PER_SEED;

const MICRO_CONTEXT_TOKENS: u32 = 16;
const TOKENS_PER_SECOND_SCALE: u128 = 1_000_000 * 1_000_000_000;

const ASSURANCE: &[&str] = &[
    "compatibility",
    "exact_replay_verified",
    "incomplete",
    "same_build_replayable",
];
const FINISH: &[&str] = &[
    "cancelled",
    "error",
    "grammar",
    "length",
    "policy",
    "stop",
    "timeout",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelReceiptBindingV1 {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub model_runtime_root: DigestHex,
    pub architecture_adapter_id: String,
    pub architecture_adapter_root: DigestHex,
    pub config_root: DigestHex,
    pub tokenizer_root: DigestHex,
    pub template_root: DigestHex,
    pub storage_precision: String,
    pub compute_precision: String,
}

impl ModelReceiptBindingV1 {
    fn validate(&self) -> Result<(), InferFailure> {
        bounded_text("architectureAdapterId", &self.architecture_adapter_id, 128)?;
        one_of(
            "storagePrecision",
            &self.storage_precision,
            super::common::PRECISIONS,
        )?;
        one_of(
            "computePrecision",
            &self.compute_precision,
            super::common::KV_PRECISIONS,
        )?;
        Ok(())
    }

    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put(
            "architectureAdapterId",
            cbor_text(&self.architecture_adapter_id),
        );
        b.put(
            "architectureAdapterRoot",
            cbor_digest(&self.architecture_adapter_root),
        );
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("computePrecision", cbor_text(&self.compute_precision));
        b.put("configRoot", cbor_digest(&self.config_root));
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("modelRuntimeRoot", cbor_digest(&self.model_runtime_root));
        b.put("storagePrecision", cbor_text(&self.storage_precision));
        b.put("templateRoot", cbor_digest(&self.template_root));
        b.put("tokenizerRoot", cbor_digest(&self.tokenizer_root));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            architecture_adapter_id: fields.text("architectureAdapterId")?,
            architecture_adapter_root: fields.digest("architectureAdapterRoot")?,
            artifact_root: fields.digest("artifactRoot")?,
            compute_precision: fields.text("computePrecision")?,
            config_root: fields.digest("configRoot")?,
            model_image_root: fields.digest("modelImageRoot")?,
            model_runtime_root: fields.digest("modelRuntimeRoot")?,
            storage_precision: fields.text("storagePrecision")?,
            template_root: fields.digest("templateRoot")?,
            tokenizer_root: fields.digest("tokenizerRoot")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineReceiptBindingV1 {
    pub engine_build_root: DigestHex,
    pub backend: String,
    pub backend_version: String,
    pub binary_sha256: DigestHex,
    pub kernel_bundle_root: DigestHex,
    pub kernel_plan_root: DigestHex,
}

impl EngineReceiptBindingV1 {
    fn validate(&self) -> Result<(), InferFailure> {
        one_of("backend", &self.backend, &["llamacpp", "native", "ollama"])?;
        bounded_text("backendVersion", &self.backend_version, 32)?;
        Ok(())
    }

    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put("backend", cbor_text(&self.backend));
        b.put("backendVersion", cbor_text(&self.backend_version));
        b.put("binarySha256", cbor_digest(&self.binary_sha256));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("kernelBundleRoot", cbor_digest(&self.kernel_bundle_root));
        b.put("kernelPlanRoot", cbor_digest(&self.kernel_plan_root));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            backend: fields.text("backend")?,
            backend_version: fields.text("backendVersion")?,
            binary_sha256: fields.digest("binarySha256")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            kernel_bundle_root: fields.digest("kernelBundleRoot")?,
            kernel_plan_root: fields.digest("kernelPlanRoot")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareReceiptBindingV1 {
    pub hardware_root: DigestHex,
    pub gpu_model: Option<String>,
    pub compute_capability: Option<String>,
    pub device_slot: String,
}

impl HardwareReceiptBindingV1 {
    fn validate(&self) -> Result<(), InferFailure> {
        super::common::device_id(&self.device_slot)?;
        if let Some(model) = &self.gpu_model {
            bounded_text("gpuModel", model, 64)?;
        }
        if let Some(cap) = &self.compute_capability {
            bounded_text("computeCapability", cap, 16)?;
        }
        Ok(())
    }

    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put_opt(
            "computeCapability",
            self.compute_capability.clone().map(cbor_text),
        );
        b.put("deviceSlot", cbor_text(&self.device_slot));
        b.put_opt("gpuModel", self.gpu_model.clone().map(cbor_text));
        b.put("hardwareRoot", cbor_digest(&self.hardware_root));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            compute_capability: fields.opt_text("computeCapability")?,
            device_slot: fields.text("deviceSlot")?,
            gpu_model: fields.opt_text("gpuModel")?,
            hardware_root: fields.digest("hardwareRoot")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementReceiptBindingV1 {
    pub placement_root: DigestHex,
    pub kv_block_size: u32,
    pub kv_precision: String,
}

impl PlacementReceiptBindingV1 {
    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        one_of(
            "kvPrecision",
            &self.kv_precision,
            super::common::KV_PRECISIONS,
        )?;
        let mut b = Builder::bare();
        b.put("kvBlockSize", cbor_u32(self.kv_block_size));
        b.put("kvPrecision", cbor_text(&self.kv_precision));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            kv_block_size: fields.u32("kvBlockSize")?,
            kv_precision: fields.text("kvPrecision")?,
            placement_root: fields.digest("placementRoot")?,
        };
        fields.finish()?;
        out.to_cbor()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptReceiptBindingV1 {
    pub messages_root: DigestHex,
    pub tools_root: Option<DigestHex>,
    pub evidence_root: Option<DigestHex>,
    pub rendered_text_root: DigestHex,
    pub token_id_root: DigestHex,
    pub prompt_token_count: u32,
    pub truncation_root: DigestHex,
}

impl PromptReceiptBindingV1 {
    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        if self.prompt_token_count > 1_048_576 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prompt token count is too large",
            ));
        }
        let mut b = Builder::bare();
        b.put_opt("evidenceRoot", self.evidence_root.as_ref().map(cbor_digest));
        b.put("messagesRoot", cbor_digest(&self.messages_root));
        b.put("promptTokenCount", cbor_u32(self.prompt_token_count));
        b.put("renderedTextRoot", cbor_digest(&self.rendered_text_root));
        b.put("tokenIdRoot", cbor_digest(&self.token_id_root));
        b.put_opt("toolsRoot", self.tools_root.as_ref().map(cbor_digest));
        b.put("truncationRoot", cbor_digest(&self.truncation_root));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            evidence_root: fields.opt_digest("evidenceRoot")?,
            messages_root: fields.digest("messagesRoot")?,
            prompt_token_count: fields.u32("promptTokenCount")?,
            rendered_text_root: fields.digest("renderedTextRoot")?,
            token_id_root: fields.digest("tokenIdRoot")?,
            tools_root: fields.opt_digest("toolsRoot")?,
            truncation_root: fields.digest("truncationRoot")?,
        };
        fields.finish()?;
        out.to_cbor()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplerReceiptBindingV1 {
    pub sampler_plan_root: DigestHex,
    pub temperature_micros: u32,
    pub seed: Option<u64>,
    pub rng: String,
}

impl SamplerReceiptBindingV1 {
    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        one_of("rng", &self.rng, &["none", "philox-4x32-v1"])?;
        if self.temperature_micros > 10_000_000 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "temperature is outside its bounds",
            ));
        }
        if (self.temperature_micros == 0) != (self.rng == "none" && self.seed.is_none()) {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "sampler receipt rng does not match temperature",
            ));
        }
        let mut b = Builder::bare();
        b.put("rng", cbor_text(&self.rng));
        b.put("samplerPlanRoot", cbor_digest(&self.sampler_plan_root));
        b.put_opt("seed", self.seed.map(cbor_u64));
        b.put("temperatureMicros", cbor_u32(self.temperature_micros));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            rng: fields.text("rng")?,
            sampler_plan_root: fields.digest("samplerPlanRoot")?,
            seed: fields.opt_u64("seed")?,
            temperature_micros: fields.u32("temperatureMicros")?,
        };
        fields.finish()?;
        out.to_cbor()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionReceiptBindingV1 {
    pub execution_plan_root: DigestHex,
    pub scheduling_mode: String,
    pub prefill_chunk_tokens: u32,
    pub kv_block_size: u32,
    pub prefix_cache_hit_tokens: u32,
    pub batch_trace_root: DigestHex,
    pub speculative_plan_root: Option<DigestHex>,
    pub event_trace_root: DigestHex,
    pub attempt: u32,
}

impl ExecutionReceiptBindingV1 {
    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        one_of(
            "schedulingMode",
            &self.scheduling_mode,
            &["continuous", "isolated"],
        )?;
        if self.prefix_cache_hit_tokens != 0 || self.speculative_plan_root.is_some() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "version 1 receipts have no prefix cache or speculation",
            ));
        }
        if self.attempt > 16 || self.prefill_chunk_tokens == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "execution receipt counters are invalid",
            ));
        }
        let mut b = Builder::bare();
        b.put("attempt", cbor_u32(self.attempt));
        b.put("batchTraceRoot", cbor_digest(&self.batch_trace_root));
        b.put("eventTraceRoot", cbor_digest(&self.event_trace_root));
        b.put("executionPlanRoot", cbor_digest(&self.execution_plan_root));
        b.put("kvBlockSize", cbor_u32(self.kv_block_size));
        b.put("prefillChunkTokens", cbor_u32(self.prefill_chunk_tokens));
        b.put(
            "prefixCacheHitTokens",
            cbor_u32(self.prefix_cache_hit_tokens),
        );
        b.put("schedulingMode", cbor_text(&self.scheduling_mode));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        if fields.optional("speculativePlanRoot")?.is_some() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "version 1 receipts have no speculation",
            ));
        }
        let out = Self {
            attempt: fields.u32("attempt")?,
            batch_trace_root: fields.digest("batchTraceRoot")?,
            event_trace_root: fields.digest("eventTraceRoot")?,
            execution_plan_root: fields.digest("executionPlanRoot")?,
            kv_block_size: fields.u32("kvBlockSize")?,
            prefill_chunk_tokens: fields.u32("prefillChunkTokens")?,
            prefix_cache_hit_tokens: fields.u32("prefixCacheHitTokens")?,
            scheduling_mode: fields.text("schedulingMode")?,
            speculative_plan_root: None,
        };
        fields.finish()?;
        out.to_cbor()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputReceiptBindingV1 {
    pub output_token_root: DigestHex,
    pub output_text_root: DigestHex,
    pub token_count: u32,
    pub finish_reason: String,
    pub structured_output_root: Option<DigestHex>,
}

impl OutputReceiptBindingV1 {
    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        one_of("finishReason", &self.finish_reason, FINISH)?;
        if self.token_count > 1_048_576 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "output token count is too large",
            ));
        }
        let mut b = Builder::bare();
        b.put("finishReason", cbor_text(&self.finish_reason));
        b.put("outputTextRoot", cbor_digest(&self.output_text_root));
        b.put("outputTokenRoot", cbor_digest(&self.output_token_root));
        b.put_opt(
            "structuredOutputRoot",
            self.structured_output_root.as_ref().map(cbor_digest),
        );
        b.put("tokenCount", cbor_u32(self.token_count));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            finish_reason: fields.text("finishReason")?,
            output_text_root: fields.digest("outputTextRoot")?,
            output_token_root: fields.digest("outputTokenRoot")?,
            structured_output_root: fields.opt_digest("structuredOutputRoot")?,
            token_count: fields.u32("tokenCount")?,
        };
        fields.finish()?;
        out.to_cbor()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimingReceiptV1 {
    pub queue_micros: u64,
    pub prefill_micros: u64,
    pub decode_micros: u64,
    pub total_micros: u64,
}

impl TimingReceiptV1 {
    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        let sum = self
            .queue_micros
            .checked_add(self.prefill_micros)
            .and_then(|v| v.checked_add(self.decode_micros))
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "timing total overflows"))?;
        if sum != self.total_micros {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "timing total must equal its parts",
            ));
        }
        let mut b = Builder::bare();
        b.put("decodeMicros", cbor_u64(self.decode_micros));
        b.put("prefillMicros", cbor_u64(self.prefill_micros));
        b.put("queueMicros", cbor_u64(self.queue_micros));
        b.put("totalMicros", cbor_u64(self.total_micros));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            decode_micros: fields.u64("decodeMicros")?,
            prefill_micros: fields.u64("prefillMicros")?,
            queue_micros: fields.u64("queueMicros")?,
            total_micros: fields.u64("totalMicros")?,
        };
        fields.finish()?;
        out.to_cbor()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceReceiptV1 {
    pub receipt_id: DigestHex,
    pub intent_root: DigestHex,
    pub model: ModelReceiptBindingV1,
    pub engine: EngineReceiptBindingV1,
    pub hardware: HardwareReceiptBindingV1,
    pub placement: PlacementReceiptBindingV1,
    pub prompt: PromptReceiptBindingV1,
    pub knowledge: Option<EvidenceBindingV1>,
    pub sampler: SamplerReceiptBindingV1,
    pub execution: ExecutionReceiptBindingV1,
    pub output: OutputReceiptBindingV1,
    pub timing: TimingReceiptV1,
    pub assurance: String,
    pub previous_receipt_root: Option<DigestHex>,
    pub extensions: BTreeMap<String, CborValue>,
    pub signatures: Vec<SignatureV1>,
}

impl InferenceReceiptV1 {
    pub fn content_cbor(&self) -> Result<CborValue, InferFailure> {
        let mut b = Builder::typed(RECEIPT_KIND);
        b.put("assurance", cbor_text(&self.assurance));
        b.put("engine", self.engine.to_cbor()?);
        b.put("execution", self.execution.to_cbor()?);
        put_extensions(&mut b, &self.extensions)?;
        b.put("hardware", self.hardware.to_cbor()?);
        b.put("intentRoot", cbor_digest(&self.intent_root));
        b.put_opt(
            "knowledge",
            self.knowledge.as_ref().map(|k| k.to_cbor()).transpose()?,
        );
        b.put("model", self.model.to_cbor()?);
        b.put("output", self.output.to_cbor()?);
        b.put("placement", self.placement.to_cbor()?);
        b.put_opt(
            "previousReceiptRoot",
            self.previous_receipt_root.as_ref().map(cbor_digest),
        );
        b.put("prompt", self.prompt.to_cbor()?);
        b.put("sampler", self.sampler.to_cbor()?);
        b.put("timing", self.timing.to_cbor()?);
        Ok(b.finish())
    }

    pub fn computed_id(&self) -> Result<DigestHex, InferFailure> {
        one_of("assurance", &self.assurance, ASSURANCE)?;
        digest_value("infer-receipt", &self.content_cbor()?)
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        let id = self.computed_id()?;
        if id != self.receipt_id {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "receipt id does not match its body",
            ));
        }
        let CborValue::Map(mut entries) = self.content_cbor()? else {
            return Err(fail(ErrorCode::ContractInvalid, "receipt must be a map"));
        };
        entries.push(("receiptId".into(), cbor_digest(&self.receipt_id)));
        entries.push(("signatures".into(), signatures_to_cbor(&self.signatures)?));
        entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        Ok(CborValue::Map(entries))
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, RECEIPT_KIND)?;
        let out = Self {
            assurance: fields.text("assurance")?,
            engine: EngineReceiptBindingV1::from_cbor(&fields.require("engine")?)?,
            execution: ExecutionReceiptBindingV1::from_cbor(&fields.require("execution")?)?,
            extensions: fields.extensions()?,
            hardware: HardwareReceiptBindingV1::from_cbor(&fields.require("hardware")?)?,
            intent_root: fields.digest("intentRoot")?,
            knowledge: match fields.optional("knowledge")? {
                None => None,
                Some(value) => Some(EvidenceBindingV1::from_cbor(&value)?),
            },
            model: ModelReceiptBindingV1::from_cbor(&fields.require("model")?)?,
            output: OutputReceiptBindingV1::from_cbor(&fields.require("output")?)?,
            placement: PlacementReceiptBindingV1::from_cbor(&fields.require("placement")?)?,
            previous_receipt_root: fields.opt_digest("previousReceiptRoot")?,
            prompt: PromptReceiptBindingV1::from_cbor(&fields.require("prompt")?)?,
            receipt_id: fields.digest("receiptId")?,
            sampler: SamplerReceiptBindingV1::from_cbor(&fields.require("sampler")?)?,
            signatures: signatures_from_cbor(&fields.array("signatures")?)?,
            timing: TimingReceiptV1::from_cbor(&fields.require("timing")?)?,
        };
        fields.finish()?;
        let id = out.computed_id()?;
        if id != out.receipt_id {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "receipt id does not match its body",
            ));
        }
        Ok(out)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayCheckReceiptV1 {
    pub receipt_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub output_token_root: DigestHex,
    pub matched: bool,
    pub assurance: String,
    pub mismatches: Vec<String>,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ReplayCheckReceiptV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        sorted_unique("mismatches", &self.mismatches)?;
        for mismatch in &self.mismatches {
            if ErrorCode::parse(mismatch).is_none() {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "replay mismatch is not a stable error code",
                ));
            }
        }
        if self.matched {
            if !self.mismatches.is_empty() || self.assurance != "exact_replay_verified" {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "a matched replay is exact_replay_verified",
                ));
            }
        } else if self.mismatches.is_empty() || self.assurance != "incomplete" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "an unmatched replay records mismatches",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(REPLAY_KIND);
        b.put("assurance", cbor_text(&self.assurance));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        put_extensions(&mut b, &self.extensions)?;
        b.put("matched", CborValue::Bool(self.matched));
        b.put(
            "mismatches",
            CborValue::Array(
                self.mismatches
                    .iter()
                    .cloned()
                    .map(CborValue::Text)
                    .collect(),
            ),
        );
        b.put("outputTokenRoot", cbor_digest(&self.output_token_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("receiptRoot", cbor_digest(&self.receipt_root));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, REPLAY_KIND)?;
        let out = Self {
            assurance: fields.text("assurance")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            extensions: fields.extensions()?,
            matched: fields.bool("matched")?,
            mismatches: text_array(&fields.array("mismatches")?, "mismatches")?,
            output_token_root: fields.digest("outputTokenRoot")?,
            placement_root: fields.digest("placementRoot")?,
            receipt_root: fields.digest("receiptRoot")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-replay-check", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConformanceReceiptV1 {
    pub model_runtime_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub adapter_id: String,
    pub logit_abs_tolerance_millionths: u32,
    pub greedy_token_parity: bool,
    pub support_level: String,
    pub notes_root: DigestHex,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ModelConformanceReceiptV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        bounded_text("adapterId", &self.adapter_id, 128)?;
        one_of(
            "supportLevel",
            &self.support_level,
            &["blessed", "conformant", "experimental"],
        )?;
        if self.logit_abs_tolerance_millionths > 100_000_000 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "logit tolerance is outside its bounds",
            ));
        }
        if self.support_level == "blessed" && !self.greedy_token_parity {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "blessed support requires greedy token parity",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(CONFORMANCE_KIND);
        b.put("adapterId", cbor_text(&self.adapter_id));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        put_extensions(&mut b, &self.extensions)?;
        b.put(
            "greedyTokenParity",
            CborValue::Bool(self.greedy_token_parity),
        );
        b.put(
            "logitAbsToleranceMillionths",
            cbor_u32(self.logit_abs_tolerance_millionths),
        );
        b.put("modelRuntimeRoot", cbor_digest(&self.model_runtime_root));
        b.put("notesRoot", cbor_digest(&self.notes_root));
        b.put("supportLevel", cbor_text(&self.support_level));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, CONFORMANCE_KIND)?;
        let out = Self {
            adapter_id: fields.text("adapterId")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            extensions: fields.extensions()?,
            greedy_token_parity: fields.bool("greedyTokenParity")?,
            logit_abs_tolerance_millionths: fields.u32("logitAbsToleranceMillionths")?,
            model_runtime_root: fields.digest("modelRuntimeRoot")?,
            notes_root: fields.digest("notesRoot")?,
            support_level: fields.text("supportLevel")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-conformance", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

const CONVERSION_SOURCE_TYPES: &[&str] = &["F16", "F32", "Q4_K", "Q5_K", "Q6_K", "Q8_0"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionReceiptV1 {
    pub source_artifact_root: DigestHex,
    pub converter_build_root: DigestHex,
    pub conversion_config_root: DigestHex,
    pub destination_artifact_root: DigestHex,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ConversionReceiptV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("validationResult", &self.validation_result, &["matched"])?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(CONVERSION_KIND);
        b.put(
            "conversionConfigRoot",
            cbor_digest(&self.conversion_config_root),
        );
        b.put(
            "converterBuildRoot",
            cbor_digest(&self.converter_build_root),
        );
        b.put(
            "destinationArtifactRoot",
            cbor_digest(&self.destination_artifact_root),
        );
        put_extensions(&mut b, &self.extensions)?;
        b.put(
            "sourceArtifactRoot",
            cbor_digest(&self.source_artifact_root),
        );
        b.put("validationResult", cbor_text(&self.validation_result));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, CONVERSION_KIND)?;
        let out = Self {
            conversion_config_root: fields.digest("conversionConfigRoot")?,
            converter_build_root: fields.digest("converterBuildRoot")?,
            destination_artifact_root: fields.digest("destinationArtifactRoot")?,
            extensions: fields.extensions()?,
            source_artifact_root: fields.digest("sourceArtifactRoot")?,
            validation_result: fields.text("validationResult")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-conversion", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerplexityReportV1 {
    pub reference_logit_root: DigestHex,
    pub candidate_logit_root: DigestHex,
    pub target_token_root: DigestHex,
    pub reporter_build_root: DigestHex,
    pub token_count: u32,
    pub reference_perplexity_micros: u64,
    pub candidate_perplexity_micros: u64,
    pub perplexity_delta_micros: i64,
    pub greedy_parity: bool,
    pub target_accuracy_millionths: u32,
    pub max_abs_logit_delta_millionths: u64,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl PerplexityReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("validationResult", &self.validation_result, &["recorded"])?;
        if self.token_count == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "perplexity token count is zero",
            ));
        }
        if self.target_accuracy_millionths > 1_000_000 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "target accuracy is outside its bounds",
            ));
        }
        let expected = i128::from(self.candidate_perplexity_micros)
            - i128::from(self.reference_perplexity_micros);
        if i128::from(self.perplexity_delta_micros) != expected {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "perplexity delta does not match",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(PERPLEXITY_KIND);
        b.put(
            "candidateLogitRoot",
            cbor_digest(&self.candidate_logit_root),
        );
        b.put(
            "candidatePerplexityMicros",
            cbor_u64(self.candidate_perplexity_micros),
        );
        put_extensions(&mut b, &self.extensions)?;
        b.put("greedyParity", CborValue::Bool(self.greedy_parity));
        b.put(
            "maxAbsLogitDeltaMillionths",
            cbor_u64(self.max_abs_logit_delta_millionths),
        );
        b.put(
            "perplexityDeltaMicros",
            cbor_i64(self.perplexity_delta_micros),
        );
        b.put(
            "referenceLogitRoot",
            cbor_digest(&self.reference_logit_root),
        );
        b.put(
            "referencePerplexityMicros",
            cbor_u64(self.reference_perplexity_micros),
        );
        b.put("reporterBuildRoot", cbor_digest(&self.reporter_build_root));
        b.put(
            "targetAccuracyMillionths",
            cbor_u32(self.target_accuracy_millionths),
        );
        b.put("targetTokenRoot", cbor_digest(&self.target_token_root));
        b.put("tokenCount", cbor_u32(self.token_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, PERPLEXITY_KIND)?;
        let out = Self {
            candidate_logit_root: fields.digest("candidateLogitRoot")?,
            candidate_perplexity_micros: fields.u64("candidatePerplexityMicros")?,
            extensions: fields.extensions()?,
            greedy_parity: fields.bool("greedyParity")?,
            max_abs_logit_delta_millionths: fields.u64("maxAbsLogitDeltaMillionths")?,
            perplexity_delta_micros: fields.i64("perplexityDeltaMicros")?,
            reference_logit_root: fields.digest("referenceLogitRoot")?,
            reference_perplexity_micros: fields.u64("referencePerplexityMicros")?,
            reporter_build_root: fields.digest("reporterBuildRoot")?,
            target_accuracy_millionths: fields.u32("targetAccuracyMillionths")?,
            target_token_root: fields.digest("targetTokenRoot")?,
            token_count: fields.u32("tokenCount")?,
            validation_result: fields.text("validationResult")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-perplexity", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryEstimateReportV1 {
    pub placement_root: DigestHex,
    pub estimator_build_root: DigestHex,
    pub declared_weight_bytes: u64,
    pub declared_kv_bytes: u64,
    pub declared_workspace_bytes: u64,
    pub declared_staging_bytes: u64,
    pub declared_overhead_bytes: u64,
    pub declared_margin_bytes: u64,
    pub declared_total_bytes: u64,
    pub measured_weight_bytes: u64,
    pub measured_kv_bytes: u64,
    pub measured_workspace_bytes: u64,
    pub measured_staging_bytes: u64,
    pub measured_overhead_bytes: u64,
    pub measured_total_bytes: u64,
    pub headroom_bytes: u64,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl MemoryEstimateReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of(
            "validationResult",
            &self.validation_result,
            &["within-bounds"],
        )?;
        let measured = byte_sum(
            &[
                self.measured_weight_bytes,
                self.measured_kv_bytes,
                self.measured_workspace_bytes,
                self.measured_staging_bytes,
                self.measured_overhead_bytes,
            ],
            "measured bytes overflow",
        )?;
        if measured != self.measured_total_bytes {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "measured bytes do not match",
            ));
        }
        let declared = byte_sum(
            &[
                self.declared_weight_bytes,
                self.declared_kv_bytes,
                self.declared_workspace_bytes,
                self.declared_staging_bytes,
                self.declared_overhead_bytes,
                self.declared_margin_bytes,
            ],
            "declared bytes overflow",
        )?;
        if declared != self.declared_total_bytes {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "declared bytes do not match",
            ));
        }
        within_bound(
            self.measured_weight_bytes,
            self.declared_weight_bytes,
            "measured weight bytes exceed the placement bound",
        )?;
        within_bound(
            self.measured_kv_bytes,
            self.declared_kv_bytes,
            "measured kv bytes exceed the placement bound",
        )?;
        within_bound(
            self.measured_workspace_bytes,
            self.declared_workspace_bytes,
            "measured workspace bytes exceed the placement bound",
        )?;
        within_bound(
            self.measured_staging_bytes,
            self.declared_staging_bytes,
            "measured staging bytes exceed the placement bound",
        )?;
        within_bound(
            self.measured_overhead_bytes,
            self.declared_overhead_bytes,
            "measured overhead bytes exceed the placement bound",
        )?;
        let headroom = self
            .declared_total_bytes
            .checked_sub(self.measured_total_bytes)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "memory headroom does not match"))?;
        if headroom != self.headroom_bytes {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "memory headroom does not match",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(MEMORY_KIND);
        b.put("declaredKvBytes", cbor_u64(self.declared_kv_bytes));
        b.put("declaredMarginBytes", cbor_u64(self.declared_margin_bytes));
        b.put(
            "declaredOverheadBytes",
            cbor_u64(self.declared_overhead_bytes),
        );
        b.put(
            "declaredStagingBytes",
            cbor_u64(self.declared_staging_bytes),
        );
        b.put("declaredTotalBytes", cbor_u64(self.declared_total_bytes));
        b.put("declaredWeightBytes", cbor_u64(self.declared_weight_bytes));
        b.put(
            "declaredWorkspaceBytes",
            cbor_u64(self.declared_workspace_bytes),
        );
        b.put(
            "estimatorBuildRoot",
            cbor_digest(&self.estimator_build_root),
        );
        put_extensions(&mut b, &self.extensions)?;
        b.put("headroomBytes", cbor_u64(self.headroom_bytes));
        b.put("measuredKvBytes", cbor_u64(self.measured_kv_bytes));
        b.put(
            "measuredOverheadBytes",
            cbor_u64(self.measured_overhead_bytes),
        );
        b.put(
            "measuredStagingBytes",
            cbor_u64(self.measured_staging_bytes),
        );
        b.put("measuredTotalBytes", cbor_u64(self.measured_total_bytes));
        b.put("measuredWeightBytes", cbor_u64(self.measured_weight_bytes));
        b.put(
            "measuredWorkspaceBytes",
            cbor_u64(self.measured_workspace_bytes),
        );
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("validationResult", cbor_text(&self.validation_result));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, MEMORY_KIND)?;
        let out = Self {
            declared_kv_bytes: fields.u64("declaredKvBytes")?,
            declared_margin_bytes: fields.u64("declaredMarginBytes")?,
            declared_overhead_bytes: fields.u64("declaredOverheadBytes")?,
            declared_staging_bytes: fields.u64("declaredStagingBytes")?,
            declared_total_bytes: fields.u64("declaredTotalBytes")?,
            declared_weight_bytes: fields.u64("declaredWeightBytes")?,
            declared_workspace_bytes: fields.u64("declaredWorkspaceBytes")?,
            estimator_build_root: fields.digest("estimatorBuildRoot")?,
            extensions: fields.extensions()?,
            headroom_bytes: fields.u64("headroomBytes")?,
            measured_kv_bytes: fields.u64("measuredKvBytes")?,
            measured_overhead_bytes: fields.u64("measuredOverheadBytes")?,
            measured_staging_bytes: fields.u64("measuredStagingBytes")?,
            measured_total_bytes: fields.u64("measuredTotalBytes")?,
            measured_weight_bytes: fields.u64("measuredWeightBytes")?,
            measured_workspace_bytes: fields.u64("measuredWorkspaceBytes")?,
            placement_root: fields.digest("placementRoot")?,
            validation_result: fields.text("validationResult")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-memory", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Tokens or requests per second, in micros. Integer division discards the remainder.
pub fn tokens_per_second_micros(count: u64, nanos: u64) -> Result<u64, InferFailure> {
    if count == 0 || nanos == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "throughput rate overflows",
        ));
    }
    let scaled = u128::from(count)
        .checked_mul(TOKENS_PER_SECOND_SCALE)
        .ok_or_else(|| fail(ErrorCode::ContractInvalid, "throughput rate overflows"))?;
    let rate = scaled / u128::from(nanos);
    u64::try_from(rate).map_err(|_| fail(ErrorCode::ContractInvalid, "throughput rate overflows"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThroughputReportV1 {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub prompt_token_root: DigestHex,
    pub output_token_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub request_count: u32,
    pub prefill_nanos: u64,
    pub decode_nanos: u64,
    pub prefill_tokens_per_second_micros: u64,
    pub decode_tokens_per_second_micros: u64,
    pub requests_per_second_micros: u64,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ThroughputReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("validationResult", &self.validation_result, &["recorded"])?;
        one_of(
            "executionMode",
            &self.execution_mode,
            &["isolated-replay", "pinned"],
        )?;
        if self.cache_policy != "off" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefix cache is off for the throughput report",
            ));
        }
        if self.concurrency != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "throughput concurrency is one",
            ));
        }
        if self.run_count != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "throughput run count is one",
            ));
        }
        if self.warm_state != "cold" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "throughput warm state is cold",
            ));
        }
        if self.request_count != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "throughput request count is one",
            ));
        }
        if self.prefill_tokens == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefill token count is zero",
            ));
        }
        if self.decode_tokens == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "decode token count is zero",
            ));
        }
        let span = self
            .prefill_tokens
            .checked_add(self.decode_tokens)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "micro context length overflows"))?;
        if span > MICRO_CONTEXT_TOKENS {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "micro prompt does not fit in the reserved context",
            ));
        }
        if self.prefill_nanos == 0 {
            return Err(fail(ErrorCode::ContractInvalid, "prefill duration is zero"));
        }
        if self.decode_nanos == 0 {
            return Err(fail(ErrorCode::ContractInvalid, "decode duration is zero"));
        }
        let total = self
            .prefill_nanos
            .checked_add(self.decode_nanos)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "throughput duration overflows"))?;
        same_rate(
            u64::from(self.prefill_tokens),
            self.prefill_nanos,
            self.prefill_tokens_per_second_micros,
            "prefill throughput does not match",
        )?;
        same_rate(
            u64::from(self.decode_tokens),
            self.decode_nanos,
            self.decode_tokens_per_second_micros,
            "decode throughput does not match",
        )?;
        same_rate(
            u64::from(self.request_count),
            total,
            self.requests_per_second_micros,
            "request throughput does not match",
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(THROUGHPUT_KIND);
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("decodeNanos", cbor_u64(self.decode_nanos));
        b.put("decodeTokens", cbor_u32(self.decode_tokens));
        b.put(
            "decodeTokensPerSecondMicros",
            cbor_u64(self.decode_tokens_per_second_micros),
        );
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("outputTokenRoot", cbor_digest(&self.output_token_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("prefillNanos", cbor_u64(self.prefill_nanos));
        b.put("prefillTokens", cbor_u32(self.prefill_tokens));
        b.put(
            "prefillTokensPerSecondMicros",
            cbor_u64(self.prefill_tokens_per_second_micros),
        );
        b.put("promptTokenRoot", cbor_digest(&self.prompt_token_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put(
            "requestsPerSecondMicros",
            cbor_u64(self.requests_per_second_micros),
        );
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, THROUGHPUT_KIND)?;
        let out = Self {
            artifact_root: fields.digest("artifactRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            decode_nanos: fields.u64("decodeNanos")?,
            decode_tokens: fields.u32("decodeTokens")?,
            decode_tokens_per_second_micros: fields.u64("decodeTokensPerSecondMicros")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            model_image_root: fields.digest("modelImageRoot")?,
            output_token_root: fields.digest("outputTokenRoot")?,
            placement_root: fields.digest("placementRoot")?,
            prefill_nanos: fields.u64("prefillNanos")?,
            prefill_tokens: fields.u32("prefillTokens")?,
            prefill_tokens_per_second_micros: fields.u64("prefillTokensPerSecondMicros")?,
            prompt_token_root: fields.digest("promptTokenRoot")?,
            request_count: fields.u32("requestCount")?,
            requests_per_second_micros: fields.u64("requestsPerSecondMicros")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-throughput", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Request latency in nanoseconds. Queue time is not included.
pub fn request_latency_nanos(prefill_nanos: u64, decode_nanos: u64) -> Result<u64, InferFailure> {
    prefill_nanos
        .checked_add(decode_nanos)
        .ok_or_else(|| fail(ErrorCode::ContractInvalid, "latency duration overflows"))
}

/// Nanoseconds from the cancel request until the sequence is terminal.
pub fn cancellation_latency_nanos(
    requested_nanos: u64,
    terminal_nanos: u64,
) -> Result<u64, InferFailure> {
    if requested_nanos == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "cancel request time is zero",
        ));
    }
    match terminal_nanos.checked_sub(requested_nanos) {
        Some(latency) if latency > 0 => Ok(latency),
        _ => Err(fail(
            ErrorCode::ContractInvalid,
            "cancel terminal is not after the request",
        )),
    }
}

/// Nanoseconds from the terminal journal event until the receipt file is durable.
pub fn receipt_finalization_nanos(
    terminal_nanos: u64,
    finalized_nanos: u64,
) -> Result<u64, InferFailure> {
    if terminal_nanos == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "receipt terminal time is zero",
        ));
    }
    match finalized_nanos.checked_sub(terminal_nanos) {
        Some(latency) if latency > 0 => Ok(latency),
        _ => Err(fail(
            ErrorCode::ContractInvalid,
            "receipt is not finalized after the terminal",
        )),
    }
}

/// Accepted-event fsync plus receipt-file fsync. The idle gap is not included.
pub fn receipt_overhead_nanos(
    accepted_write_nanos: u64,
    receipt_write_nanos: u64,
) -> Result<u64, InferFailure> {
    if accepted_write_nanos == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "accepted write time is zero",
        ));
    }
    if receipt_write_nanos == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "receipt write time is zero",
        ));
    }
    accepted_write_nanos
        .checked_add(receipt_write_nanos)
        .ok_or_else(|| fail(ErrorCode::ContractInvalid, "receipt overhead overflows"))
}

/// Decode nanoseconds per output token. Integer division discards the remainder.
pub fn time_per_output_token_nanos(
    decode_tokens: u32,
    decode_nanos: u64,
) -> Result<u64, InferFailure> {
    if decode_tokens == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "decode token count is zero",
        ));
    }
    Ok(decode_nanos / u64::from(decode_tokens))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatencyReportV1 {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub prompt_token_root: DigestHex,
    pub output_token_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub request_count: u32,
    pub prefill_nanos: u64,
    pub decode_nanos: u64,
    pub time_to_first_token_nanos: u64,
    pub time_per_output_token_nanos: u64,
    pub request_latency_nanos: u64,
    pub latency_p50_nanos: u64,
    pub latency_p95_nanos: u64,
    pub latency_p99_nanos: u64,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl LatencyReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("validationResult", &self.validation_result, &["recorded"])?;
        one_of(
            "executionMode",
            &self.execution_mode,
            &["isolated-replay", "pinned"],
        )?;
        if self.cache_policy != "off" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefix cache is off for the latency report",
            ));
        }
        if self.concurrency != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "latency concurrency is one",
            ));
        }
        if self.run_count != 1 {
            return Err(fail(ErrorCode::ContractInvalid, "latency run count is one"));
        }
        if self.warm_state != "cold" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "latency warm state is cold",
            ));
        }
        if self.request_count != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "latency request count is one",
            ));
        }
        if self.prefill_tokens == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefill token count is zero",
            ));
        }
        if self.decode_tokens == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "decode token count is zero",
            ));
        }
        let span = self
            .prefill_tokens
            .checked_add(self.decode_tokens)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "micro context length overflows"))?;
        if span > MICRO_CONTEXT_TOKENS {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "micro prompt does not fit in the reserved context",
            ));
        }
        if self.prefill_nanos == 0 {
            return Err(fail(ErrorCode::ContractInvalid, "prefill duration is zero"));
        }
        if self.decode_nanos == 0 {
            return Err(fail(ErrorCode::ContractInvalid, "decode duration is zero"));
        }
        let request = request_latency_nanos(self.prefill_nanos, self.decode_nanos)?;
        let per_token = time_per_output_token_nanos(self.decode_tokens, self.decode_nanos)?;
        same_u64(
            self.prefill_nanos,
            self.time_to_first_token_nanos,
            "time to first token does not match",
        )?;
        same_u64(
            per_token,
            self.time_per_output_token_nanos,
            "time per output token does not match",
        )?;
        same_u64(
            request,
            self.request_latency_nanos,
            "request latency does not match",
        )?;
        same_u64(
            request,
            self.latency_p50_nanos,
            "latency p50 does not match",
        )?;
        same_u64(
            request,
            self.latency_p95_nanos,
            "latency p95 does not match",
        )?;
        same_u64(
            request,
            self.latency_p99_nanos,
            "latency p99 does not match",
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(LATENCY_KIND);
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("decodeNanos", cbor_u64(self.decode_nanos));
        b.put("decodeTokens", cbor_u32(self.decode_tokens));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("latencyP50Nanos", cbor_u64(self.latency_p50_nanos));
        b.put("latencyP95Nanos", cbor_u64(self.latency_p95_nanos));
        b.put("latencyP99Nanos", cbor_u64(self.latency_p99_nanos));
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("outputTokenRoot", cbor_digest(&self.output_token_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("prefillNanos", cbor_u64(self.prefill_nanos));
        b.put("prefillTokens", cbor_u32(self.prefill_tokens));
        b.put("promptTokenRoot", cbor_digest(&self.prompt_token_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("requestLatencyNanos", cbor_u64(self.request_latency_nanos));
        b.put("runCount", cbor_u32(self.run_count));
        b.put(
            "timePerOutputTokenNanos",
            cbor_u64(self.time_per_output_token_nanos),
        );
        b.put(
            "timeToFirstTokenNanos",
            cbor_u64(self.time_to_first_token_nanos),
        );
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, LATENCY_KIND)?;
        let out = Self {
            artifact_root: fields.digest("artifactRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            decode_nanos: fields.u64("decodeNanos")?,
            decode_tokens: fields.u32("decodeTokens")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            latency_p50_nanos: fields.u64("latencyP50Nanos")?,
            latency_p95_nanos: fields.u64("latencyP95Nanos")?,
            latency_p99_nanos: fields.u64("latencyP99Nanos")?,
            model_image_root: fields.digest("modelImageRoot")?,
            output_token_root: fields.digest("outputTokenRoot")?,
            placement_root: fields.digest("placementRoot")?,
            prefill_nanos: fields.u64("prefillNanos")?,
            prefill_tokens: fields.u32("prefillTokens")?,
            prompt_token_root: fields.digest("promptTokenRoot")?,
            request_count: fields.u32("requestCount")?,
            request_latency_nanos: fields.u64("requestLatencyNanos")?,
            run_count: fields.u32("runCount")?,
            time_per_output_token_nanos: fields.u64("timePerOutputTokenNanos")?,
            time_to_first_token_nanos: fields.u64("timeToFirstTokenNanos")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-latency", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One bounded corruption campaign over the nine parser targets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuzzReportV1 {
    pub engine_build_root: DigestHex,
    pub corpus_root: DigestHex,
    pub target_count: u32,
    pub seed_count: u32,
    pub mutation_count: u32,
    pub rejected_count: u32,
    pub distinguished_count: u32,
    pub accepted_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl FuzzReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of(
            "validationResult",
            &self.validation_result,
            &["fail-closed"],
        )?;
        if self.target_count != FUZZ_SEED_COUNT
            || self.seed_count != FUZZ_SEED_COUNT
            || self.mutation_count != FUZZ_MUTATION_COUNT
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "fuzz campaign is not the nine-target slice",
            ));
        }
        if self.accepted_count != 0 {
            return Err(fail(ErrorCode::ContractInvalid, "corruption was accepted"));
        }
        let closed = self
            .rejected_count
            .checked_add(self.distinguished_count)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "fuzz case count overflows"))?;
        if closed != self.mutation_count {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "fuzz case count does not match",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(FUZZ_KIND);
        b.put("acceptedCount", cbor_u32(self.accepted_count));
        b.put("corpusRoot", cbor_digest(&self.corpus_root));
        b.put("distinguishedCount", cbor_u32(self.distinguished_count));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        put_extensions(&mut b, &self.extensions)?;
        b.put("mutationCount", cbor_u32(self.mutation_count));
        b.put("rejectedCount", cbor_u32(self.rejected_count));
        b.put("seedCount", cbor_u32(self.seed_count));
        b.put("targetCount", cbor_u32(self.target_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, FUZZ_KIND)?;
        let out = Self {
            accepted_count: fields.u32("acceptedCount")?,
            corpus_root: fields.digest("corpusRoot")?,
            distinguished_count: fields.u32("distinguishedCount")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            extensions: fields.extensions()?,
            mutation_count: fields.u32("mutationCount")?,
            rejected_count: fields.u32("rejectedCount")?,
            seed_count: fields.u32("seedCount")?,
            target_count: fields.u32("targetCount")?,
            validation_result: fields.text("validationResult")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-fuzz", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Cancellation latency for one cold micro-fixture request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CancellationReportV1 {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub prompt_token_root: DigestHex,
    pub output_token_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub cancel_stage: String,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub requested_nanos: u64,
    pub terminal_nanos: u64,
    pub cancellation_latency_nanos: u64,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl CancellationReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("validationResult", &self.validation_result, &["recorded"])?;
        one_of(
            "executionMode",
            &self.execution_mode,
            &["isolated-replay", "pinned"],
        )?;
        one_of("cancelStage", &self.cancel_stage, &["decode", "prefill"])?;
        if self.cache_policy != "off" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefix cache is off for the cancellation report",
            ));
        }
        if self.concurrency != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "cancellation concurrency is one",
            ));
        }
        if self.run_count != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "cancellation run count is one",
            ));
        }
        if self.warm_state != "cold" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "cancellation warm state is cold",
            ));
        }
        if self.request_count != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "cancellation request count is one",
            ));
        }
        if self.prefill_tokens == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefill token count is zero",
            ));
        }
        if self.cancel_stage == "prefill" && self.decode_tokens != 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefill cancel has output tokens",
            ));
        }
        if self.cancel_stage == "decode" && self.decode_tokens == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "decode cancel has no output tokens",
            ));
        }
        let span = self
            .prefill_tokens
            .checked_add(self.decode_tokens)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "micro context length overflows"))?;
        if span > MICRO_CONTEXT_TOKENS {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "micro prompt does not fit in the reserved context",
            ));
        }
        let latency = cancellation_latency_nanos(self.requested_nanos, self.terminal_nanos)?;
        same_u64(
            latency,
            self.cancellation_latency_nanos,
            "cancellation latency does not match",
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(CANCELLATION_KIND);
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("cancelStage", cbor_text(&self.cancel_stage));
        b.put(
            "cancellationLatencyNanos",
            cbor_u64(self.cancellation_latency_nanos),
        );
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("decodeTokens", cbor_u32(self.decode_tokens));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("outputTokenRoot", cbor_digest(&self.output_token_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("prefillTokens", cbor_u32(self.prefill_tokens));
        b.put("promptTokenRoot", cbor_digest(&self.prompt_token_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("requestedNanos", cbor_u64(self.requested_nanos));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("terminalNanos", cbor_u64(self.terminal_nanos));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, CANCELLATION_KIND)?;
        let out = Self {
            artifact_root: fields.digest("artifactRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            cancel_stage: fields.text("cancelStage")?,
            cancellation_latency_nanos: fields.u64("cancellationLatencyNanos")?,
            concurrency: fields.u32("concurrency")?,
            decode_tokens: fields.u32("decodeTokens")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            model_image_root: fields.digest("modelImageRoot")?,
            output_token_root: fields.digest("outputTokenRoot")?,
            placement_root: fields.digest("placementRoot")?,
            prefill_tokens: fields.u32("prefillTokens")?,
            prompt_token_root: fields.digest("promptTokenRoot")?,
            request_count: fields.u32("requestCount")?,
            requested_nanos: fields.u64("requestedNanos")?,
            run_count: fields.u32("runCount")?,
            terminal_nanos: fields.u64("terminalNanos")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-cancellation", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Wall-clock time from a terminal completion until its receipt is durable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalizationReportV1 {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub prompt_token_root: DigestHex,
    pub output_token_root: DigestHex,
    pub receipt_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub finish_reason: String,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub terminal_nanos: u64,
    pub finalized_nanos: u64,
    pub finalization_latency_nanos: u64,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl FinalizationReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("validationResult", &self.validation_result, &["recorded"])?;
        one_of(
            "executionMode",
            &self.execution_mode,
            &["isolated-replay", "pinned"],
        )?;
        one_of("finishReason", &self.finish_reason, &["length", "stop"])?;
        if self.cache_policy != "off" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefix cache is off for the finalization report",
            ));
        }
        if self.concurrency != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "finalization concurrency is one",
            ));
        }
        if self.run_count != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "finalization run count is one",
            ));
        }
        if self.warm_state != "cold" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "finalization warm state is cold",
            ));
        }
        if self.request_count != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "finalization request count is one",
            ));
        }
        if self.prefill_tokens == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefill token count is zero",
            ));
        }
        if self.finish_reason == "stop" && self.decode_tokens == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a stop receipt has no output tokens",
            ));
        }
        let span = self
            .prefill_tokens
            .checked_add(self.decode_tokens)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "micro context length overflows"))?;
        if span > MICRO_CONTEXT_TOKENS {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "micro prompt does not fit in the reserved context",
            ));
        }
        let latency = receipt_finalization_nanos(self.terminal_nanos, self.finalized_nanos)?;
        same_u64(
            latency,
            self.finalization_latency_nanos,
            "receipt finalization latency does not match",
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(FINALIZATION_KIND);
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("decodeTokens", cbor_u32(self.decode_tokens));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put(
            "finalizationLatencyNanos",
            cbor_u64(self.finalization_latency_nanos),
        );
        b.put("finalizedNanos", cbor_u64(self.finalized_nanos));
        b.put("finishReason", cbor_text(&self.finish_reason));
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("outputTokenRoot", cbor_digest(&self.output_token_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("prefillTokens", cbor_u32(self.prefill_tokens));
        b.put("promptTokenRoot", cbor_digest(&self.prompt_token_root));
        b.put("receiptRoot", cbor_digest(&self.receipt_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("terminalNanos", cbor_u64(self.terminal_nanos));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, FINALIZATION_KIND)?;
        let out = Self {
            artifact_root: fields.digest("artifactRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            decode_tokens: fields.u32("decodeTokens")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            finalization_latency_nanos: fields.u64("finalizationLatencyNanos")?,
            finalized_nanos: fields.u64("finalizedNanos")?,
            finish_reason: fields.text("finishReason")?,
            model_image_root: fields.digest("modelImageRoot")?,
            output_token_root: fields.digest("outputTokenRoot")?,
            placement_root: fields.digest("placementRoot")?,
            prefill_tokens: fields.u32("prefillTokens")?,
            prompt_token_root: fields.digest("promptTokenRoot")?,
            receipt_root: fields.digest("receiptRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            terminal_nanos: fields.u64("terminalNanos")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-finalization", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Accepted-event write plus receipt-file write for one stored receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverheadReportV1 {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub prompt_token_root: DigestHex,
    pub output_token_root: DigestHex,
    pub receipt_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub finish_reason: String,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub accepted_write_nanos: u64,
    pub receipt_write_nanos: u64,
    pub receipt_overhead_nanos: u64,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl OverheadReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("validationResult", &self.validation_result, &["recorded"])?;
        one_of(
            "executionMode",
            &self.execution_mode,
            &["isolated-replay", "pinned"],
        )?;
        one_of("finishReason", &self.finish_reason, &["length", "stop"])?;
        if self.cache_policy != "off" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefix cache is off for the overhead report",
            ));
        }
        if self.concurrency != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "overhead concurrency is one",
            ));
        }
        if self.run_count != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "overhead run count is one",
            ));
        }
        if self.warm_state != "cold" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "overhead warm state is cold",
            ));
        }
        if self.request_count != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "overhead request count is one",
            ));
        }
        if self.prefill_tokens == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefill token count is zero",
            ));
        }
        if self.finish_reason == "stop" && self.decode_tokens == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a stop receipt has no output tokens",
            ));
        }
        let span = self
            .prefill_tokens
            .checked_add(self.decode_tokens)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "micro context length overflows"))?;
        if span > MICRO_CONTEXT_TOKENS {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "micro prompt does not fit in the reserved context",
            ));
        }
        let overhead = receipt_overhead_nanos(self.accepted_write_nanos, self.receipt_write_nanos)?;
        same_u64(
            overhead,
            self.receipt_overhead_nanos,
            "receipt overhead does not match",
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(OVERHEAD_KIND);
        b.put("acceptedWriteNanos", cbor_u64(self.accepted_write_nanos));
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("decodeTokens", cbor_u32(self.decode_tokens));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("finishReason", cbor_text(&self.finish_reason));
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("outputTokenRoot", cbor_digest(&self.output_token_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("prefillTokens", cbor_u32(self.prefill_tokens));
        b.put("promptTokenRoot", cbor_digest(&self.prompt_token_root));
        b.put(
            "receiptOverheadNanos",
            cbor_u64(self.receipt_overhead_nanos),
        );
        b.put("receiptRoot", cbor_digest(&self.receipt_root));
        b.put("receiptWriteNanos", cbor_u64(self.receipt_write_nanos));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, OVERHEAD_KIND)?;
        let out = Self {
            accepted_write_nanos: fields.u64("acceptedWriteNanos")?,
            artifact_root: fields.digest("artifactRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            decode_tokens: fields.u32("decodeTokens")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            finish_reason: fields.text("finishReason")?,
            model_image_root: fields.digest("modelImageRoot")?,
            output_token_root: fields.digest("outputTokenRoot")?,
            placement_root: fields.digest("placementRoot")?,
            prefill_tokens: fields.u32("prefillTokens")?,
            prompt_token_root: fields.digest("promptTokenRoot")?,
            receipt_overhead_nanos: fields.u64("receiptOverheadNanos")?,
            receipt_root: fields.digest("receiptRoot")?,
            receipt_write_nanos: fields.u64("receiptWriteNanos")?,
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
        digest_value("infer-overhead", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

fn same_u64(computed: u64, stored: u64, message: &str) -> Result<(), InferFailure> {
    if computed == stored {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn same_rate(count: u64, nanos: u64, stored: u64, message: &str) -> Result<(), InferFailure> {
    if tokens_per_second_micros(count, nanos)? != stored {
        Err(fail(ErrorCode::ContractInvalid, message))
    } else {
        Ok(())
    }
}

fn byte_sum(values: &[u64], overflow: &str) -> Result<u64, InferFailure> {
    let mut total = 0u64;
    for value in values {
        total = total
            .checked_add(*value)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, overflow))?;
    }
    Ok(total)
}

fn within_bound(measured: u64, declared: u64, message: &str) -> Result<(), InferFailure> {
    if measured > declared {
        Err(fail(ErrorCode::ContractInvalid, message))
    } else {
        Ok(())
    }
}

/// `H(infer-conversion-config, { destinationDtype, ne, operation, sourceType })`.
pub fn conversion_config_root(source_type: &str, ne: &[u64]) -> Result<DigestHex, InferFailure> {
    one_of("sourceType", source_type, CONVERSION_SOURCE_TYPES)?;
    if ne.is_empty() || ne.len() > 4 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "gguf tensor dimension count is outside 1..=4",
        ));
    }
    if ne.contains(&0) {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "gguf tensor dimension is zero",
        ));
    }
    let block = match source_type {
        "Q8_0" => 32,
        "Q4_K" | "Q5_K" | "Q6_K" => 256,
        _ => 1,
    };
    if ne[0] % block != 0 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "gguf tensor dimension is not a multiple of the block",
        ));
    }
    let mut count = 1u64;
    for dim in ne {
        count = count.checked_mul(*dim).ok_or_else(|| {
            fail(
                ErrorCode::ModelImageInvalid,
                "gguf tensor byte length overflows",
            )
        })?;
    }
    if count.checked_mul(4).is_none() {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "gguf tensor byte length overflows",
        ));
    }
    let mut b = Builder::bare();
    b.put("destinationDtype", cbor_text("f32"));
    b.put(
        "ne",
        CborValue::Array(ne.iter().copied().map(cbor_u64).collect()),
    );
    b.put("operation", cbor_text("dequant"));
    b.put("sourceType", cbor_text(source_type));
    digest_value("infer-conversion-config", &b.finish())
}

pub fn output_token_root(token_ids: &[u32]) -> Result<DigestHex, InferFailure> {
    digest_value(
        "infer-output-tokens",
        &CborValue::Array(token_ids.iter().copied().map(cbor_u32).collect()),
    )
}

pub fn output_text_root(text: &str) -> Result<DigestHex, InferFailure> {
    digest_value(
        "infer-output-text",
        &CborValue::Bytes(text.as_bytes().to_vec()),
    )
}

/// `H(infer-logits, little-endian f32 bytes)`. The domain is not a contract kind.
pub fn logit_root(values: &[f32]) -> Result<DigestHex, InferFailure> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    digest_bytes("infer-logits", &bytes)
}
