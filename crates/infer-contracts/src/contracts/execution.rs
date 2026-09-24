use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    bounded_text, cbor_digest, cbor_text, cbor_u32, cbor_u64, expect_kind_version, one_of,
    sorted_unique, text_array, u32_array, Fields,
};

use super::common::{put_extensions, Builder, FixedPointSamplerV1};

pub const SAMPLER_PLAN_KIND: &str = "knolo.infer.sampler-plan";
pub const GRAMMAR_PLAN_KIND: &str = "knolo.infer.grammar-plan";
pub const INFERENCE_INTENT_KIND: &str = "knolo.infer.inference-intent";
pub const EXECUTION_PLAN_KIND: &str = "knolo.infer.execution-plan";
pub const INFERENCE_EVENT_KIND: &str = "knolo.infer.inference-event";

pub const SAMPLER_ORDER_V1: &[&str] = &[
    "logits",
    "banned-token-mask",
    "grammar-mask",
    "repetition-penalty",
    "presence-penalty",
    "frequency-penalty",
    "temperature",
    "top-k",
    "top-p",
    "min-p",
    "sample",
    "stop",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplerPlanV1 {
    pub settings: FixedPointSamplerV1,
    pub rng: String,
    pub seed: Option<u64>,
    pub stream: Option<u64>,
    pub tie_break: String,
    pub eos_token_ids: Vec<u32>,
    pub stop_string_roots: Vec<DigestHex>,
    pub extensions: BTreeMap<String, CborValue>,
}

impl SamplerPlanV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        self.settings.validate()?;
        one_of("tieBreak", &self.tie_break, &["lowest-token-id"])?;
        one_of("rng", &self.rng, &["none", "philox-4x32-v1"])?;
        if self.settings.temperature_micros == 0 {
            if self.rng != "none" || self.seed.is_some() || self.stream.is_some() {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "greedy sampler cannot carry an RNG seed",
                ));
            }
        } else if self.rng != "philox-4x32-v1" || self.seed.is_none() || self.stream.is_none() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "sampled generation requires philox seed and stream",
            ));
        }
        if self.eos_token_ids.len() > 64 || self.stop_string_roots.len() > 64 {
            return Err(fail(ErrorCode::ContractInvalid, "stop list is too large"));
        }
        for window in self.eos_token_ids.windows(2) {
            if window[0] >= window[1] {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "eos token ids must be strictly increasing",
                ));
            }
        }
        let stops: Vec<_> = self
            .stop_string_roots
            .iter()
            .map(|d| d.as_str().to_string())
            .collect();
        sorted_unique("stopStringRoots", &stops)?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let order = CborValue::Array(SAMPLER_ORDER_V1.iter().copied().map(cbor_text).collect());
        let mut b = Builder::typed(SAMPLER_PLAN_KIND);
        b.put(
            "eosTokenIds",
            CborValue::Array(self.eos_token_ids.iter().copied().map(cbor_u32).collect()),
        );
        put_extensions(&mut b, &self.extensions)?;
        b.put(
            "frequencyPenaltyMicros",
            cbor_u32(self.settings.frequency_penalty_micros),
        );
        b.put("maxOutputTokens", cbor_u32(self.settings.max_output_tokens));
        b.put("minPMillionths", cbor_u32(self.settings.min_p_millionths));
        b.put(
            "presencePenaltyMicros",
            cbor_u32(self.settings.presence_penalty_micros),
        );
        b.put("processingOrder", order);
        b.put(
            "repetitionPenaltyMicros",
            cbor_u32(self.settings.repetition_penalty_micros),
        );
        b.put("rng", cbor_text(&self.rng));
        b.put_opt("seed", self.seed.map(cbor_u64));
        b.put(
            "stopStringRoots",
            CborValue::Array(self.stop_string_roots.iter().map(cbor_digest).collect()),
        );
        b.put_opt("stream", self.stream.map(cbor_u64));
        b.put(
            "temperatureMicros",
            cbor_u32(self.settings.temperature_micros),
        );
        b.put("tieBreak", cbor_text(&self.tie_break));
        b.put("topK", cbor_u32(self.settings.top_k));
        b.put("topPMillionths", cbor_u32(self.settings.top_p_millionths));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, SAMPLER_PLAN_KIND)?;
        let order = text_array(&fields.array("processingOrder")?, "processingOrder")?;
        let expected: Vec<_> = SAMPLER_ORDER_V1.iter().map(|s| (*s).to_string()).collect();
        if order != expected {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "sampler processing order is not version 1",
            ));
        }
        let settings = FixedPointSamplerV1 {
            frequency_penalty_micros: fields.u32("frequencyPenaltyMicros")?,
            max_output_tokens: fields.u32("maxOutputTokens")?,
            min_p_millionths: fields.u32("minPMillionths")?,
            presence_penalty_micros: fields.u32("presencePenaltyMicros")?,
            repetition_penalty_micros: fields.u32("repetitionPenaltyMicros")?,
            temperature_micros: fields.u32("temperatureMicros")?,
            top_k: fields.u32("topK")?,
            top_p_millionths: fields.u32("topPMillionths")?,
        };
        let out = Self {
            eos_token_ids: u32_array(&fields.array("eosTokenIds")?, "eosTokenIds")?,
            extensions: fields.extensions()?,
            rng: fields.text("rng")?,
            seed: fields.opt_u64("seed")?,
            settings,
            stop_string_roots: crate::fields::digest_array(
                &fields.array("stopStringRoots")?,
                "stopStringRoots",
            )?,
            stream: fields.opt_u64("stream")?,
            tie_break: fields.text("tieBreak")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-sampler", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarPlanV1 {
    pub format: String,
    pub source_root: DigestHex,
    pub enabled: bool,
    pub extensions: BTreeMap<String, CborValue>,
}

impl GrammarPlanV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of(
            "format",
            &self.format,
            &["grammar", "json-schema", "none", "regex"],
        )?;
        if self.format == "none" && self.enabled {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "grammar format none cannot be enabled",
            ));
        }
        if self.format != "none" && !self.enabled {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a named grammar format must be enabled",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(GRAMMAR_PLAN_KIND);
        b.put("enabled", CborValue::Bool(self.enabled));
        put_extensions(&mut b, &self.extensions)?;
        b.put("format", cbor_text(&self.format));
        b.put("sourceRoot", cbor_digest(&self.source_root));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, GRAMMAR_PLAN_KIND)?;
        let out = Self {
            enabled: fields.bool("enabled")?,
            extensions: fields.extensions()?,
            format: fields.text("format")?,
            source_root: fields.digest("sourceRoot")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-grammar", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LimitsV1 {
    pub max_output_tokens: u32,
    pub max_prompt_tokens: u32,
    pub deadline_unix_micros: Option<u64>,
}

impl LimitsV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        if self.max_output_tokens == 0
            || self.max_prompt_tokens == 0
            || self.max_output_tokens > 1_048_576
            || self.max_prompt_tokens > 1_048_576
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "limits are outside their bounds",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put_opt(
            "deadlineUnixMicros",
            self.deadline_unix_micros.map(cbor_u64),
        );
        b.put("maxOutputTokens", cbor_u32(self.max_output_tokens));
        b.put("maxPromptTokens", cbor_u32(self.max_prompt_tokens));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            deadline_unix_micros: fields.opt_u64("deadlineUnixMicros")?,
            max_output_tokens: fields.u32("maxOutputTokens")?,
            max_prompt_tokens: fields.u32("maxPromptTokens")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-limits", &self.to_cbor()?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceIntentV1 {
    pub model_runtime_root: DigestHex,
    pub prompt_plan_root: DigestHex,
    pub sampler_plan_root: DigestHex,
    pub grammar_plan_root: Option<DigestHex>,
    pub evidence_binding_root: Option<DigestHex>,
    pub requested_mode: String,
    pub limits_root: DigestHex,
    pub extensions: BTreeMap<String, CborValue>,
}

impl InferenceIntentV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of(
            "requestedMode",
            &self.requested_mode,
            &["isolated-replay", "pinned", "throughput"],
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(INFERENCE_INTENT_KIND);
        b.put_opt(
            "evidenceBindingRoot",
            self.evidence_binding_root.as_ref().map(cbor_digest),
        );
        put_extensions(&mut b, &self.extensions)?;
        b.put_opt(
            "grammarPlanRoot",
            self.grammar_plan_root.as_ref().map(cbor_digest),
        );
        b.put("limitsRoot", cbor_digest(&self.limits_root));
        b.put("modelRuntimeRoot", cbor_digest(&self.model_runtime_root));
        b.put("promptPlanRoot", cbor_digest(&self.prompt_plan_root));
        b.put("requestedMode", cbor_text(&self.requested_mode));
        b.put("samplerPlanRoot", cbor_digest(&self.sampler_plan_root));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, INFERENCE_INTENT_KIND)?;
        let out = Self {
            evidence_binding_root: fields.opt_digest("evidenceBindingRoot")?,
            extensions: fields.extensions()?,
            grammar_plan_root: fields.opt_digest("grammarPlanRoot")?,
            limits_root: fields.digest("limitsRoot")?,
            model_runtime_root: fields.digest("modelRuntimeRoot")?,
            prompt_plan_root: fields.digest("promptPlanRoot")?,
            requested_mode: fields.text("requestedMode")?,
            sampler_plan_root: fields.digest("samplerPlanRoot")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-request-intent", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPlanV1 {
    pub intent_root: DigestHex,
    pub placement_root: DigestHex,
    pub kernel_plan_root: DigestHex,
    pub scheduling_mode: String,
    pub prefill_chunk_tokens: u32,
    pub kv_block_size: u32,
    pub prefix_cache_enabled: bool,
    pub prefix_cache_namespace: String,
    pub speculative_plan_root: Option<DigestHex>,
    pub receipt_policy: String,
    pub mode: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ExecutionPlanV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of(
            "schedulingMode",
            &self.scheduling_mode,
            &["continuous", "isolated"],
        )?;
        one_of(
            "receiptPolicy",
            &self.receipt_policy,
            &["atomic-verified", "compatibility", "durable-stream"],
        )?;
        one_of(
            "mode",
            &self.mode,
            &["isolated-replay", "pinned", "throughput"],
        )?;
        if self.prefix_cache_enabled {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefix cache is disabled in contract version 1",
            ));
        }
        bounded_text("prefixCacheNamespace", &self.prefix_cache_namespace, 64)?;
        if self.prefill_chunk_tokens == 0 || self.prefill_chunk_tokens > 1_048_576 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefill chunk size is invalid",
            ));
        }
        if !(16..=65_536).contains(&self.kv_block_size) || !self.kv_block_size.is_power_of_two() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "KV block size must be a power of two from 16 to 65536",
            ));
        }
        if self.speculative_plan_root.is_some() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "speculative decoding is not part of contract version 1",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut prefix = Builder::bare();
        prefix.put("enabled", CborValue::Bool(self.prefix_cache_enabled));
        prefix.put("namespace", cbor_text(&self.prefix_cache_namespace));
        let mut b = Builder::typed(EXECUTION_PLAN_KIND);
        put_extensions(&mut b, &self.extensions)?;
        b.put("intentRoot", cbor_digest(&self.intent_root));
        b.put("kernelPlanRoot", cbor_digest(&self.kernel_plan_root));
        b.put("kvBlockSize", cbor_u32(self.kv_block_size));
        b.put("mode", cbor_text(&self.mode));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("prefillChunkTokens", cbor_u32(self.prefill_chunk_tokens));
        b.put("prefixCache", prefix.finish());
        b.put("receiptPolicy", cbor_text(&self.receipt_policy));
        b.put("schedulingMode", cbor_text(&self.scheduling_mode));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, EXECUTION_PLAN_KIND)?;
        let mut prefix = fields.nested("prefixCache")?;
        let prefix_cache_enabled = prefix.bool("enabled")?;
        let prefix_cache_namespace = prefix.text("namespace")?;
        prefix.finish()?;
        let out = Self {
            extensions: fields.extensions()?,
            intent_root: fields.digest("intentRoot")?,
            kernel_plan_root: fields.digest("kernelPlanRoot")?,
            kv_block_size: fields.u32("kvBlockSize")?,
            mode: fields.text("mode")?,
            placement_root: fields.digest("placementRoot")?,
            prefill_chunk_tokens: fields.u32("prefillChunkTokens")?,
            prefix_cache_enabled,
            prefix_cache_namespace,
            receipt_policy: fields.text("receiptPolicy")?,
            scheduling_mode: fields.text("schedulingMode")?,
            speculative_plan_root: None,
        };
        if fields.optional("speculativePlanRoot")?.is_some() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "speculative decoding is not part of contract version 1",
            ));
        }
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-execution-plan", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceEventV1 {
    pub request_id: String,
    pub attempt: u32,
    pub name: String,
    pub previous_event_root: Option<DigestHex>,
    pub payload_root: DigestHex,
    pub extensions: BTreeMap<String, CborValue>,
}

impl InferenceEventV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        if self.request_id.len() > 64
            || self.request_id.is_empty()
            || !self
                .request_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(fail(ErrorCode::ContractInvalid, "request id is invalid"));
        }
        one_of(
            "name",
            &self.name,
            &[
                "accepted",
                "cancelled",
                "completed",
                "failed",
                "prefill",
                "token",
            ],
        )?;
        if self.attempt > 16 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "attempt is outside its bounds",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(INFERENCE_EVENT_KIND);
        b.put("attempt", cbor_u32(self.attempt));
        put_extensions(&mut b, &self.extensions)?;
        b.put("name", cbor_text(&self.name));
        b.put("payloadRoot", cbor_digest(&self.payload_root));
        b.put_opt(
            "previousEventRoot",
            self.previous_event_root.as_ref().map(cbor_digest),
        );
        b.put("requestId", cbor_text(&self.request_id));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, INFERENCE_EVENT_KIND)?;
        let out = Self {
            attempt: fields.u32("attempt")?,
            extensions: fields.extensions()?,
            name: fields.text("name")?,
            payload_root: fields.digest("payloadRoot")?,
            previous_event_root: fields.opt_digest("previousEventRoot")?,
            request_id: fields.text("requestId")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        let event = self.to_cbor()?;
        let mut b = Builder::bare();
        b.put("event", event);
        b.put(
            "previousEventRoot",
            match &self.previous_event_root {
                Some(root) => cbor_digest(root),
                None => CborValue::Null,
            },
        );
        digest_value("infer-execution-event", &b.finish())
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
