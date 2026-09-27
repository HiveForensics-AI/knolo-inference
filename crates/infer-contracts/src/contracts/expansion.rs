//! Domain separation, the Phase 6 records that stay off, and the Phase 8
//! records that stay off.
//!
//! Each report records one cold micro fixture. None of them separates a
//! domain, hashes a payload, approximates attention, allocates hybrid
//! attention, applies linear attention, applies an mHC connection, runs an
//! MTP head, binds a digest, opens a media part, places a large expert,
//! allocates an extreme-context page, runs a vision projector, rescales
//! RoPE, selects an exact attention kernel, applies an RMSNorm kernel,
//! applies a RoPE kernel, fuses a residual, routes an expert, places an
//! expert, selects a grouped kernel, compares a router, selects a GLM
//! adapter, blesses a workstation recipe, selects a mixed placement, or
//! applies expert capacity. The layouts are
//! specified in `spec/KIP-INFER-0116-domain-separation.md` through
//! `spec/KIP-INFER-0130-mtp.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{cbor_digest, cbor_text, cbor_u32, expect_kind_version, one_of, Fields};

use super::common::{put_extensions, Builder};
use super::exposure::{ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES};
use super::lifecycle::cold_single;
use super::resilience::ED25519_SCALAR_BYTES;

pub const MAX_WINDOW_TOKENS: u32 = 16;
pub const MAX_EXPERTS: u32 = 16;
pub const MAX_PLACEMENT_DEVICES: u32 = 2;
pub const MAX_GROUP_COUNT: u32 = 16;
pub const MAX_ROUTER_SAMPLES: u32 = 16;
pub const MAX_BENCHMARK_COUNT: u32 = 32;
pub const MAX_CAPACITY_TOKENS: u32 = 16;
pub const MAX_HYBRID_TOKENS: u32 = 16;
pub const MAX_LINEAR_TOKENS: u32 = 16;
pub const MAX_MHC_STREAMS: u32 = 16;
pub const MAX_MTP_TOKENS: u32 = 16;
pub const MAX_MULTIMODAL_PARTS: u32 = 16;
pub const MAX_LARGE_EXPERTS: u32 = 16;
pub const MAX_EXTREME_TOKENS: u32 = 16;
pub const MAX_VISION_PATCHES: u32 = 16;
pub const MAX_POSITIONAL_SCALE: u32 = 16;
pub const MAX_EXACT_TILES: u32 = 16;
pub const MAX_RMS_LAYERS: u32 = 16;
pub const MAX_ROPE_LAYERS: u32 = 16;
pub const MAX_RESIDUAL_LAYERS: u32 = 16;
pub const GLM_ADAPTER: &str = "knolo.glm.v1";

const SEPARATE_STATUSES: &[&str] = &["separated", "rejected", "unsigned-local"];
const ATTENTION_REASONS: &[&str] = &["approximation", "window", "sparsity", "quantized-kv"];
const MOE_REASONS: &[&str] = &["router", "expert", "shared"];
const EXPERT_PLACEMENT_REASONS: &[&str] = &["residency", "offload", "asymmetric"];
const GROUPED_REASONS: &[&str] = &["gemm", "routing"];
const ROUTER_REASONS: &[&str] = &["logits", "selection", "load"];
const GLM_REASONS: &[&str] = &["adapter", "inventory", "recipe"];
const WORKSTATION_REASONS: &[&str] = &["profile", "benchmark", "fallback"];
const MIXED_REASONS: &[&str] = &["single", "mixed", "fallback"];
const CAPACITY_REASONS: &[&str] = &["overflow", "drop", "balance"];
const HASH_STATUSES: &[&str] = &["hashed", "rejected", "unsigned-local"];
const HYBRID_REASONS: &[&str] = &["state", "sliding", "full"];
const LINEAR_REASONS: &[&str] = &["kda", "state", "decay"];
const MHC_REASONS: &[&str] = &["connection", "manifold", "residual"];
const MTP_REASONS: &[&str] = &["head", "draft", "accept"];
const BIND_STATUSES: &[&str] = &["bound", "rejected", "unsigned-local"];
const MULTIMODAL_REASONS: &[&str] = &["image", "audio", "video"];
const LARGE_PLACEMENT_REASONS: &[&str] = &["span", "shard", "replicate"];
const EXTREME_CACHE_REASONS: &[&str] = &["window", "yarn", "page"];
const VISION_REASONS: &[&str] = &["projector", "patch", "embed"];
const POSITIONAL_REASONS: &[&str] = &["yarn", "ntk", "rope"];
const EXACT_ATTENTION_REASONS: &[&str] = &["flash", "tile", "split"];
const RMSNORM_REASONS: &[&str] = &["norm", "weight", "eps"];
const ROPE_REASONS: &[&str] = &["rotary", "frequency", "partial"];
const RESIDUAL_REASONS: &[&str] = &["fused", "add", "norm"];

#[allow(clippy::too_many_arguments)]
fn cold_fields(
    noun: &str,
    validation_result: &str,
    allowed_validation: &[&str],
    execution_mode: &str,
    cache_policy: &str,
    concurrency: u32,
    run_count: u32,
    warm_state: &str,
    request_count: u32,
    extensions: &BTreeMap<String, CborValue>,
) -> Result<(), InferFailure> {
    one_of("validationResult", validation_result, allowed_validation)?;
    one_of(
        "executionMode",
        execution_mode,
        &["isolated-replay", "pinned"],
    )?;
    cold_single(
        noun,
        cache_policy,
        concurrency,
        run_count,
        warm_state,
        request_count,
    )?;
    if extensions.is_empty() {
        Ok(())
    } else {
        Err(fail(
            ErrorCode::ContractInvalid,
            format!("the {noun} extensions are empty"),
        ))
    }
}

fn exact(value: &str, expected: &str, message: &str) -> Result<(), InferFailure> {
    if value == expected {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn exact_u32(value: u32, expected: u32, message: &str) -> Result<(), InferFailure> {
    if value == expected {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn forbid(flag: bool, message: &str) -> Result<(), InferFailure> {
    if flag {
        Err(fail(ErrorCode::ContractInvalid, message))
    } else {
        Ok(())
    }
}

fn require_flag(flag: bool, message: &str) -> Result<(), InferFailure> {
    if flag {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn at_most(value: u32, cap: u32, message: &str) -> Result<(), InferFailure> {
    if value > cap {
        Err(fail(ErrorCode::ContractInvalid, message))
    } else {
        Ok(())
    }
}

fn distinct(left: &DigestHex, right: &DigestHex, message: &str) -> Result<(), InferFailure> {
    if left == right {
        Err(fail(ErrorCode::ContractInvalid, message))
    } else {
        Ok(())
    }
}

fn compared_counts(public_key: u32, signature: u32, scalar: u32) -> Result<(), InferFailure> {
    exact_u32(
        public_key,
        ED25519_PUBLIC_KEY_BYTES,
        "ed25519 public keys are 32 bytes",
    )?;
    exact_u32(
        signature,
        ED25519_SIGNATURE_BYTES,
        "ed25519 signatures are 64 bytes",
    )?;
    exact_u32(scalar, ED25519_SCALAR_BYTES, "ed25519 scalars are 32 bytes")
}

fn unsigned_counts(public_key: u32, signature: u32, scalar: u32) -> Result<(), InferFailure> {
    exact_u32(public_key, 0, "an unsigned release carries a public key")?;
    exact_u32(signature, 0, "an unsigned release carries signature bytes")?;
    exact_u32(scalar, 0, "an unsigned release carries a scalar")
}

fn stopped(forward_ran: bool, receipt_stored: bool, subject: &str) -> Result<(), InferFailure> {
    forbid(forward_ran, &format!("{subject} does not run the forward"))?;
    forbid(receipt_stored, &format!("{subject} stores no receipt"))
}

pub const DOMAIN_KIND: &str = "knolo.infer.domain-report";

/// Domain separation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub domain_root: DigestHex,
    pub message_root: DigestHex,
    pub separate_status: String,
    pub domain_separated: bool,
    pub payload_hashed: bool,
    pub public_key_bytes: u32,
    pub signature_bytes: u32,
    pub scalar_bytes: u32,
    pub key_material_present: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl DomainReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "domain",
            &self.validation_result,
            &["recorded", "verified"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        distinct(
            &self.domain_root,
            &self.engine_build_root,
            "the domain repeats the engine build",
        )?;
        distinct(
            &self.domain_root,
            &self.placement_root,
            "the domain repeats the placement",
        )?;
        distinct(
            &self.message_root,
            &self.engine_build_root,
            "the separated message repeats the engine build",
        )?;
        distinct(
            &self.message_root,
            &self.placement_root,
            "the separated message repeats the placement",
        )?;
        distinct(
            &self.message_root,
            &self.domain_root,
            "the separated message repeats the domain",
        )?;
        forbid(
            self.key_material_present,
            "key material stays in host storage",
        )?;
        forbid(self.payload_hashed, "the payload stays unhashed")?;
        one_of("separateStatus", &self.separate_status, SEPARATE_STATUSES)?;
        match self.separate_status.as_str() {
            "unsigned-local" => {
                unsigned_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                forbid(
                    self.domain_separated,
                    "an unsigned release separates the domain",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a separated domain is verified",
                )?;
            }
            "separated" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.domain_separated,
                    "a separated domain records the prefix",
                )?;
                exact(
                    &self.validation_result,
                    "verified",
                    "a separated domain is verified",
                )?;
            }
            "rejected" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.domain_separated,
                    "a rejected domain records the prefix",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a separated domain is verified",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field separateStatus has an unsupported value",
                ));
            }
        };
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(DOMAIN_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("domainRoot", cbor_digest(&self.domain_root));
        b.put("messageRoot", cbor_digest(&self.message_root));
        b.put("separateStatus", cbor_text(&self.separate_status));
        b.put("domainSeparated", CborValue::Bool(self.domain_separated));
        b.put("payloadHashed", CborValue::Bool(self.payload_hashed));
        b.put("publicKeyBytes", cbor_u32(self.public_key_bytes));
        b.put("signatureBytes", cbor_u32(self.signature_bytes));
        b.put("scalarBytes", cbor_u32(self.scalar_bytes));
        b.put(
            "keyMaterialPresent",
            CborValue::Bool(self.key_material_present),
        );
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, DOMAIN_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            domain_root: fields.digest("domainRoot")?,
            domain_separated: fields.bool("domainSeparated")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            key_material_present: fields.bool("keyMaterialPresent")?,
            message_root: fields.digest("messageRoot")?,
            payload_hashed: fields.bool("payloadHashed")?,
            placement_root: fields.digest("placementRoot")?,
            public_key_bytes: fields.u32("publicKeyBytes")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            scalar_bytes: fields.u32("scalarBytes")?,
            separate_status: fields.text("separateStatus")?,
            signature_bytes: fields.u32("signatureBytes")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            extensions: fields.extensions()?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-domain", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const ATTENTION_KIND: &str = "knolo.infer.attention-report";

/// Attention approximation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub window_tokens: u32,
    pub code: String,
    pub retryable: bool,
    pub attention_exact: bool,
    pub approximated: bool,
    pub window_applied: bool,
    pub sparsity_applied: bool,
    pub kv_quantized: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl AttentionReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "attention",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "an attention modification is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "an attention modification is not retryable")?;
        require_flag(self.attention_exact, "attention stays exact")?;
        forbid(self.approximated, "an approximation is not applied")?;
        forbid(self.window_applied, "a sliding window is not applied")?;
        forbid(self.sparsity_applied, "a sparsity rule is not applied")?;
        forbid(self.kv_quantized, "quantized KV is not applied")?;
        stopped(
            self.forward_ran,
            self.receipt_stored,
            "an attention modification",
        )?;
        at_most(
            self.window_tokens,
            MAX_WINDOW_TOKENS,
            "window span exceeds the record cap",
        )?;
        one_of("reason", &self.reason, ATTENTION_REASONS)?;
        match self.reason.as_str() {
            "window" => {
                if self.window_tokens == 0 {
                    Err(fail(
                        ErrorCode::ContractInvalid,
                        "a window refusal names a span",
                    ))
                } else {
                    Ok(())
                }
            }
            "approximation" => exact_u32(
                self.window_tokens,
                0,
                "an approximation refusal carries no window",
            ),
            "sparsity" => exact_u32(
                self.window_tokens,
                0,
                "a sparsity refusal carries no window",
            ),
            "quantized-kv" => exact_u32(
                self.window_tokens,
                0,
                "a quantized-kv refusal carries no window",
            ),
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                "field reason has an unsupported value",
            )),
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(ATTENTION_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("windowTokens", cbor_u32(self.window_tokens));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("attentionExact", CborValue::Bool(self.attention_exact));
        b.put("approximated", CborValue::Bool(self.approximated));
        b.put("windowApplied", CborValue::Bool(self.window_applied));
        b.put("sparsityApplied", CborValue::Bool(self.sparsity_applied));
        b.put("kvQuantized", CborValue::Bool(self.kv_quantized));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, ATTENTION_KIND)?;
        let out = Self {
            approximated: fields.bool("approximated")?,
            attention_exact: fields.bool("attentionExact")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            forward_ran: fields.bool("forwardRan")?,
            kv_quantized: fields.bool("kvQuantized")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            sparsity_applied: fields.bool("sparsityApplied")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            window_applied: fields.bool("windowApplied")?,
            window_tokens: fields.u32("windowTokens")?,
            extensions: fields.extensions()?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-attention", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const MOE_KIND: &str = "knolo.infer.moe-report";

/// Mixture of experts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub requested_experts: u32,
    pub selected_experts: u32,
    pub code: String,
    pub retryable: bool,
    pub routed: bool,
    pub shared_used: bool,
    pub tie_broken: bool,
    pub grouped: bool,
    pub expert_placed: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl MoeReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "moe",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "a mixture of experts is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "a mixture of experts is not retryable")?;
        forbid(self.routed, "a router does not run")?;
        forbid(self.shared_used, "a shared expert is not used")?;
        forbid(self.tie_broken, "expert ties are not broken")?;
        forbid(self.grouped, "grouped kernels stay off")?;
        forbid(self.expert_placed, "an expert is not placed")?;
        exact_u32(self.selected_experts, 0, "selected experts stay zero")?;
        stopped(
            self.forward_ran,
            self.receipt_stored,
            "a mixture of experts",
        )?;
        at_most(
            self.requested_experts,
            MAX_EXPERTS,
            "expert count exceeds the record cap",
        )?;
        one_of("reason", &self.reason, MOE_REASONS)?;
        let named = match self.reason.as_str() {
            "router" => "a router refusal names an expert",
            "expert" => "an expert refusal names an expert",
            "shared" => "a shared-expert refusal names an expert",
            _ => "field reason has an unsupported value",
        };
        if self.reason != "router" && self.reason != "expert" && self.reason != "shared" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "field reason has an unsupported value",
            ));
        }
        if self.requested_experts == 0 {
            Err(fail(ErrorCode::ContractInvalid, named))
        } else {
            Ok(())
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(MOE_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("requestedExperts", cbor_u32(self.requested_experts));
        b.put("selectedExperts", cbor_u32(self.selected_experts));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("routed", CborValue::Bool(self.routed));
        b.put("sharedUsed", CborValue::Bool(self.shared_used));
        b.put("tieBroken", CborValue::Bool(self.tie_broken));
        b.put("grouped", CborValue::Bool(self.grouped));
        b.put("expertPlaced", CborValue::Bool(self.expert_placed));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, MOE_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            expert_placed: fields.bool("expertPlaced")?,
            forward_ran: fields.bool("forwardRan")?,
            grouped: fields.bool("grouped")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            requested_experts: fields.u32("requestedExperts")?,
            retryable: fields.bool("retryable")?,
            routed: fields.bool("routed")?,
            run_count: fields.u32("runCount")?,
            selected_experts: fields.u32("selectedExperts")?,
            shared_used: fields.bool("sharedUsed")?,
            tie_broken: fields.bool("tieBroken")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            extensions: fields.extensions()?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-moe", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const EXPERT_PLACEMENT_KIND: &str = "knolo.infer.expert-placement-report";

/// Expert placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertPlacementReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub device_count: u32,
    pub experts_placed: u32,
    pub code: String,
    pub retryable: bool,
    pub cpu_offload: bool,
    pub resident_moved: bool,
    pub tensor_parallel: bool,
    pub automatic_fallback: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ExpertPlacementReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "expert-placement",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "an expert placement is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "an expert placement is not retryable")?;
        forbid(self.cpu_offload, "experts are not offloaded")?;
        forbid(self.resident_moved, "resident experts are not moved")?;
        forbid(self.tensor_parallel, "tensor parallel stays off")?;
        forbid(self.automatic_fallback, "placement does not fall back")?;
        exact_u32(self.experts_placed, 0, "placed experts stay zero")?;
        stopped(self.forward_ran, self.receipt_stored, "an expert placement")?;
        at_most(
            self.device_count,
            MAX_PLACEMENT_DEVICES,
            "device count exceeds the record cap",
        )?;
        one_of("reason", &self.reason, EXPERT_PLACEMENT_REASONS)?;
        match self.reason.as_str() {
            "residency" => exact_u32(self.device_count, 1, "a residency refusal names one device"),
            "offload" => exact_u32(self.device_count, 1, "an offload refusal names one device"),
            "asymmetric" => exact_u32(
                self.device_count,
                2,
                "an asymmetric refusal names two devices",
            ),
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                "field reason has an unsupported value",
            )),
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(EXPERT_PLACEMENT_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("deviceCount", cbor_u32(self.device_count));
        b.put("expertsPlaced", cbor_u32(self.experts_placed));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("cpuOffload", CborValue::Bool(self.cpu_offload));
        b.put("residentMoved", CborValue::Bool(self.resident_moved));
        b.put("tensorParallel", CborValue::Bool(self.tensor_parallel));
        b.put(
            "automaticFallback",
            CborValue::Bool(self.automatic_fallback),
        );
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, EXPERT_PLACEMENT_KIND)?;
        let out = Self {
            automatic_fallback: fields.bool("automaticFallback")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            cpu_offload: fields.bool("cpuOffload")?,
            device_count: fields.u32("deviceCount")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            experts_placed: fields.u32("expertsPlaced")?,
            forward_ran: fields.bool("forwardRan")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            resident_moved: fields.bool("residentMoved")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            tensor_parallel: fields.bool("tensorParallel")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            extensions: fields.extensions()?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-expert-placement", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const GROUPED_KERNEL_KIND: &str = "knolo.infer.grouped-kernel-report";

/// Grouped kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedKernelReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub group_count: u32,
    pub code: String,
    pub retryable: bool,
    pub kernel_selected: bool,
    pub grouped: bool,
    pub routing_ran: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl GroupedKernelReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "grouped-kernel",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "UNSUPPORTED_KERNEL",
            "a grouped kernel is UNSUPPORTED_KERNEL",
        )?;
        forbid(self.retryable, "a grouped kernel is not retryable")?;
        forbid(self.kernel_selected, "a grouped kernel is not selected")?;
        forbid(self.grouped, "a grouped gemm is not selected")?;
        forbid(self.routing_ran, "top-k routing does not run")?;
        stopped(self.forward_ran, self.receipt_stored, "a grouped kernel")?;
        at_most(
            self.group_count,
            MAX_GROUP_COUNT,
            "group count exceeds the record cap",
        )?;
        one_of("reason", &self.reason, GROUPED_REASONS)?;
        match self.reason.as_str() {
            "gemm" => {
                if self.group_count == 0 {
                    Err(fail(
                        ErrorCode::ContractInvalid,
                        "a grouped gemm names a group",
                    ))
                } else {
                    Ok(())
                }
            }
            "routing" => exact_u32(self.group_count, 0, "a routing refusal carries no group"),
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                "field reason has an unsupported value",
            )),
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(GROUPED_KERNEL_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("groupCount", cbor_u32(self.group_count));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("kernelSelected", CborValue::Bool(self.kernel_selected));
        b.put("grouped", CborValue::Bool(self.grouped));
        b.put("routingRan", CborValue::Bool(self.routing_ran));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, GROUPED_KERNEL_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            forward_ran: fields.bool("forwardRan")?,
            group_count: fields.u32("groupCount")?,
            grouped: fields.bool("grouped")?,
            kernel_selected: fields.bool("kernelSelected")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            routing_ran: fields.bool("routingRan")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            extensions: fields.extensions()?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-grouped-kernel", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const ROUTER_KIND: &str = "knolo.infer.router-report";

/// Router conformance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reference_root: DigestHex,
    pub candidate_root: DigestHex,
    pub reason: String,
    pub sample_count: u32,
    pub compared: bool,
    pub parity: bool,
    pub load_recorded: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl RouterReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "router",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        distinct(
            &self.reference_root,
            &self.engine_build_root,
            "the reference repeats the engine build",
        )?;
        distinct(
            &self.reference_root,
            &self.placement_root,
            "the reference repeats the placement",
        )?;
        distinct(
            &self.candidate_root,
            &self.engine_build_root,
            "the candidate repeats the engine build",
        )?;
        distinct(
            &self.candidate_root,
            &self.placement_root,
            "the candidate repeats the placement",
        )?;
        distinct(
            &self.candidate_root,
            &self.reference_root,
            "the candidate repeats the reference",
        )?;
        forbid(self.compared, "router outputs are not compared")?;
        forbid(self.parity, "router parity is not claimed")?;
        forbid(self.load_recorded, "expert load is not recorded")?;
        stopped(self.forward_ran, self.receipt_stored, "a router comparison")?;
        at_most(
            self.sample_count,
            MAX_ROUTER_SAMPLES,
            "router sample exceeds the record cap",
        )?;
        one_of("reason", &self.reason, ROUTER_REASONS)?;
        let named = match self.reason.as_str() {
            "logits" => "a logits record names a sample",
            "selection" => "a selection record names a sample",
            "load" => "a load record names a sample",
            _ => "field reason has an unsupported value",
        };
        if !matches!(self.reason.as_str(), "logits" | "selection" | "load") {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "field reason has an unsupported value",
            ));
        }
        if self.sample_count == 0 {
            Err(fail(ErrorCode::ContractInvalid, named))
        } else {
            Ok(())
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(ROUTER_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("referenceRoot", cbor_digest(&self.reference_root));
        b.put("candidateRoot", cbor_digest(&self.candidate_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("sampleCount", cbor_u32(self.sample_count));
        b.put("compared", CborValue::Bool(self.compared));
        b.put("parity", CborValue::Bool(self.parity));
        b.put("loadRecorded", CborValue::Bool(self.load_recorded));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, ROUTER_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            candidate_root: fields.digest("candidateRoot")?,
            compared: fields.bool("compared")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            forward_ran: fields.bool("forwardRan")?,
            load_recorded: fields.bool("loadRecorded")?,
            parity: fields.bool("parity")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            reference_root: fields.digest("referenceRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            sample_count: fields.u32("sampleCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            extensions: fields.extensions()?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-router", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const GLM_KIND: &str = "knolo.infer.glm-report";

/// GLM-4.7 adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlmReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub rejected_adapter: String,
    pub code: String,
    pub retryable: bool,
    pub weights_opened: bool,
    pub inventory_read: bool,
    pub recipe_blessed: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl GlmReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "glm",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        if self.rejected_adapter == "knolo.micro.v1" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the micro adapter is compiled in",
            ));
        }
        if self.rejected_adapter != GLM_ADAPTER {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a glm record names knolo.glm.v1",
            ));
        }
        exact(
            &self.code,
            "UNSUPPORTED_ARCHITECTURE",
            "an unsupported glm adapter is UNSUPPORTED_ARCHITECTURE",
        )?;
        forbid(self.retryable, "a glm adapter is not retryable")?;
        forbid(self.recipe_blessed, "a glm recipe is not blessed")?;
        stopped(self.forward_ran, self.receipt_stored, "a glm adapter")?;
        one_of("reason", &self.reason, GLM_REASONS)?;
        match self.reason.as_str() {
            "adapter" => {
                forbid(
                    self.weights_opened,
                    "an adapter refusal does not open weights",
                )?;
                forbid(
                    self.inventory_read,
                    "an adapter refusal does not read the inventory",
                )
            }
            "inventory" => {
                require_flag(
                    self.weights_opened,
                    "an inventory refusal opened the weights",
                )?;
                forbid(
                    self.inventory_read,
                    "an inventory refusal does not read the inventory",
                )
            }
            "recipe" => {
                require_flag(self.weights_opened, "a recipe refusal opened the weights")?;
                require_flag(self.inventory_read, "a recipe refusal read the inventory")
            }
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                "field reason has an unsupported value",
            )),
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(GLM_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("rejectedAdapter", cbor_text(&self.rejected_adapter));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("weightsOpened", CborValue::Bool(self.weights_opened));
        b.put("inventoryRead", CborValue::Bool(self.inventory_read));
        b.put("recipeBlessed", CborValue::Bool(self.recipe_blessed));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, GLM_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            forward_ran: fields.bool("forwardRan")?,
            inventory_read: fields.bool("inventoryRead")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            recipe_blessed: fields.bool("recipeBlessed")?,
            rejected_adapter: fields.text("rejectedAdapter")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            weights_opened: fields.bool("weightsOpened")?,
            extensions: fields.extensions()?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-glm", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const WORKSTATION_KIND: &str = "knolo.infer.workstation-report";

/// Workstation recipe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkstationReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub profile_root: DigestHex,
    pub benchmark_root: DigestHex,
    pub reason: String,
    pub benchmark_count: u32,
    pub blessed: bool,
    pub benchmark_recorded: bool,
    pub fallback_selected: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl WorkstationReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "workstation",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        distinct(
            &self.profile_root,
            &self.engine_build_root,
            "the profile repeats the engine build",
        )?;
        distinct(
            &self.profile_root,
            &self.placement_root,
            "the profile repeats the placement",
        )?;
        distinct(
            &self.benchmark_root,
            &self.engine_build_root,
            "the benchmark repeats the engine build",
        )?;
        distinct(
            &self.benchmark_root,
            &self.placement_root,
            "the benchmark repeats the placement",
        )?;
        distinct(
            &self.benchmark_root,
            &self.profile_root,
            "the benchmark repeats the profile",
        )?;
        forbid(self.blessed, "a workstation recipe is not blessed")?;
        forbid(
            self.benchmark_recorded,
            "a workstation benchmark does not run",
        )?;
        forbid(
            self.fallback_selected,
            "a workstation recipe does not fall back",
        )?;
        stopped(
            self.forward_ran,
            self.receipt_stored,
            "a workstation recipe",
        )?;
        at_most(
            self.benchmark_count,
            MAX_BENCHMARK_COUNT,
            "benchmark count exceeds the record cap",
        )?;
        one_of("reason", &self.reason, WORKSTATION_REASONS)?;
        match self.reason.as_str() {
            "benchmark" => {
                if self.benchmark_count == 0 {
                    Err(fail(
                        ErrorCode::ContractInvalid,
                        "a benchmark record names a suite",
                    ))
                } else {
                    Ok(())
                }
            }
            "profile" => exact_u32(
                self.benchmark_count,
                0,
                "a profile record carries no benchmark",
            ),
            "fallback" => exact_u32(
                self.benchmark_count,
                0,
                "a fallback record carries no benchmark",
            ),
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                "field reason has an unsupported value",
            )),
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(WORKSTATION_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("profileRoot", cbor_digest(&self.profile_root));
        b.put("benchmarkRoot", cbor_digest(&self.benchmark_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("benchmarkCount", cbor_u32(self.benchmark_count));
        b.put("blessed", CborValue::Bool(self.blessed));
        b.put(
            "benchmarkRecorded",
            CborValue::Bool(self.benchmark_recorded),
        );
        b.put("fallbackSelected", CborValue::Bool(self.fallback_selected));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, WORKSTATION_KIND)?;
        let out = Self {
            benchmark_count: fields.u32("benchmarkCount")?,
            benchmark_recorded: fields.bool("benchmarkRecorded")?,
            benchmark_root: fields.digest("benchmarkRoot")?,
            blessed: fields.bool("blessed")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            fallback_selected: fields.bool("fallbackSelected")?,
            forward_ran: fields.bool("forwardRan")?,
            placement_root: fields.digest("placementRoot")?,
            profile_root: fields.digest("profileRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            extensions: fields.extensions()?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-workstation", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const MIXED_PLACEMENT_KIND: &str = "knolo.infer.mixed-placement-report";

/// Mixed placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedPlacementReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub device_count: u32,
    pub code: String,
    pub retryable: bool,
    pub single_recorded: bool,
    pub mixed_selected: bool,
    pub automatic_fallback: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl MixedPlacementReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "mixed-placement",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        forbid(self.retryable, "a placement record is not retryable")?;
        forbid(self.mixed_selected, "mixed placement is not selected")?;
        forbid(self.automatic_fallback, "placement does not fall back")?;
        stopped(self.forward_ran, self.receipt_stored, "a placement record")?;
        at_most(
            self.device_count,
            MAX_PLACEMENT_DEVICES,
            "device count exceeds the record cap",
        )?;
        one_of("reason", &self.reason, MIXED_REASONS)?;
        match self.reason.as_str() {
            "single" => {
                exact_u32(self.device_count, 1, "a single placement names one device")?;
                exact(&self.code, "none", "a single placement is not a refusal")?;
                require_flag(
                    self.single_recorded,
                    "a single placement records one device",
                )
            }
            "mixed" => {
                exact_u32(self.device_count, 2, "a mixed placement names two devices")?;
                exact(
                    &self.code,
                    "CONTRACT_INVALID",
                    "a mixed placement is CONTRACT_INVALID",
                )?;
                forbid(
                    self.single_recorded,
                    "only a single placement records one device",
                )
            }
            "fallback" => {
                exact_u32(
                    self.device_count,
                    1,
                    "a fallback placement names one device",
                )?;
                exact(
                    &self.code,
                    "CONTRACT_INVALID",
                    "an automatic fallback is CONTRACT_INVALID",
                )?;
                forbid(
                    self.single_recorded,
                    "only a single placement records one device",
                )
            }
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                "field reason has an unsupported value",
            )),
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(MIXED_PLACEMENT_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("deviceCount", cbor_u32(self.device_count));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("singleRecorded", CborValue::Bool(self.single_recorded));
        b.put("mixedSelected", CborValue::Bool(self.mixed_selected));
        b.put(
            "automaticFallback",
            CborValue::Bool(self.automatic_fallback),
        );
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, MIXED_PLACEMENT_KIND)?;
        let out = Self {
            automatic_fallback: fields.bool("automaticFallback")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            device_count: fields.u32("deviceCount")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            forward_ran: fields.bool("forwardRan")?,
            mixed_selected: fields.bool("mixedSelected")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            single_recorded: fields.bool("singleRecorded")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            extensions: fields.extensions()?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-mixed-placement", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const CAPACITY_KIND: &str = "knolo.infer.capacity-report";

/// Expert capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapacityReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub requested_tokens: u32,
    pub capacity_tokens: u32,
    pub code: String,
    pub retryable: bool,
    pub overflowed: bool,
    pub dropped: bool,
    pub balanced: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl CapacityReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "capacity",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "expert capacity is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "expert capacity is not retryable")?;
        forbid(self.overflowed, "expert overflow is not placed")?;
        forbid(self.dropped, "expert tokens are not dropped")?;
        forbid(self.balanced, "expert load is not balanced")?;
        exact_u32(self.capacity_tokens, 0, "expert capacity stays zero")?;
        stopped(self.forward_ran, self.receipt_stored, "expert capacity")?;
        at_most(
            self.requested_tokens,
            MAX_CAPACITY_TOKENS,
            "capacity tokens exceed the record cap",
        )?;
        one_of("reason", &self.reason, CAPACITY_REASONS)?;
        let named = match self.reason.as_str() {
            "overflow" => "an overflow refusal names a token count",
            "drop" => "a drop refusal names a token count",
            "balance" => "a balance refusal names a token count",
            _ => "field reason has an unsupported value",
        };
        if !matches!(self.reason.as_str(), "overflow" | "drop" | "balance") {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "field reason has an unsupported value",
            ));
        }
        if self.requested_tokens == 0 {
            Err(fail(ErrorCode::ContractInvalid, named))
        } else {
            Ok(())
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(CAPACITY_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("requestedTokens", cbor_u32(self.requested_tokens));
        b.put("capacityTokens", cbor_u32(self.capacity_tokens));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("overflowed", CborValue::Bool(self.overflowed));
        b.put("dropped", CborValue::Bool(self.dropped));
        b.put("balanced", CborValue::Bool(self.balanced));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, CAPACITY_KIND)?;
        let out = Self {
            balanced: fields.bool("balanced")?,
            cache_policy: fields.text("cachePolicy")?,
            capacity_tokens: fields.u32("capacityTokens")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            dropped: fields.bool("dropped")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            forward_ran: fields.bool("forwardRan")?,
            overflowed: fields.bool("overflowed")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            requested_tokens: fields.u32("requestedTokens")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            extensions: fields.extensions()?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-capacity", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const PAYLOAD_KIND: &str = "knolo.infer.payload-report";

/// Payload hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub payload_root: DigestHex,
    pub domain_root: DigestHex,
    pub hash_status: String,
    pub payload_hashed: bool,
    pub digest_bound: bool,
    pub public_key_bytes: u32,
    pub signature_bytes: u32,
    pub scalar_bytes: u32,
    pub key_material_present: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl PayloadReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "payload",
            &self.validation_result,
            &["recorded", "verified"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        distinct(
            &self.payload_root,
            &self.engine_build_root,
            "the hashed payload repeats the engine build",
        )?;
        distinct(
            &self.payload_root,
            &self.placement_root,
            "the hashed payload repeats the placement",
        )?;
        distinct(
            &self.domain_root,
            &self.engine_build_root,
            "the payload domain repeats the engine build",
        )?;
        distinct(
            &self.domain_root,
            &self.placement_root,
            "the payload domain repeats the placement",
        )?;
        distinct(
            &self.domain_root,
            &self.payload_root,
            "the payload domain repeats the payload",
        )?;
        forbid(
            self.key_material_present,
            "key material stays in host storage",
        )?;
        forbid(self.digest_bound, "the digest stays unbound")?;
        one_of("hashStatus", &self.hash_status, HASH_STATUSES)?;
        match self.hash_status.as_str() {
            "unsigned-local" => {
                unsigned_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                forbid(
                    self.payload_hashed,
                    "an unsigned release hashes the payload",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a hashed payload is verified",
                )?;
            }
            "hashed" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(self.payload_hashed, "a hashed payload records the digest")?;
                exact(
                    &self.validation_result,
                    "verified",
                    "a hashed payload is verified",
                )?;
            }
            "rejected" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(self.payload_hashed, "a rejected payload records the digest")?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a hashed payload is verified",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field hashStatus has an unsupported value",
                ));
            }
        };
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(PAYLOAD_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("payloadRoot", cbor_digest(&self.payload_root));
        b.put("domainRoot", cbor_digest(&self.domain_root));
        b.put("hashStatus", cbor_text(&self.hash_status));
        b.put("payloadHashed", CborValue::Bool(self.payload_hashed));
        b.put("digestBound", CborValue::Bool(self.digest_bound));
        b.put("publicKeyBytes", cbor_u32(self.public_key_bytes));
        b.put("signatureBytes", cbor_u32(self.signature_bytes));
        b.put("scalarBytes", cbor_u32(self.scalar_bytes));
        b.put(
            "keyMaterialPresent",
            CborValue::Bool(self.key_material_present),
        );
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, PAYLOAD_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            digest_bound: fields.bool("digestBound")?,
            domain_root: fields.digest("domainRoot")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            hash_status: fields.text("hashStatus")?,
            key_material_present: fields.bool("keyMaterialPresent")?,
            payload_hashed: fields.bool("payloadHashed")?,
            payload_root: fields.digest("payloadRoot")?,
            placement_root: fields.digest("placementRoot")?,
            public_key_bytes: fields.u32("publicKeyBytes")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            scalar_bytes: fields.u32("scalarBytes")?,
            signature_bytes: fields.u32("signatureBytes")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            extensions: fields.extensions()?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-payload", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const HYBRID_KIND: &str = "knolo.infer.hybrid-report";

/// Hybrid attention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub state_tokens: u32,
    pub code: String,
    pub retryable: bool,
    pub attention_exact: bool,
    pub hybrid_allocated: bool,
    pub sliding_applied: bool,
    pub full_mixed: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl HybridReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "hybrid",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "hybrid attention is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "hybrid attention is not retryable")?;
        require_flag(self.attention_exact, "attention stays exact")?;
        forbid(self.hybrid_allocated, "hybrid state is not allocated")?;
        forbid(self.sliding_applied, "a sliding layer is not applied")?;
        forbid(self.full_mixed, "full attention is not mixed")?;
        stopped(self.forward_ran, self.receipt_stored, "hybrid attention")?;
        at_most(
            self.state_tokens,
            MAX_HYBRID_TOKENS,
            "hybrid state exceeds the record cap",
        )?;
        one_of("reason", &self.reason, HYBRID_REASONS)?;
        match self.reason.as_str() {
            "state" => {
                if self.state_tokens == 0 {
                    Err(fail(
                        ErrorCode::ContractInvalid,
                        "a state refusal names a span",
                    ))
                } else {
                    Ok(())
                }
            }
            "sliding" => exact_u32(self.state_tokens, 0, "a sliding refusal carries no state"),
            "full" => exact_u32(self.state_tokens, 0, "a full refusal carries no state"),
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                "field reason has an unsupported value",
            )),
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(HYBRID_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("stateTokens", cbor_u32(self.state_tokens));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("attentionExact", CborValue::Bool(self.attention_exact));
        b.put("hybridAllocated", CborValue::Bool(self.hybrid_allocated));
        b.put("slidingApplied", CborValue::Bool(self.sliding_applied));
        b.put("fullMixed", CborValue::Bool(self.full_mixed));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, HYBRID_KIND)?;
        let out = Self {
            attention_exact: fields.bool("attentionExact")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            forward_ran: fields.bool("forwardRan")?,
            full_mixed: fields.bool("fullMixed")?,
            hybrid_allocated: fields.bool("hybridAllocated")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            sliding_applied: fields.bool("slidingApplied")?,
            state_tokens: fields.u32("stateTokens")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            extensions: fields.extensions()?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-hybrid", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const LINEAR_KIND: &str = "knolo.infer.linear-report";

/// Linear attention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub state_tokens: u32,
    pub code: String,
    pub retryable: bool,
    pub attention_exact: bool,
    pub linear_applied: bool,
    pub kda_selected: bool,
    pub decay_applied: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl LinearReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "linear",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "linear attention is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "linear attention is not retryable")?;
        require_flag(self.attention_exact, "attention stays exact")?;
        forbid(self.linear_applied, "linear attention is not applied")?;
        forbid(self.kda_selected, "a KDA kernel is not selected")?;
        forbid(self.decay_applied, "a decay rule is not applied")?;
        stopped(self.forward_ran, self.receipt_stored, "linear attention")?;
        at_most(
            self.state_tokens,
            MAX_LINEAR_TOKENS,
            "linear state exceeds the record cap",
        )?;
        one_of("reason", &self.reason, LINEAR_REASONS)?;
        match self.reason.as_str() {
            "state" => {
                if self.state_tokens == 0 {
                    Err(fail(
                        ErrorCode::ContractInvalid,
                        "a state refusal names a span",
                    ))
                } else {
                    Ok(())
                }
            }
            "kda" => exact_u32(self.state_tokens, 0, "a KDA refusal carries no state"),
            "decay" => exact_u32(self.state_tokens, 0, "a decay refusal carries no state"),
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                "field reason has an unsupported value",
            )),
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(LINEAR_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("stateTokens", cbor_u32(self.state_tokens));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("attentionExact", CborValue::Bool(self.attention_exact));
        b.put("linearApplied", CborValue::Bool(self.linear_applied));
        b.put("kdaSelected", CborValue::Bool(self.kda_selected));
        b.put("decayApplied", CborValue::Bool(self.decay_applied));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, LINEAR_KIND)?;
        let out = Self {
            attention_exact: fields.bool("attentionExact")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            decay_applied: fields.bool("decayApplied")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            forward_ran: fields.bool("forwardRan")?,
            kda_selected: fields.bool("kdaSelected")?,
            linear_applied: fields.bool("linearApplied")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            state_tokens: fields.u32("stateTokens")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            extensions: fields.extensions()?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-linear", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const MHC_KIND: &str = "knolo.infer.mhc-report";

/// mHC support.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MhcReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub requested_streams: u32,
    pub applied_streams: u32,
    pub code: String,
    pub retryable: bool,
    pub connected: bool,
    pub manifold_applied: bool,
    pub residual_mapped: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl MhcReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "mhc",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "an mHC connection is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "an mHC connection is not retryable")?;
        forbid(self.connected, "an mHC connection is not applied")?;
        forbid(
            self.manifold_applied,
            "a manifold constraint is not applied",
        )?;
        forbid(self.residual_mapped, "a residual map is not applied")?;
        exact_u32(self.applied_streams, 0, "mHC streams stay zero")?;
        stopped(self.forward_ran, self.receipt_stored, "an mHC connection")?;
        at_most(
            self.requested_streams,
            MAX_MHC_STREAMS,
            "mHC streams exceed the record cap",
        )?;
        one_of("reason", &self.reason, MHC_REASONS)?;
        let named = match self.reason.as_str() {
            "connection" => "a connection refusal names a stream count",
            "manifold" => "a manifold refusal names a stream count",
            "residual" => "a residual refusal names a stream count",
            _ => "field reason has an unsupported value",
        };
        if !matches!(self.reason.as_str(), "connection" | "manifold" | "residual") {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "field reason has an unsupported value",
            ));
        }
        if self.requested_streams == 0 {
            Err(fail(ErrorCode::ContractInvalid, named))
        } else {
            Ok(())
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(MHC_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("requestedStreams", cbor_u32(self.requested_streams));
        b.put("appliedStreams", cbor_u32(self.applied_streams));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("connected", CborValue::Bool(self.connected));
        b.put("manifoldApplied", CborValue::Bool(self.manifold_applied));
        b.put("residualMapped", CborValue::Bool(self.residual_mapped));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, MHC_KIND)?;
        let out = Self {
            applied_streams: fields.u32("appliedStreams")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            connected: fields.bool("connected")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            forward_ran: fields.bool("forwardRan")?,
            manifold_applied: fields.bool("manifoldApplied")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            requested_streams: fields.u32("requestedStreams")?,
            residual_mapped: fields.bool("residualMapped")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            extensions: fields.extensions()?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-mhc", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const MTP_KIND: &str = "knolo.infer.mtp-report";

/// MTP speculation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub proposal_tokens: u32,
    pub accepted_tokens: u32,
    pub code: String,
    pub retryable: bool,
    pub head_ran: bool,
    pub drafted: bool,
    pub accepted: bool,
    pub speculated: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl MtpReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "mtp",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "an MTP head is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "an MTP head is not retryable")?;
        forbid(self.head_ran, "an MTP head does not run")?;
        forbid(self.drafted, "an MTP draft is not produced")?;
        forbid(self.accepted, "an MTP proposal is not accepted")?;
        forbid(self.speculated, "speculation stays off")?;
        exact_u32(self.accepted_tokens, 0, "accepted MTP tokens stay zero")?;
        stopped(self.forward_ran, self.receipt_stored, "an MTP head")?;
        at_most(
            self.proposal_tokens,
            MAX_MTP_TOKENS,
            "MTP tokens exceed the record cap",
        )?;
        one_of("reason", &self.reason, MTP_REASONS)?;
        let named = match self.reason.as_str() {
            "head" => "a head refusal names a proposal",
            "draft" => "a draft refusal names a proposal",
            "accept" => "an accept refusal names a proposal",
            _ => "field reason has an unsupported value",
        };
        if !matches!(self.reason.as_str(), "head" | "draft" | "accept") {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "field reason has an unsupported value",
            ));
        }
        if self.proposal_tokens == 0 {
            Err(fail(ErrorCode::ContractInvalid, named))
        } else {
            Ok(())
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(MTP_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("proposalTokens", cbor_u32(self.proposal_tokens));
        b.put("acceptedTokens", cbor_u32(self.accepted_tokens));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("headRan", CborValue::Bool(self.head_ran));
        b.put("drafted", CborValue::Bool(self.drafted));
        b.put("accepted", CborValue::Bool(self.accepted));
        b.put("speculated", CborValue::Bool(self.speculated));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, MTP_KIND)?;
        let out = Self {
            accepted: fields.bool("accepted")?,
            accepted_tokens: fields.u32("acceptedTokens")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            drafted: fields.bool("drafted")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            forward_ran: fields.bool("forwardRan")?,
            head_ran: fields.bool("headRan")?,
            placement_root: fields.digest("placementRoot")?,
            proposal_tokens: fields.u32("proposalTokens")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            speculated: fields.bool("speculated")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            extensions: fields.extensions()?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-mtp", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const BINDING_KIND: &str = "knolo.infer.binding-report";

/// Digest binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub digest_root: DigestHex,
    pub evidence_root: DigestHex,
    pub bind_status: String,
    pub digest_bound: bool,
    pub evidence_bound: bool,
    pub public_key_bytes: u32,
    pub signature_bytes: u32,
    pub scalar_bytes: u32,
    pub key_material_present: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl BindingReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "binding",
            &self.validation_result,
            &["recorded", "verified"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        distinct(
            &self.digest_root,
            &self.engine_build_root,
            "the bound digest repeats the engine build",
        )?;
        distinct(
            &self.digest_root,
            &self.placement_root,
            "the bound digest repeats the placement",
        )?;
        distinct(
            &self.evidence_root,
            &self.engine_build_root,
            "the evidence root repeats the engine build",
        )?;
        distinct(
            &self.evidence_root,
            &self.placement_root,
            "the evidence root repeats the placement",
        )?;
        distinct(
            &self.evidence_root,
            &self.digest_root,
            "the evidence root repeats the digest",
        )?;
        forbid(
            self.key_material_present,
            "key material stays in host storage",
        )?;
        forbid(self.evidence_bound, "evidence stays unbound")?;
        one_of("bindStatus", &self.bind_status, BIND_STATUSES)?;
        match self.bind_status.as_str() {
            "unsigned-local" => {
                unsigned_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                forbid(self.digest_bound, "an unsigned release binds the digest")?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a bound digest is verified",
                )?;
            }
            "bound" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(self.digest_bound, "a bound digest records the binding")?;
                exact(
                    &self.validation_result,
                    "verified",
                    "a bound digest is verified",
                )?;
            }
            "rejected" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(self.digest_bound, "a rejected digest records the binding")?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a bound digest is verified",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field bindStatus has an unsupported value",
                ));
            }
        };
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(BINDING_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("digestRoot", cbor_digest(&self.digest_root));
        b.put("evidenceRoot", cbor_digest(&self.evidence_root));
        b.put("bindStatus", cbor_text(&self.bind_status));
        b.put("digestBound", CborValue::Bool(self.digest_bound));
        b.put("evidenceBound", CborValue::Bool(self.evidence_bound));
        b.put("publicKeyBytes", cbor_u32(self.public_key_bytes));
        b.put("signatureBytes", cbor_u32(self.signature_bytes));
        b.put("scalarBytes", cbor_u32(self.scalar_bytes));
        b.put(
            "keyMaterialPresent",
            CborValue::Bool(self.key_material_present),
        );
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, BINDING_KIND)?;
        let out = Self {
            bind_status: fields.text("bindStatus")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            digest_bound: fields.bool("digestBound")?,
            digest_root: fields.digest("digestRoot")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            evidence_bound: fields.bool("evidenceBound")?,
            evidence_root: fields.digest("evidenceRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            key_material_present: fields.bool("keyMaterialPresent")?,
            placement_root: fields.digest("placementRoot")?,
            public_key_bytes: fields.u32("publicKeyBytes")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            scalar_bytes: fields.u32("scalarBytes")?,
            signature_bytes: fields.u32("signatureBytes")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-binding", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const MULTIMODAL_KIND: &str = "knolo.infer.multimodal-report";

/// Multimodal input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultimodalReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub requested_parts: u32,
    pub accepted_parts: u32,
    pub code: String,
    pub retryable: bool,
    pub media_opened: bool,
    pub projected: bool,
    pub embedded: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl MultimodalReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "multimodal",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "multimodal input is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "multimodal input is not retryable")?;
        forbid(self.media_opened, "a media part is not opened")?;
        forbid(self.projected, "a projector is not applied")?;
        forbid(self.embedded, "a media embedding is not stored")?;
        exact_u32(
            self.accepted_parts,
            0,
            "accepted multimodal parts stay zero",
        )?;
        stopped(self.forward_ran, self.receipt_stored, "multimodal input")?;
        at_most(
            self.requested_parts,
            MAX_MULTIMODAL_PARTS,
            "multimodal parts exceed the record cap",
        )?;
        one_of("reason", &self.reason, MULTIMODAL_REASONS)?;
        let named = match self.reason.as_str() {
            "image" => "an image refusal names a part count",
            "audio" => "an audio refusal names a part count",
            "video" => "a video refusal names a part count",
            _ => "field reason has an unsupported value",
        };
        if self.requested_parts == 0 {
            Err(fail(ErrorCode::ContractInvalid, named))
        } else {
            Ok(())
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(MULTIMODAL_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("requestedParts", cbor_u32(self.requested_parts));
        b.put("acceptedParts", cbor_u32(self.accepted_parts));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("mediaOpened", CborValue::Bool(self.media_opened));
        b.put("projected", CborValue::Bool(self.projected));
        b.put("embedded", CborValue::Bool(self.embedded));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, MULTIMODAL_KIND)?;
        let out = Self {
            accepted_parts: fields.u32("acceptedParts")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            embedded: fields.bool("embedded")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            media_opened: fields.bool("mediaOpened")?,
            placement_root: fields.digest("placementRoot")?,
            projected: fields.bool("projected")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            requested_parts: fields.u32("requestedParts")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-multimodal", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const LARGE_PLACEMENT_KIND: &str = "knolo.infer.large-placement-report";

/// Large expert placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LargePlacementReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub requested_experts: u32,
    pub placed_experts: u32,
    pub code: String,
    pub retryable: bool,
    pub spanned: bool,
    pub sharded: bool,
    pub replicated: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl LargePlacementReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "large-placement",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "large placement is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "large placement is not retryable")?;
        forbid(self.spanned, "a span is not applied")?;
        forbid(self.sharded, "a shard is not applied")?;
        forbid(self.replicated, "a replica is not applied")?;
        exact_u32(self.placed_experts, 0, "placed experts stay zero")?;
        stopped(self.forward_ran, self.receipt_stored, "large placement")?;
        at_most(
            self.requested_experts,
            MAX_LARGE_EXPERTS,
            "large placement exceeds the record cap",
        )?;
        one_of("reason", &self.reason, LARGE_PLACEMENT_REASONS)?;
        let named = match self.reason.as_str() {
            "span" => "a span refusal names an expert count",
            "shard" => "a shard refusal names an expert count",
            "replicate" => "a replica refusal names an expert count",
            _ => "field reason has an unsupported value",
        };
        if self.requested_experts == 0 {
            Err(fail(ErrorCode::ContractInvalid, named))
        } else {
            Ok(())
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(LARGE_PLACEMENT_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("requestedExperts", cbor_u32(self.requested_experts));
        b.put("placedExperts", cbor_u32(self.placed_experts));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("spanned", CborValue::Bool(self.spanned));
        b.put("sharded", CborValue::Bool(self.sharded));
        b.put("replicated", CborValue::Bool(self.replicated));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, LARGE_PLACEMENT_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            placed_experts: fields.u32("placedExperts")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            replicated: fields.bool("replicated")?,
            request_count: fields.u32("requestCount")?,
            requested_experts: fields.u32("requestedExperts")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            sharded: fields.bool("sharded")?,
            spanned: fields.bool("spanned")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-large-placement", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const EXTREME_CACHE_KIND: &str = "knolo.infer.extreme-cache-report";

/// Extreme-context cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtremeCacheReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub requested_tokens: u32,
    pub cached_tokens: u32,
    pub code: String,
    pub retryable: bool,
    pub windowed: bool,
    pub yarn_applied: bool,
    pub paged: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ExtremeCacheReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "extreme-cache",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "extreme-context cache is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "extreme-context cache is not retryable")?;
        forbid(self.windowed, "an extreme window is not applied")?;
        forbid(self.yarn_applied, "a YaRN scale is not applied")?;
        forbid(self.paged, "an extreme page is not allocated")?;
        exact_u32(
            self.cached_tokens,
            0,
            "cached extreme-context tokens stay zero",
        )?;
        stopped(
            self.forward_ran,
            self.receipt_stored,
            "extreme-context cache",
        )?;
        at_most(
            self.requested_tokens,
            MAX_EXTREME_TOKENS,
            "extreme-context tokens exceed the record cap",
        )?;
        one_of("reason", &self.reason, EXTREME_CACHE_REASONS)?;
        let named = match self.reason.as_str() {
            "window" => "a window refusal names a token count",
            "yarn" => "a YaRN refusal names a token count",
            "page" => "a page refusal names a token count",
            _ => "field reason has an unsupported value",
        };
        if self.requested_tokens == 0 {
            Err(fail(ErrorCode::ContractInvalid, named))
        } else {
            Ok(())
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(EXTREME_CACHE_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("requestedTokens", cbor_u32(self.requested_tokens));
        b.put("cachedTokens", cbor_u32(self.cached_tokens));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("windowed", CborValue::Bool(self.windowed));
        b.put("yarnApplied", CborValue::Bool(self.yarn_applied));
        b.put("paged", CborValue::Bool(self.paged));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, EXTREME_CACHE_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            cached_tokens: fields.u32("cachedTokens")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            paged: fields.bool("paged")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            requested_tokens: fields.u32("requestedTokens")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            windowed: fields.bool("windowed")?,
            yarn_applied: fields.bool("yarnApplied")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-extreme-cache", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const VISION_KIND: &str = "knolo.infer.vision-report";

/// Vision projector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub requested_patches: u32,
    pub projected_tokens: u32,
    pub code: String,
    pub retryable: bool,
    pub projector_applied: bool,
    pub patches_opened: bool,
    pub embedded: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl VisionReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "vision",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "a vision projector is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "a vision projector is not retryable")?;
        forbid(self.projector_applied, "a projector is not applied")?;
        forbid(self.patches_opened, "a patch is not opened")?;
        forbid(self.embedded, "a vision embedding is not stored")?;
        exact_u32(
            self.projected_tokens,
            0,
            "projected vision tokens stay zero",
        )?;
        stopped(self.forward_ran, self.receipt_stored, "a vision projector")?;
        at_most(
            self.requested_patches,
            MAX_VISION_PATCHES,
            "vision patches exceed the record cap",
        )?;
        one_of("reason", &self.reason, VISION_REASONS)?;
        let named = match self.reason.as_str() {
            "projector" => "a projector refusal names a patch count",
            "patch" => "a patch refusal names a patch count",
            "embed" => "an embed refusal names a patch count",
            _ => "field reason has an unsupported value",
        };
        if self.requested_patches == 0 {
            Err(fail(ErrorCode::ContractInvalid, named))
        } else {
            Ok(())
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(VISION_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("requestedPatches", cbor_u32(self.requested_patches));
        b.put("projectedTokens", cbor_u32(self.projected_tokens));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("projectorApplied", CborValue::Bool(self.projector_applied));
        b.put("patchesOpened", CborValue::Bool(self.patches_opened));
        b.put("embedded", CborValue::Bool(self.embedded));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, VISION_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            embedded: fields.bool("embedded")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            patches_opened: fields.bool("patchesOpened")?,
            placement_root: fields.digest("placementRoot")?,
            projected_tokens: fields.u32("projectedTokens")?,
            projector_applied: fields.bool("projectorApplied")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            requested_patches: fields.u32("requestedPatches")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-vision", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const POSITIONAL_KIND: &str = "knolo.infer.positional-report";

/// Positional extension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionalReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub requested_scale: u32,
    pub applied_scale: u32,
    pub code: String,
    pub retryable: bool,
    pub yarn_applied: bool,
    pub ntk_applied: bool,
    pub rope_scaled: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl PositionalReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "positional",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "positional extension is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "positional extension is not retryable")?;
        forbid(self.yarn_applied, "a YaRN scale is not applied")?;
        forbid(self.ntk_applied, "an NTK scale is not applied")?;
        forbid(self.rope_scaled, "RoPE is not rescaled")?;
        exact_u32(self.applied_scale, 0, "applied positional scale stays zero")?;
        stopped(
            self.forward_ran,
            self.receipt_stored,
            "positional extension",
        )?;
        at_most(
            self.requested_scale,
            MAX_POSITIONAL_SCALE,
            "positional scale exceeds the record cap",
        )?;
        one_of("reason", &self.reason, POSITIONAL_REASONS)?;
        let named = match self.reason.as_str() {
            "yarn" => "a YaRN refusal names a scale",
            "ntk" => "an NTK refusal names a scale",
            "rope" => "a RoPE refusal names a scale",
            _ => "field reason has an unsupported value",
        };
        if self.requested_scale == 0 {
            Err(fail(ErrorCode::ContractInvalid, named))
        } else {
            Ok(())
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(POSITIONAL_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("requestedScale", cbor_u32(self.requested_scale));
        b.put("appliedScale", cbor_u32(self.applied_scale));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("yarnApplied", CborValue::Bool(self.yarn_applied));
        b.put("ntkApplied", CborValue::Bool(self.ntk_applied));
        b.put("ropeScaled", CborValue::Bool(self.rope_scaled));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, POSITIONAL_KIND)?;
        let out = Self {
            applied_scale: fields.u32("appliedScale")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            ntk_applied: fields.bool("ntkApplied")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            requested_scale: fields.u32("requestedScale")?,
            retryable: fields.bool("retryable")?,
            rope_scaled: fields.bool("ropeScaled")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            yarn_applied: fields.bool("yarnApplied")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-positional", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const EXACT_ATTENTION_KIND: &str = "knolo.infer.exact-attention-report";

/// Exact attention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactAttentionReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub requested_tiles: u32,
    pub selected_tiles: u32,
    pub code: String,
    pub retryable: bool,
    pub flash_selected: bool,
    pub tiled: bool,
    pub split: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ExactAttentionReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "exact-attention",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "UNSUPPORTED_KERNEL",
            "an exact attention kernel is UNSUPPORTED_KERNEL",
        )?;
        forbid(self.retryable, "an exact attention kernel is not retryable")?;
        forbid(self.flash_selected, "a flash kernel is not selected")?;
        forbid(self.tiled, "an attention tile is not selected")?;
        forbid(self.split, "a split kernel is not selected")?;
        exact_u32(self.selected_tiles, 0, "selected attention tiles stay zero")?;
        stopped(
            self.forward_ran,
            self.receipt_stored,
            "an exact attention kernel",
        )?;
        at_most(
            self.requested_tiles,
            MAX_EXACT_TILES,
            "exact attention tiles exceed the record cap",
        )?;
        one_of("reason", &self.reason, EXACT_ATTENTION_REASONS)?;
        let named = match self.reason.as_str() {
            "flash" => "a flash refusal names a tile count",
            "tile" => "a tile refusal names a tile count",
            "split" => "a split refusal names a tile count",
            _ => "field reason has an unsupported value",
        };
        if self.requested_tiles == 0 {
            Err(fail(ErrorCode::ContractInvalid, named))
        } else {
            Ok(())
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(EXACT_ATTENTION_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("requestedTiles", cbor_u32(self.requested_tiles));
        b.put("selectedTiles", cbor_u32(self.selected_tiles));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("flashSelected", CborValue::Bool(self.flash_selected));
        b.put("tiled", CborValue::Bool(self.tiled));
        b.put("split", CborValue::Bool(self.split));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, EXACT_ATTENTION_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            flash_selected: fields.bool("flashSelected")?,
            forward_ran: fields.bool("forwardRan")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            requested_tiles: fields.u32("requestedTiles")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            selected_tiles: fields.u32("selectedTiles")?,
            split: fields.bool("split")?,
            tiled: fields.bool("tiled")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-exact-attention", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const RMSNORM_KIND: &str = "knolo.infer.rmsnorm-report";

/// RMSNorm kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmsnormReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub requested_layers: u32,
    pub applied_layers: u32,
    pub code: String,
    pub retryable: bool,
    pub norm_applied: bool,
    pub weight_applied: bool,
    pub eps_applied: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl RmsnormReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "rmsnorm",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "UNSUPPORTED_KERNEL",
            "an RMSNorm kernel is UNSUPPORTED_KERNEL",
        )?;
        forbid(self.retryable, "an RMSNorm kernel is not retryable")?;
        forbid(self.norm_applied, "a norm is not applied")?;
        forbid(self.weight_applied, "a norm weight is not applied")?;
        forbid(self.eps_applied, "an epsilon override is not applied")?;
        exact_u32(self.applied_layers, 0, "applied RMSNorm layers stay zero")?;
        stopped(self.forward_ran, self.receipt_stored, "an RMSNorm kernel")?;
        at_most(
            self.requested_layers,
            MAX_RMS_LAYERS,
            "RMSNorm layers exceed the record cap",
        )?;
        one_of("reason", &self.reason, RMSNORM_REASONS)?;
        let named = match self.reason.as_str() {
            "norm" => "a norm refusal names a layer count",
            "weight" => "a weight refusal names a layer count",
            "eps" => "an epsilon refusal names a layer count",
            _ => "field reason has an unsupported value",
        };
        if self.requested_layers == 0 {
            Err(fail(ErrorCode::ContractInvalid, named))
        } else {
            Ok(())
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(RMSNORM_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("requestedLayers", cbor_u32(self.requested_layers));
        b.put("appliedLayers", cbor_u32(self.applied_layers));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("normApplied", CborValue::Bool(self.norm_applied));
        b.put("weightApplied", CborValue::Bool(self.weight_applied));
        b.put("epsApplied", CborValue::Bool(self.eps_applied));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, RMSNORM_KIND)?;
        let out = Self {
            applied_layers: fields.u32("appliedLayers")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            eps_applied: fields.bool("epsApplied")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            norm_applied: fields.bool("normApplied")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            requested_layers: fields.u32("requestedLayers")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            weight_applied: fields.bool("weightApplied")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-rmsnorm", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const ROPE_KIND: &str = "knolo.infer.rope-report";

/// RoPE kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RopeReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub requested_layers: u32,
    pub applied_layers: u32,
    pub code: String,
    pub retryable: bool,
    pub rotary_applied: bool,
    pub frequency_scaled: bool,
    pub partial_applied: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl RopeReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "rope",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "UNSUPPORTED_KERNEL",
            "a RoPE kernel is UNSUPPORTED_KERNEL",
        )?;
        forbid(self.retryable, "a RoPE kernel is not retryable")?;
        forbid(self.rotary_applied, "a rotary embedding is not applied")?;
        forbid(self.frequency_scaled, "a frequency is not scaled")?;
        forbid(self.partial_applied, "a partial rotation is not applied")?;
        exact_u32(self.applied_layers, 0, "applied RoPE layers stay zero")?;
        stopped(self.forward_ran, self.receipt_stored, "a RoPE kernel")?;
        at_most(
            self.requested_layers,
            MAX_ROPE_LAYERS,
            "RoPE layers exceed the record cap",
        )?;
        one_of("reason", &self.reason, ROPE_REASONS)?;
        let named = match self.reason.as_str() {
            "rotary" => "a rotary refusal names a layer count",
            "frequency" => "a frequency refusal names a layer count",
            "partial" => "a partial refusal names a layer count",
            _ => "field reason has an unsupported value",
        };
        if self.requested_layers == 0 {
            Err(fail(ErrorCode::ContractInvalid, named))
        } else {
            Ok(())
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(ROPE_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("requestedLayers", cbor_u32(self.requested_layers));
        b.put("appliedLayers", cbor_u32(self.applied_layers));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("rotaryApplied", CborValue::Bool(self.rotary_applied));
        b.put("frequencyScaled", CborValue::Bool(self.frequency_scaled));
        b.put("partialApplied", CborValue::Bool(self.partial_applied));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, ROPE_KIND)?;
        let out = Self {
            applied_layers: fields.u32("appliedLayers")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            frequency_scaled: fields.bool("frequencyScaled")?,
            partial_applied: fields.bool("partialApplied")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            requested_layers: fields.u32("requestedLayers")?,
            retryable: fields.bool("retryable")?,
            rotary_applied: fields.bool("rotaryApplied")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-rope", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub const RESIDUAL_KIND: &str = "knolo.infer.residual-report";

/// Fused residual.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub requested_layers: u32,
    pub applied_layers: u32,
    pub code: String,
    pub retryable: bool,
    pub fused_applied: bool,
    pub added: bool,
    pub norm_folded: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ResidualReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "residual",
            &self.validation_result,
            &["recorded"],
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        exact(
            &self.code,
            "UNSUPPORTED_KERNEL",
            "a fused residual is UNSUPPORTED_KERNEL",
        )?;
        forbid(self.retryable, "a fused residual is not retryable")?;
        forbid(self.fused_applied, "a fused residual is not applied")?;
        forbid(self.added, "a residual add is not applied")?;
        forbid(self.norm_folded, "a folded norm is not applied")?;
        exact_u32(self.applied_layers, 0, "applied residual layers stay zero")?;
        stopped(self.forward_ran, self.receipt_stored, "a fused residual")?;
        at_most(
            self.requested_layers,
            MAX_RESIDUAL_LAYERS,
            "residual layers exceed the record cap",
        )?;
        one_of("reason", &self.reason, RESIDUAL_REASONS)?;
        let named = match self.reason.as_str() {
            "fused" => "a fused refusal names a layer count",
            "add" => "an add refusal names a layer count",
            "norm" => "a folded-norm refusal names a layer count",
            _ => "field reason has an unsupported value",
        };
        if self.requested_layers == 0 {
            Err(fail(ErrorCode::ContractInvalid, named))
        } else {
            Ok(())
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(RESIDUAL_KIND);
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("requestedLayers", cbor_u32(self.requested_layers));
        b.put("appliedLayers", cbor_u32(self.applied_layers));
        b.put("code", cbor_text(&self.code));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("fusedApplied", CborValue::Bool(self.fused_applied));
        b.put("added", CborValue::Bool(self.added));
        b.put("normFolded", CborValue::Bool(self.norm_folded));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        put_extensions(&mut b, &self.extensions)?;
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, RESIDUAL_KIND)?;
        let out = Self {
            added: fields.bool("added")?,
            applied_layers: fields.u32("appliedLayers")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            fused_applied: fields.bool("fusedApplied")?,
            norm_folded: fields.bool("normFolded")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            requested_layers: fields.u32("requestedLayers")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-residual", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
