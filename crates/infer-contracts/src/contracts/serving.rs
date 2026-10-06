//! Receipt verification and the advanced-serving records that stay off.
//!
//! Each report records one cold micro fixture. None of them verifies a
//! receipt, runs speculation, captures a CUDA graph, loads a second model,
//! or starts a secondary service. The layouts are specified in
//! `spec/KIP-INFER-0111-receipt-verification.md`,
//! `spec/KIP-INFER-0112-speculative-decoding.md`,
//! `spec/KIP-INFER-0113-cuda-graph.md`,
//! `spec/KIP-INFER-0114-multi-model.md`, and
//! `spec/KIP-INFER-0115-secondary-service.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{cbor_digest, cbor_text, cbor_u32, expect_kind_version, one_of, Fields};

use super::common::{put_extensions, Builder};
use super::exposure::{ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES};
use super::lifecycle::cold_single;
use super::resilience::ED25519_SCALAR_BYTES;

pub const RECEIPT_VERIFY_KIND: &str = "knolo.infer.receipt-verify-report";
pub const SPECULATIVE_KIND: &str = "knolo.infer.speculative-report";
pub const GRAPH_KIND: &str = "knolo.infer.graph-report";
pub const MULTI_MODEL_KIND: &str = "knolo.infer.multi-model-report";
pub const SECONDARY_KIND: &str = "knolo.infer.secondary-report";

pub const MAX_PROPOSAL_TOKENS: u32 = 16;
pub const MAX_DEVICE_COUNT: u32 = 2;

const VERIFY_STATUSES: &[&str] = &["verified", "rejected", "unsigned-local"];
const SPECULATIVE_REASONS: &[&str] = &["draft", "mtp", "plan"];
const GRAPH_REASONS: &[&str] = &["identity", "kernel", "workspace", "shape"];
const MULTI_MODEL_REASONS: &[&str] = &["second", "replace", "parallel"];
const SECONDARY_REASONS: &[&str] = &["tiny", "embeddings", "overflow"];

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

fn distinct(left: &DigestHex, right: &DigestHex, message: &str) -> Result<(), InferFailure> {
    if left == right {
        Err(fail(ErrorCode::ContractInvalid, message))
    } else {
        Ok(())
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

fn proposal_len(reason: &str, tokens: u32) -> Result<(), InferFailure> {
    at_most(
        tokens,
        MAX_PROPOSAL_TOKENS,
        "speculative proposal exceeds the record cap",
    )?;
    match reason {
        "plan" => exact_u32(tokens, 0, "a plan refusal carries no proposal"),
        "draft" | "mtp" => {
            if tokens == 0 {
                let message = if reason == "draft" {
                    "a draft refusal names a proposal"
                } else {
                    "an mtp refusal names a proposal"
                };
                Err(fail(ErrorCode::ContractInvalid, message))
            } else {
                Ok(())
            }
        }
        _ => Err(fail(
            ErrorCode::ContractInvalid,
            "field reason has an unsupported value",
        )),
    }
}

fn device_count(reason: &str, count: u32) -> Result<(), InferFailure> {
    at_most(
        count,
        MAX_DEVICE_COUNT,
        "device count exceeds the record cap",
    )?;
    match reason {
        "second" => exact_u32(count, 1, "a second-model refusal names one device"),
        "replace" => exact_u32(count, 1, "a replace refusal names one device"),
        "parallel" => exact_u32(count, 2, "a parallel refusal names two devices"),
        _ => Err(fail(
            ErrorCode::ContractInvalid,
            "field reason has an unsupported value",
        )),
    }
}

fn secondary_flags(
    reason: &str,
    model_named: bool,
    embeddings_requested: bool,
    overflow_requested: bool,
) -> Result<(), InferFailure> {
    match reason {
        "tiny" => {
            require_flag(model_named, "a tiny-model request names the model")?;
            forbid(
                embeddings_requested,
                "a tiny-model request does not ask for embeddings",
            )?;
            forbid(
                overflow_requested,
                "a tiny-model request does not ask for overflow",
            )
        }
        "embeddings" => {
            forbid(model_named, "an embeddings request does not name a model")?;
            require_flag(
                embeddings_requested,
                "an embeddings request asks for embeddings",
            )?;
            forbid(
                overflow_requested,
                "an embeddings request does not ask for overflow",
            )
        }
        "overflow" => {
            require_flag(model_named, "an overflow request names the model")?;
            forbid(
                embeddings_requested,
                "an overflow request does not ask for embeddings",
            )?;
            require_flag(overflow_requested, "an overflow request asks for overflow")
        }
        _ => Err(fail(
            ErrorCode::ContractInvalid,
            "field reason has an unsupported value",
        )),
    }
}

/// Host-supplied Ed25519 receipt verification. The receipt is not verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptVerifyReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub verify_status: String,
    pub receipt_verified: bool,
    pub domain_separated: bool,
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

impl ReceiptVerifyReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "receipt-verify",
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
            &self.release_root,
            &self.engine_build_root,
            "the release repeats the engine build",
        )?;
        distinct(
            &self.message_root,
            &self.engine_build_root,
            "the verified message repeats the engine build",
        )?;
        distinct(
            &self.message_root,
            &self.release_root,
            "the verified message repeats the release",
        )?;
        forbid(
            self.key_material_present,
            "key material stays in host storage",
        )?;
        forbid(self.domain_separated, "the domain stays unseparated")?;
        one_of("verifyStatus", &self.verify_status, VERIFY_STATUSES)?;
        match self.verify_status.as_str() {
            "unsigned-local" => {
                unsigned_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                forbid(
                    self.receipt_verified,
                    "an unsigned release verifies the receipt",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a verified receipt is verified",
                )?;
            }
            "verified" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.receipt_verified,
                    "a verified receipt records the check",
                )?;
                exact(
                    &self.validation_result,
                    "verified",
                    "a verified receipt is verified",
                )?;
            }
            "rejected" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.receipt_verified,
                    "a rejected receipt records the check",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a verified receipt is verified",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field verifyStatus has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(RECEIPT_VERIFY_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("domainSeparated", CborValue::Bool(self.domain_separated));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put(
            "keyMaterialPresent",
            CborValue::Bool(self.key_material_present),
        );
        b.put("messageRoot", cbor_digest(&self.message_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("publicKeyBytes", cbor_u32(self.public_key_bytes));
        b.put("receiptVerified", CborValue::Bool(self.receipt_verified));
        b.put("releaseRoot", cbor_digest(&self.release_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("scalarBytes", cbor_u32(self.scalar_bytes));
        b.put("signatureBytes", cbor_u32(self.signature_bytes));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("verifyStatus", cbor_text(&self.verify_status));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, RECEIPT_VERIFY_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            domain_separated: fields.bool("domainSeparated")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            key_material_present: fields.bool("keyMaterialPresent")?,
            message_root: fields.digest("messageRoot")?,
            placement_root: fields.digest("placementRoot")?,
            public_key_bytes: fields.u32("publicKeyBytes")?,
            receipt_verified: fields.bool("receiptVerified")?,
            release_root: fields.digest("releaseRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            scalar_bytes: fields.u32("scalarBytes")?,
            signature_bytes: fields.u32("signatureBytes")?,
            validation_result: fields.text("validationResult")?,
            verify_status: fields.text("verifyStatus")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-receipt-verify", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One speculative request the engine did not run. Speculation stays off.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub target_root: DigestHex,
    pub proposal_root: DigestHex,
    pub reason: String,
    pub proposal_tokens: u32,
    pub accepted_tokens: u32,
    pub rejected_tokens: u32,
    pub code: String,
    pub retryable: bool,
    pub speculated: bool,
    pub distribution_changed: bool,
    pub cache_affected: bool,
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

impl SpeculativeReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "speculative",
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
            &self.target_root,
            &self.engine_build_root,
            "the target repeats the engine build",
        )?;
        distinct(
            &self.target_root,
            &self.placement_root,
            "the target repeats the placement",
        )?;
        distinct(
            &self.proposal_root,
            &self.engine_build_root,
            "the proposal repeats the engine build",
        )?;
        distinct(
            &self.proposal_root,
            &self.placement_root,
            "the proposal repeats the placement",
        )?;
        distinct(
            &self.proposal_root,
            &self.target_root,
            "the proposal repeats the target",
        )?;
        one_of("reason", &self.reason, SPECULATIVE_REASONS)?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "a speculative refusal is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "a speculative refusal is not retryable")?;
        forbid(self.speculated, "a speculative refusal does not speculate")?;
        exact_u32(
            self.accepted_tokens,
            0,
            "accepted tokens stay zero while speculation is off",
        )?;
        exact_u32(
            self.rejected_tokens,
            0,
            "rejected tokens stay zero while speculation is off",
        )?;
        forbid(
            self.distribution_changed,
            "speculation does not change the target distribution",
        )?;
        forbid(self.cache_affected, "speculation does not affect the cache")?;
        stopped(
            self.forward_ran,
            self.receipt_stored,
            "a speculative refusal",
        )?;
        proposal_len(&self.reason, self.proposal_tokens)?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(SPECULATIVE_KIND);
        b.put("acceptedTokens", cbor_u32(self.accepted_tokens));
        b.put("cacheAffected", CborValue::Bool(self.cache_affected));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put(
            "distributionChanged",
            CborValue::Bool(self.distribution_changed),
        );
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("proposalRoot", cbor_digest(&self.proposal_root));
        b.put("proposalTokens", cbor_u32(self.proposal_tokens));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("rejectedTokens", cbor_u32(self.rejected_tokens));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("speculated", CborValue::Bool(self.speculated));
        b.put("targetRoot", cbor_digest(&self.target_root));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, SPECULATIVE_KIND)?;
        let out = Self {
            accepted_tokens: fields.u32("acceptedTokens")?,
            cache_affected: fields.bool("cacheAffected")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            distribution_changed: fields.bool("distributionChanged")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            placement_root: fields.digest("placementRoot")?,
            proposal_root: fields.digest("proposalRoot")?,
            proposal_tokens: fields.u32("proposalTokens")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            rejected_tokens: fields.u32("rejectedTokens")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            speculated: fields.bool("speculated")?,
            target_root: fields.digest("targetRoot")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-speculative", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One CUDA graph constraint that was not captured. Graphs stay off.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub device: String,
    pub code: String,
    pub retryable: bool,
    pub graph_captured: bool,
    pub captured_nodes: u32,
    pub workspace_bytes: u32,
    pub shape_buckets: u32,
    pub kernel_repeated: bool,
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

impl GraphReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "graph",
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
        one_of("reason", &self.reason, GRAPH_REASONS)?;
        exact(&self.device, "slot-0", "a cuda graph record names slot-0")?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "a cuda graph record is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "a cuda graph record is not retryable")?;
        forbid(self.graph_captured, "cuda graphs stay off")?;
        exact_u32(
            self.captured_nodes,
            0,
            "captured nodes stay zero while graphs are off",
        )?;
        exact_u32(
            self.workspace_bytes,
            0,
            "graph workspace stays zero while graphs are off",
        )?;
        exact_u32(
            self.shape_buckets,
            0,
            "shape buckets stay zero while graphs are off",
        )?;
        forbid(
            self.kernel_repeated,
            "a cuda graph record does not repeat a kernel",
        )?;
        stopped(self.forward_ran, self.receipt_stored, "a cuda graph record")?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(GRAPH_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("capturedNodes", cbor_u32(self.captured_nodes));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("device", cbor_text(&self.device));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("graphCaptured", CborValue::Bool(self.graph_captured));
        b.put("kernelRepeated", CborValue::Bool(self.kernel_repeated));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("shapeBuckets", cbor_u32(self.shape_buckets));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("workspaceBytes", cbor_u32(self.workspace_bytes));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, GRAPH_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            captured_nodes: fields.u32("capturedNodes")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            device: fields.text("device")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            graph_captured: fields.bool("graphCaptured")?,
            kernel_repeated: fields.bool("kernelRepeated")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            shape_buckets: fields.u32("shapeBuckets")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            workspace_bytes: fields.u32("workspaceBytes")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-graph", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One second model the worker did not load. One resident model stays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiModelReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub resident_root: DigestHex,
    pub incoming_root: DigestHex,
    pub reason: String,
    pub device_count: u32,
    pub models_loaded: u32,
    pub code: String,
    pub retryable: bool,
    pub second_loaded: bool,
    pub resident_replaced: bool,
    pub tensor_parallel: bool,
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

impl MultiModelReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "multi-model",
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
            &self.resident_root,
            &self.engine_build_root,
            "the resident model repeats the engine build",
        )?;
        distinct(
            &self.resident_root,
            &self.placement_root,
            "the resident model repeats the placement",
        )?;
        distinct(
            &self.incoming_root,
            &self.engine_build_root,
            "the incoming model repeats the engine build",
        )?;
        distinct(
            &self.incoming_root,
            &self.placement_root,
            "the incoming model repeats the placement",
        )?;
        distinct(
            &self.incoming_root,
            &self.resident_root,
            "the incoming model repeats the resident model",
        )?;
        one_of("reason", &self.reason, MULTI_MODEL_REASONS)?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "a multi-model record is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "a multi-model record is not retryable")?;
        exact_u32(
            self.models_loaded,
            1,
            "a multi-model record keeps one resident model",
        )?;
        forbid(
            self.second_loaded,
            "a multi-model record does not load a second model",
        )?;
        forbid(
            self.resident_replaced,
            "a multi-model record does not replace the resident model",
        )?;
        forbid(self.tensor_parallel, "tensor parallel stays off")?;
        stopped(
            self.forward_ran,
            self.receipt_stored,
            "a multi-model record",
        )?;
        device_count(&self.reason, self.device_count)?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(MULTI_MODEL_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("deviceCount", cbor_u32(self.device_count));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("incomingRoot", cbor_digest(&self.incoming_root));
        b.put("modelsLoaded", cbor_u32(self.models_loaded));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("residentReplaced", CborValue::Bool(self.resident_replaced));
        b.put("residentRoot", cbor_digest(&self.resident_root));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("secondLoaded", CborValue::Bool(self.second_loaded));
        b.put("tensorParallel", CborValue::Bool(self.tensor_parallel));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, MULTI_MODEL_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            device_count: fields.u32("deviceCount")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            incoming_root: fields.digest("incomingRoot")?,
            models_loaded: fields.u32("modelsLoaded")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            resident_replaced: fields.bool("residentReplaced")?,
            resident_root: fields.digest("residentRoot")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            second_loaded: fields.bool("secondLoaded")?,
            tensor_parallel: fields.bool("tensorParallel")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-multi-model", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One secondary-slot service the worker did not start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecondaryReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub primary_root: DigestHex,
    pub secondary_root: DigestHex,
    pub reason: String,
    pub secondary_slot: String,
    pub code: String,
    pub retryable: bool,
    pub model_named: bool,
    pub embeddings_requested: bool,
    pub overflow_requested: bool,
    pub service_started: bool,
    pub embeddings_ran: bool,
    pub overflow_placed: bool,
    pub device_opened: bool,
    pub tensor_parallel: bool,
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

impl SecondaryReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "secondary",
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
            &self.primary_root,
            &self.engine_build_root,
            "the primary model repeats the engine build",
        )?;
        distinct(
            &self.primary_root,
            &self.placement_root,
            "the primary model repeats the placement",
        )?;
        distinct(
            &self.secondary_root,
            &self.engine_build_root,
            "the secondary model repeats the engine build",
        )?;
        distinct(
            &self.secondary_root,
            &self.placement_root,
            "the secondary model repeats the placement",
        )?;
        distinct(
            &self.secondary_root,
            &self.primary_root,
            "the secondary model repeats the primary model",
        )?;
        one_of("reason", &self.reason, SECONDARY_REASONS)?;
        exact(
            &self.secondary_slot,
            "slot-1",
            "a secondary service names slot-1",
        )?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "a secondary service is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "a secondary service is not retryable")?;
        forbid(self.service_started, "a secondary service does not start")?;
        forbid(
            self.device_opened,
            "a secondary service does not open a device",
        )?;
        forbid(self.tensor_parallel, "tensor parallel stays off")?;
        forbid(
            self.embeddings_ran,
            "a secondary service does not run embeddings",
        )?;
        forbid(
            self.overflow_placed,
            "a secondary service does not place overflow",
        )?;
        stopped(self.forward_ran, self.receipt_stored, "a secondary service")?;
        secondary_flags(
            &self.reason,
            self.model_named,
            self.embeddings_requested,
            self.overflow_requested,
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(SECONDARY_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("deviceOpened", CborValue::Bool(self.device_opened));
        b.put("embeddingsRan", CborValue::Bool(self.embeddings_ran));
        b.put(
            "embeddingsRequested",
            CborValue::Bool(self.embeddings_requested),
        );
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("modelNamed", CborValue::Bool(self.model_named));
        b.put("overflowPlaced", CborValue::Bool(self.overflow_placed));
        b.put(
            "overflowRequested",
            CborValue::Bool(self.overflow_requested),
        );
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("primaryRoot", cbor_digest(&self.primary_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("secondaryRoot", cbor_digest(&self.secondary_root));
        b.put("secondarySlot", cbor_text(&self.secondary_slot));
        b.put("serviceStarted", CborValue::Bool(self.service_started));
        b.put("tensorParallel", CborValue::Bool(self.tensor_parallel));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, SECONDARY_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            device_opened: fields.bool("deviceOpened")?,
            embeddings_ran: fields.bool("embeddingsRan")?,
            embeddings_requested: fields.bool("embeddingsRequested")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            model_named: fields.bool("modelNamed")?,
            overflow_placed: fields.bool("overflowPlaced")?,
            overflow_requested: fields.bool("overflowRequested")?,
            placement_root: fields.digest("placementRoot")?,
            primary_root: fields.digest("primaryRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            secondary_root: fields.digest("secondaryRoot")?,
            secondary_slot: fields.text("secondarySlot")?,
            service_started: fields.bool("serviceStarted")?,
            tensor_parallel: fields.bool("tensorParallel")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-secondary", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
