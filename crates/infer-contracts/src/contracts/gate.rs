//! Signature check and the compile refusals that stop a forward.
//!
//! Each report records one cold micro fixture. None of them checks a
//! signature, clears the Ed25519 cofactor, truncates a prompt, compiles a
//! prompt, or opens a weight file. The layouts are specified in
//! `spec/KIP-INFER-0096-signature-check.md`,
//! `spec/KIP-INFER-0097-context-limit.md`,
//! `spec/KIP-INFER-0098-prompt-compilation.md`,
//! `spec/KIP-INFER-0099-model-image-invalid.md`, and
//! `spec/KIP-INFER-0100-model-image-signature.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{cbor_digest, cbor_text, cbor_u32, expect_kind_version, one_of, Fields};

use super::common::{put_extensions, Builder};
use super::exposure::{ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES};
use super::lifecycle::cold_single;
use super::product::MICRO_CONTEXT;
use super::resilience::ED25519_SCALAR_BYTES;

pub const SIGNATURE_CHECK_KIND: &str = "knolo.infer.signature-check-report";
pub const CONTEXT_LIMIT_KIND: &str = "knolo.infer.context-limit-report";
pub const PROMPT_COMPILATION_KIND: &str = "knolo.infer.prompt-compilation-report";
pub const IMAGE_INVALID_KIND: &str = "knolo.infer.image-invalid-report";
pub const IMAGE_SIGNATURE_KIND: &str = "knolo.infer.image-signature-report";

pub const MAX_CONTEXT_PROMPT: u32 = 64;
pub const MAX_REJECTED_TOKEN: u32 = 1024;
pub const MAX_SIGNATURE_RECORD_BYTES: u32 = 256;
pub const MAX_SIGNATURE_RECORD_COUNT: u32 = 16;

const CHECK_STATUSES: &[&str] = &["checked", "rejected", "unsigned-local"];
const CONTEXT_REASONS: &[&str] = &["prompt", "budget", "overflow"];
const PROMPT_FAILURES: &[&str] = &["empty", "vocab", "size"];
const IMAGE_REASONS: &[&str] = &["empty", "canonical", "format", "inventory"];
const IMAGE_SIGNATURE_REASONS: &[&str] = &["algorithm", "length", "count"];

#[allow(clippy::too_many_arguments)]
fn accept_report(
    noun: &str,
    validation_result: &str,
    allowed_validation: &[&str],
    execution_mode_value: &str,
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
        execution_mode_value,
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

fn count_exact(value: u32, expected: u32, message: &str) -> Result<(), InferFailure> {
    if value == expected {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn compared_counts(public_key: u32, signature: u32, scalar: u32) -> Result<(), InferFailure> {
    count_exact(
        public_key,
        ED25519_PUBLIC_KEY_BYTES,
        "ed25519 public keys are 32 bytes",
    )?;
    count_exact(
        signature,
        ED25519_SIGNATURE_BYTES,
        "ed25519 signatures are 64 bytes",
    )?;
    count_exact(scalar, ED25519_SCALAR_BYTES, "ed25519 scalars are 32 bytes")
}

fn unsigned_counts(public_key: u32, signature: u32, scalar: u32) -> Result<(), InferFailure> {
    count_exact(public_key, 0, "an unsigned release carries a public key")?;
    count_exact(signature, 0, "an unsigned release carries signature bytes")?;
    count_exact(scalar, 0, "an unsigned release carries a scalar")
}

fn stopped(forward_ran: bool, receipt_stored: bool, subject: &str) -> Result<(), InferFailure> {
    forbid(forward_ran, &format!("{subject} does not run the forward"))?;
    forbid(receipt_stored, &format!("{subject} stores no receipt"))
}

/// Host-supplied Ed25519 signature check. The signature is not checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureCheckReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub check_status: String,
    pub signature_checked: bool,
    pub cofactor_cleared: bool,
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

impl SignatureCheckReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "signature-check",
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
            "the signed message repeats the engine build",
        )?;
        distinct(
            &self.message_root,
            &self.release_root,
            "the signed message repeats the release",
        )?;
        forbid(
            self.key_material_present,
            "key material stays in host storage",
        )?;
        forbid(self.cofactor_cleared, "the cofactor stays uncleared")?;
        one_of("checkStatus", &self.check_status, CHECK_STATUSES)?;
        match self.check_status.as_str() {
            "unsigned-local" => {
                unsigned_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                forbid(
                    self.signature_checked,
                    "an unsigned release checks a signature",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a checked signature is verified",
                )?;
            }
            "checked" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.signature_checked,
                    "a checked signature records the check",
                )?;
                exact(
                    &self.validation_result,
                    "verified",
                    "a checked signature is verified",
                )?;
            }
            "rejected" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.signature_checked,
                    "a rejected signature records the check",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a checked signature is verified",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field checkStatus has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(SIGNATURE_CHECK_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("checkStatus", cbor_text(&self.check_status));
        b.put("cofactorCleared", CborValue::Bool(self.cofactor_cleared));
        b.put("concurrency", cbor_u32(self.concurrency));
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
        b.put("releaseRoot", cbor_digest(&self.release_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("scalarBytes", cbor_u32(self.scalar_bytes));
        b.put("signatureBytes", cbor_u32(self.signature_bytes));
        b.put("signatureChecked", CborValue::Bool(self.signature_checked));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, SIGNATURE_CHECK_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            check_status: fields.text("checkStatus")?,
            cofactor_cleared: fields.bool("cofactorCleared")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            key_material_present: fields.bool("keyMaterialPresent")?,
            message_root: fields.digest("messageRoot")?,
            placement_root: fields.digest("placementRoot")?,
            public_key_bytes: fields.u32("publicKeyBytes")?,
            release_root: fields.digest("releaseRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            scalar_bytes: fields.u32("scalarBytes")?,
            signature_bytes: fields.u32("signatureBytes")?,
            signature_checked: fields.bool("signatureChecked")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-signature-check", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One prompt that does not fit the micro context. Tokens are not truncated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextLimitReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub prompt_tokens: u32,
    pub reserved_tokens: u32,
    pub context_tokens: u32,
    pub truncated: bool,
    pub code: String,
    pub retryable: bool,
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

impl ContextLimitReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "context-limit",
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
        one_of("reason", &self.reason, CONTEXT_REASONS)?;
        if self.prompt_tokens == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a context limit has a prompt",
            ));
        }
        count_exact(
            self.context_tokens,
            MICRO_CONTEXT,
            "the context-limit report is the micro fixture",
        )?;
        forbid(self.truncated, "tokens are not truncated")?;
        exact(
            &self.code,
            "CONTEXT_LIMIT_EXCEEDED",
            "a context limit is CONTEXT_LIMIT_EXCEEDED",
        )?;
        forbid(self.retryable, "a context limit is not retryable")?;
        stopped(self.forward_ran, self.receipt_stored, "a context limit")?;
        match self.reason.as_str() {
            "prompt" => {
                if self.prompt_tokens <= MICRO_CONTEXT {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a prompt that fits is not a prompt limit",
                    ));
                }
                if self.prompt_tokens > MAX_CONTEXT_PROMPT {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "prompt tokens exceed the record cap",
                    ));
                }
                if self.reserved_tokens > MICRO_CONTEXT {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a prompt limit does not reserve past the context",
                    ));
                }
            }
            "budget" => {
                if self.prompt_tokens > MICRO_CONTEXT {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a budget limit prompt fits in the context",
                    ));
                }
                if self.reserved_tokens == 0 || self.reserved_tokens > MICRO_CONTEXT {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a budget limit reserves at most the context",
                    ));
                }
                if self.prompt_tokens + self.reserved_tokens <= MICRO_CONTEXT {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a budget that fits is not a context limit",
                    ));
                }
            }
            "overflow" => {
                let sum = u64::from(self.prompt_tokens) + u64::from(self.reserved_tokens);
                if sum <= u64::from(u32::MAX) {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "an overflowing context does not fit in u32",
                    ));
                }
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field reason has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(CONTEXT_LIMIT_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("contextTokens", cbor_u32(self.context_tokens));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("promptTokens", cbor_u32(self.prompt_tokens));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("reservedTokens", cbor_u32(self.reserved_tokens));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("truncated", CborValue::Bool(self.truncated));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, CONTEXT_LIMIT_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            context_tokens: fields.u32("contextTokens")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            placement_root: fields.digest("placementRoot")?,
            prompt_tokens: fields.u32("promptTokens")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            reserved_tokens: fields.u32("reservedTokens")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            truncated: fields.bool("truncated")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-context-limit", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One prompt the compiler refused. The prompt is not compiled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptCompilationReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub failure: String,
    pub token_count: u32,
    pub rejected_token: u32,
    pub code: String,
    pub retryable: bool,
    pub template_rendered: bool,
    pub tokenizer_parsed: bool,
    pub prompt_compiled: bool,
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

impl PromptCompilationReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "prompt-compilation",
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
        one_of("failure", &self.failure, PROMPT_FAILURES)?;
        exact(
            &self.code,
            "PROMPT_COMPILATION_FAILED",
            "a prompt failure is PROMPT_COMPILATION_FAILED",
        )?;
        forbid(self.retryable, "a prompt failure is not retryable")?;
        forbid(
            self.prompt_compiled,
            "a prompt failure does not compile the prompt",
        )?;
        stopped(self.forward_ran, self.receipt_stored, "a prompt failure")?;
        require_flag(
            self.template_rendered,
            "a prompt failure rendered the template",
        )?;
        require_flag(
            self.tokenizer_parsed,
            "a prompt failure parsed the tokenizer",
        )?;
        match self.failure.as_str() {
            "empty" => {
                count_exact(self.token_count, 0, "an empty prompt has no tokens")?;
                count_exact(
                    self.rejected_token,
                    0,
                    "an empty prompt has no rejected token",
                )?;
            }
            "vocab" => {
                if self.token_count == 0 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a vocabulary failure has a prompt",
                    ));
                }
                if self.token_count > MICRO_CONTEXT {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "the prompt compilation is the micro fixture",
                    ));
                }
                if self.rejected_token < MICRO_CONTEXT {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a vocabulary failure is outside the micro vocabulary",
                    ));
                }
                if self.rejected_token > MAX_REJECTED_TOKEN {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a rejected token exceeds the record cap",
                    ));
                }
            }
            "size" => {
                if self.token_count != 0 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a prompt that is too large has no token count",
                    ));
                }
                count_exact(
                    self.rejected_token,
                    0,
                    "a prompt that is too large has no rejected token",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field failure has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(PROMPT_COMPILATION_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("failure", cbor_text(&self.failure));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("promptCompiled", CborValue::Bool(self.prompt_compiled));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("rejectedToken", cbor_u32(self.rejected_token));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("templateRendered", CborValue::Bool(self.template_rendered));
        b.put("tokenCount", cbor_u32(self.token_count));
        b.put("tokenizerParsed", CborValue::Bool(self.tokenizer_parsed));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, PROMPT_COMPILATION_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            failure: fields.text("failure")?,
            forward_ran: fields.bool("forwardRan")?,
            placement_root: fields.digest("placementRoot")?,
            prompt_compiled: fields.bool("promptCompiled")?,
            receipt_stored: fields.bool("receiptStored")?,
            rejected_token: fields.u32("rejectedToken")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            template_rendered: fields.bool("templateRendered")?,
            token_count: fields.u32("tokenCount")?,
            tokenizer_parsed: fields.bool("tokenizerParsed")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-prompt-compilation", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One model image the compiler refused. The image is not executed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInvalidReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub image_root: DigestHex,
    pub reason: String,
    pub code: String,
    pub retryable: bool,
    pub image_parsed: bool,
    pub weights_opened: bool,
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

impl ImageInvalidReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "image-invalid",
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
            &self.image_root,
            &self.engine_build_root,
            "the image repeats the engine build",
        )?;
        distinct(
            &self.image_root,
            &self.placement_root,
            "the image repeats the placement",
        )?;
        one_of("reason", &self.reason, IMAGE_REASONS)?;
        exact(
            &self.code,
            "MODEL_IMAGE_INVALID",
            "an invalid image is MODEL_IMAGE_INVALID",
        )?;
        forbid(self.retryable, "an invalid image is not retryable")?;
        stopped(self.forward_ran, self.receipt_stored, "an invalid image")?;
        match self.reason.as_str() {
            "empty" => {
                forbid(self.image_parsed, "an empty image is not parsed")?;
                forbid(self.weights_opened, "an empty image does not open weights")?;
            }
            "canonical" => {
                forbid(self.image_parsed, "a non-canonical image is not parsed")?;
                forbid(
                    self.weights_opened,
                    "a non-canonical image does not open weights",
                )?;
            }
            "format" => {
                require_flag(self.image_parsed, "a format refusal parsed the image")?;
                forbid(
                    self.weights_opened,
                    "a format refusal does not open weights",
                )?;
            }
            "inventory" => {
                require_flag(self.image_parsed, "an inventory refusal parsed the image")?;
                require_flag(
                    self.weights_opened,
                    "an inventory refusal opens the weights",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field reason has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(IMAGE_INVALID_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("imageParsed", CborValue::Bool(self.image_parsed));
        b.put("imageRoot", cbor_digest(&self.image_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("weightsOpened", CborValue::Bool(self.weights_opened));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, IMAGE_INVALID_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            image_parsed: fields.bool("imageParsed")?,
            image_root: fields.digest("imageRoot")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            weights_opened: fields.bool("weightsOpened")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-image-invalid", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One signature block the image compiler refused. The key is not read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSignatureReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub image_root: DigestHex,
    pub reason: String,
    pub signature_count: u32,
    pub signature_bytes: u32,
    pub code: String,
    pub retryable: bool,
    pub algorithm_accepted: bool,
    pub weights_opened: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
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

impl ImageSignatureReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "image-signature",
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
            &self.image_root,
            &self.engine_build_root,
            "the image repeats the engine build",
        )?;
        distinct(
            &self.image_root,
            &self.placement_root,
            "the image repeats the placement",
        )?;
        one_of("reason", &self.reason, IMAGE_SIGNATURE_REASONS)?;
        exact(
            &self.code,
            "MODEL_IMAGE_SIGNATURE_INVALID",
            "an invalid signature block is MODEL_IMAGE_SIGNATURE_INVALID",
        )?;
        forbid(
            self.retryable,
            "an invalid signature block is not retryable",
        )?;
        forbid(
            self.key_material_present,
            "key material stays in host storage",
        )?;
        forbid(
            self.weights_opened,
            "a signature refusal does not open weights",
        )?;
        stopped(self.forward_ran, self.receipt_stored, "a signature refusal")?;
        match self.reason.as_str() {
            "algorithm" => {
                count_exact(
                    self.signature_count,
                    1,
                    "an algorithm refusal carries one signature",
                )?;
                count_exact(
                    self.signature_bytes,
                    0,
                    "an algorithm refusal does not read the signature",
                )?;
                forbid(
                    self.algorithm_accepted,
                    "an algorithm refusal does not accept the algorithm",
                )?;
            }
            "length" => {
                count_exact(
                    self.signature_count,
                    1,
                    "a length refusal carries one signature",
                )?;
                if self.signature_bytes == 0 || self.signature_bytes == ED25519_SIGNATURE_BYTES {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a length refusal read a signature that is not 64 bytes",
                    ));
                }
                if self.signature_bytes > MAX_SIGNATURE_RECORD_BYTES {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "signature bytes exceed the record cap",
                    ));
                }
                require_flag(
                    self.algorithm_accepted,
                    "a length refusal accepted the algorithm",
                )?;
            }
            "count" => {
                if self.signature_count <= 8 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "eight signatures are still a shape check",
                    ));
                }
                if self.signature_count > MAX_SIGNATURE_RECORD_COUNT {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "signature count exceeds the record cap",
                    ));
                }
                count_exact(
                    self.signature_bytes,
                    0,
                    "a count refusal does not read the signature",
                )?;
                forbid(
                    self.algorithm_accepted,
                    "a count refusal does not accept the algorithm",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field reason has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(IMAGE_SIGNATURE_KIND);
        b.put(
            "algorithmAccepted",
            CborValue::Bool(self.algorithm_accepted),
        );
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("imageRoot", cbor_digest(&self.image_root));
        b.put(
            "keyMaterialPresent",
            CborValue::Bool(self.key_material_present),
        );
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("signatureBytes", cbor_u32(self.signature_bytes));
        b.put("signatureCount", cbor_u32(self.signature_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("weightsOpened", CborValue::Bool(self.weights_opened));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, IMAGE_SIGNATURE_KIND)?;
        let out = Self {
            algorithm_accepted: fields.bool("algorithmAccepted")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            image_root: fields.digest("imageRoot")?,
            key_material_present: fields.bool("keyMaterialPresent")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            signature_bytes: fields.u32("signatureBytes")?,
            signature_count: fields.u32("signatureCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            weights_opened: fields.bool("weightsOpened")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-image-signature", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
