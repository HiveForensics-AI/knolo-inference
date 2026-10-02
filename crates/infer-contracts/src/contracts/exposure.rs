//! Cache side-channel, signature equation, safe error, and redacted log reports.
//!
//! Each report records one cold micro fixture. None of them allocates a
//! prefix index, computes an Ed25519 curve, returns a prompt, or writes a
//! log line. The layouts are specified in
//! `spec/KIP-INFER-0060-cache-channel.md`,
//! `spec/KIP-INFER-0061-signature-equation.md`,
//! `spec/KIP-INFER-0062-safe-error.md`, and
//! `spec/KIP-INFER-0063-redacted-log.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    bounded_text, cbor_digest, cbor_text, cbor_u32, expect_kind_version, one_of, Fields,
};

use super::common::{put_extensions, Builder};
use super::lifecycle::cold_single;

pub const CACHE_CHANNEL_KIND: &str = "knolo.infer.cache-channel-report";
pub const EQUATION_KIND: &str = "knolo.infer.equation-report";
pub const SAFE_ERROR_KIND: &str = "knolo.infer.safe-error-report";
pub const REDACTION_KIND: &str = "knolo.infer.redaction-report";

/// Ed25519 public keys are 32 bytes.
pub const ED25519_PUBLIC_KEY_BYTES: u32 = 32;

/// Ed25519 signatures are 64 bytes.
pub const ED25519_SIGNATURE_BYTES: u32 = 64;

const LOG_STAGES: &[&str] = &[
    "api",
    "prompt",
    "admission",
    "prefill",
    "decode",
    "finalize",
];

fn execution_mode(value: &str) -> Result<(), InferFailure> {
    one_of("executionMode", value, &["isolated-replay", "pinned"])
}

#[allow(clippy::too_many_arguments)]
fn accept_exposure(
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
    execution_mode(execution_mode_value)?;
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

fn request_token(value: &str) -> Result<(), InferFailure> {
    bounded_text("requestId", value, 64)?;
    if value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    {
        Ok(())
    } else {
        Err(fail(
            ErrorCode::ContractInvalid,
            "the request id is a token",
        ))
    }
}

fn partial_receipt(
    value: &str,
    engine: &DigestHex,
    placement: &DigestHex,
) -> Result<(), InferFailure> {
    if value == "absent" {
        return Ok(());
    }
    let Ok(root) = DigestHex::parse(value) else {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "a partial receipt is absent or a root",
        ));
    };
    distinct(
        &root,
        engine,
        "the partial receipt repeats the engine build",
    )?;
    distinct(
        &root,
        placement,
        "the partial receipt repeats the placement",
    )
}

/// Prefix-cache policy for one cold micro fixture. The index is not allocated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheChannelReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub tenant_root: DigestHex,
    pub project_root: DigestHex,
    pub sharing_policy: String,
    pub cross_tenant: bool,
    pub existence_disclosure: String,
    pub metrics_scope: String,
    pub prefix_allocated: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl CacheChannelReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_exposure(
            "cache channel",
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
            &self.tenant_root,
            &self.engine_build_root,
            "the tenant repeats the engine build",
        )?;
        distinct(
            &self.project_root,
            &self.engine_build_root,
            "the project repeats the engine build",
        )?;
        distinct(
            &self.project_root,
            &self.tenant_root,
            "the project repeats the tenant",
        )?;
        exact(
            &self.sharing_policy,
            "isolated",
            "cache sharing is isolated",
        )?;
        forbid(self.cross_tenant, "cross-tenant sharing is off")?;
        exact(
            &self.existence_disclosure,
            "hidden",
            "another tenant prefix stays hidden",
        )?;
        exact(
            &self.metrics_scope,
            "aggregate",
            "cache metrics stay aggregated",
        )?;
        forbid(self.prefix_allocated, "the prefix index stays unallocated")?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(CACHE_CHANNEL_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("crossTenant", CborValue::Bool(self.cross_tenant));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        b.put("existenceDisclosure", cbor_text(&self.existence_disclosure));
        put_extensions(&mut b, &self.extensions)?;
        b.put("metricsScope", cbor_text(&self.metrics_scope));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("prefixAllocated", CborValue::Bool(self.prefix_allocated));
        b.put("projectRoot", cbor_digest(&self.project_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("sharingPolicy", cbor_text(&self.sharing_policy));
        b.put("tenantRoot", cbor_digest(&self.tenant_root));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, CACHE_CHANNEL_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            cross_tenant: fields.bool("crossTenant")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            existence_disclosure: fields.text("existenceDisclosure")?,
            extensions: fields.extensions()?,
            metrics_scope: fields.text("metricsScope")?,
            placement_root: fields.digest("placementRoot")?,
            prefix_allocated: fields.bool("prefixAllocated")?,
            project_root: fields.digest("projectRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            sharing_policy: fields.text("sharingPolicy")?,
            tenant_root: fields.digest("tenantRoot")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-cache-channel", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Host-supplied Ed25519 equation status. This crate does not compute the curve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EquationReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub equation_status: String,
    pub equation_evaluated: bool,
    pub public_key_bytes: u32,
    pub signature_bytes: u32,
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

impl EquationReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_exposure(
            "equation",
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
        match self.equation_status.as_str() {
            "unsigned-local" => {
                if self.public_key_bytes != 0 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "an unsigned release carries a public key",
                    ));
                }
                if self.signature_bytes != 0 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "an unsigned release carries signature bytes",
                    ));
                }
                forbid(
                    self.equation_evaluated,
                    "an unsigned release evaluates an equation",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only an accepted equation is verified",
                )?;
            }
            "accepted" | "rejected" => {
                if self.public_key_bytes != ED25519_PUBLIC_KEY_BYTES {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "ed25519 public keys are 32 bytes",
                    ));
                }
                if self.signature_bytes != ED25519_SIGNATURE_BYTES {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "ed25519 signatures are 64 bytes",
                    ));
                }
                require_flag(
                    self.equation_evaluated,
                    "a checked release evaluates the equation",
                )?;
                if self.equation_status == "accepted" {
                    exact(
                        &self.validation_result,
                        "verified",
                        "an accepted equation is verified",
                    )?;
                } else {
                    exact(
                        &self.validation_result,
                        "recorded",
                        "only an accepted equation is verified",
                    )?;
                }
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field equationStatus has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(EQUATION_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put(
            "equationEvaluated",
            CborValue::Bool(self.equation_evaluated),
        );
        b.put("equationStatus", cbor_text(&self.equation_status));
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
        b.put("signatureBytes", cbor_u32(self.signature_bytes));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, EQUATION_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            equation_evaluated: fields.bool("equationEvaluated")?,
            equation_status: fields.text("equationStatus")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            key_material_present: fields.bool("keyMaterialPresent")?,
            message_root: fields.digest("messageRoot")?,
            placement_root: fields.digest("placementRoot")?,
            public_key_bytes: fields.u32("publicKeyBytes")?,
            release_root: fields.digest("releaseRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            signature_bytes: fields.u32("signatureBytes")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-equation", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One stable failure without prompt text or a secret.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeErrorReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub code: String,
    pub message: String,
    pub retryable: bool,
    pub request_id: String,
    pub attempt: u32,
    pub prompt_present: bool,
    pub secret_present: bool,
    pub partial_receipt: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl SafeErrorReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_exposure(
            "safe error",
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
        request_token(&self.request_id)?;
        let Some(code) = ErrorCode::parse(&self.code) else {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "field code has an unsupported value",
            ));
        };
        exact(
            &self.message,
            code.as_str(),
            "the safe message is the stable code",
        )?;
        if self.retryable != code.retryable() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "retryability follows the stable code",
            ));
        }
        if self.attempt != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the safe error attempt is one",
            ));
        }
        forbid(self.prompt_present, "a safe error omits the prompt")?;
        forbid(self.secret_present, "a safe error omits secrets")?;
        partial_receipt(
            &self.partial_receipt,
            &self.engine_build_root,
            &self.placement_root,
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(SAFE_ERROR_KIND);
        b.put("attempt", cbor_u32(self.attempt));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("message", cbor_text(&self.message));
        b.put("partialReceipt", cbor_text(&self.partial_receipt));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("promptPresent", CborValue::Bool(self.prompt_present));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("requestId", cbor_text(&self.request_id));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("secretPresent", CborValue::Bool(self.secret_present));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, SAFE_ERROR_KIND)?;
        let out = Self {
            attempt: fields.u32("attempt")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            message: fields.text("message")?,
            partial_receipt: fields.text("partialReceipt")?,
            placement_root: fields.digest("placementRoot")?,
            prompt_present: fields.bool("promptPresent")?,
            request_count: fields.u32("requestCount")?,
            request_id: fields.text("requestId")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            secret_present: fields.bool("secretPresent")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-safe-error", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One redacted completion log line. The line is not written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedactionReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub stage: String,
    pub request_id: String,
    pub prompt_plan_root: DigestHex,
    pub receipt_root: DigestHex,
    pub format: String,
    pub redacted: bool,
    pub prompt: String,
    pub output: String,
    pub token_ids: String,
    pub alias: String,
    pub path: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl RedactionReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_exposure(
            "redaction",
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
        one_of("stage", &self.stage, LOG_STAGES)?;
        request_token(&self.request_id)?;
        distinct(
            &self.prompt_plan_root,
            &self.engine_build_root,
            "the prompt plan repeats the engine build",
        )?;
        distinct(
            &self.receipt_root,
            &self.engine_build_root,
            "the receipt repeats the engine build",
        )?;
        distinct(
            &self.receipt_root,
            &self.prompt_plan_root,
            "the receipt repeats the prompt plan",
        )?;
        exact(
            &self.format,
            "json-line",
            "a completion log is one json line",
        )?;
        require_flag(self.redacted, "completion logs are redacted")?;
        exact(&self.prompt, "omitted", "ordinary logs omit prompt text")?;
        exact(&self.output, "omitted", "ordinary logs omit output text")?;
        exact(&self.token_ids, "omitted", "ordinary logs omit token ids")?;
        exact(&self.alias, "omitted", "ordinary logs omit the alias")?;
        exact(&self.path, "omitted", "ordinary logs omit a path")?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(REDACTION_KIND);
        b.put("alias", cbor_text(&self.alias));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("format", cbor_text(&self.format));
        b.put("output", cbor_text(&self.output));
        b.put("path", cbor_text(&self.path));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("prompt", cbor_text(&self.prompt));
        b.put("promptPlanRoot", cbor_digest(&self.prompt_plan_root));
        b.put("receiptRoot", cbor_digest(&self.receipt_root));
        b.put("redacted", CborValue::Bool(self.redacted));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("requestId", cbor_text(&self.request_id));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("stage", cbor_text(&self.stage));
        b.put("tokenIds", cbor_text(&self.token_ids));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, REDACTION_KIND)?;
        let out = Self {
            alias: fields.text("alias")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            format: fields.text("format")?,
            output: fields.text("output")?,
            path: fields.text("path")?,
            placement_root: fields.digest("placementRoot")?,
            prompt: fields.text("prompt")?,
            prompt_plan_root: fields.digest("promptPlanRoot")?,
            receipt_root: fields.digest("receiptRoot")?,
            redacted: fields.bool("redacted")?,
            request_count: fields.u32("requestCount")?,
            request_id: fields.text("requestId")?,
            run_count: fields.u32("runCount")?,
            stage: fields.text("stage")?,
            token_ids: fields.text("tokenIds")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-redaction", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
