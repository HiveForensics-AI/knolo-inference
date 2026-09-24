//! Curve, rollback, client disconnect, and receipt-store reports.
//!
//! Each report records one cold micro fixture. None of them multiplies the
//! Ed25519 base point, rewrites a lockfile, closes a socket, or writes a
//! receipt. The layouts are specified in
//! `spec/KIP-INFER-0064-curve.md`,
//! `spec/KIP-INFER-0065-rollback.md`,
//! `spec/KIP-INFER-0066-disconnect.md`, and
//! `spec/KIP-INFER-0067-receipt-store.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    bounded_text, cbor_digest, cbor_text, cbor_u32, expect_kind_version, one_of, Fields,
};

use super::common::{put_extensions, Builder};
use super::exposure::{ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES};
use super::lifecycle::cold_single;

pub const CURVE_KIND: &str = "knolo.infer.curve-report";
pub const ROLLBACK_KIND: &str = "knolo.infer.rollback-report";
pub const DISCONNECT_KIND: &str = "knolo.infer.disconnect-report";
pub const RECEIPT_STORE_KIND: &str = "knolo.infer.receipt-store-report";

const MICRO_TOKENS: u32 = 16;
const DISCONNECT_STAGES: &[&str] = &["queue", "prefill", "decode", "stream"];
const ROLLBACK_REASONS: &[&str] = &["pin-mismatch", "failed-load", "operator"];

fn execution_mode(value: &str) -> Result<(), InferFailure> {
    one_of("executionMode", value, &["isolated-replay", "pinned"])
}

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

/// Host-supplied Ed25519 curve result. The base point is not multiplied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurveReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub curve_status: String,
    pub curve_computed: bool,
    pub public_key_bytes: u32,
    pub signature_bytes: u32,
    pub point_checked: bool,
    pub base_multiplied: bool,
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

impl CurveReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "curve",
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
        forbid(self.base_multiplied, "the base point stays unmultiplied")?;
        match self.curve_status.as_str() {
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
                    self.curve_computed,
                    "an unsigned release computes the curve",
                )?;
                forbid(self.point_checked, "an unsigned release checks a point")?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only an on-curve point is verified",
                )?;
            }
            "on-curve" | "off-curve" => {
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
                require_flag(self.curve_computed, "a checked release computes the curve")?;
                require_flag(self.point_checked, "a checked release checks the point")?;
                if self.curve_status == "on-curve" {
                    exact(
                        &self.validation_result,
                        "verified",
                        "an on-curve point is verified",
                    )?;
                } else {
                    exact(
                        &self.validation_result,
                        "recorded",
                        "only an on-curve point is verified",
                    )?;
                }
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field curveStatus has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(CURVE_KIND);
        b.put("baseMultiplied", CborValue::Bool(self.base_multiplied));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("curveComputed", CborValue::Bool(self.curve_computed));
        b.put("curveStatus", cbor_text(&self.curve_status));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put(
            "keyMaterialPresent",
            CborValue::Bool(self.key_material_present),
        );
        b.put("messageRoot", cbor_digest(&self.message_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("pointChecked", CborValue::Bool(self.point_checked));
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
        expect_kind_version(&mut fields, CURVE_KIND)?;
        let out = Self {
            base_multiplied: fields.bool("baseMultiplied")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            curve_computed: fields.bool("curveComputed")?,
            curve_status: fields.text("curveStatus")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            key_material_present: fields.bool("keyMaterialPresent")?,
            message_root: fields.digest("messageRoot")?,
            placement_root: fields.digest("placementRoot")?,
            point_checked: fields.bool("pointChecked")?,
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
        digest_value("infer-curve", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One rollback of the infer-local pin. The lockfile is not rewritten.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RollbackReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub previous_image_root: DigestHex,
    pub incoming_image_root: DigestHex,
    pub previous_artifact_root: DigestHex,
    pub incoming_artifact_root: DigestHex,
    pub reason: String,
    pub lockfile_mutated: bool,
    pub core_lock_touched: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl RollbackReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "rollback",
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
            &self.previous_image_root,
            &self.engine_build_root,
            "the previous image repeats the engine build",
        )?;
        distinct(
            &self.incoming_image_root,
            &self.engine_build_root,
            "the incoming image repeats the engine build",
        )?;
        distinct(
            &self.incoming_image_root,
            &self.previous_image_root,
            "the incoming image repeats the previous image",
        )?;
        distinct(
            &self.previous_artifact_root,
            &self.engine_build_root,
            "the previous artifact repeats the engine build",
        )?;
        distinct(
            &self.incoming_artifact_root,
            &self.engine_build_root,
            "the incoming artifact repeats the engine build",
        )?;
        distinct(
            &self.incoming_artifact_root,
            &self.previous_artifact_root,
            "the incoming artifact repeats the previous artifact",
        )?;
        forbid(
            self.lockfile_mutated,
            "a rollback record leaves the lockfile unchanged",
        )?;
        forbid(
            self.core_lock_touched,
            "a rollback record leaves the core lockfile unchanged",
        )?;
        one_of("reason", &self.reason, ROLLBACK_REASONS)?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(ROLLBACK_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("coreLockTouched", CborValue::Bool(self.core_lock_touched));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put(
            "incomingArtifactRoot",
            cbor_digest(&self.incoming_artifact_root),
        );
        b.put("incomingImageRoot", cbor_digest(&self.incoming_image_root));
        b.put("lockfileMutated", CborValue::Bool(self.lockfile_mutated));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put(
            "previousArtifactRoot",
            cbor_digest(&self.previous_artifact_root),
        );
        b.put("previousImageRoot", cbor_digest(&self.previous_image_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, ROLLBACK_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            core_lock_touched: fields.bool("coreLockTouched")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            incoming_artifact_root: fields.digest("incomingArtifactRoot")?,
            incoming_image_root: fields.digest("incomingImageRoot")?,
            lockfile_mutated: fields.bool("lockfileMutated")?,
            placement_root: fields.digest("placementRoot")?,
            previous_artifact_root: fields.digest("previousArtifactRoot")?,
            previous_image_root: fields.digest("previousImageRoot")?,
            reason: fields.text("reason")?,
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
        digest_value("infer-rollback", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One client disconnect. The listener stays up and the socket is not closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisconnectReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub request_id: String,
    pub stage: String,
    pub outcome: String,
    pub code: String,
    pub listener_up: bool,
    pub socket_closed: bool,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl DisconnectReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "disconnect",
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
        one_of("stage", &self.stage, DISCONNECT_STAGES)?;
        exact(
            &self.outcome,
            "cancelled",
            "a client disconnect cancels the request",
        )?;
        exact(
            &self.code,
            "REQUEST_CANCELLED",
            "a client disconnect is REQUEST_CANCELLED",
        )?;
        require_flag(self.listener_up, "the listener stays up")?;
        forbid(
            self.socket_closed,
            "a disconnect record does not close the socket",
        )?;
        if self.prompt_tokens == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the disconnect has no prompt",
            ));
        }
        if self.prompt_tokens > MICRO_TOKENS {
            return Err(fail(
                ErrorCode::ContextLimitExceeded,
                "the disconnect is the micro fixture",
            ));
        }
        if self.stage == "queue" && self.output_tokens != 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a queued disconnect has no output",
            ));
        }
        if self.prompt_tokens.saturating_add(self.output_tokens) > MICRO_TOKENS {
            return Err(fail(
                ErrorCode::ContextLimitExceeded,
                "the disconnect is the micro fixture",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(DISCONNECT_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("listenerUp", CborValue::Bool(self.listener_up));
        b.put("outcome", cbor_text(&self.outcome));
        b.put("outputTokens", cbor_u32(self.output_tokens));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("promptTokens", cbor_u32(self.prompt_tokens));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("requestId", cbor_text(&self.request_id));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("socketClosed", CborValue::Bool(self.socket_closed));
        b.put("stage", cbor_text(&self.stage));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, DISCONNECT_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            listener_up: fields.bool("listenerUp")?,
            outcome: fields.text("outcome")?,
            output_tokens: fields.u32("outputTokens")?,
            placement_root: fields.digest("placementRoot")?,
            prompt_tokens: fields.u32("promptTokens")?,
            request_count: fields.u32("requestCount")?,
            request_id: fields.text("requestId")?,
            run_count: fields.u32("runCount")?,
            socket_closed: fields.bool("socketClosed")?,
            stage: fields.text("stage")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-disconnect", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One receipt-store failure. The receipt file is not written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptStoreReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub request_id: String,
    pub code: String,
    pub retryable: bool,
    pub receipt_stored: bool,
    pub journal_event: String,
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

impl ReceiptStoreReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "receipt store",
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
        exact(
            &self.code,
            "RECEIPT_PERSIST_FAILED",
            "a receipt-store failure is RECEIPT_PERSIST_FAILED",
        )?;
        require_flag(self.retryable, "a receipt-store failure is retryable")?;
        forbid(
            self.receipt_stored,
            "a receipt-store failure stores no receipt",
        )?;
        exact(&self.journal_event, "failed", "the journal seals failed")?;
        partial_receipt(
            &self.partial_receipt,
            &self.engine_build_root,
            &self.placement_root,
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(RECEIPT_STORE_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("journalEvent", cbor_text(&self.journal_event));
        b.put("partialReceipt", cbor_text(&self.partial_receipt));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("requestId", cbor_text(&self.request_id));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, RECEIPT_STORE_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            journal_event: fields.text("journalEvent")?,
            partial_receipt: fields.text("partialReceipt")?,
            placement_root: fields.digest("placementRoot")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            request_id: fields.text("requestId")?,
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
        digest_value("infer-receipt-store", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
