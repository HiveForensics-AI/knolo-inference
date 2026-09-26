//! Cofactor clear and the refusals that stop before a forward.
//!
//! Each report records one cold micro fixture. None of them clears the
//! Ed25519 cofactor, signs a receipt, opens a missing artifact, reads a
//! receipt file, selects a backend, or hashes a digest. The layouts are
//! specified in `spec/KIP-INFER-0101-cofactor-clearing.md`,
//! `spec/KIP-INFER-0102-artifact-missing.md`,
//! `spec/KIP-INFER-0103-receipt-required.md`,
//! `spec/KIP-INFER-0104-backend-not-allowed.md`, and
//! `spec/KIP-INFER-0105-digest-invalid.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex, INFER_DOMAINS};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{cbor_digest, cbor_text, cbor_u32, expect_kind_version, one_of, Fields};

use super::common::{put_extensions, Builder};
use super::exposure::{ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES};
use super::lifecycle::cold_single;
use super::product::LOCAL_WEIGHT_SOURCE;
use super::resilience::ED25519_SCALAR_BYTES;

pub const COFACTOR_KIND: &str = "knolo.infer.cofactor-report";
pub const ARTIFACT_MISSING_KIND: &str = "knolo.infer.artifact-missing-report";
pub const RECEIPT_REQUIRED_KIND: &str = "knolo.infer.receipt-required-report";
pub const BACKEND_KIND: &str = "knolo.infer.backend-report";
pub const DIGEST_INVALID_KIND: &str = "knolo.infer.digest-invalid-report";

pub const MAX_DIGEST_RECORD_HEX: u32 = 128;
pub const MAX_DIGEST_DOMAIN_BYTES: u32 = 64;
pub const RECEIPT_HTTP_STATUS: u32 = 404;

const CLEAR_STATUSES: &[&str] = &["cleared", "rejected", "unsigned-local"];
const ARTIFACT_REASONS: &[&str] = &["lock", "alias", "image", "weights"];
const RECEIPT_REASONS: &[&str] = &["file", "request", "journal"];
const BACKEND_SURFACES: &[&str] = &["run", "measure"];
const DIGEST_REASONS: &[&str] = &["prefix", "length", "alphabet", "domain"];
const UNCHECKED_DOMAIN: &str = "none";

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
    accept_report(
        noun,
        validation_result,
        allowed_validation,
        execution_mode,
        cache_policy,
        concurrency,
        run_count,
        warm_state,
        request_count,
        extensions,
    )
}

/// Host-supplied Ed25519 cofactor clear. The cofactor is not cleared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CofactorReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub clear_status: String,
    pub cofactor_cleared: bool,
    pub receipt_signed: bool,
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

impl CofactorReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "cofactor",
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
        forbid(self.receipt_signed, "the receipt stays unsigned")?;
        one_of("clearStatus", &self.clear_status, CLEAR_STATUSES)?;
        match self.clear_status.as_str() {
            "unsigned-local" => {
                unsigned_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                forbid(
                    self.cofactor_cleared,
                    "an unsigned release clears the cofactor",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a cleared cofactor is verified",
                )?;
            }
            "cleared" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.cofactor_cleared,
                    "a cleared cofactor records the clear",
                )?;
                exact(
                    &self.validation_result,
                    "verified",
                    "a cleared cofactor is verified",
                )?;
            }
            "rejected" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.cofactor_cleared,
                    "a rejected cofactor records the clear",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a cleared cofactor is verified",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field clearStatus has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(COFACTOR_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("clearStatus", cbor_text(&self.clear_status));
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
        b.put("receiptSigned", CborValue::Bool(self.receipt_signed));
        b.put("releaseRoot", cbor_digest(&self.release_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("scalarBytes", cbor_u32(self.scalar_bytes));
        b.put("signatureBytes", cbor_u32(self.signature_bytes));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, COFACTOR_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            clear_status: fields.text("clearStatus")?,
            cofactor_cleared: fields.bool("cofactorCleared")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            key_material_present: fields.bool("keyMaterialPresent")?,
            message_root: fields.digest("messageRoot")?,
            placement_root: fields.digest("placementRoot")?,
            public_key_bytes: fields.u32("publicKeyBytes")?,
            receipt_signed: fields.bool("receiptSigned")?,
            release_root: fields.digest("releaseRoot")?,
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
        digest_value("infer-cofactor", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One pinned artifact the loader did not find. The file is not opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactMissingReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub reason: String,
    pub code: String,
    pub retryable: bool,
    pub source_provider: String,
    pub lock_present: bool,
    pub alias_pinned: bool,
    pub image_opened: bool,
    pub weights_opened: bool,
    pub downloaded: bool,
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

impl ArtifactMissingReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "artifact-missing",
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
        distinct(
            &self.artifact_root,
            &self.engine_build_root,
            "the artifact repeats the engine build",
        )?;
        distinct(
            &self.artifact_root,
            &self.placement_root,
            "the artifact repeats the placement",
        )?;
        distinct(
            &self.artifact_root,
            &self.image_root,
            "the artifact repeats the image",
        )?;
        one_of("reason", &self.reason, ARTIFACT_REASONS)?;
        exact(
            &self.code,
            "MODEL_ARTIFACT_MISSING",
            "a missing artifact is MODEL_ARTIFACT_MISSING",
        )?;
        forbid(self.retryable, "a missing artifact is not retryable")?;
        exact(
            &self.source_provider,
            LOCAL_WEIGHT_SOURCE,
            "the source provider is local",
        )?;
        forbid(self.downloaded, "pull does not download")?;
        stopped(self.forward_ran, self.receipt_stored, "a missing artifact")?;
        match self.reason.as_str() {
            "lock" => {
                forbid(self.lock_present, "a missing lockfile is not present")?;
                forbid(self.alias_pinned, "a missing lockfile has no pinned alias")?;
                forbid(
                    self.image_opened,
                    "a missing lockfile does not open the image",
                )?;
                forbid(
                    self.weights_opened,
                    "a missing lockfile does not open the weights",
                )?;
            }
            "alias" => {
                require_flag(
                    self.lock_present,
                    "an unpinned alias was read from the lockfile",
                )?;
                forbid(self.alias_pinned, "an unpinned alias is not pinned")?;
                forbid(
                    self.image_opened,
                    "an unpinned alias does not open the image",
                )?;
                forbid(
                    self.weights_opened,
                    "an unpinned alias does not open the weights",
                )?;
            }
            "image" => {
                require_flag(
                    self.lock_present,
                    "a missing image was read from the lockfile",
                )?;
                require_flag(self.alias_pinned, "a missing image names a pinned alias")?;
                forbid(self.image_opened, "a missing image is not opened")?;
                forbid(
                    self.weights_opened,
                    "a missing image does not open the weights",
                )?;
            }
            "weights" => {
                require_flag(
                    self.lock_present,
                    "a missing weight artifact was read from the lockfile",
                )?;
                require_flag(
                    self.alias_pinned,
                    "a missing weight artifact names a pinned alias",
                )?;
                require_flag(
                    self.image_opened,
                    "a missing weight artifact opened the image",
                )?;
                forbid(
                    self.weights_opened,
                    "a missing weight artifact is not opened",
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
        let mut b = Builder::typed(ARTIFACT_MISSING_KIND);
        b.put("aliasPinned", CborValue::Bool(self.alias_pinned));
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("downloaded", CborValue::Bool(self.downloaded));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("imageOpened", CborValue::Bool(self.image_opened));
        b.put("imageRoot", cbor_digest(&self.image_root));
        b.put("lockPresent", CborValue::Bool(self.lock_present));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("sourceProvider", cbor_text(&self.source_provider));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("weightsOpened", CborValue::Bool(self.weights_opened));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, ARTIFACT_MISSING_KIND)?;
        let out = Self {
            alias_pinned: fields.bool("aliasPinned")?,
            artifact_root: fields.digest("artifactRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            downloaded: fields.bool("downloaded")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            image_opened: fields.bool("imageOpened")?,
            image_root: fields.digest("imageRoot")?,
            lock_present: fields.bool("lockPresent")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            source_provider: fields.text("sourceProvider")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            weights_opened: fields.bool("weightsOpened")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-artifact-missing", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One receipt the verifier could not use. The file is not read here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptRequiredReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub receipt_root: DigestHex,
    pub reason: String,
    pub code: String,
    pub retryable: bool,
    pub http_status: u32,
    pub receipt_read: bool,
    pub request_bound: bool,
    pub journal_opened: bool,
    pub event_count: u32,
    pub listener_up: bool,
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

impl ReceiptRequiredReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "receipt-required",
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
            &self.receipt_root,
            &self.engine_build_root,
            "the receipt repeats the engine build",
        )?;
        distinct(
            &self.receipt_root,
            &self.placement_root,
            "the receipt repeats the placement",
        )?;
        one_of("reason", &self.reason, RECEIPT_REASONS)?;
        exact(
            &self.code,
            "RECEIPT_REQUIRED",
            "a missing receipt is RECEIPT_REQUIRED",
        )?;
        forbid(self.retryable, "a missing receipt is not retryable")?;
        exact_u32(
            self.event_count,
            0,
            "a required receipt has no journal events",
        )?;
        require_flag(self.listener_up, "the listener stays up")?;
        stopped(self.forward_ran, self.receipt_stored, "a missing receipt")?;
        match self.reason.as_str() {
            "file" => {
                forbid(self.receipt_read, "a missing receipt file is not read")?;
                forbid(
                    self.request_bound,
                    "a missing receipt file has no request id",
                )?;
                forbid(
                    self.journal_opened,
                    "a missing receipt file does not open the journal",
                )?;
                exact_u32(
                    self.http_status,
                    RECEIPT_HTTP_STATUS,
                    "a missing receipt file is HTTP 404",
                )?;
            }
            "request" => {
                require_flag(self.receipt_read, "a missing request id was read")?;
                forbid(self.request_bound, "a missing request id is not bound")?;
                forbid(
                    self.journal_opened,
                    "a missing request id does not open the journal",
                )?;
                exact_u32(
                    self.http_status,
                    0,
                    "a verify-path refusal has no HTTP status",
                )?;
            }
            "journal" => {
                require_flag(
                    self.receipt_read,
                    "an empty journal was read from the receipt",
                )?;
                require_flag(self.request_bound, "an empty journal names a request id")?;
                require_flag(self.journal_opened, "an empty journal was opened")?;
                exact_u32(
                    self.http_status,
                    0,
                    "a verify-path refusal has no HTTP status",
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
        let mut b = Builder::typed(RECEIPT_REQUIRED_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("eventCount", cbor_u32(self.event_count));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("httpStatus", cbor_u32(self.http_status));
        b.put("journalOpened", CborValue::Bool(self.journal_opened));
        b.put("listenerUp", CborValue::Bool(self.listener_up));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptRead", CborValue::Bool(self.receipt_read));
        b.put("receiptRoot", cbor_digest(&self.receipt_root));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestBound", CborValue::Bool(self.request_bound));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, RECEIPT_REQUIRED_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            event_count: fields.u32("eventCount")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            http_status: fields.u32("httpStatus")?,
            journal_opened: fields.bool("journalOpened")?,
            listener_up: fields.bool("listenerUp")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_read: fields.bool("receiptRead")?,
            receipt_root: fields.digest("receiptRoot")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_bound: fields.bool("requestBound")?,
            request_count: fields.u32("requestCount")?,
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
        digest_value("infer-receipt-required", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One throughput mode the reference refused. The backend is not selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub surface: String,
    pub requested_mode: String,
    pub code: String,
    pub retryable: bool,
    pub backend_selected: bool,
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

impl BackendReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "backend",
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
        one_of("surface", &self.surface, BACKEND_SURFACES)?;
        exact(
            &self.requested_mode,
            "throughput",
            "the refused mode is throughput",
        )?;
        exact(
            &self.code,
            "BACKEND_NOT_ALLOWED",
            "a refused backend is BACKEND_NOT_ALLOWED",
        )?;
        forbid(self.retryable, "a refused backend is not retryable")?;
        forbid(self.backend_selected, "the backend is not selected")?;
        stopped(self.forward_ran, self.receipt_stored, "a refused backend")?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(BACKEND_KIND);
        b.put("backendSelected", CborValue::Bool(self.backend_selected));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("requestedMode", cbor_text(&self.requested_mode));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("surface", cbor_text(&self.surface));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, BACKEND_KIND)?;
        let out = Self {
            backend_selected: fields.bool("backendSelected")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            placement_root: fields.digest("placementRoot")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            requested_mode: fields.text("requestedMode")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            surface: fields.text("surface")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-backend", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One digest text or domain the parser refused. The payload is not hashed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DigestInvalidReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub hex_length: u32,
    pub domain: String,
    pub code: String,
    pub retryable: bool,
    pub prefix_accepted: bool,
    pub length_accepted: bool,
    pub alphabet_accepted: bool,
    pub domain_accepted: bool,
    pub hashed: bool,
    pub file_opened: bool,
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

impl DigestInvalidReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "digest-invalid",
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
        one_of("reason", &self.reason, DIGEST_REASONS)?;
        exact(
            &self.code,
            "DIGEST_INVALID",
            "an invalid digest is DIGEST_INVALID",
        )?;
        forbid(self.retryable, "an invalid digest is not retryable")?;
        forbid(self.domain_accepted, "the domain stays unaccepted")?;
        forbid(self.hashed, "a digest refusal does not hash the payload")?;
        forbid(self.file_opened, "a digest refusal does not open a receipt")?;
        stopped(self.forward_ran, self.receipt_stored, "a digest refusal")?;
        match self.reason.as_str() {
            "prefix" => {
                exact_u32(self.hex_length, 0, "a prefix refusal carries no hex")?;
                forbid(
                    self.prefix_accepted,
                    "a prefix refusal does not accept the prefix",
                )?;
                forbid(
                    self.length_accepted,
                    "a prefix refusal does not accept the length",
                )?;
                forbid(
                    self.alphabet_accepted,
                    "a prefix refusal does not accept the alphabet",
                )?;
                exact(&self.domain, UNCHECKED_DOMAIN, "the domain was not checked")?;
            }
            "length" => {
                if self.hex_length == 0 || self.hex_length == 64 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a length refusal is not 64 lowercase hex characters",
                    ));
                }
                if self.hex_length > MAX_DIGEST_RECORD_HEX {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "digest hex exceeds the record cap",
                    ));
                }
                require_flag(self.prefix_accepted, "a length refusal accepted the prefix")?;
                forbid(
                    self.length_accepted,
                    "a length refusal does not accept the length",
                )?;
                forbid(
                    self.alphabet_accepted,
                    "a length refusal does not accept the alphabet",
                )?;
                exact(&self.domain, UNCHECKED_DOMAIN, "the domain was not checked")?;
            }
            "alphabet" => {
                exact_u32(self.hex_length, 64, "an alphabet refusal is 64 characters")?;
                require_flag(
                    self.prefix_accepted,
                    "an alphabet refusal accepted the prefix",
                )?;
                require_flag(
                    self.length_accepted,
                    "an alphabet refusal accepted the length",
                )?;
                forbid(
                    self.alphabet_accepted,
                    "an alphabet refusal does not accept the alphabet",
                )?;
                exact(&self.domain, UNCHECKED_DOMAIN, "the domain was not checked")?;
            }
            "domain" => {
                exact_u32(self.hex_length, 64, "a domain refusal is 64 hex characters")?;
                require_flag(self.prefix_accepted, "a domain refusal accepted the prefix")?;
                require_flag(self.length_accepted, "a domain refusal accepted the length")?;
                require_flag(
                    self.alphabet_accepted,
                    "a domain refusal accepted the alphabet",
                )?;
                if self.domain == UNCHECKED_DOMAIN || self.domain.is_empty() {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a domain refusal names the domain",
                    ));
                }
                if self.domain.len() > MAX_DIGEST_DOMAIN_BYTES as usize {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "domain exceeds the record cap",
                    ));
                }
                if !self
                    .domain
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
                {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a domain refusal names a domain label",
                    ));
                }
                if INFER_DOMAINS.contains(&self.domain.as_str()) {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a domain refusal names a domain outside the allowlist",
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
        let mut b = Builder::typed(DIGEST_INVALID_KIND);
        b.put("alphabetAccepted", CborValue::Bool(self.alphabet_accepted));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("domain", cbor_text(&self.domain));
        b.put("domainAccepted", CborValue::Bool(self.domain_accepted));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("fileOpened", CborValue::Bool(self.file_opened));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("hashed", CborValue::Bool(self.hashed));
        b.put("hexLength", cbor_u32(self.hex_length));
        b.put("lengthAccepted", CborValue::Bool(self.length_accepted));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("prefixAccepted", CborValue::Bool(self.prefix_accepted));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, DIGEST_INVALID_KIND)?;
        let out = Self {
            alphabet_accepted: fields.bool("alphabetAccepted")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            domain: fields.text("domain")?,
            domain_accepted: fields.bool("domainAccepted")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            file_opened: fields.bool("fileOpened")?,
            forward_ran: fields.bool("forwardRan")?,
            hashed: fields.bool("hashed")?,
            hex_length: fields.u32("hexLength")?,
            length_accepted: fields.bool("lengthAccepted")?,
            placement_root: fields.digest("placementRoot")?,
            prefix_accepted: fields.bool("prefixAccepted")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
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
        digest_value("infer-digest-invalid", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
