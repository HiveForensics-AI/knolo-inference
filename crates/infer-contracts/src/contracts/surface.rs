//! Receipt signature and the refusals that stop before a forward.
//!
//! Each report records one cold micro fixture. None of them signs a
//! receipt, decodes a rejected document, compiles a grammar, or executes
//! a tool. The layouts are specified in
//! `spec/KIP-INFER-0106-receipt-signing.md`,
//! `spec/KIP-INFER-0107-canonical-cbor.md`,
//! `spec/KIP-INFER-0108-contract-invalid.md`,
//! `spec/KIP-INFER-0109-grammar-refusal.md`, and
//! `spec/KIP-INFER-0110-tool-refusal.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{cbor_digest, cbor_text, cbor_u32, expect_kind_version, one_of, Fields};

use super::common::{put_extensions, Builder};
use super::exposure::{ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES};
use super::lifecycle::cold_single;
use super::resilience::ED25519_SCALAR_BYTES;

pub const RECEIPT_SIGN_KIND: &str = "knolo.infer.receipt-sign-report";
pub const CANONICAL_CBOR_KIND: &str = "knolo.infer.canonical-cbor-report";
pub const CONTRACT_INVALID_KIND: &str = "knolo.infer.contract-invalid-report";
pub const GRAMMAR_REFUSAL_KIND: &str = "knolo.infer.grammar-refusal-report";
pub const TOOL_REFUSAL_KIND: &str = "knolo.infer.tool-refusal-report";

pub const MAX_CBOR_MAJOR: u32 = 7;
pub const MAX_CBOR_ADDITIONAL: u32 = 31;
pub const MAX_MAP_ADDITIONAL: u32 = 23;
pub const SHORTEST_ADDITIONAL_MIN: u32 = 24;
pub const SHORTEST_ADDITIONAL_MAX: u32 = 27;
pub const MAX_FIELD_RECORD_BYTES: u32 = 80;
pub const MAX_GRAMMAR_SOURCE_BYTES: u32 = 4096;
pub const MAX_TOOL_NAME_BYTES: u32 = 64;

const SIGN_STATUSES: &[&str] = &["signed", "rejected", "unsigned-local"];
const CANONICAL_REASONS: &[&str] = &["indefinite", "tag", "order", "shortest"];
const CONTRACT_REASONS: &[&str] = &["missing", "type", "unknown", "value"];
const GRAMMAR_REASONS: &[&str] = &["schema", "automaton", "mask"];
const TOOL_REASONS: &[&str] = &["name", "object", "execute"];

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

fn tag_header(major: u32, additional: u32) -> bool {
    if major == 6 {
        additional <= MAX_CBOR_ADDITIONAL
    } else {
        major == 7 && additional <= MAX_CBOR_ADDITIONAL && !matches!(additional, 20 | 21 | 22 | 31)
    }
}

/// Host-supplied Ed25519 receipt signature. The receipt is not signed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptSignReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub sign_status: String,
    pub receipt_signed: bool,
    pub receipt_verified: bool,
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

impl ReceiptSignReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "receipt-sign",
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
        forbid(self.receipt_verified, "the receipt stays unverified")?;
        one_of("signStatus", &self.sign_status, SIGN_STATUSES)?;
        match self.sign_status.as_str() {
            "unsigned-local" => {
                unsigned_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                forbid(self.receipt_signed, "an unsigned release signs the receipt")?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a signed receipt is verified",
                )?;
            }
            "signed" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.receipt_signed,
                    "a signed receipt records the signature",
                )?;
                exact(
                    &self.validation_result,
                    "verified",
                    "a signed receipt is verified",
                )?;
            }
            "rejected" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.receipt_signed,
                    "a rejected receipt records the signature",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a signed receipt is verified",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field signStatus has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(RECEIPT_SIGN_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
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
        b.put("receiptVerified", CborValue::Bool(self.receipt_verified));
        b.put("releaseRoot", cbor_digest(&self.release_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("scalarBytes", cbor_u32(self.scalar_bytes));
        b.put("signStatus", cbor_text(&self.sign_status));
        b.put("signatureBytes", cbor_u32(self.signature_bytes));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, RECEIPT_SIGN_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            key_material_present: fields.bool("keyMaterialPresent")?,
            message_root: fields.digest("messageRoot")?,
            placement_root: fields.digest("placementRoot")?,
            public_key_bytes: fields.u32("publicKeyBytes")?,
            receipt_signed: fields.bool("receiptSigned")?,
            receipt_verified: fields.bool("receiptVerified")?,
            release_root: fields.digest("releaseRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            scalar_bytes: fields.u32("scalarBytes")?,
            sign_status: fields.text("signStatus")?,
            signature_bytes: fields.u32("signatureBytes")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-receipt-sign", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One canonical-CBOR header the decoder refused. The document is not decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalCborReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub major: u32,
    pub additional_info: u32,
    pub code: String,
    pub retryable: bool,
    pub argument_read: bool,
    pub value_accepted: bool,
    pub keys_ordered: bool,
    pub reencoded: bool,
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

impl CanonicalCborReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "canonical-cbor",
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
        one_of("reason", &self.reason, CANONICAL_REASONS)?;
        exact(
            &self.code,
            "CANONICAL_CBOR_INVALID",
            "an invalid canonical document is CANONICAL_CBOR_INVALID",
        )?;
        forbid(
            self.retryable,
            "an invalid canonical document is not retryable",
        )?;
        forbid(
            self.value_accepted,
            "a canonical refusal does not accept the value",
        )?;
        forbid(
            self.keys_ordered,
            "a canonical refusal does not order the keys",
        )?;
        forbid(
            self.reencoded,
            "a canonical refusal does not re-encode the document",
        )?;
        stopped(self.forward_ran, self.receipt_stored, "a canonical refusal")?;
        at_most(
            self.major,
            MAX_CBOR_MAJOR,
            "CBOR major exceeds the record cap",
        )?;
        at_most(
            self.additional_info,
            MAX_CBOR_ADDITIONAL,
            "additional information exceeds the record cap",
        )?;
        match self.reason.as_str() {
            "indefinite" => {
                exact_u32(
                    self.additional_info,
                    31,
                    "an indefinite refusal is additional information 31",
                )?;
                if !matches!(self.major, 0..=5 | 7) {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "an indefinite refusal is not a tag",
                    ));
                }
                forbid(
                    self.argument_read,
                    "an indefinite refusal does not read the argument",
                )?;
            }
            "tag" => {
                if !tag_header(self.major, self.additional_info) {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a tag refusal is a tag or a rejected simple value",
                    ));
                }
                forbid(
                    self.argument_read,
                    "a tag refusal does not read the argument",
                )?;
            }
            "order" => {
                exact_u32(self.major, 5, "an order refusal is a map")?;
                at_most(
                    self.additional_info,
                    MAX_MAP_ADDITIONAL,
                    "map length exceeds the record cap",
                )?;
                require_flag(self.argument_read, "an order refusal read the map length")?;
            }
            "shortest" => {
                if self.major > 5 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a shortest refusal is an integer or a length",
                    ));
                }
                if !(SHORTEST_ADDITIONAL_MIN..=SHORTEST_ADDITIONAL_MAX)
                    .contains(&self.additional_info)
                {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a shortest refusal is a wide argument",
                    ));
                }
                require_flag(self.argument_read, "a shortest refusal read the argument")?;
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
        let mut b = Builder::typed(CANONICAL_CBOR_KIND);
        b.put("additionalInfo", cbor_u32(self.additional_info));
        b.put("argumentRead", CborValue::Bool(self.argument_read));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("keysOrdered", CborValue::Bool(self.keys_ordered));
        b.put("major", cbor_u32(self.major));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("reencoded", CborValue::Bool(self.reencoded));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("valueAccepted", CborValue::Bool(self.value_accepted));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, CANONICAL_CBOR_KIND)?;
        let out = Self {
            additional_info: fields.u32("additionalInfo")?,
            argument_read: fields.bool("argumentRead")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            keys_ordered: fields.bool("keysOrdered")?,
            major: fields.u32("major")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            reencoded: fields.bool("reencoded")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            value_accepted: fields.bool("valueAccepted")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-canonical-cbor", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One contract field the decoder refused. The document is not accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractInvalidReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub field_bytes: u32,
    pub code: String,
    pub retryable: bool,
    pub field_present: bool,
    pub type_accepted: bool,
    pub value_accepted: bool,
    pub decoded: bool,
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

impl ContractInvalidReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "contract-invalid",
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
        one_of("reason", &self.reason, CONTRACT_REASONS)?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "an invalid contract is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "an invalid contract is not retryable")?;
        forbid(
            self.decoded,
            "a contract refusal does not decode the document",
        )?;
        forbid(
            self.value_accepted,
            "a contract refusal does not accept the value",
        )?;
        stopped(self.forward_ran, self.receipt_stored, "a contract refusal")?;
        match self.reason.as_str() {
            "missing" => {
                exact_u32(self.field_bytes, 0, "a missing field carries no bytes")?;
                forbid(self.field_present, "a missing field is not present")?;
                forbid(
                    self.type_accepted,
                    "a missing field does not accept the type",
                )?;
            }
            "type" => {
                if self.field_bytes == 0 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a type refusal names the field",
                    ));
                }
                at_most(
                    self.field_bytes,
                    MAX_FIELD_RECORD_BYTES,
                    "field exceeds the record cap",
                )?;
                require_flag(self.field_present, "a type refusal is present")?;
                forbid(
                    self.type_accepted,
                    "a type refusal does not accept the type",
                )?;
            }
            "unknown" => {
                if self.field_bytes == 0 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "an unknown field names the field",
                    ));
                }
                at_most(
                    self.field_bytes,
                    MAX_FIELD_RECORD_BYTES,
                    "field exceeds the record cap",
                )?;
                require_flag(self.field_present, "an unknown field is present")?;
                forbid(
                    self.type_accepted,
                    "an unknown field does not accept the type",
                )?;
            }
            "value" => {
                if self.field_bytes == 0 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a value refusal names the field",
                    ));
                }
                at_most(
                    self.field_bytes,
                    MAX_FIELD_RECORD_BYTES,
                    "field exceeds the record cap",
                )?;
                require_flag(self.field_present, "a value refusal is present")?;
                require_flag(self.type_accepted, "a value refusal accepted the type")?;
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
        let mut b = Builder::typed(CONTRACT_INVALID_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("decoded", CborValue::Bool(self.decoded));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("fieldBytes", cbor_u32(self.field_bytes));
        b.put("fieldPresent", CborValue::Bool(self.field_present));
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("typeAccepted", CborValue::Bool(self.type_accepted));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("valueAccepted", CborValue::Bool(self.value_accepted));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, CONTRACT_INVALID_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            decoded: fields.bool("decoded")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            field_bytes: fields.u32("fieldBytes")?,
            field_present: fields.bool("fieldPresent")?,
            forward_ran: fields.bool("forwardRan")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            type_accepted: fields.bool("typeAccepted")?,
            validation_result: fields.text("validationResult")?,
            value_accepted: fields.bool("valueAccepted")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-contract-invalid", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One grammar the compiler did not build. The grammar is not compiled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarRefusalReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub source_bytes: u32,
    pub code: String,
    pub retryable: bool,
    pub source_opened: bool,
    pub grammar_rooted: bool,
    pub mask_applied: bool,
    pub grammar_compiled: bool,
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

impl GrammarRefusalReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "grammar-refusal",
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
        one_of("reason", &self.reason, GRAMMAR_REASONS)?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "a grammar refusal is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "a grammar refusal is not retryable")?;
        forbid(self.mask_applied, "a grammar refusal does not apply a mask")?;
        forbid(
            self.grammar_compiled,
            "a grammar refusal does not compile the grammar",
        )?;
        stopped(self.forward_ran, self.receipt_stored, "a grammar refusal")?;
        match self.reason.as_str() {
            "schema" => {
                exact_u32(self.source_bytes, 0, "a schema refusal carries no source")?;
                forbid(
                    self.source_opened,
                    "a schema refusal does not open the source",
                )?;
                forbid(
                    self.grammar_rooted,
                    "a schema refusal does not root the grammar",
                )?;
            }
            "automaton" => {
                if self.source_bytes == 0 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "an automaton refusal names the source",
                    ));
                }
                at_most(
                    self.source_bytes,
                    MAX_GRAMMAR_SOURCE_BYTES,
                    "grammar source exceeds the record cap",
                )?;
                require_flag(self.source_opened, "an automaton refusal opened the source")?;
                forbid(
                    self.grammar_rooted,
                    "an automaton refusal does not root the grammar",
                )?;
            }
            "mask" => {
                if self.source_bytes == 0 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a mask refusal names the source",
                    ));
                }
                at_most(
                    self.source_bytes,
                    MAX_GRAMMAR_SOURCE_BYTES,
                    "grammar source exceeds the record cap",
                )?;
                require_flag(self.source_opened, "a mask refusal opened the source")?;
                require_flag(self.grammar_rooted, "a mask refusal rooted the grammar")?;
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
        let mut b = Builder::typed(GRAMMAR_REFUSAL_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("grammarCompiled", CborValue::Bool(self.grammar_compiled));
        b.put("grammarRooted", CborValue::Bool(self.grammar_rooted));
        b.put("maskApplied", CborValue::Bool(self.mask_applied));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("sourceBytes", cbor_u32(self.source_bytes));
        b.put("sourceOpened", CborValue::Bool(self.source_opened));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, GRAMMAR_REFUSAL_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            grammar_compiled: fields.bool("grammarCompiled")?,
            grammar_rooted: fields.bool("grammarRooted")?,
            mask_applied: fields.bool("maskApplied")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            source_bytes: fields.u32("sourceBytes")?,
            source_opened: fields.bool("sourceOpened")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-grammar-refusal", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One tool call the engine did not execute. The tool is not called.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRefusalReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub object_root: DigestHex,
    pub reason: String,
    pub name_bytes: u32,
    pub code: String,
    pub retryable: bool,
    pub name_accepted: bool,
    pub object_generated: bool,
    pub tool_executed: bool,
    pub authority_checked: bool,
    pub budget_checked: bool,
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

impl ToolRefusalReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        cold_fields(
            "tool-refusal",
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
            &self.object_root,
            &self.engine_build_root,
            "the tool object repeats the engine build",
        )?;
        distinct(
            &self.object_root,
            &self.placement_root,
            "the tool object repeats the placement",
        )?;
        one_of("reason", &self.reason, TOOL_REASONS)?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "a tool refusal is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "a tool refusal is not retryable")?;
        forbid(
            self.tool_executed,
            "a tool refusal does not execute the tool",
        )?;
        forbid(
            self.authority_checked,
            "a tool refusal does not check authority",
        )?;
        forbid(
            self.budget_checked,
            "a tool refusal does not check a budget",
        )?;
        stopped(self.forward_ran, self.receipt_stored, "a tool refusal")?;
        match self.reason.as_str() {
            "name" => {
                exact_u32(self.name_bytes, 0, "a name refusal carries no name")?;
                forbid(
                    self.name_accepted,
                    "a name refusal does not accept the name",
                )?;
                forbid(
                    self.object_generated,
                    "a name refusal does not generate an object",
                )?;
            }
            "object" => {
                if self.name_bytes == 0 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "an object refusal names the tool",
                    ));
                }
                at_most(
                    self.name_bytes,
                    MAX_TOOL_NAME_BYTES,
                    "tool name exceeds the record cap",
                )?;
                require_flag(self.name_accepted, "an object refusal accepted the name")?;
                forbid(
                    self.object_generated,
                    "an object refusal does not generate an object",
                )?;
            }
            "execute" => {
                if self.name_bytes == 0 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "an execute refusal names the tool",
                    ));
                }
                at_most(
                    self.name_bytes,
                    MAX_TOOL_NAME_BYTES,
                    "tool name exceeds the record cap",
                )?;
                require_flag(self.name_accepted, "an execute refusal accepted the name")?;
                require_flag(
                    self.object_generated,
                    "an execute refusal generated the object",
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
        let mut b = Builder::typed(TOOL_REFUSAL_KIND);
        b.put("authorityChecked", CborValue::Bool(self.authority_checked));
        b.put("budgetChecked", CborValue::Bool(self.budget_checked));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("nameAccepted", CborValue::Bool(self.name_accepted));
        b.put("nameBytes", cbor_u32(self.name_bytes));
        b.put("objectGenerated", CborValue::Bool(self.object_generated));
        b.put("objectRoot", cbor_digest(&self.object_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("toolExecuted", CborValue::Bool(self.tool_executed));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, TOOL_REFUSAL_KIND)?;
        let out = Self {
            authority_checked: fields.bool("authorityChecked")?,
            budget_checked: fields.bool("budgetChecked")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            name_accepted: fields.bool("nameAccepted")?,
            name_bytes: fields.u32("nameBytes")?,
            object_generated: fields.bool("objectGenerated")?,
            object_root: fields.digest("objectRoot")?,
            placement_root: fields.digest("placementRoot")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            tool_executed: fields.bool("toolExecuted")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-tool-refusal", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
