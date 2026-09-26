//! Scalar reduction and the image checks that stop a forward.
//!
//! Each report records one cold micro fixture. None of them reduces an
//! Ed25519 scalar, multiplies a public key, opens a weight file, parses a
//! tokenizer, or renders a template. The layouts are specified in
//! `spec/KIP-INFER-0086-scalar-reduction.md`,
//! `spec/KIP-INFER-0087-digest-mismatch.md`,
//! `spec/KIP-INFER-0088-tokenizer-invalid.md`,
//! `spec/KIP-INFER-0089-template-invalid.md`, and
//! `spec/KIP-INFER-0090-unsupported-architecture.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{cbor_digest, cbor_text, cbor_u32, expect_kind_version, one_of, Fields};

use super::common::{put_extensions, Builder};
use super::exposure::{ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES};
use super::lifecycle::cold_single;
use super::product::MICRO_ADAPTER;
use super::resilience::ED25519_SCALAR_BYTES;

pub const SCALAR_KIND: &str = "knolo.infer.scalar-report";
pub const DIGEST_MISMATCH_KIND: &str = "knolo.infer.digest-mismatch-report";
pub const TOKENIZER_INVALID_KIND: &str = "knolo.infer.tokenizer-invalid-report";
pub const TEMPLATE_INVALID_KIND: &str = "knolo.infer.template-invalid-report";
pub const ARCHITECTURE_KIND: &str = "knolo.infer.architecture-report";

const SCALAR_STATUSES: &[&str] = &["reduced", "rejected", "unsigned-local"];
const MISMATCHES: &[&str] = &["size", "digest"];
const TOKENIZER_FAILURES: &[&str] = &["root", "file", "special", "encode"];
const TEMPLATE_FAILURES: &[&str] = &["root", "encoding", "grammar", "cap"];
const DEFERRED_ADAPTER: &str = "knolo.llama.v1";

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

fn stopped(
    retryable: bool,
    forward_ran: bool,
    receipt_stored: bool,
    subject: &str,
) -> Result<(), InferFailure> {
    forbid(retryable, &format!("{subject} is not retryable"))?;
    forbid(forward_ran, &format!("{subject} does not run the forward"))?;
    forbid(receipt_stored, &format!("{subject} stores no receipt"))
}

/// Host-supplied Ed25519 scalar reduction. The scalar is not reduced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub scalar_status: String,
    pub scalar_reduced: bool,
    pub public_multiplied: bool,
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

impl ScalarReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "scalar",
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
        forbid(
            self.public_multiplied,
            "the public-key multiplication stays uncomputed",
        )?;
        one_of("scalarStatus", &self.scalar_status, SCALAR_STATUSES)?;
        match self.scalar_status.as_str() {
            "unsigned-local" => {
                unsigned_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                forbid(
                    self.scalar_reduced,
                    "an unsigned release reduces the scalar",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a reduced scalar is verified",
                )?;
            }
            "reduced" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.scalar_reduced,
                    "a reduced scalar records the reduction",
                )?;
                exact(
                    &self.validation_result,
                    "verified",
                    "a reduced scalar is verified",
                )?;
            }
            "rejected" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.scalar_reduced,
                    "a rejected scalar records the reduction",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a reduced scalar is verified",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field scalarStatus has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(SCALAR_KIND);
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
        b.put("publicMultiplied", CborValue::Bool(self.public_multiplied));
        b.put("releaseRoot", cbor_digest(&self.release_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("scalarBytes", cbor_u32(self.scalar_bytes));
        b.put("scalarReduced", CborValue::Bool(self.scalar_reduced));
        b.put("scalarStatus", cbor_text(&self.scalar_status));
        b.put("signatureBytes", cbor_u32(self.signature_bytes));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, SCALAR_KIND)?;
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
            public_multiplied: fields.bool("publicMultiplied")?,
            release_root: fields.digest("releaseRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            scalar_bytes: fields.u32("scalarBytes")?,
            scalar_reduced: fields.bool("scalarReduced")?,
            scalar_status: fields.text("scalarStatus")?,
            signature_bytes: fields.u32("signatureBytes")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-scalar", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One weight file whose size or digest did not match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DigestMismatchReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub artifact_root: DigestHex,
    pub mismatch: String,
    pub code: String,
    pub retryable: bool,
    pub header_parsed: bool,
    pub body_read: bool,
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

impl DigestMismatchReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "digest-mismatch",
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
            &self.artifact_root,
            &self.engine_build_root,
            "the artifact repeats the engine build",
        )?;
        distinct(
            &self.artifact_root,
            &self.placement_root,
            "the artifact repeats the placement",
        )?;
        one_of("mismatch", &self.mismatch, MISMATCHES)?;
        exact(
            &self.code,
            "MODEL_DIGEST_MISMATCH",
            "a digest mismatch is MODEL_DIGEST_MISMATCH",
        )?;
        stopped(
            self.retryable,
            self.forward_ran,
            self.receipt_stored,
            "a digest mismatch",
        )?;
        forbid(
            self.header_parsed,
            "a digest mismatch does not parse the header",
        )?;
        match self.mismatch.as_str() {
            "size" => forbid(self.body_read, "a size mismatch does not read the body")?,
            "digest" => require_flag(self.body_read, "a digest mismatch reads the body")?,
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field mismatch has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(DIGEST_MISMATCH_KIND);
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("bodyRead", CborValue::Bool(self.body_read));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("headerParsed", CborValue::Bool(self.header_parsed));
        b.put("mismatch", cbor_text(&self.mismatch));
        b.put("placementRoot", cbor_digest(&self.placement_root));
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
        expect_kind_version(&mut fields, DIGEST_MISMATCH_KIND)?;
        let out = Self {
            artifact_root: fields.digest("artifactRoot")?,
            body_read: fields.bool("bodyRead")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            header_parsed: fields.bool("headerParsed")?,
            mismatch: fields.text("mismatch")?,
            placement_root: fields.digest("placementRoot")?,
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
        digest_value("infer-digest-mismatch", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One tokenizer the prompt compiler refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerInvalidReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub tokenizer_root: DigestHex,
    pub failure: String,
    pub code: String,
    pub retryable: bool,
    pub tokenizer_parsed: bool,
    pub template_rendered: bool,
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

impl TokenizerInvalidReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "tokenizer",
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
            &self.tokenizer_root,
            &self.engine_build_root,
            "the tokenizer repeats the engine build",
        )?;
        distinct(
            &self.tokenizer_root,
            &self.placement_root,
            "the tokenizer repeats the placement",
        )?;
        one_of("failure", &self.failure, TOKENIZER_FAILURES)?;
        exact(
            &self.code,
            "TOKENIZER_INVALID",
            "a tokenizer failure is TOKENIZER_INVALID",
        )?;
        stopped(
            self.retryable,
            self.forward_ran,
            self.receipt_stored,
            "a tokenizer failure",
        )?;
        forbid(
            self.prompt_compiled,
            "a tokenizer failure does not compile the prompt",
        )?;
        match self.failure.as_str() {
            "root" => {
                forbid(
                    self.tokenizer_parsed,
                    "a root mismatch does not parse the tokenizer",
                )?;
                forbid(
                    self.template_rendered,
                    "a root mismatch does not render the template",
                )?;
            }
            "file" => {
                forbid(self.tokenizer_parsed, "a tokenizer file is not accepted")?;
                require_flag(
                    self.template_rendered,
                    "a tokenizer file is checked after the template renders",
                )?;
            }
            "special" => {
                require_flag(
                    self.tokenizer_parsed,
                    "a special token is checked after the tokenizer parses",
                )?;
                require_flag(
                    self.template_rendered,
                    "a special token is checked after the template renders",
                )?;
            }
            "encode" => {
                require_flag(
                    self.tokenizer_parsed,
                    "an encode failure parsed the tokenizer",
                )?;
                require_flag(
                    self.template_rendered,
                    "an encode failure rendered the template",
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
        let mut b = Builder::typed(TOKENIZER_INVALID_KIND);
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
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("templateRendered", CborValue::Bool(self.template_rendered));
        b.put("tokenizerParsed", CborValue::Bool(self.tokenizer_parsed));
        b.put("tokenizerRoot", cbor_digest(&self.tokenizer_root));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, TOKENIZER_INVALID_KIND)?;
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
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            template_rendered: fields.bool("templateRendered")?,
            tokenizer_parsed: fields.bool("tokenizerParsed")?,
            tokenizer_root: fields.digest("tokenizerRoot")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-tokenizer-invalid", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One template the prompt compiler refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateInvalidReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub template_root: DigestHex,
    pub failure: String,
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

impl TemplateInvalidReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "template",
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
            &self.template_root,
            &self.engine_build_root,
            "the template repeats the engine build",
        )?;
        distinct(
            &self.template_root,
            &self.placement_root,
            "the template repeats the placement",
        )?;
        one_of("failure", &self.failure, TEMPLATE_FAILURES)?;
        exact(
            &self.code,
            "TEMPLATE_INVALID",
            "a template failure is TEMPLATE_INVALID",
        )?;
        stopped(
            self.retryable,
            self.forward_ran,
            self.receipt_stored,
            "a template failure",
        )?;
        forbid(
            self.template_rendered,
            "a template failure does not render the prompt",
        )?;
        forbid(
            self.tokenizer_parsed,
            "a template failure does not parse the tokenizer",
        )?;
        forbid(
            self.prompt_compiled,
            "a template failure does not compile the prompt",
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(TEMPLATE_INVALID_KIND);
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
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("templateRendered", CborValue::Bool(self.template_rendered));
        b.put("templateRoot", cbor_digest(&self.template_root));
        b.put("tokenizerParsed", CborValue::Bool(self.tokenizer_parsed));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, TEMPLATE_INVALID_KIND)?;
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
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            template_rendered: fields.bool("templateRendered")?,
            template_root: fields.digest("templateRoot")?,
            tokenizer_parsed: fields.bool("tokenizerParsed")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-template-invalid", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One architecture adapter that is not compiled in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub rejected_adapter: String,
    pub code: String,
    pub retryable: bool,
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

impl ArchitectureReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "architecture",
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
        if self.rejected_adapter == MICRO_ADAPTER {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the micro adapter is compiled in",
            ));
        }
        exact(
            &self.rejected_adapter,
            DEFERRED_ADAPTER,
            "field rejectedAdapter has an unsupported value",
        )?;
        exact(
            &self.code,
            "UNSUPPORTED_ARCHITECTURE",
            "an unsupported architecture is UNSUPPORTED_ARCHITECTURE",
        )?;
        stopped(
            self.retryable,
            self.forward_ran,
            self.receipt_stored,
            "an unsupported architecture",
        )?;
        forbid(
            self.weights_opened,
            "an unsupported architecture does not open weights",
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(ARCHITECTURE_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("rejectedAdapter", cbor_text(&self.rejected_adapter));
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
        expect_kind_version(&mut fields, ARCHITECTURE_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            placement_root: fields.digest("placementRoot")?,
            receipt_stored: fields.bool("receiptStored")?,
            rejected_adapter: fields.text("rejectedAdapter")?,
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
        digest_value("infer-architecture", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
