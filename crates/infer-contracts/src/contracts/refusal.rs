//! Public-key multiplication and the runtime refusals that stop a forward.
//!
//! Each report records one cold micro fixture. None of them multiplies a
//! public key, checks a signature, opens a weight file, selects a kernel,
//! opens a device, or allocates memory. The layouts are specified in
//! `spec/KIP-INFER-0091-public-multiplication.md`,
//! `spec/KIP-INFER-0092-unsupported-quantization.md`,
//! `spec/KIP-INFER-0093-unsupported-kernel.md`,
//! `spec/KIP-INFER-0094-placement-unsatisfiable.md`, and
//! `spec/KIP-INFER-0095-insufficient-memory.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    cbor_digest, cbor_text, cbor_u32, cbor_u64, expect_kind_version, one_of, Fields,
};

use super::census::MAX_PEAK_BYTES;
use super::common::{put_extensions, Builder};
use super::exposure::{ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES};
use super::lifecycle::cold_single;
use super::resilience::ED25519_SCALAR_BYTES;

pub const PUBLIC_KIND: &str = "knolo.infer.public-report";
pub const QUANTIZATION_KIND: &str = "knolo.infer.quantization-report";
pub const KERNEL_KIND: &str = "knolo.infer.kernel-report";
pub const PLACEMENT_REFUSAL_KIND: &str = "knolo.infer.placement-refusal-report";
pub const MEMORY_REFUSAL_KIND: &str = "knolo.infer.memory-refusal-report";

const PUBLIC_STATUSES: &[&str] = &["multiplied", "rejected", "unsigned-local"];
const QUANTIZATION_REASONS: &[&str] = &["precision", "dtype", "ggml", "version"];
const KERNEL_REASONS: &[&str] = &["backend", "feature"];
const PLACEMENT_REASONS: &[&str] = &["device", "toolkit", "capability", "bytes"];
const MEMORY_REASONS: &[&str] = &["pool", "output", "admission"];

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

/// Host-supplied Ed25519 public-key multiplication. The public key is not multiplied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub public_status: String,
    pub public_multiplied: bool,
    pub signature_checked: bool,
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

impl PublicReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "public",
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
            self.signature_checked,
            "the signature check stays uncomputed",
        )?;
        one_of("publicStatus", &self.public_status, PUBLIC_STATUSES)?;
        match self.public_status.as_str() {
            "unsigned-local" => {
                unsigned_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                forbid(
                    self.public_multiplied,
                    "an unsigned release multiplies the public key",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a multiplied public key is verified",
                )?;
            }
            "multiplied" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.public_multiplied,
                    "a multiplied public key records the multiplication",
                )?;
                exact(
                    &self.validation_result,
                    "verified",
                    "a multiplied public key is verified",
                )?;
            }
            "rejected" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.public_multiplied,
                    "a rejected public key records the multiplication",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a multiplied public key is verified",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field publicStatus has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(PUBLIC_KIND);
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
        b.put("publicStatus", cbor_text(&self.public_status));
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
        expect_kind_version(&mut fields, PUBLIC_KIND)?;
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
            public_status: fields.text("publicStatus")?,
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
        digest_value("infer-public", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One weight tensor whose precision, dtype, ggml type, or version is outside the allowlist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizationReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub artifact_root: DigestHex,
    pub reason: String,
    pub code: String,
    pub retryable: bool,
    pub weights_opened: bool,
    pub payload_read: bool,
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

impl QuantizationReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "quantization",
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
        one_of("reason", &self.reason, QUANTIZATION_REASONS)?;
        exact(
            &self.code,
            "UNSUPPORTED_QUANTIZATION",
            "an unsupported quantization is UNSUPPORTED_QUANTIZATION",
        )?;
        stopped(
            self.retryable,
            self.forward_ran,
            self.receipt_stored,
            "an unsupported quantization",
        )?;
        match self.reason.as_str() {
            "precision" => {
                forbid(
                    self.weights_opened,
                    "a precision refusal does not open weights",
                )?;
                forbid(
                    self.payload_read,
                    "a precision refusal does not read the payload",
                )?;
            }
            "dtype" => {
                require_flag(self.weights_opened, "a dtype refusal opens the weights")?;
                require_flag(self.payload_read, "a dtype refusal reads the payload")?;
            }
            "ggml" => {
                require_flag(self.weights_opened, "a ggml type opens the weights")?;
                forbid(self.payload_read, "a ggml type does not read the payload")?;
            }
            "version" => {
                require_flag(
                    self.weights_opened,
                    "a quantization version opens the weights",
                )?;
                forbid(
                    self.payload_read,
                    "a quantization version does not read the payload",
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
        let mut b = Builder::typed(QUANTIZATION_KIND);
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("payloadRead", CborValue::Bool(self.payload_read));
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
        expect_kind_version(&mut fields, QUANTIZATION_KIND)?;
        let out = Self {
            artifact_root: fields.digest("artifactRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            payload_read: fields.bool("payloadRead")?,
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
        digest_value("infer-quantization", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One backend the micro adapter does not select.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub code: String,
    pub retryable: bool,
    pub cuda_requested: bool,
    pub kernel_selected: bool,
    pub device_opened: bool,
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

impl KernelReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "kernel",
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
        one_of("reason", &self.reason, KERNEL_REASONS)?;
        exact(
            &self.code,
            "UNSUPPORTED_KERNEL",
            "an unsupported kernel is UNSUPPORTED_KERNEL",
        )?;
        stopped(
            self.retryable,
            self.forward_ran,
            self.receipt_stored,
            "an unsupported kernel",
        )?;
        forbid(
            self.kernel_selected,
            "an unsupported kernel is not selected",
        )?;
        forbid(
            self.device_opened,
            "an unsupported kernel does not open a device",
        )?;
        match self.reason.as_str() {
            "backend" => forbid(
                self.cuda_requested,
                "a foreign backend does not request cuda",
            )?,
            "feature" => require_flag(
                self.cuda_requested,
                "a missing feature requests candle-cuda",
            )?,
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
        let mut b = Builder::typed(KERNEL_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("cudaRequested", CborValue::Bool(self.cuda_requested));
        b.put("deviceOpened", CborValue::Bool(self.device_opened));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("kernelSelected", CborValue::Bool(self.kernel_selected));
        b.put("placementRoot", cbor_digest(&self.placement_root));
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
        expect_kind_version(&mut fields, KERNEL_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            cuda_requested: fields.bool("cudaRequested")?,
            device_opened: fields.bool("deviceOpened")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            kernel_selected: fields.bool("kernelSelected")?,
            placement_root: fields.digest("placementRoot")?,
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
        digest_value("infer-kernel", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One placement the micro fixture cannot satisfy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementRefusalReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub code: String,
    pub retryable: bool,
    pub probe_reached: bool,
    pub slot_visible: bool,
    pub device_opened: bool,
    pub cpu_fallback: bool,
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

impl PlacementRefusalReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "placement-refusal",
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
        one_of("reason", &self.reason, PLACEMENT_REASONS)?;
        exact(
            &self.code,
            "PLACEMENT_UNSATISFIABLE",
            "an unsatisfiable placement is PLACEMENT_UNSATISFIABLE",
        )?;
        stopped(
            self.retryable,
            self.forward_ran,
            self.receipt_stored,
            "an unsatisfiable placement",
        )?;
        forbid(
            self.device_opened,
            "an unsatisfiable placement does not open a device",
        )?;
        forbid(
            self.cpu_fallback,
            "an unsatisfiable placement does not fall back",
        )?;
        match self.reason.as_str() {
            "toolkit" => {
                forbid(
                    self.probe_reached,
                    "a toolkit refusal does not probe the device",
                )?;
                forbid(
                    self.slot_visible,
                    "a toolkit refusal does not claim a visible device",
                )?;
            }
            "device" => {
                require_flag(self.probe_reached, "a device refusal probes slot-0")?;
                forbid(self.slot_visible, "a missing device is not visible")?;
            }
            "capability" => {
                require_flag(self.probe_reached, "a capability refusal probes slot-0")?;
                require_flag(self.slot_visible, "a capability refusal sees slot-0")?;
            }
            "bytes" => {
                require_flag(self.probe_reached, "a byte refusal probes slot-0")?;
                require_flag(self.slot_visible, "a byte refusal sees slot-0")?;
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
        let mut b = Builder::typed(PLACEMENT_REFUSAL_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("cpuFallback", CborValue::Bool(self.cpu_fallback));
        b.put("deviceOpened", CborValue::Bool(self.device_opened));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("probeReached", CborValue::Bool(self.probe_reached));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("slotVisible", CborValue::Bool(self.slot_visible));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, PLACEMENT_REFUSAL_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            cpu_fallback: fields.bool("cpuFallback")?,
            device_opened: fields.bool("deviceOpened")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            placement_root: fields.digest("placementRoot")?,
            probe_reached: fields.bool("probeReached")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            slot_visible: fields.bool("slotVisible")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-placement-refusal", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One cold fixture that needed more host memory than the bound allows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRefusalReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub reason: String,
    pub free_bytes: u64,
    pub needed_bytes: u64,
    pub code: String,
    pub retryable: bool,
    pub resident_full: bool,
    pub queue_held: bool,
    pub allocated: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub listener_up: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl MemoryRefusalReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "memory-refusal",
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
        one_of("reason", &self.reason, MEMORY_REASONS)?;
        if self.free_bytes != 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "an insufficient memory record has no free bytes",
            ));
        }
        if self.needed_bytes == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "an insufficient memory record needs bytes",
            ));
        }
        if self.needed_bytes > MAX_PEAK_BYTES {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "needed bytes exceed 64 MiB",
            ));
        }
        exact(
            &self.code,
            "INSUFFICIENT_MEMORY",
            "an insufficient memory record is INSUFFICIENT_MEMORY",
        )?;
        require_flag(self.retryable, "an insufficient memory record is retryable")?;
        forbid(
            self.forward_ran,
            "an insufficient memory record does not run the forward",
        )?;
        forbid(
            self.receipt_stored,
            "an insufficient memory record stores no receipt",
        )?;
        require_flag(self.listener_up, "the listener stays up")?;
        forbid(
            self.allocated,
            "an insufficient memory record does not allocate",
        )?;
        match self.reason.as_str() {
            "pool" => {
                require_flag(self.resident_full, "a full pool has no free page")?;
                forbid(self.queue_held, "a full pool does not hold the queue")?;
            }
            "output" => {
                forbid(self.resident_full, "an output cap is not a full pool")?;
                forbid(self.queue_held, "an output cap does not hold the queue")?;
            }
            "admission" => {
                forbid(self.resident_full, "an admission cap is not a full pool")?;
                require_flag(self.queue_held, "an admission cap holds the queue")?;
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
        let mut b = Builder::typed(MEMORY_REFUSAL_KIND);
        b.put("allocated", CborValue::Bool(self.allocated));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("freeBytes", cbor_u64(self.free_bytes));
        b.put("listenerUp", CborValue::Bool(self.listener_up));
        b.put("neededBytes", cbor_u64(self.needed_bytes));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("queueHeld", CborValue::Bool(self.queue_held));
        b.put("reason", cbor_text(&self.reason));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("residentFull", CborValue::Bool(self.resident_full));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, MEMORY_REFUSAL_KIND)?;
        let out = Self {
            allocated: fields.bool("allocated")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            free_bytes: fields.u64("freeBytes")?,
            listener_up: fields.bool("listenerUp")?,
            needed_bytes: fields.u64("neededBytes")?,
            placement_root: fields.digest("placementRoot")?,
            queue_held: fields.bool("queueHeld")?,
            reason: fields.text("reason")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            resident_full: fields.bool("residentFull")?,
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
        digest_value("infer-memory-refusal", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
