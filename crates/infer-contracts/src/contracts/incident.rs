//! Point comparison, CUDA out-of-memory, CUDA fault, request timeout, and
//! worker-start reports.
//!
//! Each report records one cold micro fixture. None of them compares the
//! Ed25519 points, allocates device memory, captures a CUDA graph, or spawns
//! a worker. The layouts are specified in `spec/KIP-INFER-0076-point-comparison.md`,
//! `spec/KIP-INFER-0077-cuda-oom.md`, `spec/KIP-INFER-0078-cuda-fault.md`,
//! `spec/KIP-INFER-0079-request-timeout.md`, and
//! `spec/KIP-INFER-0080-worker-start.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    bounded_text, cbor_digest, cbor_text, cbor_u32, cbor_u64, expect_kind_version, one_of, Fields,
};

use super::census::MAX_PEAK_BYTES;
use super::common::{put_extensions, Builder};
use super::exposure::{ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES};
use super::lifecycle::cold_single;
use super::resilience::ED25519_SCALAR_BYTES;

pub const EQUALITY_KIND: &str = "knolo.infer.equality-report";
pub const OOM_KIND: &str = "knolo.infer.oom-report";
pub const FAULT_KIND: &str = "knolo.infer.fault-report";
pub const TIMEOUT_KIND: &str = "knolo.infer.timeout-report";
pub const WORKER_START_KIND: &str = "knolo.infer.worker-start-report";

/// HTTP status of a request that reached `REQUEST_TIMEOUT`.
pub const HTTP_TIMEOUT_STATUS: u32 = 504;

const EQUALITY_STATUSES: &[&str] = &["equal", "unequal", "unsigned-local"];
const FAULT_CLASSES: &[&str] = &["kernel", "device"];
const TIMEOUT_STAGES: &[&str] = &["admission", "completion", "cancellation"];
const START_FAILURES: &[&str] = &["missing-binary", "not-ready", "socket"];

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

/// Host-supplied comparison of the two Ed25519 points. The points are not read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EqualityReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub equality_status: String,
    pub points_compared: bool,
    pub points_equal: bool,
    pub challenge_hashed: bool,
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

impl EqualityReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "equality",
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
        forbid(self.challenge_hashed, "the challenge hash stays uncomputed")?;
        one_of("equalityStatus", &self.equality_status, EQUALITY_STATUSES)?;
        match self.equality_status.as_str() {
            "unsigned-local" => {
                unsigned_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                forbid(
                    self.points_compared,
                    "an unsigned release compares the points",
                )?;
                forbid(
                    self.points_equal,
                    "an unsigned release says the points are equal",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only an equal point comparison is verified",
                )?;
            }
            "equal" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.points_compared,
                    "a compared release compares the points",
                )?;
                require_flag(
                    self.points_equal,
                    "an equal comparison says the points are equal",
                )?;
                exact(
                    &self.validation_result,
                    "verified",
                    "an equal point comparison is verified",
                )?;
            }
            "unequal" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.points_compared,
                    "a compared release compares the points",
                )?;
                forbid(
                    self.points_equal,
                    "an unequal comparison says the points differ",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only an equal point comparison is verified",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field equalityStatus has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(EQUALITY_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("challengeHashed", CborValue::Bool(self.challenge_hashed));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("equalityStatus", cbor_text(&self.equality_status));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put(
            "keyMaterialPresent",
            CborValue::Bool(self.key_material_present),
        );
        b.put("messageRoot", cbor_digest(&self.message_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("pointsCompared", CborValue::Bool(self.points_compared));
        b.put("pointsEqual", CborValue::Bool(self.points_equal));
        b.put("publicKeyBytes", cbor_u32(self.public_key_bytes));
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
        expect_kind_version(&mut fields, EQUALITY_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            challenge_hashed: fields.bool("challengeHashed")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            equality_status: fields.text("equalityStatus")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            key_material_present: fields.bool("keyMaterialPresent")?,
            message_root: fields.digest("messageRoot")?,
            placement_root: fields.digest("placementRoot")?,
            points_compared: fields.bool("pointsCompared")?,
            points_equal: fields.bool("pointsEqual")?,
            public_key_bytes: fields.u32("publicKeyBytes")?,
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
        digest_value("infer-equality", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One CUDA out-of-memory result. Device memory is not allocated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OomReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub device: String,
    pub free_bytes: u64,
    pub needed_bytes: u64,
    pub code: String,
    pub retryable: bool,
    pub receipt_stored: bool,
    pub listener_up: bool,
    pub supervisor_exited: bool,
    pub cpu_fallback: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl OomReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "oom",
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
        exact(&self.device, "slot-0", "a cuda oom record names slot-0")?;
        if self.free_bytes != 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a cuda oom record has no free device bytes",
            ));
        }
        if self.needed_bytes == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a cuda oom record needs device bytes",
            ));
        }
        if self.needed_bytes > MAX_PEAK_BYTES {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "needed bytes exceed 64 MiB",
            ));
        }
        exact(&self.code, "CUDA_OOM", "a cuda oom record is CUDA_OOM")?;
        require_flag(self.retryable, "a cuda oom record is retryable")?;
        forbid(self.receipt_stored, "a cuda oom record stores no receipt")?;
        require_flag(self.listener_up, "the listener stays up")?;
        forbid(
            self.supervisor_exited,
            "a cuda oom does not exit the supervisor",
        )?;
        forbid(self.cpu_fallback, "a cuda oom does not fall back to cpu")?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(OOM_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("cpuFallback", CborValue::Bool(self.cpu_fallback));
        b.put("device", cbor_text(&self.device));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("freeBytes", cbor_u64(self.free_bytes));
        b.put("listenerUp", CborValue::Bool(self.listener_up));
        b.put("neededBytes", cbor_u64(self.needed_bytes));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("supervisorExited", CborValue::Bool(self.supervisor_exited));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, OOM_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            cpu_fallback: fields.bool("cpuFallback")?,
            device: fields.text("device")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            free_bytes: fields.u64("freeBytes")?,
            listener_up: fields.bool("listenerUp")?,
            needed_bytes: fields.u64("neededBytes")?,
            placement_root: fields.digest("placementRoot")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            supervisor_exited: fields.bool("supervisorExited")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-oom", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One CUDA kernel or device fault. The device is not queried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub device: String,
    pub fault_class: String,
    pub code: String,
    pub retryable: bool,
    pub receipt_stored: bool,
    pub listener_up: bool,
    pub supervisor_exited: bool,
    pub cpu_fallback: bool,
    pub graph_captured: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl FaultReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "fault",
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
        exact(&self.device, "slot-0", "a cuda fault record names slot-0")?;
        one_of("faultClass", &self.fault_class, FAULT_CLASSES)?;
        exact(
            &self.code,
            "CUDA_FAULT",
            "a cuda fault record is CUDA_FAULT",
        )?;
        forbid(self.retryable, "a cuda fault record is not retryable")?;
        forbid(self.receipt_stored, "a cuda fault record stores no receipt")?;
        require_flag(self.listener_up, "the listener stays up")?;
        forbid(
            self.supervisor_exited,
            "a cuda fault does not exit the supervisor",
        )?;
        forbid(self.cpu_fallback, "a cuda fault does not fall back to cpu")?;
        forbid(self.graph_captured, "cuda graphs stay off")?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(FAULT_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("cpuFallback", CborValue::Bool(self.cpu_fallback));
        b.put("device", cbor_text(&self.device));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("faultClass", cbor_text(&self.fault_class));
        b.put("graphCaptured", CborValue::Bool(self.graph_captured));
        b.put("listenerUp", CborValue::Bool(self.listener_up));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("supervisorExited", CborValue::Bool(self.supervisor_exited));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, FAULT_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            cpu_fallback: fields.bool("cpuFallback")?,
            device: fields.text("device")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            fault_class: fields.text("faultClass")?,
            graph_captured: fields.bool("graphCaptured")?,
            listener_up: fields.bool("listenerUp")?,
            placement_root: fields.digest("placementRoot")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            supervisor_exited: fields.bool("supervisorExited")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-fault", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One request that reached `REQUEST_TIMEOUT`. No timer is armed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeoutReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub request_id: String,
    pub timeout_stage: String,
    pub waited_nanos: u64,
    pub code: String,
    pub retryable: bool,
    pub receipt_stored: bool,
    pub listener_up: bool,
    pub worker_lost: bool,
    pub http_status: u32,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl TimeoutReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "timeout",
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
        one_of("timeoutStage", &self.timeout_stage, TIMEOUT_STAGES)?;
        if self.waited_nanos == 0 {
            return Err(fail(ErrorCode::ContractInvalid, "a timeout record waits"));
        }
        exact(
            &self.code,
            "REQUEST_TIMEOUT",
            "a timeout record is REQUEST_TIMEOUT",
        )?;
        require_flag(self.retryable, "a timeout record is retryable")?;
        forbid(self.receipt_stored, "a timeout record stores no receipt")?;
        require_flag(self.listener_up, "the listener stays up")?;
        forbid(self.worker_lost, "a timeout is not a lost worker")?;
        count_exact(
            self.http_status,
            HTTP_TIMEOUT_STATUS,
            "a timeout record is HTTP 504",
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(TIMEOUT_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("httpStatus", cbor_u32(self.http_status));
        b.put("listenerUp", CborValue::Bool(self.listener_up));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("requestId", cbor_text(&self.request_id));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("timeoutStage", cbor_text(&self.timeout_stage));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("waitedNanos", cbor_u64(self.waited_nanos));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("workerLost", CborValue::Bool(self.worker_lost));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, TIMEOUT_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            http_status: fields.u32("httpStatus")?,
            listener_up: fields.bool("listenerUp")?,
            placement_root: fields.digest("placementRoot")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            request_id: fields.text("requestId")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            timeout_stage: fields.text("timeoutStage")?,
            validation_result: fields.text("validationResult")?,
            waited_nanos: fields.u64("waitedNanos")?,
            warm_state: fields.text("warmState")?,
            worker_lost: fields.bool("workerLost")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-timeout", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One worker that did not become ready. No process is spawned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerStartReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub start_failure: String,
    pub code: String,
    pub retryable: bool,
    pub receipt_stored: bool,
    pub listener_up: bool,
    pub process_spawned: bool,
    pub worker_ready: bool,
    pub restart_count: u32,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl WorkerStartReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "worker-start",
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
        one_of("startFailure", &self.start_failure, START_FAILURES)?;
        exact(
            &self.code,
            "WORKER_START_FAILED",
            "a start failure is WORKER_START_FAILED",
        )?;
        require_flag(self.retryable, "a start failure is retryable")?;
        forbid(self.receipt_stored, "a start failure stores no receipt")?;
        forbid(self.worker_ready, "a start failure leaves the worker down")?;
        count_exact(
            self.restart_count,
            0,
            "a start failure does not count a restart",
        )?;
        match self.start_failure.as_str() {
            "missing-binary" => {
                forbid(self.listener_up, "a missing worker binary does not bind")?;
                forbid(
                    self.process_spawned,
                    "a missing worker binary does not spawn",
                )?;
            }
            "not-ready" => {
                require_flag(self.listener_up, "the listener stays up")?;
                require_flag(
                    self.process_spawned,
                    "a worker that is not ready was spawned",
                )?;
            }
            "socket" => {
                require_flag(self.listener_up, "the listener stays up")?;
                forbid(
                    self.process_spawned,
                    "a socket failure does not spawn the worker",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field startFailure has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(WORKER_START_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("listenerUp", CborValue::Bool(self.listener_up));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("processSpawned", CborValue::Bool(self.process_spawned));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("restartCount", cbor_u32(self.restart_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("startFailure", cbor_text(&self.start_failure));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("workerReady", CborValue::Bool(self.worker_ready));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, WORKER_START_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            listener_up: fields.bool("listenerUp")?,
            placement_root: fields.digest("placementRoot")?,
            process_spawned: fields.bool("processSpawned")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            restart_count: fields.u32("restartCount")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            start_failure: fields.text("startFailure")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            worker_ready: fields.bool("workerReady")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-worker-start", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
