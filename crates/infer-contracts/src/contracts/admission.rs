//! Point-addition, duplicate-request, concurrent-load, and prefix-eviction reports.
//!
//! Each report records one cold micro fixture. None of them compares the
//! Ed25519 points, admits a second copy of a request id, replaces a live
//! worker, or allocates a prefix index. The layouts are specified in
//! `spec/KIP-INFER-0072-point-addition.md`,
//! `spec/KIP-INFER-0073-duplicate-request.md`,
//! `spec/KIP-INFER-0074-concurrent-load.md`, and
//! `spec/KIP-INFER-0075-prefix-eviction.md`.

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
use super::resilience::ED25519_SCALAR_BYTES;

pub const POINT_KIND: &str = "knolo.infer.point-report";
pub const DUPLICATE_KIND: &str = "knolo.infer.duplicate-report";
pub const CONCURRENT_LOAD_KIND: &str = "knolo.infer.concurrent-load-report";
pub const EVICTION_KIND: &str = "knolo.infer.eviction-report";

/// A micro admission records at most 16 requests.
pub const MAX_ADMISSION_REQUESTS: u32 = 16;

const LOAD_PHASES: &[&str] = &["ready", "unloading", "down"];

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

fn count_exact(value: u32, expected: u32, message: &str) -> Result<(), InferFailure> {
    if value == expected {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn at_most(value: u32, max: u32, message: &str) -> Result<(), InferFailure> {
    if value > max {
        Err(fail(ErrorCode::ContractInvalid, message))
    } else {
        Ok(())
    }
}

/// Host-supplied public-key point addition. The points are not compared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub point_status: String,
    pub point_added: bool,
    pub public_key_bytes: u32,
    pub signature_bytes: u32,
    pub scalar_bytes: u32,
    pub points_equal: bool,
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

impl PointReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "point",
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
        forbid(self.points_equal, "the points stay uncompared")?;
        match self.point_status.as_str() {
            "unsigned-local" => {
                count_exact(
                    self.public_key_bytes,
                    0,
                    "an unsigned release carries a public key",
                )?;
                count_exact(
                    self.signature_bytes,
                    0,
                    "an unsigned release carries signature bytes",
                )?;
                count_exact(self.scalar_bytes, 0, "an unsigned release carries a scalar")?;
                forbid(
                    self.point_added,
                    "an unsigned release adds the public-key point",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only an added public-key point is verified",
                )?;
            }
            "added" | "rejected" => {
                count_exact(
                    self.public_key_bytes,
                    ED25519_PUBLIC_KEY_BYTES,
                    "ed25519 public keys are 32 bytes",
                )?;
                count_exact(
                    self.signature_bytes,
                    ED25519_SIGNATURE_BYTES,
                    "ed25519 signatures are 64 bytes",
                )?;
                count_exact(
                    self.scalar_bytes,
                    ED25519_SCALAR_BYTES,
                    "ed25519 scalars are 32 bytes",
                )?;
                require_flag(
                    self.point_added,
                    "an added release adds the public-key point",
                )?;
                if self.point_status == "added" {
                    exact(
                        &self.validation_result,
                        "verified",
                        "an added public-key point is verified",
                    )?;
                } else {
                    exact(
                        &self.validation_result,
                        "recorded",
                        "only an added public-key point is verified",
                    )?;
                }
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field pointStatus has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(POINT_KIND);
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
        b.put("pointAdded", CborValue::Bool(self.point_added));
        b.put("pointStatus", cbor_text(&self.point_status));
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
        expect_kind_version(&mut fields, POINT_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            key_material_present: fields.bool("keyMaterialPresent")?,
            message_root: fields.digest("messageRoot")?,
            placement_root: fields.digest("placementRoot")?,
            point_added: fields.bool("pointAdded")?,
            point_status: fields.text("pointStatus")?,
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
        digest_value("infer-point", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One duplicate request id. The second copy does not start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplicateReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub request_id: String,
    pub occupant_root: DigestHex,
    pub code: String,
    pub retryable: bool,
    pub duplicate_started: bool,
    pub occupant_kept: bool,
    pub second_journal: bool,
    pub listener_up: bool,
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

impl DuplicateReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "duplicate",
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
        distinct(
            &self.occupant_root,
            &self.engine_build_root,
            "the occupant repeats the engine build",
        )?;
        distinct(
            &self.occupant_root,
            &self.placement_root,
            "the occupant repeats the placement",
        )?;
        exact(
            &self.code,
            "CONTRACT_INVALID",
            "a duplicate request id is CONTRACT_INVALID",
        )?;
        forbid(self.retryable, "a duplicate request id is not retryable")?;
        forbid(self.duplicate_started, "a duplicate request does not start")?;
        require_flag(self.occupant_kept, "the admitted request stays")?;
        forbid(self.second_journal, "a duplicate request opens no journal")?;
        require_flag(self.listener_up, "the listener stays up")?;
        forbid(self.receipt_stored, "a duplicate request stores no receipt")?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(DUPLICATE_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("duplicateStarted", CborValue::Bool(self.duplicate_started));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("listenerUp", CborValue::Bool(self.listener_up));
        b.put("occupantKept", CborValue::Bool(self.occupant_kept));
        b.put("occupantRoot", cbor_digest(&self.occupant_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("requestId", cbor_text(&self.request_id));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("secondJournal", CborValue::Bool(self.second_journal));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, DUPLICATE_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            duplicate_started: fields.bool("duplicateStarted")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            listener_up: fields.bool("listenerUp")?,
            occupant_kept: fields.bool("occupantKept")?,
            occupant_root: fields.digest("occupantRoot")?,
            placement_root: fields.digest("placementRoot")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            request_id: fields.text("requestId")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            second_journal: fields.bool("secondJournal")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-duplicate", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One concurrent load. A live worker is not replaced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcurrentLoadReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub phase: String,
    pub lifecycle: String,
    pub inflight: u32,
    pub worker_started: bool,
    pub second_worker: bool,
    pub worker_replaced: bool,
    pub restart_count: u32,
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

impl ConcurrentLoadReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "concurrent-load",
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
        one_of("phase", &self.phase, LOAD_PHASES)?;
        forbid(self.second_worker, "a concurrent load starts one worker")?;
        forbid(
            self.worker_replaced,
            "a concurrent load does not replace a live worker",
        )?;
        if self.restart_count != 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a concurrent load does not count a restart",
            ));
        }
        require_flag(self.listener_up, "the listener stays up")?;
        at_most(
            self.inflight,
            MAX_ADMISSION_REQUESTS,
            "a concurrent load admits at most 16 requests",
        )?;
        match self.phase.as_str() {
            "ready" => {
                exact(&self.lifecycle, "serving", "a ready load stays serving")?;
                forbid(self.worker_started, "a ready worker is left running")?;
            }
            "unloading" => {
                exact(
                    &self.lifecycle,
                    "unloading",
                    "an unloading load stays unloading",
                )?;
                forbid(
                    self.worker_started,
                    "an unloading load does not start a worker",
                )?;
                if self.inflight == 0 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "an unloading load still has admitted work",
                    ));
                }
            }
            "down" => {
                exact(&self.lifecycle, "serving", "a down load returns to serving")?;
                require_flag(self.worker_started, "a down worker is started")?;
                if self.inflight != 0 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a down load has no admitted work",
                    ));
                }
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field phase has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(CONCURRENT_LOAD_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("inflight", cbor_u32(self.inflight));
        b.put("lifecycle", cbor_text(&self.lifecycle));
        b.put("listenerUp", CborValue::Bool(self.listener_up));
        b.put("phase", cbor_text(&self.phase));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("restartCount", cbor_u32(self.restart_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("secondWorker", CborValue::Bool(self.second_worker));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("workerReplaced", CborValue::Bool(self.worker_replaced));
        b.put("workerStarted", CborValue::Bool(self.worker_started));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, CONCURRENT_LOAD_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            inflight: fields.u32("inflight")?,
            lifecycle: fields.text("lifecycle")?,
            listener_up: fields.bool("listenerUp")?,
            phase: fields.text("phase")?,
            placement_root: fields.digest("placementRoot")?,
            request_count: fields.u32("requestCount")?,
            restart_count: fields.u32("restartCount")?,
            run_count: fields.u32("runCount")?,
            second_worker: fields.bool("secondWorker")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            worker_replaced: fields.bool("workerReplaced")?,
            worker_started: fields.bool("workerStarted")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-concurrent-load", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Prefix eviction under load while the index stays unallocated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvictionReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub load_requests: u32,
    pub evicted_pages: u32,
    pub evicted_tokens: u32,
    pub active_evicted: bool,
    pub prefix_allocated: bool,
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

impl EvictionReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "eviction",
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
        if self.load_requests == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "eviction under load names a request",
            ));
        }
        at_most(
            self.load_requests,
            MAX_ADMISSION_REQUESTS,
            "eviction under load admits at most 16 requests",
        )?;
        count_exact(
            self.evicted_pages,
            0,
            "prefix eviction is zero while the cache is off",
        )?;
        count_exact(
            self.evicted_tokens,
            0,
            "evicted tokens are zero while the cache is off",
        )?;
        forbid(self.active_evicted, "an active sequence is not evicted")?;
        forbid(self.prefix_allocated, "prefix cache is not allocated")?;
        require_flag(self.listener_up, "the listener stays up")?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(EVICTION_KIND);
        b.put("activeEvicted", CborValue::Bool(self.active_evicted));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("evictedPages", cbor_u32(self.evicted_pages));
        b.put("evictedTokens", cbor_u32(self.evicted_tokens));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("listenerUp", CborValue::Bool(self.listener_up));
        b.put("loadRequests", cbor_u32(self.load_requests));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("prefixAllocated", CborValue::Bool(self.prefix_allocated));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, EVICTION_KIND)?;
        let out = Self {
            active_evicted: fields.bool("activeEvicted")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            evicted_pages: fields.u32("evictedPages")?,
            evicted_tokens: fields.u32("evictedTokens")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            listener_up: fields.bool("listenerUp")?,
            load_requests: fields.u32("loadRequests")?,
            placement_root: fields.digest("placementRoot")?,
            prefix_allocated: fields.bool("prefixAllocated")?,
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
        digest_value("infer-eviction", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
