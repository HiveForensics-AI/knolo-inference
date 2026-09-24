//! Base-point, disk-full, queued-unload, and daemon-restart reports.
//!
//! Each report records one cold micro fixture. None of them adds the
//! public-key point, fills a disk, unloads a worker, or spawns a process.
//! The layouts are specified in
//! `spec/KIP-INFER-0068-base-point.md`,
//! `spec/KIP-INFER-0069-disk-full.md`,
//! `spec/KIP-INFER-0070-queued-unload.md`, and
//! `spec/KIP-INFER-0071-daemon-restart.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    bounded_text, cbor_digest, cbor_text, cbor_u32, cbor_u64, expect_kind_version, one_of, Fields,
};

use super::common::{put_extensions, Builder};
use super::exposure::{ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES};
use super::lifecycle::cold_single;

pub const BASE_KIND: &str = "knolo.infer.base-report";
pub const DISK_KIND: &str = "knolo.infer.disk-report";
pub const UNLOAD_KIND: &str = "knolo.infer.unload-report";
pub const RESTART_KIND: &str = "knolo.infer.restart-report";

/// An Ed25519 scalar is 32 bytes.
pub const ED25519_SCALAR_BYTES: u32 = 32;

/// A disk-full record names at most 1 MiB of missing space.
pub const MAX_DISK_NEEDED_BYTES: u64 = 1024 * 1024;

/// The micro fixture records at most 16 queued or admitted requests.
pub const MAX_UNLOAD_QUEUE: u32 = 16;

const DISK_STORES: &[&str] = &["journal", "receipt", "trace", "lock"];
const RESTART_REASONS: &[&str] = &["stale-lock", "clean-exit", "killed"];

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

/// Host-supplied base-point multiplication. The public-key point is not added.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub base_status: String,
    pub base_multiplied: bool,
    pub public_key_bytes: u32,
    pub signature_bytes: u32,
    pub scalar_bytes: u32,
    pub point_added: bool,
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

impl BaseReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "base",
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
        forbid(self.point_added, "the public-key point stays unadded")?;
        match self.base_status.as_str() {
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
                    self.base_multiplied,
                    "an unsigned release multiplies the base point",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a multiplied base point is verified",
                )?;
            }
            "multiplied" | "rejected" => {
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
                    self.base_multiplied,
                    "a multiplied release multiplies the base point",
                )?;
                if self.base_status == "multiplied" {
                    exact(
                        &self.validation_result,
                        "verified",
                        "a multiplied base point is verified",
                    )?;
                } else {
                    exact(
                        &self.validation_result,
                        "recorded",
                        "only a multiplied base point is verified",
                    )?;
                }
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field baseStatus has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(BASE_KIND);
        b.put("baseMultiplied", CborValue::Bool(self.base_multiplied));
        b.put("baseStatus", cbor_text(&self.base_status));
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
        expect_kind_version(&mut fields, BASE_KIND)?;
        let out = Self {
            base_multiplied: fields.bool("baseMultiplied")?,
            base_status: fields.text("baseStatus")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            key_material_present: fields.bool("keyMaterialPresent")?,
            message_root: fields.digest("messageRoot")?,
            placement_root: fields.digest("placementRoot")?,
            point_added: fields.bool("pointAdded")?,
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
        digest_value("infer-base", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One full disk. The target file is not written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub request_id: String,
    pub store: String,
    pub free_bytes: u64,
    pub needed_bytes: u64,
    pub file_written: bool,
    pub space_reclaimed: bool,
    pub listener_up: bool,
    pub code: String,
    pub retryable: bool,
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

impl DiskReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "disk",
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
        one_of("store", &self.store, DISK_STORES)?;
        if self.free_bytes != 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a disk-full record has no free bytes",
            ));
        }
        if self.needed_bytes == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a disk-full record needs space",
            ));
        }
        if self.needed_bytes > MAX_DISK_NEEDED_BYTES {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "needed bytes exceed 1 MiB",
            ));
        }
        forbid(self.file_written, "a disk-full record writes no file")?;
        forbid(self.space_reclaimed, "a disk-full record reclaims no space")?;
        let durable = matches!(self.store.as_str(), "journal" | "receipt" | "lock");
        if durable {
            exact(
                &self.code,
                "RECEIPT_PERSIST_FAILED",
                "a durable store failure is RECEIPT_PERSIST_FAILED",
            )?;
            require_flag(self.retryable, "a durable store failure is retryable")?;
            forbid(
                self.receipt_stored,
                "a full durable store stores no receipt",
            )?;
        } else {
            exact(
                &self.code,
                "none",
                "a trace write does not fail the completion",
            )?;
            forbid(self.retryable, "a trace write is not a retryable failure")?;
            require_flag(
                self.receipt_stored,
                "a full trace leaves the receipt stored",
            )?;
        }
        if self.store == "lock" {
            forbid(self.listener_up, "a lock store that is full does not bind")?;
        } else {
            require_flag(self.listener_up, "the listener stays up")?;
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(DISK_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("fileWritten", CborValue::Bool(self.file_written));
        b.put("freeBytes", cbor_u64(self.free_bytes));
        b.put("listenerUp", CborValue::Bool(self.listener_up));
        b.put("neededBytes", cbor_u64(self.needed_bytes));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("requestId", cbor_text(&self.request_id));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("spaceReclaimed", CborValue::Bool(self.space_reclaimed));
        b.put("store", cbor_text(&self.store));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, DISK_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            file_written: fields.bool("fileWritten")?,
            free_bytes: fields.u64("freeBytes")?,
            listener_up: fields.bool("listenerUp")?,
            needed_bytes: fields.u64("neededBytes")?,
            placement_root: fields.digest("placementRoot")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            request_id: fields.text("requestId")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            space_reclaimed: fields.bool("spaceReclaimed")?,
            store: fields.text("store")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-disk", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One unload while requests are still queued. The worker is not unloaded here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnloadReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub request_id: String,
    pub queued_requests: u32,
    pub admitted_requests: u32,
    pub queued_started: bool,
    pub admitted_finished: bool,
    pub lifecycle: String,
    pub worker_exited: bool,
    pub code: String,
    pub retryable: bool,
    pub listener_up: bool,
    pub model_reloaded: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl UnloadReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "unload",
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
        if self.queued_requests == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "an unload under queue has a queued request",
            ));
        }
        if self.queued_requests > MAX_UNLOAD_QUEUE {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the unload queue is at most 16",
            ));
        }
        if self.admitted_requests > MAX_UNLOAD_QUEUE {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "admitted work is at most 16",
            ));
        }
        forbid(self.queued_started, "queued work does not start")?;
        require_flag(self.admitted_finished, "admitted work finishes")?;
        exact(
            &self.code,
            "SERVICE_UNLOADED",
            "a queued unload is SERVICE_UNLOADED",
        )?;
        forbid(self.retryable, "a queued unload is not retryable")?;
        require_flag(self.listener_up, "the listener stays up")?;
        forbid(
            self.model_reloaded,
            "a queued unload does not load the model",
        )?;
        if self.admitted_requests == 0 {
            exact(
                &self.lifecycle,
                "unloaded",
                "an empty admission is unloaded",
            )?;
            require_flag(self.worker_exited, "an empty admission exits the worker")?;
        } else {
            exact(
                &self.lifecycle,
                "unloading",
                "admitted work keeps the lifecycle unloading",
            )?;
            forbid(
                self.worker_exited,
                "the worker stays until admitted work finishes",
            )?;
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(UNLOAD_KIND);
        b.put("admittedFinished", CborValue::Bool(self.admitted_finished));
        b.put("admittedRequests", cbor_u32(self.admitted_requests));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("lifecycle", cbor_text(&self.lifecycle));
        b.put("listenerUp", CborValue::Bool(self.listener_up));
        b.put("modelReloaded", CborValue::Bool(self.model_reloaded));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("queuedRequests", cbor_u32(self.queued_requests));
        b.put("queuedStarted", CborValue::Bool(self.queued_started));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("requestId", cbor_text(&self.request_id));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("workerExited", CborValue::Bool(self.worker_exited));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, UNLOAD_KIND)?;
        let out = Self {
            admitted_finished: fields.bool("admittedFinished")?,
            admitted_requests: fields.u32("admittedRequests")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            lifecycle: fields.text("lifecycle")?,
            listener_up: fields.bool("listenerUp")?,
            model_reloaded: fields.bool("modelReloaded")?,
            placement_root: fields.digest("placementRoot")?,
            queued_requests: fields.u32("queuedRequests")?,
            queued_started: fields.bool("queuedStarted")?,
            request_count: fields.u32("requestCount")?,
            request_id: fields.text("requestId")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            worker_exited: fields.bool("workerExited")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-unload", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One daemon restart. This record does not spawn a process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub previous_owner_root: DigestHex,
    pub incoming_owner_root: DigestHex,
    pub reason: String,
    pub lock_replaced: bool,
    pub journals_sealed: bool,
    pub listener_up: bool,
    pub worker_restart_count: u32,
    pub process_spawned: bool,
    pub second_worker: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl RestartReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "restart",
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
            &self.previous_owner_root,
            &self.engine_build_root,
            "the previous owner repeats the engine build",
        )?;
        distinct(
            &self.incoming_owner_root,
            &self.engine_build_root,
            "the incoming owner repeats the engine build",
        )?;
        distinct(
            &self.incoming_owner_root,
            &self.previous_owner_root,
            "the incoming owner repeats the previous owner",
        )?;
        distinct(
            &self.previous_owner_root,
            &self.placement_root,
            "the previous owner repeats the placement",
        )?;
        distinct(
            &self.incoming_owner_root,
            &self.placement_root,
            "the incoming owner repeats the placement",
        )?;
        forbid(
            self.process_spawned,
            "a restart record does not spawn a process",
        )?;
        forbid(self.second_worker, "a restart record starts one worker")?;
        require_flag(self.listener_up, "the listener comes back")?;
        if self.worker_restart_count != 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a new process starts the restart counter at zero",
            ));
        }
        one_of("reason", &self.reason, RESTART_REASONS)?;
        match self.reason.as_str() {
            "stale-lock" | "killed" => {
                require_flag(self.lock_replaced, "a stale owner is replaced")?;
                require_flag(self.journals_sealed, "a stale owner seals open journals")?;
            }
            "clean-exit" => {
                forbid(
                    self.lock_replaced,
                    "a clean exit does not replace a held lock",
                )?;
                forbid(self.journals_sealed, "a clean exit has no open journal")?;
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
        let mut b = Builder::typed(RESTART_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("incomingOwnerRoot", cbor_digest(&self.incoming_owner_root));
        b.put("journalsSealed", CborValue::Bool(self.journals_sealed));
        b.put("listenerUp", CborValue::Bool(self.listener_up));
        b.put("lockReplaced", CborValue::Bool(self.lock_replaced));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("previousOwnerRoot", cbor_digest(&self.previous_owner_root));
        b.put("processSpawned", CborValue::Bool(self.process_spawned));
        b.put("reason", cbor_text(&self.reason));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("secondWorker", CborValue::Bool(self.second_worker));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("workerRestartCount", cbor_u32(self.worker_restart_count));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, RESTART_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            incoming_owner_root: fields.digest("incomingOwnerRoot")?,
            journals_sealed: fields.bool("journalsSealed")?,
            listener_up: fields.bool("listenerUp")?,
            lock_replaced: fields.bool("lockReplaced")?,
            placement_root: fields.digest("placementRoot")?,
            previous_owner_root: fields.digest("previousOwnerRoot")?,
            process_spawned: fields.bool("processSpawned")?,
            reason: fields.text("reason")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            second_worker: fields.bool("secondWorker")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            worker_restart_count: fields.u32("workerRestartCount")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-restart", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
