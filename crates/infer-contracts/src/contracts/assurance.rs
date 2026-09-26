//! Challenge hash, replay mismatch, worker-lost, and draining reports.
//!
//! Each report records one cold micro fixture. None of them hashes the
//! Ed25519 challenge, reduces a scalar, reruns a receipt, kills a worker, or
//! drains the listener. The layouts are specified in
//! `spec/KIP-INFER-0081-challenge-hash.md`,
//! `spec/KIP-INFER-0082-replay-environment.md`,
//! `spec/KIP-INFER-0083-replay-output.md`,
//! `spec/KIP-INFER-0084-worker-lost.md`, and
//! `spec/KIP-INFER-0085-service-draining.md`.

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

pub const CHALLENGE_KIND: &str = "knolo.infer.challenge-report";
pub const REPLAY_ENVIRONMENT_KIND: &str = "knolo.infer.replay-environment-report";
pub const REPLAY_OUTPUT_KIND: &str = "knolo.infer.replay-output-report";
pub const WORKER_LOST_KIND: &str = "knolo.infer.worker-lost-report";
pub const DRAINING_KIND: &str = "knolo.infer.draining-report";

/// HTTP status of `WORKER_LOST` and `SERVICE_DRAINING`.
pub const HTTP_SERVICE_UNAVAILABLE: u32 = 503;

const CHALLENGE_STATUSES: &[&str] = &["hashed", "rejected", "unsigned-local"];
const MISMATCH_FIELDS: &[&str] = &["prompt", "sampler", "model", "engine", "placement"];
const DRAIN_LIFECYCLES: &[&str] = &["draining", "drained"];

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

/// Host-supplied Ed25519 challenge hash. The challenge is not hashed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChallengeReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub release_root: DigestHex,
    pub message_root: DigestHex,
    pub challenge_status: String,
    pub challenge_hashed: bool,
    pub scalar_reduced: bool,
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

impl ChallengeReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "challenge",
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
        forbid(self.scalar_reduced, "the scalar reduction stays uncomputed")?;
        one_of(
            "challengeStatus",
            &self.challenge_status,
            CHALLENGE_STATUSES,
        )?;
        match self.challenge_status.as_str() {
            "unsigned-local" => {
                unsigned_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                forbid(
                    self.challenge_hashed,
                    "an unsigned release hashes the challenge",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a hashed challenge is verified",
                )?;
            }
            "hashed" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(self.challenge_hashed, "a hashed challenge records the hash")?;
                exact(
                    &self.validation_result,
                    "verified",
                    "a hashed challenge is verified",
                )?;
            }
            "rejected" => {
                compared_counts(
                    self.public_key_bytes,
                    self.signature_bytes,
                    self.scalar_bytes,
                )?;
                require_flag(
                    self.challenge_hashed,
                    "a rejected challenge records the hash",
                )?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "only a hashed challenge is verified",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field challengeStatus has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(CHALLENGE_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("challengeHashed", CborValue::Bool(self.challenge_hashed));
        b.put("challengeStatus", cbor_text(&self.challenge_status));
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
        b.put("scalarReduced", CborValue::Bool(self.scalar_reduced));
        b.put("signatureBytes", cbor_u32(self.signature_bytes));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, CHALLENGE_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            challenge_hashed: fields.bool("challengeHashed")?,
            challenge_status: fields.text("challengeStatus")?,
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
            scalar_reduced: fields.bool("scalarReduced")?,
            signature_bytes: fields.u32("signatureBytes")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-challenge", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One replay whose environment did not match the original receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayEnvironmentReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub receipt_root: DigestHex,
    pub mismatched_field: String,
    pub code: String,
    pub retryable: bool,
    pub check_stored: bool,
    pub forward_ran: bool,
    pub output_compared: bool,
    pub assurance: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ReplayEnvironmentReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "replay-environment",
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
        one_of("mismatchedField", &self.mismatched_field, MISMATCH_FIELDS)?;
        exact(
            &self.code,
            "REPLAY_ENVIRONMENT_MISMATCH",
            "an environment mismatch is REPLAY_ENVIRONMENT_MISMATCH",
        )?;
        forbid(self.retryable, "an environment mismatch is not retryable")?;
        forbid(
            self.check_stored,
            "an environment mismatch stores no replay check",
        )?;
        forbid(
            self.forward_ran,
            "an environment mismatch does not run the forward",
        )?;
        forbid(
            self.output_compared,
            "an environment mismatch does not compare output tokens",
        )?;
        exact(
            &self.assurance,
            "incomplete",
            "an environment mismatch is incomplete",
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(REPLAY_ENVIRONMENT_KIND);
        b.put("assurance", cbor_text(&self.assurance));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("checkStored", CborValue::Bool(self.check_stored));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("mismatchedField", cbor_text(&self.mismatched_field));
        b.put("outputCompared", CborValue::Bool(self.output_compared));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("receiptRoot", cbor_digest(&self.receipt_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, REPLAY_ENVIRONMENT_KIND)?;
        let out = Self {
            assurance: fields.text("assurance")?,
            cache_policy: fields.text("cachePolicy")?,
            check_stored: fields.bool("checkStored")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            mismatched_field: fields.text("mismatchedField")?,
            output_compared: fields.bool("outputCompared")?,
            placement_root: fields.digest("placementRoot")?,
            receipt_root: fields.digest("receiptRoot")?,
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
        digest_value("infer-replay-environment", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One replay whose output tokens did not match the original receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayOutputReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub receipt_root: DigestHex,
    pub candidate_output_root: DigestHex,
    pub code: String,
    pub retryable: bool,
    pub check_stored: bool,
    pub forward_ran: bool,
    pub environment_matched: bool,
    pub assurance: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ReplayOutputReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "replay-output",
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
            &self.candidate_output_root,
            &self.receipt_root,
            "the candidate output repeats the receipt",
        )?;
        exact(
            &self.code,
            "REPLAY_OUTPUT_MISMATCH",
            "an output mismatch is REPLAY_OUTPUT_MISMATCH",
        )?;
        forbid(self.retryable, "an output mismatch is not retryable")?;
        forbid(
            self.check_stored,
            "an output mismatch stores no replay check",
        )?;
        require_flag(self.forward_ran, "an output mismatch ran the forward")?;
        require_flag(
            self.environment_matched,
            "an output mismatch follows a matching environment",
        )?;
        exact(
            &self.assurance,
            "incomplete",
            "an output mismatch is incomplete",
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(REPLAY_OUTPUT_KIND);
        b.put("assurance", cbor_text(&self.assurance));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put(
            "candidateOutputRoot",
            cbor_digest(&self.candidate_output_root),
        );
        b.put("checkStored", CborValue::Bool(self.check_stored));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put(
            "environmentMatched",
            CborValue::Bool(self.environment_matched),
        );
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("forwardRan", CborValue::Bool(self.forward_ran));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("receiptRoot", cbor_digest(&self.receipt_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, REPLAY_OUTPUT_KIND)?;
        let out = Self {
            assurance: fields.text("assurance")?,
            cache_policy: fields.text("cachePolicy")?,
            candidate_output_root: fields.digest("candidateOutputRoot")?,
            check_stored: fields.bool("checkStored")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            environment_matched: fields.bool("environmentMatched")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            forward_ran: fields.bool("forwardRan")?,
            placement_root: fields.digest("placementRoot")?,
            receipt_root: fields.digest("receiptRoot")?,
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
        digest_value("infer-replay-output", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One ready worker that exited during a request. No process is killed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerLostReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub request_id: String,
    pub code: String,
    pub retryable: bool,
    pub receipt_stored: bool,
    pub listener_up: bool,
    pub supervisor_exited: bool,
    pub http_status: u32,
    pub journal_sealed: bool,
    pub restart_counted: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl WorkerLostReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "worker-lost",
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
        exact(&self.code, "WORKER_LOST", "a lost worker is WORKER_LOST")?;
        require_flag(self.retryable, "a lost worker is retryable")?;
        forbid(self.receipt_stored, "a lost worker stores no receipt")?;
        require_flag(self.listener_up, "the listener stays up")?;
        forbid(
            self.supervisor_exited,
            "a lost worker does not exit the supervisor",
        )?;
        count_exact(
            self.http_status,
            HTTP_SERVICE_UNAVAILABLE,
            "a lost worker is HTTP 503",
        )?;
        require_flag(self.journal_sealed, "a lost worker seals the open journal")?;
        require_flag(self.restart_counted, "a lost worker counts a restart")?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(WORKER_LOST_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("httpStatus", cbor_u32(self.http_status));
        b.put("journalSealed", CborValue::Bool(self.journal_sealed));
        b.put("listenerUp", CborValue::Bool(self.listener_up));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("requestId", cbor_text(&self.request_id));
        b.put("restartCounted", CborValue::Bool(self.restart_counted));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("supervisorExited", CborValue::Bool(self.supervisor_exited));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, WORKER_LOST_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            http_status: fields.u32("httpStatus")?,
            journal_sealed: fields.bool("journalSealed")?,
            listener_up: fields.bool("listenerUp")?,
            placement_root: fields.digest("placementRoot")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            request_id: fields.text("requestId")?,
            restart_counted: fields.bool("restartCounted")?,
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
        digest_value("infer-worker-lost", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One new completion refused because the service is draining.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrainingReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub lifecycle: String,
    pub code: String,
    pub retryable: bool,
    pub receipt_stored: bool,
    pub body_parsed: bool,
    pub listener_up: bool,
    pub worker_loaded: bool,
    pub restart_counted: bool,
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

impl DrainingReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "draining",
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
        one_of("lifecycle", &self.lifecycle, DRAIN_LIFECYCLES)?;
        exact(
            &self.code,
            "SERVICE_DRAINING",
            "a draining refusal is SERVICE_DRAINING",
        )?;
        require_flag(self.retryable, "a draining refusal is retryable")?;
        forbid(self.receipt_stored, "a draining refusal stores no receipt")?;
        forbid(
            self.body_parsed,
            "a draining refusal does not parse the body",
        )?;
        require_flag(self.listener_up, "the listener stays up")?;
        require_flag(
            self.worker_loaded,
            "a draining refusal leaves the worker loaded",
        )?;
        forbid(
            self.restart_counted,
            "a draining refusal does not count a restart",
        )?;
        count_exact(
            self.http_status,
            HTTP_SERVICE_UNAVAILABLE,
            "a draining refusal is HTTP 503",
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(DRAINING_KIND);
        b.put("bodyParsed", CborValue::Bool(self.body_parsed));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("code", cbor_text(&self.code));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("httpStatus", cbor_u32(self.http_status));
        b.put("lifecycle", cbor_text(&self.lifecycle));
        b.put("listenerUp", CborValue::Bool(self.listener_up));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("receiptStored", CborValue::Bool(self.receipt_stored));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("restartCounted", CborValue::Bool(self.restart_counted));
        b.put("retryable", CborValue::Bool(self.retryable));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("workerLoaded", CborValue::Bool(self.worker_loaded));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, DRAINING_KIND)?;
        let out = Self {
            body_parsed: fields.bool("bodyParsed")?,
            cache_policy: fields.text("cachePolicy")?,
            code: fields.text("code")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            http_status: fields.u32("httpStatus")?,
            lifecycle: fields.text("lifecycle")?,
            listener_up: fields.bool("listenerUp")?,
            placement_root: fields.digest("placementRoot")?,
            receipt_stored: fields.bool("receiptStored")?,
            request_count: fields.u32("requestCount")?,
            restart_counted: fields.bool("restartCounted")?,
            retryable: fields.bool("retryable")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            worker_loaded: fields.bool("workerLoaded")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-draining", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
