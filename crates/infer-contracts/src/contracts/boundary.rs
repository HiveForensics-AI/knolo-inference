//! Host-key, receipt-key, sandbox, and API boundary reports.
//!
//! Each report records one cold micro fixture. None of them reads key
//! bytes, loads a secret, applies a sandbox, or binds a socket. The layouts
//! are specified in `spec/KIP-INFER-0056-host-key.md`,
//! `spec/KIP-INFER-0057-receipt-key.md`,
//! `spec/KIP-INFER-0058-sandbox-profile.md`, and
//! `spec/KIP-INFER-0059-api-boundary.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    bounded_text, cbor_digest, cbor_text, cbor_u32, cbor_u64, expect_kind_version, one_of, Fields,
};

use super::common::{put_extensions, Builder};
use super::lifecycle::cold_single;

pub const HOST_KEY_KIND: &str = "knolo.infer.host-key-report";
pub const RECEIPT_KEY_KIND: &str = "knolo.infer.receipt-key-report";
pub const SANDBOX_KIND: &str = "knolo.infer.sandbox-report";
pub const API_KIND: &str = "knolo.infer.api-report";

/// Worker memory above this size is not a micro-fixture sandbox report.
pub const MAX_SANDBOX_MEMORY: u64 = 64 * 1024 * 1024;

/// Shared memory above this size is outside the worker profile.
pub const MAX_SHARED_MEMORY: u64 = 4096;

/// Request body above this size is not a micro-fixture API report.
pub const MAX_API_BODY: u64 = 1024 * 1024;

/// Requests per minute above this rate are outside the micro-fixture API.
pub const MAX_API_RATE: u32 = 256;

fn execution_mode(value: &str) -> Result<(), InferFailure> {
    one_of("executionMode", value, &["isolated-replay", "pinned"])
}

#[allow(clippy::too_many_arguments)]
fn accept_boundary(
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

/// Host-supplied signature match. The Ed25519 equation is not evaluated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostKeyReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub release_root: DigestHex,
    pub host_key_root: DigestHex,
    pub signature_status: String,
    pub key_id: String,
    pub signature_bytes: u32,
    pub key_verified: bool,
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

impl HostKeyReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_boundary(
            "host key",
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
            &self.host_key_root,
            &self.engine_build_root,
            "the host key repeats the engine build",
        )?;
        distinct(
            &self.host_key_root,
            &self.release_root,
            "the host key repeats the release",
        )?;
        forbid(
            self.key_material_present,
            "key material stays in host storage",
        )?;
        match self.signature_status.as_str() {
            "unsigned-local" => {
                if !self.key_id.is_empty() {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "an unsigned release names a key",
                    ));
                }
                if self.signature_bytes != 0 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "an unsigned release carries signature bytes",
                    ));
                }
                forbid(self.key_verified, "an unsigned release verifies a key")?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "an unsigned release is recorded",
                )?;
            }
            "matched" => {
                bounded_text("keyId", &self.key_id, 128)?;
                if self.signature_bytes != 64 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "ed25519 signatures are 64 bytes",
                    ));
                }
                require_flag(self.key_verified, "a matched release verifies the host key")?;
                exact(
                    &self.validation_result,
                    "verified",
                    "a matched release is verified",
                )?;
            }
            "rejected" => {
                bounded_text("keyId", &self.key_id, 128)?;
                if self.signature_bytes != 64 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "ed25519 signatures are 64 bytes",
                    ));
                }
                forbid(self.key_verified, "a rejected release verifies a key")?;
                exact(
                    &self.validation_result,
                    "recorded",
                    "a rejected release is recorded",
                )?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field signatureStatus has an unsupported value",
                ));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(HOST_KEY_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("hostKeyRoot", cbor_digest(&self.host_key_root));
        b.put("keyId", cbor_text(&self.key_id));
        b.put(
            "keyMaterialPresent",
            CborValue::Bool(self.key_material_present),
        );
        b.put("keyVerified", CborValue::Bool(self.key_verified));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("releaseRoot", cbor_digest(&self.release_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("signatureBytes", cbor_u32(self.signature_bytes));
        b.put("signatureStatus", cbor_text(&self.signature_status));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, HOST_KEY_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            host_key_root: fields.digest("hostKeyRoot")?,
            key_id: fields.text("keyId")?,
            key_material_present: fields.bool("keyMaterialPresent")?,
            key_verified: fields.bool("keyVerified")?,
            placement_root: fields.digest("placementRoot")?,
            release_root: fields.digest("releaseRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            signature_bytes: fields.u32("signatureBytes")?,
            signature_status: fields.text("signatureStatus")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-host-key", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Receipt signing key stays in host storage. The secret is not loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptKeyReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub receipt_root: DigestHex,
    pub custody: String,
    pub key_material_serialized: bool,
    pub key_id: String,
    pub signature_status: String,
    pub signature_bytes: u32,
    pub rotation: String,
    pub previous_key_id: String,
    pub trusted_metadata: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ReceiptKeyReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_boundary(
            "receipt key",
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
        exact(
            &self.custody,
            "host-store",
            "receipt keys stay in host storage",
        )?;
        forbid(
            self.key_material_serialized,
            "key material stays out of the receipt",
        )?;
        match self.signature_status.as_str() {
            "unsigned-local" => self.validate_unsigned(),
            "shape-checked" => self.validate_shaped(),
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                "field signatureStatus has an unsupported value",
            )),
        }
    }

    fn validate_unsigned(&self) -> Result<(), InferFailure> {
        if !self.key_id.is_empty() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "an unsigned receipt names a key",
            ));
        }
        if self.signature_bytes != 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "an unsigned receipt carries signature bytes",
            ));
        }
        exact(
            &self.rotation,
            "current",
            "an unsigned receipt rotates a key",
        )?;
        if !self.previous_key_id.is_empty() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "an unsigned receipt names a previous key",
            ));
        }
        forbid(self.trusted_metadata, "an unsigned receipt names rotation")
    }

    fn validate_shaped(&self) -> Result<(), InferFailure> {
        bounded_text("keyId", &self.key_id, 128)?;
        if self.signature_bytes != 64 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "ed25519 signatures are 64 bytes",
            ));
        }
        match self.rotation.as_str() {
            "current" => {
                if !self.previous_key_id.is_empty() {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "a current key names a previous key",
                    ));
                }
                forbid(self.trusted_metadata, "a current key names rotation")
            }
            "rotated" => {
                bounded_text("previousKeyId", &self.previous_key_id, 128)?;
                if self.previous_key_id == self.key_id {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "rotation repeats the current key",
                    ));
                }
                require_flag(self.trusted_metadata, "rotation uses trusted metadata")
            }
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                "field rotation has an unsupported value",
            )),
        }
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(RECEIPT_KEY_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("custody", cbor_text(&self.custody));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("keyId", cbor_text(&self.key_id));
        b.put(
            "keyMaterialSerialized",
            CborValue::Bool(self.key_material_serialized),
        );
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("previousKeyId", cbor_text(&self.previous_key_id));
        b.put("receiptRoot", cbor_digest(&self.receipt_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("rotation", cbor_text(&self.rotation));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("signatureBytes", cbor_u32(self.signature_bytes));
        b.put("signatureStatus", cbor_text(&self.signature_status));
        b.put("trustedMetadata", CborValue::Bool(self.trusted_metadata));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, RECEIPT_KEY_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            custody: fields.text("custody")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            key_id: fields.text("keyId")?,
            key_material_serialized: fields.bool("keyMaterialSerialized")?,
            placement_root: fields.digest("placementRoot")?,
            previous_key_id: fields.text("previousKeyId")?,
            receipt_root: fields.digest("receiptRoot")?,
            request_count: fields.u32("requestCount")?,
            rotation: fields.text("rotation")?,
            run_count: fields.u32("runCount")?,
            signature_bytes: fields.u32("signatureBytes")?,
            signature_status: fields.text("signatureStatus")?,
            trusted_metadata: fields.bool("trustedMetadata")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-receipt-key", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Linux worker profile for one cold micro fixture. The profile is not applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub user_class: String,
    pub network: String,
    pub model_cas: String,
    pub scratch: String,
    pub seccomp: String,
    pub memory_limit_bytes: u64,
    pub process_group: String,
    pub parent_death: String,
    pub shared_memory_bytes: u64,
    pub arguments: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl SandboxReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_boundary(
            "sandbox",
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
        exact(
            &self.user_class,
            "unprivileged",
            "the worker user is unprivileged",
        )?;
        exact(&self.network, "none", "the worker has no external network")?;
        exact(&self.model_cas, "read-only", "the model cas is read-only")?;
        exact(
            &self.scratch,
            "worker-only",
            "worker scratch is the only writable path",
        )?;
        exact(
            &self.seccomp,
            "deferred",
            "seccomp stays deferred until the profile is stable",
        )?;
        if self.memory_limit_bytes == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the worker memory limit is zero",
            ));
        }
        if self.memory_limit_bytes > MAX_SANDBOX_MEMORY {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "worker memory exceeds 64 MiB",
            ));
        }
        exact(
            &self.process_group,
            "isolated",
            "the worker process group is isolated",
        )?;
        exact(
            &self.parent_death,
            "socket-eof",
            "the worker lifetime is the supervisor socket",
        )?;
        if self.shared_memory_bytes > MAX_SHARED_MEMORY {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "shared memory exceeds the worker bound",
            ));
        }
        exact(
            &self.arguments,
            "direct-array",
            "the worker arguments are a direct array",
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(SANDBOX_KIND);
        b.put("arguments", cbor_text(&self.arguments));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("memoryLimitBytes", cbor_u64(self.memory_limit_bytes));
        b.put("modelCas", cbor_text(&self.model_cas));
        b.put("network", cbor_text(&self.network));
        b.put("parentDeath", cbor_text(&self.parent_death));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("processGroup", cbor_text(&self.process_group));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("scratch", cbor_text(&self.scratch));
        b.put("seccomp", cbor_text(&self.seccomp));
        b.put("sharedMemoryBytes", cbor_u64(self.shared_memory_bytes));
        b.put("userClass", cbor_text(&self.user_class));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, SANDBOX_KIND)?;
        let out = Self {
            arguments: fields.text("arguments")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            memory_limit_bytes: fields.u64("memoryLimitBytes")?,
            model_cas: fields.text("modelCas")?,
            network: fields.text("network")?,
            parent_death: fields.text("parentDeath")?,
            placement_root: fields.digest("placementRoot")?,
            process_group: fields.text("processGroup")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            scratch: fields.text("scratch")?,
            seccomp: fields.text("seccomp")?,
            shared_memory_bytes: fields.u64("sharedMemoryBytes")?,
            user_class: fields.text("userClass")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-sandbox", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// API boundary for one cold micro fixture. The listener is not bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub tenant_root: DigestHex,
    pub bind: String,
    pub remote_explicit: bool,
    pub auth: String,
    pub body_limit_bytes: u64,
    pub rate_per_minute: u32,
    pub concurrency_limit: u32,
    pub prompt_log: String,
    pub metrics_labels: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ApiReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_boundary(
            "api",
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
        match self.bind.as_str() {
            "localhost" => {
                forbid(self.remote_explicit, "localhost does not set a remote bind")?;
            }
            "remote" => {
                require_flag(self.remote_explicit, "a remote bind is explicit")?;
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field bind has an unsupported value",
                ));
            }
        }
        exact(&self.auth, "hook", "authentication stays a hook")?;
        if self.body_limit_bytes == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the request body limit is zero",
            ));
        }
        if self.body_limit_bytes > MAX_API_BODY {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the request body exceeds 1 MiB",
            ));
        }
        if self.rate_per_minute == 0 {
            return Err(fail(ErrorCode::ContractInvalid, "the api rate is zero"));
        }
        if self.rate_per_minute > MAX_API_RATE {
            return Err(fail(ErrorCode::ContractInvalid, "the api rate exceeds 256"));
        }
        if self.concurrency_limit != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the api concurrency limit is one",
            ));
        }
        exact(
            &self.prompt_log,
            "omitted",
            "ordinary logs omit prompt text",
        )?;
        exact(
            &self.metrics_labels,
            "counts",
            "metrics labels omit user content",
        )?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(API_KIND);
        b.put("auth", cbor_text(&self.auth));
        b.put("bind", cbor_text(&self.bind));
        b.put("bodyLimitBytes", cbor_u64(self.body_limit_bytes));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("concurrencyLimit", cbor_u32(self.concurrency_limit));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("metricsLabels", cbor_text(&self.metrics_labels));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("promptLog", cbor_text(&self.prompt_log));
        b.put("ratePerMinute", cbor_u32(self.rate_per_minute));
        b.put("remoteExplicit", CborValue::Bool(self.remote_explicit));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("tenantRoot", cbor_digest(&self.tenant_root));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, API_KIND)?;
        let out = Self {
            auth: fields.text("auth")?,
            bind: fields.text("bind")?,
            body_limit_bytes: fields.u64("bodyLimitBytes")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            concurrency_limit: fields.u32("concurrencyLimit")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            metrics_labels: fields.text("metricsLabels")?,
            placement_root: fields.digest("placementRoot")?,
            prompt_log: fields.text("promptLog")?,
            rate_per_minute: fields.u32("ratePerMinute")?,
            remote_explicit: fields.bool("remoteExplicit")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            tenant_root: fields.digest("tenantRoot")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-api", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
