//! Release-gate reports for one cold micro fixture.
//!
//! The manifest names the notice, the SBOM, and the binary set. The binary
//! report names the supervisor and the worker. The reproducible report names
//! the lock and the instruction source. The signature report records an
//! unsigned local release or one shape-checked Ed25519 block. None of them
//! verifies a key. The layouts are specified in
//! `spec/KIP-INFER-0052-release-manifest.md`,
//! `spec/KIP-INFER-0053-binary-inventory.md`,
//! `spec/KIP-INFER-0054-reproducible-build.md`, and
//! `spec/KIP-INFER-0055-signature-gate.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    bounded_text, cbor_digest, cbor_text, cbor_u32, cbor_u64, expect_kind_version, one_of, Fields,
};

use super::common::{put_extensions, Builder};
use super::lifecycle::cold_single;

pub const RELEASE_KIND: &str = "knolo.infer.release-report";
pub const BINARY_KIND: &str = "knolo.infer.binary-report";
pub const REPRODUCIBLE_KIND: &str = "knolo.infer.reproducible-report";
pub const SIGNATURE_KIND: &str = "knolo.infer.signature-report";

pub const SUPERVISOR_BINARY: &str = "knolo-infer";
pub const WORKER_BINARY: &str = "knolo-infer-worker";

/// A release binary above this size is not recorded.
pub const MAX_BINARY_BYTES: u64 = 64 * 1024 * 1024;

/// Pinned build instructions above this count are not recorded.
pub const MAX_BUILD_INSTRUCTIONS: u32 = 32;

const FEATURE_SETS: &[&str] = &["cpu", "cuda"];
const BUILD_PROFILES: &[&str] = &["debug", "release"];
const SIGNATURE_STATUSES: &[&str] = &["shape-checked", "unsigned-local"];

fn recorded(value: &str) -> Result<(), InferFailure> {
    one_of("validationResult", value, &["recorded"])
}

fn execution_mode(value: &str) -> Result<(), InferFailure> {
    one_of("executionMode", value, &["isolated-replay", "pinned"])
}

#[allow(clippy::too_many_arguments)]
fn accept_report(
    noun: &str,
    validation_result: &str,
    execution_mode_value: &str,
    cache_policy: &str,
    concurrency: u32,
    run_count: u32,
    warm_state: &str,
    request_count: u32,
    extensions: &BTreeMap<String, CborValue>,
) -> Result<(), InferFailure> {
    recorded(validation_result)?;
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

fn signature_status(status: &str, count: u32) -> Result<(), InferFailure> {
    one_of("signatureStatus", status, SIGNATURE_STATUSES)?;
    match status {
        "unsigned-local" if count != 0 => Err(fail(
            ErrorCode::ContractInvalid,
            "an unsigned release carries a signature",
        )),
        "shape-checked" if count != 1 => Err(fail(
            ErrorCode::ContractInvalid,
            "a shape-checked release carries one signature",
        )),
        _ => Ok(()),
    }
}

/// Signed release manifest for one cold micro fixture. The key is not checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub notice_root: DigestHex,
    pub sbom_root: DigestHex,
    pub binary_set_root: DigestHex,
    pub signature_status: String,
    pub signature_count: u32,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ReleaseReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "release",
            &self.validation_result,
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        if self.notice_root == self.sbom_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the sbom repeats the notice",
            ));
        }
        if self.binary_set_root == self.engine_build_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the binary set repeats the engine build",
            ));
        }
        if self.binary_set_root == self.notice_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the binary set repeats the notice",
            ));
        }
        if self.binary_set_root == self.sbom_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the binary set repeats the sbom",
            ));
        }
        signature_status(&self.signature_status, self.signature_count)
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(RELEASE_KIND);
        b.put("binarySetRoot", cbor_digest(&self.binary_set_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("noticeRoot", cbor_digest(&self.notice_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("sbomRoot", cbor_digest(&self.sbom_root));
        b.put("signatureCount", cbor_u32(self.signature_count));
        b.put("signatureStatus", cbor_text(&self.signature_status));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, RELEASE_KIND)?;
        let out = Self {
            binary_set_root: fields.digest("binarySetRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            notice_root: fields.digest("noticeRoot")?,
            placement_root: fields.digest("placementRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            sbom_root: fields.digest("sbomRoot")?,
            signature_count: fields.u32("signatureCount")?,
            signature_status: fields.text("signatureStatus")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-release", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Supervisor and worker hashes for one cold micro fixture. The files are not opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub supervisor_name: String,
    pub supervisor_hash: DigestHex,
    pub supervisor_bytes: u64,
    pub worker_name: String,
    pub worker_hash: DigestHex,
    pub worker_bytes: u64,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl BinaryReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "binary",
            &self.validation_result,
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        if self.supervisor_name != SUPERVISOR_BINARY {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the supervisor binary is knolo-infer",
            ));
        }
        if self.worker_name != WORKER_BINARY {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the worker binary is knolo-infer-worker",
            ));
        }
        if self.supervisor_hash == self.worker_hash {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the worker hash repeats the supervisor",
            ));
        }
        if self.supervisor_hash == self.engine_build_root
            || self.worker_hash == self.engine_build_root
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the binary hash repeats the engine build",
            ));
        }
        binary_size("supervisor", self.supervisor_bytes)?;
        binary_size("worker", self.worker_bytes)?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(BINARY_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("supervisorBytes", cbor_u64(self.supervisor_bytes));
        b.put("supervisorHash", cbor_digest(&self.supervisor_hash));
        b.put("supervisorName", cbor_text(&self.supervisor_name));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        b.put("workerBytes", cbor_u64(self.worker_bytes));
        b.put("workerHash", cbor_digest(&self.worker_hash));
        b.put("workerName", cbor_text(&self.worker_name));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, BINARY_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            placement_root: fields.digest("placementRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            supervisor_bytes: fields.u64("supervisorBytes")?,
            supervisor_hash: fields.digest("supervisorHash")?,
            supervisor_name: fields.text("supervisorName")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
            worker_bytes: fields.u64("workerBytes")?,
            worker_hash: fields.digest("workerHash")?,
            worker_name: fields.text("workerName")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-binary", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

fn binary_size(noun: &str, bytes: u64) -> Result<(), InferFailure> {
    if bytes == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("the {noun} binary is empty"),
        ));
    }
    if bytes > MAX_BINARY_BYTES {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("the {noun} binary exceeds 64 MiB"),
        ));
    }
    Ok(())
}

/// Pinned build instructions for one cold micro fixture. Cargo is not run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReproducibleReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub lock_root: DigestHex,
    pub source_root: DigestHex,
    pub feature_set: String,
    pub build_profile: String,
    pub instruction_count: u32,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ReproducibleReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "reproducible",
            &self.validation_result,
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        one_of("featureSet", &self.feature_set, FEATURE_SETS)?;
        one_of("buildProfile", &self.build_profile, BUILD_PROFILES)?;
        if self.lock_root == self.engine_build_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the lock repeats the engine build",
            ));
        }
        if self.source_root == self.engine_build_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the source repeats the engine build",
            ));
        }
        if self.source_root == self.lock_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the source repeats the lock",
            ));
        }
        if self.instruction_count == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the build names no instruction",
            ));
        }
        if self.instruction_count > MAX_BUILD_INSTRUCTIONS {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the build instructions are too large",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(REPRODUCIBLE_KIND);
        b.put("buildProfile", cbor_text(&self.build_profile));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("featureSet", cbor_text(&self.feature_set));
        b.put("instructionCount", cbor_u32(self.instruction_count));
        b.put("lockRoot", cbor_digest(&self.lock_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("sourceRoot", cbor_digest(&self.source_root));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, REPRODUCIBLE_KIND)?;
        let out = Self {
            build_profile: fields.text("buildProfile")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            feature_set: fields.text("featureSet")?,
            instruction_count: fields.u32("instructionCount")?,
            lock_root: fields.digest("lockRoot")?,
            placement_root: fields.digest("placementRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            source_root: fields.digest("sourceRoot")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-reproducible", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Signature shape for one cold release. The key is not checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub release_root: DigestHex,
    pub signature_status: String,
    pub signature_count: u32,
    pub key_id: String,
    pub signature_bytes: u32,
    pub key_verified: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl SignatureReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        accept_report(
            "signature",
            &self.validation_result,
            &self.execution_mode,
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
            &self.extensions,
        )?;
        if self.release_root == self.engine_build_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the release repeats the engine build",
            ));
        }
        signature_status(&self.signature_status, self.signature_count)?;
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
            }
            "shape-checked" => {
                bounded_text("keyId", &self.key_id, 128)?;
                if self.signature_bytes != 64 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "ed25519 signatures are 64 bytes",
                    ));
                }
            }
            _ => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "field signatureStatus has an unsupported value",
                ));
            }
        }
        if self.key_verified {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "signature keys stay unverified",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(SIGNATURE_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("keyId", cbor_text(&self.key_id));
        b.put("keyVerified", CborValue::Bool(self.key_verified));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("releaseRoot", cbor_digest(&self.release_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("signatureBytes", cbor_u32(self.signature_bytes));
        b.put("signatureCount", cbor_u32(self.signature_count));
        b.put("signatureStatus", cbor_text(&self.signature_status));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, SIGNATURE_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            key_id: fields.text("keyId")?,
            key_verified: fields.bool("keyVerified")?,
            placement_root: fields.digest("placementRoot")?,
            release_root: fields.digest("releaseRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            signature_bytes: fields.u32("signatureBytes")?,
            signature_count: fields.u32("signatureCount")?,
            signature_status: fields.text("signatureStatus")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-signature", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
