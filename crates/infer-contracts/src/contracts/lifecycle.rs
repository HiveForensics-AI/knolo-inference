//! Model swap, verification, and load reports.
//!
//! These documents record caller-supplied durations. They do not open a
//! model file. The layouts are specified in `spec/KIP-INFER-0034-model-swap.md`,
//! `spec/KIP-INFER-0035-model-verification.md`, and
//! `spec/KIP-INFER-0036-model-load.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    cbor_digest, cbor_text, cbor_u32, cbor_u64, expect_kind_version, one_of, Fields,
};

use super::common::{put_extensions, Builder};

pub const SWAP_KIND: &str = "knolo.infer.swap-report";
pub const VERIFICATION_KIND: &str = "knolo.infer.verification-report";
pub const LOAD_KIND: &str = "knolo.infer.load-report";

/// Parser cap shared with the GGUF and weight readers. A larger count is not a report.
pub const MAX_VERIFIED_BYTES: u64 = 32 * 1024 * 1024;

/// Unload plus incoming verification plus incoming load.
pub fn model_swap_nanos(
    unload_nanos: u64,
    verify_nanos: u64,
    load_nanos: u64,
) -> Result<u64, InferFailure> {
    if unload_nanos == 0 {
        return Err(fail(ErrorCode::ContractInvalid, "unload time is zero"));
    }
    if verify_nanos == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "incoming verification time is zero",
        ));
    }
    if load_nanos == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "incoming load time is zero",
        ));
    }
    unload_nanos
        .checked_add(verify_nanos)
        .and_then(|sum| sum.checked_add(load_nanos))
        .ok_or_else(|| fail(ErrorCode::ContractInvalid, "model swap time overflows"))
}

/// One cold replacement of a resident image by the incoming micro fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapReportV1 {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub resident_model_image_root: DigestHex,
    pub resident_artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub unload_nanos: u64,
    pub incoming_verification_nanos: u64,
    pub incoming_load_nanos: u64,
    pub model_swap_nanos: u64,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl SwapReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("validationResult", &self.validation_result, &["recorded"])?;
        one_of(
            "executionMode",
            &self.execution_mode,
            &["isolated-replay", "pinned"],
        )?;
        cold_single(
            "swap",
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
        )?;
        if self.resident_model_image_root == self.model_image_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a swap replaces a different model image",
            ));
        }
        if self.resident_artifact_root == self.artifact_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a swap replaces a different artifact",
            ));
        }
        let swap = model_swap_nanos(
            self.unload_nanos,
            self.incoming_verification_nanos,
            self.incoming_load_nanos,
        )?;
        if swap != self.model_swap_nanos {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "model swap time does not match",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(SWAP_KIND);
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("incomingLoadNanos", cbor_u64(self.incoming_load_nanos));
        b.put(
            "incomingVerificationNanos",
            cbor_u64(self.incoming_verification_nanos),
        );
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("modelSwapNanos", cbor_u64(self.model_swap_nanos));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put(
            "residentArtifactRoot",
            cbor_digest(&self.resident_artifact_root),
        );
        b.put(
            "residentModelImageRoot",
            cbor_digest(&self.resident_model_image_root),
        );
        b.put("runCount", cbor_u32(self.run_count));
        b.put("unloadNanos", cbor_u64(self.unload_nanos));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, SWAP_KIND)?;
        let out = Self {
            artifact_root: fields.digest("artifactRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            incoming_load_nanos: fields.u64("incomingLoadNanos")?,
            incoming_verification_nanos: fields.u64("incomingVerificationNanos")?,
            model_image_root: fields.digest("modelImageRoot")?,
            model_swap_nanos: fields.u64("modelSwapNanos")?,
            placement_root: fields.digest("placementRoot")?,
            request_count: fields.u32("requestCount")?,
            resident_artifact_root: fields.digest("residentArtifactRoot")?,
            resident_model_image_root: fields.digest("residentModelImageRoot")?,
            run_count: fields.u32("runCount")?,
            unload_nanos: fields.u64("unloadNanos")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-swap", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One cold verification of the micro fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationReportV1 {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub verified_bytes: u64,
    pub model_verification_nanos: u64,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl VerificationReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("validationResult", &self.validation_result, &["recorded"])?;
        one_of(
            "executionMode",
            &self.execution_mode,
            &["isolated-replay", "pinned"],
        )?;
        cold_single(
            "verification",
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
        )?;
        if self.verified_bytes == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "verified byte count is zero",
            ));
        }
        if self.verified_bytes > MAX_VERIFIED_BYTES {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "verified bytes exceed the parser cap",
            ));
        }
        if self.model_verification_nanos == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "model verification time is zero",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(VERIFICATION_KIND);
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put(
            "modelVerificationNanos",
            cbor_u64(self.model_verification_nanos),
        );
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("verifiedBytes", cbor_u64(self.verified_bytes));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, VERIFICATION_KIND)?;
        let out = Self {
            artifact_root: fields.digest("artifactRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            model_image_root: fields.digest("modelImageRoot")?,
            model_verification_nanos: fields.u64("modelVerificationNanos")?,
            placement_root: fields.digest("placementRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            verified_bytes: fields.u64("verifiedBytes")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-verification", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One cold load of the micro fixture after verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadReportV1 {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub model_load_nanos: u64,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl LoadReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("validationResult", &self.validation_result, &["recorded"])?;
        one_of(
            "executionMode",
            &self.execution_mode,
            &["isolated-replay", "pinned"],
        )?;
        cold_single(
            "load",
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
        )?;
        if self.model_load_nanos == 0 {
            return Err(fail(ErrorCode::ContractInvalid, "model load time is zero"));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(LOAD_KIND);
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("modelLoadNanos", cbor_u64(self.model_load_nanos));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, LOAD_KIND)?;
        let out = Self {
            artifact_root: fields.digest("artifactRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            model_image_root: fields.digest("modelImageRoot")?,
            model_load_nanos: fields.u64("modelLoadNanos")?,
            placement_root: fields.digest("placementRoot")?,
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
        digest_value("infer-load", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

pub(super) fn cold_single(
    noun: &str,
    cache_policy: &str,
    concurrency: u32,
    run_count: u32,
    warm_state: &str,
    request_count: u32,
) -> Result<(), InferFailure> {
    if cache_policy != "off" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("prefix cache is off for the {noun} report"),
        ));
    }
    if concurrency != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("{noun} concurrency is one"),
        ));
    }
    if run_count != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("{noun} run count is one"),
        ));
    }
    if warm_state != "cold" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("{noun} warm state is cold"),
        ));
    }
    if request_count != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("{noun} request count is one"),
        ));
    }
    Ok(())
}
