//! Peak memory, KV utilization, and prefix-reuse reports.
//!
//! These documents record caller-supplied counts. They do not sample a
//! process, allocate a page pool, or allocate a prefix index. The layouts
//! are specified in `spec/KIP-INFER-0037-peak-memory.md`,
//! `spec/KIP-INFER-0038-kv-utilization.md`, and
//! `spec/KIP-INFER-0039-prefix-reuse.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    cbor_digest, cbor_text, cbor_u32, cbor_u64, expect_kind_version, one_of, Fields,
};

use super::common::{put_extensions, Builder};
use super::lifecycle::cold_single;

pub const PEAK_KIND: &str = "knolo.infer.peak-report";
pub const KV_KIND: &str = "knolo.infer.kv-report";
pub const PREFIX_KIND: &str = "knolo.infer.prefix-report";

/// Host or device high-water mark above this size is not a micro-fixture report.
/// The same bound refuses a KV page pool.
pub const MAX_PEAK_BYTES: u64 = 64 * 1024 * 1024;

/// Pages in the micro-fixture pool.
pub const MICRO_KV_PAGES: u32 = 8;

/// Tokens in one micro-fixture page. The context is one page.
pub const MICRO_KV_PAGE_TOKENS: u32 = 16;

/// Token slots occupied, in millionths of the eight-page pool.
pub fn kv_utilization_millionths(peak_tokens: u32, peak_pages: u32) -> Result<u32, InferFailure> {
    if peak_tokens == 0 {
        return Err(fail(ErrorCode::ContractInvalid, "kv peak tokens are zero"));
    }
    if peak_tokens > MICRO_KV_PAGE_TOKENS {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "kv peak tokens exceed the micro context",
        ));
    }
    if peak_pages != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "the micro fixture occupies one page",
        ));
    }
    let slots = MICRO_KV_PAGES * MICRO_KV_PAGE_TOKENS;
    Ok(peak_tokens * 1_000_000 / slots)
}

/// One cold peak of host RAM and device VRAM for the micro fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeakReportV1 {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub device: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub peak_ram_bytes: u64,
    pub peak_vram_bytes: u64,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl PeakReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("validationResult", &self.validation_result, &["recorded"])?;
        one_of(
            "executionMode",
            &self.execution_mode,
            &["isolated-replay", "pinned"],
        )?;
        cold_single(
            "peak",
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
        )?;
        one_of("device", &self.device, &["cpu", "slot-0"])?;
        if self.peak_ram_bytes == 0 {
            return Err(fail(ErrorCode::ContractInvalid, "peak ram is zero"));
        }
        if self.peak_ram_bytes > MAX_PEAK_BYTES {
            return Err(fail(ErrorCode::ContractInvalid, "peak ram exceeds 64 MiB"));
        }
        if self.device == "cpu" && self.peak_vram_bytes != 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "cpu placement has no device memory",
            ));
        }
        if self.device == "slot-0" && self.peak_vram_bytes == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "slot-0 placement records device memory",
            ));
        }
        if self.peak_vram_bytes > MAX_PEAK_BYTES {
            return Err(fail(ErrorCode::ContractInvalid, "peak vram exceeds 64 MiB"));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(PEAK_KIND);
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("device", cbor_text(&self.device));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("peakRamBytes", cbor_u64(self.peak_ram_bytes));
        b.put("peakVramBytes", cbor_u64(self.peak_vram_bytes));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, PEAK_KIND)?;
        let out = Self {
            artifact_root: fields.digest("artifactRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            device: fields.text("device")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            model_image_root: fields.digest("modelImageRoot")?,
            peak_ram_bytes: fields.u64("peakRamBytes")?,
            peak_vram_bytes: fields.u64("peakVramBytes")?,
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
        digest_value("infer-peak", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One cold occupancy of the micro-fixture page pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvReportV1 {
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
    pub page_total: u32,
    pub page_size_tokens: u32,
    pub peak_pages: u32,
    pub peak_tokens: u32,
    pub utilization_millionths: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl KvReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("validationResult", &self.validation_result, &["recorded"])?;
        one_of(
            "executionMode",
            &self.execution_mode,
            &["isolated-replay", "pinned"],
        )?;
        cold_single(
            "kv",
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
        )?;
        if self.page_total != MICRO_KV_PAGES || self.page_size_tokens != MICRO_KV_PAGE_TOKENS {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the kv report is the micro fixture",
            ));
        }
        let utilization = kv_utilization_millionths(self.peak_tokens, self.peak_pages)?;
        if utilization != self.utilization_millionths {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv utilization does not match",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(KV_KIND);
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("pageSizeTokens", cbor_u32(self.page_size_tokens));
        b.put("pageTotal", cbor_u32(self.page_total));
        b.put("peakPages", cbor_u32(self.peak_pages));
        b.put("peakTokens", cbor_u32(self.peak_tokens));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put(
            "utilizationMillionths",
            cbor_u32(self.utilization_millionths),
        );
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, KV_KIND)?;
        let out = Self {
            artifact_root: fields.digest("artifactRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            model_image_root: fields.digest("modelImageRoot")?,
            page_size_tokens: fields.u32("pageSizeTokens")?,
            page_total: fields.u32("pageTotal")?,
            peak_pages: fields.u32("peakPages")?,
            peak_tokens: fields.u32("peakTokens")?,
            placement_root: fields.digest("placementRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            utilization_millionths: fields.u32("utilizationMillionths")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-kv", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// One cold prefix-reuse count while the cache stays off.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixReportV1 {
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
    pub lookup_count: u32,
    pub hit_count: u32,
    pub miss_count: u32,
    pub reused_tokens: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl PrefixReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("validationResult", &self.validation_result, &["recorded"])?;
        one_of(
            "executionMode",
            &self.execution_mode,
            &["isolated-replay", "pinned"],
        )?;
        cold_single(
            "prefix",
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
        )?;
        if self.lookup_count != 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefix lookups are zero while the cache is off",
            ));
        }
        if self.hit_count != 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefix hits are zero while the cache is off",
            ));
        }
        if self.miss_count != 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefix misses are zero while the cache is off",
            ));
        }
        if self.reused_tokens != 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "reused tokens are zero while the cache is off",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(PREFIX_KIND);
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("hitCount", cbor_u32(self.hit_count));
        b.put("lookupCount", cbor_u32(self.lookup_count));
        b.put("missCount", cbor_u32(self.miss_count));
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("reusedTokens", cbor_u32(self.reused_tokens));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, PREFIX_KIND)?;
        let out = Self {
            artifact_root: fields.digest("artifactRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            hit_count: fields.u32("hitCount")?,
            lookup_count: fields.u32("lookupCount")?,
            miss_count: fields.u32("missCount")?,
            model_image_root: fields.digest("modelImageRoot")?,
            placement_root: fields.digest("placementRoot")?,
            request_count: fields.u32("requestCount")?,
            reused_tokens: fields.u32("reusedTokens")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-prefix", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
