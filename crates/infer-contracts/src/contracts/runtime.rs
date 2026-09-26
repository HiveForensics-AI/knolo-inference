use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    bounded_text, cbor_digest, cbor_text, cbor_u32, cbor_u64, digest_array, expect_kind_version,
    one_of, sorted_unique, text_array, Fields,
};

use super::common::{
    device_id, digest_field_array, put_extensions, require_sorted_precisions, string_field_array,
    u64_array_sum, Builder, KV_PRECISIONS, PRECISIONS,
};

pub const ENGINE_BUILD_KIND: &str = "knolo.infer.engine-build";
pub const KERNEL_BUNDLE_KIND: &str = "knolo.infer.kernel-bundle";
pub const HARDWARE_PROBE_KIND: &str = "knolo.infer.hardware-probe";
pub const PLACEMENT_PLAN_KIND: &str = "knolo.infer.placement-plan";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineBuildDescriptorV1 {
    pub binary_sha256: DigestHex,
    pub source_commit: String,
    pub cargo_lock_root: DigestHex,
    pub rustc_version: String,
    pub target_triple: String,
    pub build_profile: String,
    pub feature_set: Vec<String>,
    pub tensor_backend: String,
    pub tensor_backend_version: String,
    pub kernel_bundle_root: DigestHex,
    pub extensions: BTreeMap<String, CborValue>,
}

impl EngineBuildDescriptorV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        if self.source_commit != "unknown"
            && (self.source_commit.len() != 40
                || !self
                    .source_commit
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()))
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "source commit must be 40 lowercase hex or unknown",
            ));
        }
        bounded_text("rustcVersion", &self.rustc_version, 32)?;
        if self.target_triple.len() > 64
            || self.target_triple.is_empty()
            || !self
                .target_triple
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.')
        {
            return Err(fail(ErrorCode::ContractInvalid, "target triple is invalid"));
        }
        one_of("buildProfile", &self.build_profile, &["debug", "release"])?;
        one_of(
            "tensorBackend",
            &self.tensor_backend,
            &["candle-cpu", "candle-cuda", "reference-f32"],
        )?;
        bounded_text("tensorBackendVersion", &self.tensor_backend_version, 32)?;
        sorted_unique("featureSet", &self.feature_set)?;
        if self.feature_set.len() > 64 {
            return Err(fail(ErrorCode::ContractInvalid, "feature set is too large"));
        }
        for feature in &self.feature_set {
            if feature.len() > 64
                || !feature
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
            {
                return Err(fail(ErrorCode::ContractInvalid, "feature name is invalid"));
            }
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(ENGINE_BUILD_KIND);
        b.put("binarySha256", cbor_digest(&self.binary_sha256));
        b.put("buildProfile", cbor_text(&self.build_profile));
        b.put("cargoLockRoot", cbor_digest(&self.cargo_lock_root));
        put_extensions(&mut b, &self.extensions)?;
        b.put("featureSet", string_field_array(&self.feature_set));
        b.put("kernelBundleRoot", cbor_digest(&self.kernel_bundle_root));
        b.put("rustcVersion", cbor_text(&self.rustc_version));
        b.put("sourceCommit", cbor_text(&self.source_commit));
        b.put("targetTriple", cbor_text(&self.target_triple));
        b.put("tensorBackend", cbor_text(&self.tensor_backend));
        b.put(
            "tensorBackendVersion",
            cbor_text(&self.tensor_backend_version),
        );
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, ENGINE_BUILD_KIND)?;
        let features = text_array(&fields.array("featureSet")?, "featureSet")?;
        let out = Self {
            binary_sha256: fields.digest("binarySha256")?,
            build_profile: fields.text("buildProfile")?,
            cargo_lock_root: fields.digest("cargoLockRoot")?,
            extensions: fields.extensions()?,
            feature_set: features,
            kernel_bundle_root: fields.digest("kernelBundleRoot")?,
            rustc_version: fields.text("rustcVersion")?,
            source_commit: fields.text("sourceCommit")?,
            target_triple: fields.text("targetTriple")?,
            tensor_backend: fields.text("tensorBackend")?,
            tensor_backend_version: fields.text("tensorBackendVersion")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-engine-build", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JitKernelV1 {
    pub source_root: DigestHex,
    pub compiler_version: String,
    pub flags_root: DigestHex,
    pub target_architecture: String,
    pub code_object_root: DigestHex,
}

impl JitKernelV1 {
    fn validate(&self) -> Result<(), InferFailure> {
        bounded_text("compilerVersion", &self.compiler_version, 32)?;
        bounded_text("targetArchitecture", &self.target_architecture, 32)?;
        Ok(())
    }

    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put("codeObjectRoot", cbor_digest(&self.code_object_root));
        b.put("compilerVersion", cbor_text(&self.compiler_version));
        b.put("flagsRoot", cbor_digest(&self.flags_root));
        b.put("sourceRoot", cbor_digest(&self.source_root));
        b.put("targetArchitecture", cbor_text(&self.target_architecture));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            code_object_root: fields.digest("codeObjectRoot")?,
            compiler_version: fields.text("compilerVersion")?,
            flags_root: fields.digest("flagsRoot")?,
            source_root: fields.digest("sourceRoot")?,
            target_architecture: fields.text("targetArchitecture")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelBundleDescriptorV1 {
    pub cuda_architectures: Vec<String>,
    pub cuda_toolkit_version: String,
    pub source_root: DigestHex,
    pub compiler_flags: Vec<String>,
    pub build_mode: String,
    pub code_object_root: DigestHex,
    pub jit: Option<JitKernelV1>,
    pub extensions: BTreeMap<String, CborValue>,
}

impl KernelBundleDescriptorV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("buildMode", &self.build_mode, &["cpu", "cuda", "jit"])?;
        sorted_unique("cudaArchitectures", &self.cuda_architectures)?;
        sorted_unique("compilerFlags", &self.compiler_flags)?;
        if self.compiler_flags.len() > 64 || self.cuda_architectures.len() > 16 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kernel bundle list is too large",
            ));
        }
        for flag in &self.compiler_flags {
            if flag.is_empty()
                || flag.len() > 128
                || flag.chars().any(|c| c.is_whitespace() || c.is_control())
            {
                return Err(fail(ErrorCode::ContractInvalid, "compiler flag is invalid"));
            }
        }
        for arch in &self.cuda_architectures {
            bounded_text("cudaArchitectures", arch, 32)?;
        }
        match self.build_mode.as_str() {
            "cpu" => {
                if !self.cuda_architectures.is_empty()
                    || self.cuda_toolkit_version != "none"
                    || self.jit.is_some()
                {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "cpu kernel bundle must not name CUDA or JIT",
                    ));
                }
            }
            "cuda" => {
                bounded_text("cudaToolkitVersion", &self.cuda_toolkit_version, 32)?;
                if self.cuda_architectures.is_empty()
                    || self.jit.is_some()
                    || self.cuda_toolkit_version == "none"
                {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "cuda kernel bundle is incomplete",
                    ));
                }
            }
            "jit" if self.jit.is_none() => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "jit kernel bundle is missing jit metadata",
                ));
            }
            "jit" => {}
            _ => {}
        }
        if let Some(jit) = &self.jit {
            jit.validate()?;
        }
        if self.build_mode != "cuda" && self.build_mode != "cpu" {
            bounded_text("cudaToolkitVersion", &self.cuda_toolkit_version, 32)?;
        }
        if self.build_mode == "cpu" {
            // already checked "none"
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(KERNEL_BUNDLE_KIND);
        b.put("buildMode", cbor_text(&self.build_mode));
        b.put("codeObjectRoot", cbor_digest(&self.code_object_root));
        b.put("compilerFlags", string_field_array(&self.compiler_flags));
        b.put(
            "cudaArchitectures",
            string_field_array(&self.cuda_architectures),
        );
        b.put("cudaToolkitVersion", cbor_text(&self.cuda_toolkit_version));
        put_extensions(&mut b, &self.extensions)?;
        b.put_opt(
            "jit",
            self.jit.as_ref().map(|jit| jit.to_cbor()).transpose()?,
        );
        b.put("sourceRoot", cbor_digest(&self.source_root));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, KERNEL_BUNDLE_KIND)?;
        let out = Self {
            build_mode: fields.text("buildMode")?,
            code_object_root: fields.digest("codeObjectRoot")?,
            compiler_flags: text_array(&fields.array("compilerFlags")?, "compilerFlags")?,
            cuda_architectures: text_array(
                &fields.array("cudaArchitectures")?,
                "cudaArchitectures",
            )?,
            cuda_toolkit_version: fields.text("cudaToolkitVersion")?,
            extensions: fields.extensions()?,
            jit: match fields.optional("jit")? {
                None => None,
                Some(value) => Some(JitKernelV1::from_cbor(&value)?),
            },
            source_root: fields.digest("sourceRoot")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-kernel-bundle", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuProbeV1 {
    pub device: String,
    pub vendor: String,
    pub model: String,
    pub vram_total_bytes: u64,
    pub vram_available_bytes: u64,
    pub compute_capability: String,
    pub driver_version: String,
    pub runtime_version: String,
    pub precisions: Vec<String>,
}

impl GpuProbeV1 {
    fn validate(&self) -> Result<(), InferFailure> {
        device_id(&self.device)?;
        if self.device == "cpu" {
            return Err(fail(ErrorCode::ContractInvalid, "gpu device cannot be cpu"));
        }
        bounded_text("vendor", &self.vendor, 32)?;
        bounded_text("model", &self.model, 64)?;
        if looks_like_uuid(&self.model) || looks_like_uuid(&self.vendor) {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "hardware probe must not contain a UUID",
            ));
        }
        if self.vram_available_bytes > self.vram_total_bytes {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "available VRAM exceeds total",
            ));
        }
        validate_capability(&self.compute_capability)?;
        bounded_token("driverVersion", &self.driver_version)?;
        bounded_token("runtimeVersion", &self.runtime_version)?;
        require_sorted_precisions(&self.precisions, PRECISIONS)?;
        Ok(())
    }

    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put("computeCapability", cbor_text(&self.compute_capability));
        b.put("device", cbor_text(&self.device));
        b.put("driverVersion", cbor_text(&self.driver_version));
        b.put("model", cbor_text(&self.model));
        b.put("precisions", string_field_array(&self.precisions));
        b.put("runtimeVersion", cbor_text(&self.runtime_version));
        b.put("vendor", cbor_text(&self.vendor));
        b.put("vramAvailableBytes", cbor_u64(self.vram_available_bytes));
        b.put("vramTotalBytes", cbor_u64(self.vram_total_bytes));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            compute_capability: fields.text("computeCapability")?,
            device: fields.text("device")?,
            driver_version: fields.text("driverVersion")?,
            model: fields.text("model")?,
            precisions: text_array(&fields.array("precisions")?, "precisions")?,
            runtime_version: fields.text("runtimeVersion")?,
            vendor: fields.text("vendor")?,
            vram_available_bytes: fields.u64("vramAvailableBytes")?,
            vram_total_bytes: fields.u64("vramTotalBytes")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareProbeV1 {
    pub cpu_architecture: String,
    pub cpu_model_class: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    pub ram_total_bytes: u64,
    pub ram_available_bytes: u64,
    pub numa_node_count: Option<u32>,
    pub gpus: Vec<GpuProbeV1>,
    pub storage_class: String,
    pub storage_available_bytes: u64,
    pub kernel_bundles: Vec<DigestHex>,
    pub extensions: BTreeMap<String, CborValue>,
}

impl HardwareProbeV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of(
            "cpuArchitecture",
            &self.cpu_architecture,
            &["aarch64", "unknown", "x86_64"],
        )?;
        bounded_text("cpuModelClass", &self.cpu_model_class, 64)?;
        if looks_like_uuid(&self.cpu_model_class) {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "hardware probe must not contain a UUID",
            ));
        }
        if self.physical_cores == 0
            || self.logical_cores < self.physical_cores
            || self.logical_cores > 4096
        {
            return Err(fail(ErrorCode::ContractInvalid, "core counts are invalid"));
        }
        if self.ram_available_bytes > self.ram_total_bytes {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "available RAM exceeds total",
            ));
        }
        if let Some(nodes) = self.numa_node_count {
            if nodes == 0 || nodes > 64 {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "NUMA node count is invalid",
                ));
            }
        }
        one_of(
            "storageClass",
            &self.storage_class,
            &["hdd", "memory", "ssd", "unknown"],
        )?;
        if self.gpus.len() > 16 || self.kernel_bundles.len() > 32 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "hardware list is too large",
            ));
        }
        let devices: Vec<_> = self.gpus.iter().map(|gpu| gpu.device.clone()).collect();
        sorted_unique("gpus", &devices)?;
        for gpu in &self.gpus {
            gpu.validate()?;
        }
        let digests: Vec<_> = self
            .kernel_bundles
            .iter()
            .map(|d| d.as_str().to_string())
            .collect();
        sorted_unique("kernelBundles", &digests)?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut gpus = Vec::with_capacity(self.gpus.len());
        for gpu in &self.gpus {
            gpus.push(gpu.to_cbor()?);
        }
        let mut b = Builder::typed(HARDWARE_PROBE_KIND);
        b.put("cpuArchitecture", cbor_text(&self.cpu_architecture));
        b.put("cpuModelClass", cbor_text(&self.cpu_model_class));
        put_extensions(&mut b, &self.extensions)?;
        b.put("gpus", CborValue::Array(gpus));
        b.put("kernelBundles", digest_field_array(&self.kernel_bundles));
        b.put("logicalCores", cbor_u32(self.logical_cores));
        b.put_opt("numaNodeCount", self.numa_node_count.map(cbor_u32));
        b.put("physicalCores", cbor_u32(self.physical_cores));
        b.put("ramAvailableBytes", cbor_u64(self.ram_available_bytes));
        b.put("ramTotalBytes", cbor_u64(self.ram_total_bytes));
        b.put(
            "storageAvailableBytes",
            cbor_u64(self.storage_available_bytes),
        );
        b.put("storageClass", cbor_text(&self.storage_class));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, HARDWARE_PROBE_KIND)?;
        let gpu_values = fields.array("gpus")?;
        let mut gpus = Vec::with_capacity(gpu_values.len());
        for item in &gpu_values {
            gpus.push(GpuProbeV1::from_cbor(item)?);
        }
        let bundles = digest_array(&fields.array("kernelBundles")?, "kernelBundles")?;
        let out = Self {
            cpu_architecture: fields.text("cpuArchitecture")?,
            cpu_model_class: fields.text("cpuModelClass")?,
            extensions: fields.extensions()?,
            gpus,
            kernel_bundles: bundles,
            logical_cores: fields.u32("logicalCores")?,
            numa_node_count: match fields.optional("numaNodeCount")? {
                None => None,
                Some(value) => Some(crate::fields::expect_u32("numaNodeCount", &value)?),
            },
            physical_cores: fields.u32("physicalCores")?,
            ram_available_bytes: fields.u64("ramAvailableBytes")?,
            ram_total_bytes: fields.u64("ramTotalBytes")?,
            storage_available_bytes: fields.u64("storageAvailableBytes")?,
            storage_class: fields.text("storageClass")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-hardware", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorGroupV1 {
    pub name: String,
    pub device: String,
    pub compute_precision: String,
    pub storage_precision: String,
}

impl TensorGroupV1 {
    fn validate(&self) -> Result<(), InferFailure> {
        bounded_text("tensorGroup", &self.name, 64)?;
        device_id(&self.device)?;
        one_of("computePrecision", &self.compute_precision, KV_PRECISIONS)?;
        one_of("storagePrecision", &self.storage_precision, PRECISIONS)?;
        Ok(())
    }

    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put("computePrecision", cbor_text(&self.compute_precision));
        b.put("device", cbor_text(&self.device));
        b.put("name", cbor_text(&self.name));
        b.put("storagePrecision", cbor_text(&self.storage_precision));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            compute_precision: fields.text("computePrecision")?,
            device: fields.text("device")?,
            name: fields.text("name")?,
            storage_precision: fields.text("storagePrecision")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementPlanV1 {
    pub model_runtime_root: DigestHex,
    pub devices: Vec<String>,
    pub tensor_groups: Vec<TensorGroupV1>,
    pub context_reservation_tokens: u32,
    pub kv_block_size: u32,
    pub kv_precision: String,
    pub workspace_bytes: u64,
    pub graph_capture_mode: String,
    pub safety_margin_bytes: u64,
    pub expected_weight_bytes: u64,
    pub expected_kv_bytes: u64,
    pub expected_workspace_bytes: u64,
    pub expected_staging_bytes: u64,
    pub expected_overhead_bytes: u64,
    pub expected_total_bytes: u64,
    pub rejection_reason: Option<String>,
    pub extensions: BTreeMap<String, CborValue>,
}

impl PlacementPlanV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        if self.devices.is_empty() || self.devices.len() > 16 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "placement devices are empty or too large",
            ));
        }
        sorted_unique("devices", &self.devices)?;
        for device in &self.devices {
            device_id(device)?;
        }
        if self.tensor_groups.is_empty() || self.tensor_groups.len() > 256 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "tensor groups are empty or too large",
            ));
        }
        let names: Vec<_> = self.tensor_groups.iter().map(|g| g.name.clone()).collect();
        sorted_unique("tensorGroups", &names)?;
        for group in &self.tensor_groups {
            group.validate()?;
            if !self.devices.iter().any(|device| device == &group.device) {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "tensor group names an unselected device",
                ));
            }
        }
        if self.context_reservation_tokens == 0 || self.context_reservation_tokens > 1_048_576 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "context reservation is invalid",
            ));
        }
        if !(16..=65_536).contains(&self.kv_block_size) || !self.kv_block_size.is_power_of_two() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "KV block size must be a power of two from 16 to 65536",
            ));
        }
        one_of("kvPrecision", &self.kv_precision, KV_PRECISIONS)?;
        one_of(
            "graphCaptureMode",
            &self.graph_capture_mode,
            &["decode-buckets", "off"],
        )?;
        if self.workspace_bytes != self.expected_workspace_bytes {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "workspace bytes do not match the expected total",
            ));
        }
        let total = u64_array_sum(&[
            self.expected_weight_bytes,
            self.expected_kv_bytes,
            self.expected_workspace_bytes,
            self.expected_staging_bytes,
            self.expected_overhead_bytes,
            self.safety_margin_bytes,
        ])?;
        if total != self.expected_total_bytes {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "placement byte total does not match its parts",
            ));
        }
        if let Some(reason) = &self.rejection_reason {
            one_of(
                "rejectionReason",
                reason,
                &["INSUFFICIENT_MEMORY", "PLACEMENT_UNSATISFIABLE"],
            )?;
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut groups = Vec::with_capacity(self.tensor_groups.len());
        for group in &self.tensor_groups {
            groups.push(group.to_cbor()?);
        }
        let mut b = Builder::typed(PLACEMENT_PLAN_KIND);
        b.put(
            "contextReservationTokens",
            cbor_u32(self.context_reservation_tokens),
        );
        b.put("devices", string_field_array(&self.devices));
        b.put("expectedKvBytes", cbor_u64(self.expected_kv_bytes));
        b.put(
            "expectedOverheadBytes",
            cbor_u64(self.expected_overhead_bytes),
        );
        b.put(
            "expectedStagingBytes",
            cbor_u64(self.expected_staging_bytes),
        );
        b.put("expectedTotalBytes", cbor_u64(self.expected_total_bytes));
        b.put("expectedWeightBytes", cbor_u64(self.expected_weight_bytes));
        b.put(
            "expectedWorkspaceBytes",
            cbor_u64(self.expected_workspace_bytes),
        );
        put_extensions(&mut b, &self.extensions)?;
        b.put("graphCaptureMode", cbor_text(&self.graph_capture_mode));
        b.put("kvBlockSize", cbor_u32(self.kv_block_size));
        b.put("kvPrecision", cbor_text(&self.kv_precision));
        b.put("modelRuntimeRoot", cbor_digest(&self.model_runtime_root));
        b.put_opt(
            "rejectionReason",
            self.rejection_reason.clone().map(cbor_text),
        );
        b.put("safetyMarginBytes", cbor_u64(self.safety_margin_bytes));
        b.put("tensorGroups", CborValue::Array(groups));
        b.put("workspaceBytes", cbor_u64(self.workspace_bytes));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, PLACEMENT_PLAN_KIND)?;
        let group_values = fields.array("tensorGroups")?;
        let mut tensor_groups = Vec::with_capacity(group_values.len());
        for item in &group_values {
            tensor_groups.push(TensorGroupV1::from_cbor(item)?);
        }
        let out = Self {
            context_reservation_tokens: fields.u32("contextReservationTokens")?,
            devices: text_array(&fields.array("devices")?, "devices")?,
            expected_kv_bytes: fields.u64("expectedKvBytes")?,
            expected_overhead_bytes: fields.u64("expectedOverheadBytes")?,
            expected_staging_bytes: fields.u64("expectedStagingBytes")?,
            expected_total_bytes: fields.u64("expectedTotalBytes")?,
            expected_weight_bytes: fields.u64("expectedWeightBytes")?,
            expected_workspace_bytes: fields.u64("expectedWorkspaceBytes")?,
            extensions: fields.extensions()?,
            graph_capture_mode: fields.text("graphCaptureMode")?,
            kv_block_size: fields.u32("kvBlockSize")?,
            kv_precision: fields.text("kvPrecision")?,
            model_runtime_root: fields.digest("modelRuntimeRoot")?,
            rejection_reason: fields.opt_text("rejectionReason")?,
            safety_margin_bytes: fields.u64("safetyMarginBytes")?,
            tensor_groups,
            workspace_bytes: fields.u64("workspaceBytes")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-placement", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

fn bounded_token(field: &str, value: &str) -> Result<(), InferFailure> {
    if value.is_empty()
        || value.len() > 32
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("field {field} is invalid"),
        ));
    }
    Ok(())
}

fn validate_capability(value: &str) -> Result<(), InferFailure> {
    let Some((major, minor)) = value.split_once('.') else {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "compute capability must be major.minor",
        ));
    };
    if major.is_empty()
        || minor.is_empty()
        || major.len() > 3
        || minor.len() > 3
        || !major.bytes().all(|b| b.is_ascii_digit())
        || !minor.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "compute capability must be major.minor",
        ));
    }
    Ok(())
}

fn looks_like_uuid(value: &str) -> bool {
    let parts: Vec<_> = value.split('-').collect();
    parts.len() == 5
        && parts[0].len() == 8
        && parts[1].len() == 4
        && parts[2].len() == 4
        && parts[3].len() == 4
        && parts[4].len() == 12
        && parts
            .iter()
            .all(|part| part.bytes().all(|b| b.is_ascii_hexdigit()))
}
