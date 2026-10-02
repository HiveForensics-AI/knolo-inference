use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    bounded_text, cbor_text, cbor_u32, cbor_u64, expect_kind_version, one_of, sorted_unique, Fields,
};

use super::common::{
    artifact_files_cbor, artifact_root, put_extensions, signatures_from_cbor, signatures_to_cbor,
    string_field_array, validate_files, ArtifactFileV1, Builder, EmbeddedArtifactV1,
    FixedPointSamplerV1, SpecialTokensV1, PRECISIONS,
};

pub const MODEL_IMAGE_KIND: &str = "knolo.infer.model-image";
pub const MODEL_ARTIFACT_KIND: &str = "knolo.infer.model-artifact-set";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureRefV1 {
    pub family: String,
    pub adapter: String,
}

impl ArchitectureRefV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        bounded_text("family", &self.family, 128)?;
        bounded_text("adapter", &self.adapter, 128)?;
        if !self.adapter.starts_with("knolo.")
            || !self
                .adapter
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'.')
        {
            return Err(fail(
                ErrorCode::UnsupportedArchitecture,
                "adapter id must be a knolo.* identifier",
            ));
        }
        Ok(())
    }

    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put("adapter", cbor_text(&self.adapter));
        b.put("family", cbor_text(&self.family));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            adapter: fields.text("adapter")?,
            family: fields.text("family")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpecV1 {
    pub name: String,
    pub shape: Vec<u32>,
    pub dtype: String,
}

impl TensorSpecV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        if self.name.is_empty()
            || self.name.len() > 256
            || self.name.contains("..")
            || self.name.contains('/')
            || self.name.contains('\\')
            || !self
                .name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
        {
            return Err(fail(ErrorCode::ContractInvalid, "tensor name is invalid"));
        }
        one_of("dtype", &self.dtype, PRECISIONS)?;
        if self.shape.is_empty() || self.shape.len() > 8 || self.shape.contains(&0) {
            return Err(fail(ErrorCode::ContractInvalid, "tensor shape is invalid"));
        }
        let mut product = 1u64;
        for dim in &self.shape {
            product = product
                .checked_mul(u64::from(*dim))
                .ok_or_else(|| fail(ErrorCode::ContractInvalid, "tensor shape overflows"))?;
        }
        let _ = product;
        Ok(())
    }

    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put("dtype", cbor_text(&self.dtype));
        b.put("name", cbor_text(&self.name));
        b.put(
            "shape",
            CborValue::Array(self.shape.iter().copied().map(cbor_u32).collect()),
        );
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let shape_values = fields.array("shape")?;
        let mut shape = Vec::with_capacity(shape_values.len());
        for item in &shape_values {
            shape.push(crate::fields::expect_u32("shape", item)?);
        }
        let out = Self {
            dtype: fields.text("dtype")?,
            name: fields.text("name")?,
            shape,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LicenseV1 {
    pub id: String,
    pub acceptance_required: bool,
}

impl LicenseV1 {
    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        bounded_text("license", &self.id, 128)?;
        let mut b = Builder::bare();
        b.put(
            "acceptanceRequired",
            CborValue::Bool(self.acceptance_required),
        );
        b.put("id", cbor_text(&self.id));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            acceptance_required: fields.bool("acceptanceRequired")?,
            id: fields.text("id")?,
        };
        fields.finish()?;
        bounded_text("license", &out.id, 128)?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceHintV1 {
    pub provider: String,
    pub repository: String,
    pub revision: String,
}

impl SourceHintV1 {
    fn validate(&self) -> Result<(), InferFailure> {
        bounded_text("provider", &self.provider, 64)?;
        bounded_text("repository", &self.repository, 256)?;
        bounded_text("revision", &self.revision, 128)?;
        Ok(())
    }

    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put("provider", cbor_text(&self.provider));
        b.put("repository", cbor_text(&self.repository));
        b.put("revision", cbor_text(&self.revision));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            provider: fields.text("provider")?,
            repository: fields.text("repository")?,
            revision: fields.text("revision")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceRequirementsV1 {
    pub minimum_ram_bytes: u64,
    pub minimum_vram_bytes: u64,
}

impl ResourceRequirementsV1 {
    fn to_cbor(&self) -> CborValue {
        let mut b = Builder::bare();
        b.put("minimumRamBytes", cbor_u64(self.minimum_ram_bytes));
        b.put("minimumVramBytes", cbor_u64(self.minimum_vram_bytes));
        b.finish()
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            minimum_ram_bytes: fields.u64("minimumRamBytes")?,
            minimum_vram_bytes: fields.u64("minimumVramBytes")?,
        };
        fields.finish()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementHintsV1 {
    pub profile: Option<String>,
    pub prefer_device: Option<String>,
}

impl PlacementHintsV1 {
    fn validate(&self) -> Result<(), InferFailure> {
        if self.profile.is_none() && self.prefer_device.is_none() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "placement hints are empty",
            ));
        }
        if let Some(profile) = &self.profile {
            bounded_text("profile", profile, 64)?;
        }
        if let Some(device) = &self.prefer_device {
            super::common::device_id(device)?;
        }
        Ok(())
    }

    fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put_opt("preferDevice", self.prefer_device.clone().map(cbor_text));
        b.put_opt("profile", self.profile.clone().map(cbor_text));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            prefer_device: fields.opt_text("preferDevice")?,
            profile: fields.opt_text("profile")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelImageV1 {
    pub name: String,
    pub variant: String,
    pub architecture: ArchitectureRefV1,
    pub format: String,
    pub files: Vec<ArtifactFileV1>,
    pub tokenizer: EmbeddedArtifactV1,
    pub template: EmbeddedArtifactV1,
    pub special_tokens: SpecialTokensV1,
    pub generation_defaults: FixedPointSamplerV1,
    pub capabilities: Vec<String>,
    pub license: LicenseV1,
    pub sources: Vec<SourceHintV1>,
    pub tensor_inventory: Vec<TensorSpecV1>,
    pub precisions: Vec<String>,
    pub requirements: ResourceRequirementsV1,
    pub placement_hints: Option<PlacementHintsV1>,
    pub extensions: BTreeMap<String, CborValue>,
    pub signatures: Vec<super::common::SignatureV1>,
}

impl ModelImageV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        bounded_text("name", &self.name, 256)?;
        bounded_text("variant", &self.variant, 128)?;
        self.architecture.validate()?;
        one_of("format", &self.format, &["gguf", "safetensors"])?;
        validate_files(&self.files)?;
        self.tokenizer
            .validate("infer-tokenizer", ErrorCode::TokenizerInvalid)?;
        self.template
            .validate("infer-template", ErrorCode::TemplateInvalid)?;
        self.special_tokens.validate()?;
        self.generation_defaults.validate()?;
        if self.capabilities.is_empty() || self.capabilities.len() > 32 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "capabilities are empty or too large",
            ));
        }
        sorted_unique("capabilities", &self.capabilities)?;
        for capability in &self.capabilities {
            bounded_text("capabilities", capability, 64)?;
        }
        if self.sources.len() > 16 {
            return Err(fail(ErrorCode::ContractInvalid, "too many sources"));
        }
        for source in &self.sources {
            source.validate()?;
        }
        if self.tensor_inventory.is_empty() || self.tensor_inventory.len() > 100_000 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "tensor inventory is empty or too large",
            ));
        }
        let names: Vec<_> = self
            .tensor_inventory
            .iter()
            .map(|t| t.name.clone())
            .collect();
        sorted_unique("tensorInventory", &names)?;
        for tensor in &self.tensor_inventory {
            tensor.validate()?;
        }
        if self.precisions.is_empty() {
            return Err(fail(ErrorCode::ContractInvalid, "precisions are empty"));
        }
        super::common::require_sorted_precisions(&self.precisions, PRECISIONS)?;
        if let Some(hints) = &self.placement_hints {
            hints.validate()?;
        }
        if self.signatures.len() > 8 {
            return Err(fail(ErrorCode::ContractInvalid, "too many signatures"));
        }
        for signature in &self.signatures {
            signature.validate()?;
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut weights = Builder::bare();
        weights.put("files", artifact_files_cbor(&self.files)?);
        weights.put("format", cbor_text(&self.format));
        let mut tensors = Vec::with_capacity(self.tensor_inventory.len());
        for tensor in &self.tensor_inventory {
            tensors.push(tensor.to_cbor()?);
        }
        let mut sources = Vec::with_capacity(self.sources.len());
        for source in &self.sources {
            sources.push(source.to_cbor()?);
        }
        let mut b = Builder::typed(MODEL_IMAGE_KIND);
        b.put("architecture", self.architecture.to_cbor()?);
        b.put("capabilities", string_field_array(&self.capabilities));
        put_extensions(&mut b, &self.extensions)?;
        b.put("generationDefaults", self.generation_defaults.to_cbor()?);
        b.put("license", self.license.to_cbor()?);
        b.put("name", cbor_text(&self.name));
        b.put_opt(
            "placementHints",
            self.placement_hints
                .as_ref()
                .map(|hints| hints.to_cbor())
                .transpose()?,
        );
        b.put("precisions", string_field_array(&self.precisions));
        b.put("requirements", self.requirements.to_cbor());
        b.put("signatures", signatures_to_cbor(&self.signatures)?);
        b.put("sources", CborValue::Array(sources));
        b.put("specialTokens", self.special_tokens.to_cbor()?);
        b.put("template", self.template.to_cbor());
        b.put("tensorInventory", CborValue::Array(tensors));
        b.put("tokenizer", self.tokenizer.to_cbor());
        b.put("variant", cbor_text(&self.variant));
        b.put("weights", weights.finish());
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, MODEL_IMAGE_KIND)?;
        let mut weights = fields.nested("weights")?;
        let format = weights.text("format")?;
        let file_values = weights.array("files")?;
        weights.finish()?;
        let mut files = Vec::with_capacity(file_values.len());
        for item in &file_values {
            files.push(ArtifactFileV1::from_cbor(item)?);
        }
        let tensor_values = fields.array("tensorInventory")?;
        let mut tensor_inventory = Vec::with_capacity(tensor_values.len());
        for item in &tensor_values {
            tensor_inventory.push(TensorSpecV1::from_cbor(item)?);
        }
        let source_values = fields.array("sources")?;
        let mut sources = Vec::with_capacity(source_values.len());
        for item in &source_values {
            sources.push(SourceHintV1::from_cbor(item)?);
        }
        let capability_values = fields.array("capabilities")?;
        let capabilities = crate::fields::text_array(&capability_values, "capabilities")?;
        let precision_values = fields.array("precisions")?;
        let precisions = crate::fields::text_array(&precision_values, "precisions")?;
        let out = Self {
            architecture: ArchitectureRefV1::from_cbor(&fields.require("architecture")?)?,
            capabilities,
            extensions: fields.extensions()?,
            files,
            format,
            generation_defaults: FixedPointSamplerV1::from_cbor(
                &fields.require("generationDefaults")?,
            )?,
            license: LicenseV1::from_cbor(&fields.require("license")?)?,
            name: fields.text("name")?,
            placement_hints: match fields.optional("placementHints")? {
                None => None,
                Some(value) => Some(PlacementHintsV1::from_cbor(&value)?),
            },
            precisions,
            requirements: ResourceRequirementsV1::from_cbor(&fields.require("requirements")?)?,
            signatures: signatures_from_cbor(&fields.array("signatures")?)?,
            sources,
            special_tokens: SpecialTokensV1::from_cbor(&fields.require("specialTokens")?)?,
            template: EmbeddedArtifactV1::from_cbor(&fields.require("template")?)?,
            tensor_inventory,
            tokenizer: EmbeddedArtifactV1::from_cbor(&fields.require("tokenizer")?)?,
            variant: fields.text("variant")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }

    pub fn content_cbor(&self) -> Result<CborValue, InferFailure> {
        let CborValue::Map(entries) = self.to_cbor()? else {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "model image must be a map",
            ));
        };
        let entries = entries
            .into_iter()
            .filter(|(key, _)| key != "signatures")
            .collect();
        Ok(CborValue::Map(entries))
    }

    pub fn image_root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-model-image", &self.content_cbor()?)
    }

    pub fn artifact_root(&self) -> Result<DigestHex, InferFailure> {
        artifact_root(&self.files)
    }

    pub fn config_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put("architecture", self.architecture.to_cbor()?);
        b.put("capabilities", string_field_array(&self.capabilities));
        b.put("generationDefaults", self.generation_defaults.to_cbor()?);
        b.put("precisions", string_field_array(&self.precisions));
        b.put("requirements", self.requirements.to_cbor());
        b.put("specialTokens", self.special_tokens.to_cbor()?);
        Ok(b.finish())
    }

    pub fn config_root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-config", &self.config_cbor()?)
    }

    pub fn architecture_adapter_root(&self) -> Result<DigestHex, InferFailure> {
        self.architecture.validate()?;
        let mut b = Builder::bare();
        b.put("adapter", cbor_text(&self.architecture.adapter));
        digest_value("infer-config", &b.finish())
    }

    pub fn special_tokens_root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-special-tokens", &self.special_tokens.to_cbor()?)
    }

    pub fn runtime_root(&self) -> Result<DigestHex, InferFailure> {
        let mut b = Builder::bare();
        b.put(
            "architectureAdapterRoot",
            cbor_text(self.architecture_adapter_root()?.as_str()),
        );
        b.put("artifactRoot", cbor_text(self.artifact_root()?.as_str()));
        b.put("configRoot", cbor_text(self.config_root()?.as_str()));
        b.put("modelImageRoot", cbor_text(self.image_root()?.as_str()));
        b.put("templateRoot", cbor_text(self.template.root.as_str()));
        b.put("tokenizerRoot", cbor_text(self.tokenizer.root.as_str()));
        digest_value("infer-model-runtime", &b.finish())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifactSetV1 {
    pub files: Vec<ArtifactFileV1>,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ModelArtifactSetV1 {
    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        validate_files(&self.files)?;
        let mut b = Builder::typed(MODEL_ARTIFACT_KIND);
        put_extensions(&mut b, &self.extensions)?;
        b.put("files", artifact_files_cbor(&self.files)?);
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, MODEL_ARTIFACT_KIND)?;
        let file_values = fields.array("files")?;
        let mut files = Vec::with_capacity(file_values.len());
        for item in &file_values {
            files.push(ArtifactFileV1::from_cbor(item)?);
        }
        let out = Self {
            extensions: fields.extensions()?,
            files,
        };
        fields.finish()?;
        validate_files(&out.files)?;
        Ok(out)
    }

    pub fn artifact_root(&self) -> Result<DigestHex, InferFailure> {
        artifact_root(&self.files)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
