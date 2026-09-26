//! Authoring manifests. JSON is strict. YAML is the subset in `yaml`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

use infer_contracts::{fail, ErrorCode, InferFailure, MODEL_IMAGE_KIND};

use crate::io::read_utf8_limited;
use crate::json::parse_strict_json;
use crate::yaml::parse_yaml_subset;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Authoring {
    pub kind: String,
    pub version: u32,
    pub name: String,
    pub variant: String,
    pub architecture: ArchitectureAuthoring,
    pub weights: WeightsAuthoring,
    pub tokenizer: EmbeddedAuthoring,
    pub template: EmbeddedAuthoring,
    pub special_tokens: SpecialAuthoring,
    pub generation_defaults: SamplerAuthoring,
    pub capabilities: Vec<String>,
    pub license: LicenseAuthoring,
    pub sources: Vec<SourceAuthoring>,
    pub tensor_inventory: Vec<TensorAuthoring>,
    pub precisions: Vec<String>,
    pub requirements: RequirementsAuthoring,
    #[serde(default)]
    pub placement_hints: Option<PlacementAuthoring>,
    #[serde(default)]
    pub extensions: BTreeMap<String, Value>,
    #[serde(default)]
    pub signatures: Vec<SignatureAuthoring>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchitectureAuthoring {
    pub family: String,
    pub adapter: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightsAuthoring {
    pub format: String,
    pub files: Vec<WeightFileAuthoring>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightFileAuthoring {
    pub path: String,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddedAuthoring {
    pub embedded: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialAuthoring {
    #[serde(default)]
    pub bos: Option<u32>,
    #[serde(default)]
    pub eos: Option<u32>,
    #[serde(default)]
    pub pad: Option<u32>,
    #[serde(default)]
    pub unk: Option<u32>,
    #[serde(default)]
    pub additional: BTreeMap<String, u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SamplerAuthoring {
    pub temperature_micros: u32,
    pub top_p_millionths: u32,
    pub min_p_millionths: u32,
    pub repetition_penalty_micros: u32,
    pub presence_penalty_micros: u32,
    pub frequency_penalty_micros: u32,
    pub top_k: u32,
    pub max_output_tokens: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LicenseAuthoring {
    pub id: String,
    pub acceptance_required: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceAuthoring {
    pub provider: String,
    pub repository: String,
    pub revision: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorAuthoring {
    pub name: String,
    pub shape: Vec<u32>,
    pub dtype: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RequirementsAuthoring {
    pub minimum_ram_bytes: u64,
    pub minimum_vram_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlacementAuthoring {
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub prefer_device: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SignatureAuthoring {
    pub algorithm: String,
    pub key_id: String,
    pub signature: String,
}

pub struct LoadedAuthoring {
    pub manifest: Authoring,
    pub base: PathBuf,
}

pub fn load_manifest(path: &Path) -> Result<LoadedAuthoring, InferFailure> {
    let text = read_utf8_limited(path, ErrorCode::ModelImageInvalid)?;
    let value = match path.extension().and_then(|ext| ext.to_str()) {
        Some("json") => parse_strict_json(&text, ErrorCode::ModelImageInvalid)?,
        Some("yaml" | "yml") => parse_yaml_subset(&text)?,
        _ => {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                "manifest must be .json, .yaml, or .yml",
            ))
        }
    };
    let manifest: Authoring = serde_json::from_value(value).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("authoring manifest: {err}"),
        )
    })?;
    if manifest.kind != MODEL_IMAGE_KIND || manifest.version != 1 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "authoring kind and version must be knolo.infer.model-image 1",
        ));
    }
    if manifest.weights.format != "safetensors" {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "this milestone compiles safetensors manifests only",
        ));
    }
    let base = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    };
    Ok(LoadedAuthoring { manifest, base })
}
