//! Build a canonical model image from an authoring manifest and local files.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, ArchitectureRefV1, ArtifactFileV1, CborValue, EmbeddedArtifactV1, ErrorCode,
    FixedPointSamplerV1, InferFailure, LicenseV1, ModelImageV1, PlacementHintsV1,
    ResourceRequirementsV1, SignatureV1, SourceHintV1, SpecialTokensV1, TensorSpecV1,
    MAX_EMBEDDED_BYTES,
};

use crate::authoring::{load_manifest, Authoring, WeightFileAuthoring};
use crate::io::{hash_regular_file, read_bytes_limited, write_atomic};
use crate::map_image;
use crate::paths::{check_relative_posix, resolve_inside};
use crate::verify::{verify_image, verify_weights};
use infer_contracts::decode_hex;

#[derive(Debug)]
pub struct CompiledModel {
    pub bytes: Vec<u8>,
    pub image_root: infer_contracts::DigestHex,
    pub artifact_root: infer_contracts::DigestHex,
    pub runtime_root: infer_contracts::DigestHex,
}

pub fn compile_manifest(path: &Path) -> Result<CompiledModel, InferFailure> {
    let loaded = load_manifest(path)?;
    let image = build_image(&loaded.manifest, &loaded.base)?;
    let bytes = image.to_bytes().map_err(map_image)?;
    let verification = verify_image(&bytes)?;
    verify_weights(&verification.image, &loaded.base)?;
    Ok(CompiledModel {
        bytes,
        image_root: verification.image_root,
        artifact_root: verification.artifact_root,
        runtime_root: verification.runtime_root,
    })
}

pub fn write_model_image(path: &Path, bytes: &[u8]) -> Result<(), InferFailure> {
    write_atomic(path, bytes, ErrorCode::ModelImageInvalid)
}

fn build_image(manifest: &Authoring, base: &Path) -> Result<ModelImageV1, InferFailure> {
    if manifest.weights.files.is_empty() || manifest.weights.files.len() > 1024 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "weight file list is empty or too large",
        ));
    }
    if manifest.tensor_inventory.is_empty() || manifest.tensor_inventory.len() > 100_000 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "tensor inventory is empty or too large",
        ));
    }
    let mut seen_paths: Vec<String> = manifest
        .weights
        .files
        .iter()
        .map(|file| file.path.clone())
        .collect();
    for path in &seen_paths {
        check_relative_posix(path, ErrorCode::ModelImageInvalid)?;
    }
    sort_unique_text("weight path", &mut seen_paths)?;
    let mut files = Vec::with_capacity(manifest.weights.files.len());
    for file in &manifest.weights.files {
        files.push(load_weight(base, file)?);
    }
    files.sort_by(|left, right| left.path.as_bytes().cmp(right.path.as_bytes()));

    let mut capabilities = manifest.capabilities.clone();
    sort_unique_text("capability", &mut capabilities)?;
    let mut precisions = manifest.precisions.clone();
    sort_unique_text("precision", &mut precisions)?;

    let mut tensor_inventory = Vec::with_capacity(manifest.tensor_inventory.len());
    for tensor in &manifest.tensor_inventory {
        tensor_inventory.push(TensorSpecV1 {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            dtype: tensor.dtype.clone(),
        });
    }
    tensor_inventory.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
    let mut names: Vec<String> = tensor_inventory
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect();
    sort_unique_text("tensor", &mut names)?;

    let mut sources = Vec::with_capacity(manifest.sources.len());
    for source in &manifest.sources {
        sources.push(SourceHintV1 {
            provider: source.provider.clone(),
            repository: source.repository.clone(),
            revision: source.revision.clone(),
        });
    }
    let mut signatures = Vec::with_capacity(manifest.signatures.len());
    for signature in &manifest.signatures {
        signatures.push(SignatureV1 {
            algorithm: signature.algorithm.clone(),
            key_id: signature.key_id.clone(),
            signature: decode_hex(&signature.signature).map_err(map_image)?,
        });
    }
    let placement_hints = manifest
        .placement_hints
        .as_ref()
        .map(|hints| PlacementHintsV1 {
            profile: hints.profile.clone(),
            prefer_device: hints.prefer_device.clone(),
        });
    let mut extensions = BTreeMap::new();
    for (key, value) in &manifest.extensions {
        extensions.insert(key.clone(), json_to_cbor(value, 0)?);
    }

    Ok(ModelImageV1 {
        name: manifest.name.clone(),
        variant: manifest.variant.clone(),
        architecture: ArchitectureRefV1 {
            family: manifest.architecture.family.clone(),
            adapter: manifest.architecture.adapter.clone(),
        },
        format: manifest.weights.format.clone(),
        files,
        tokenizer: read_embedded(
            base,
            &manifest.tokenizer.embedded,
            "infer-tokenizer",
            ErrorCode::TokenizerInvalid,
        )?,
        template: read_embedded(
            base,
            &manifest.template.embedded,
            "infer-template",
            ErrorCode::TemplateInvalid,
        )?,
        special_tokens: SpecialTokensV1 {
            bos: manifest.special_tokens.bos,
            eos: manifest.special_tokens.eos,
            pad: manifest.special_tokens.pad,
            unk: manifest.special_tokens.unk,
            additional: manifest.special_tokens.additional.clone(),
        },
        generation_defaults: FixedPointSamplerV1 {
            temperature_micros: manifest.generation_defaults.temperature_micros,
            top_p_millionths: manifest.generation_defaults.top_p_millionths,
            min_p_millionths: manifest.generation_defaults.min_p_millionths,
            repetition_penalty_micros: manifest.generation_defaults.repetition_penalty_micros,
            presence_penalty_micros: manifest.generation_defaults.presence_penalty_micros,
            frequency_penalty_micros: manifest.generation_defaults.frequency_penalty_micros,
            top_k: manifest.generation_defaults.top_k,
            max_output_tokens: manifest.generation_defaults.max_output_tokens,
        },
        capabilities,
        license: LicenseV1 {
            id: manifest.license.id.clone(),
            acceptance_required: manifest.license.acceptance_required,
        },
        sources,
        tensor_inventory,
        precisions,
        requirements: ResourceRequirementsV1 {
            minimum_ram_bytes: manifest.requirements.minimum_ram_bytes,
            minimum_vram_bytes: manifest.requirements.minimum_vram_bytes,
        },
        placement_hints,
        extensions,
        signatures,
    })
}

fn load_weight(base: &Path, file: &WeightFileAuthoring) -> Result<ArtifactFileV1, InferFailure> {
    let path = resolve_inside(
        base,
        &file.path,
        ErrorCode::ModelImageInvalid,
        ErrorCode::ModelArtifactMissing,
    )?;
    let hashed = hash_regular_file(&path, ErrorCode::ModelImageInvalid)?;
    if hashed.size == 0 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            format!("weight file is empty: {}", file.path),
        ));
    }
    // size 0 in authoring means "fill the measured size". A non-zero claim must match.
    if let Some(size) = file.size_bytes {
        if size != 0 && size != hashed.size {
            return Err(fail(
                ErrorCode::ModelDigestMismatch,
                format!("declared size does not match {}", file.path),
            ));
        }
    }
    if let Some(digest) = &file.sha256 {
        let parsed = infer_contracts::DigestHex::parse(digest)?;
        if parsed != hashed.sha256 {
            return Err(fail(
                ErrorCode::ModelDigestMismatch,
                format!("declared digest does not match {}", file.path),
            ));
        }
    }
    Ok(ArtifactFileV1 {
        path: file.path.clone(),
        size_bytes: hashed.size,
        sha256: hashed.sha256,
    })
}

fn read_embedded(
    base: &Path,
    relative: &str,
    domain: &str,
    code: ErrorCode,
) -> Result<EmbeddedArtifactV1, InferFailure> {
    let path = resolve_inside(base, relative, code, code)?;
    let bytes = read_bytes_limited(&path, MAX_EMBEDDED_BYTES as u64, code)?;
    EmbeddedArtifactV1::new(domain, bytes).map_err(|err| {
        if err.code == ErrorCode::ContractInvalid {
            fail(code, err.message)
        } else {
            err
        }
    })
}

fn sort_unique_text(label: &str, values: &mut [String]) -> Result<(), InferFailure> {
    values.sort();
    for pair in values.windows(2) {
        if pair[0] == pair[1] {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                format!("duplicate {label}"),
            ));
        }
    }
    Ok(())
}

fn json_to_cbor(value: &serde_json::Value, depth: usize) -> Result<CborValue, InferFailure> {
    if depth > 32 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "extension nesting exceeds the depth limit",
        ));
    }
    match value {
        serde_json::Value::Null => Err(fail(
            ErrorCode::ModelImageInvalid,
            "extensions cannot contain null",
        )),
        serde_json::Value::Bool(item) => Ok(CborValue::Bool(*item)),
        serde_json::Value::Number(number) => {
            if let Some(value) = number.as_u64() {
                Ok(CborValue::Integer(i128::from(value)))
            } else if let Some(value) = number.as_i64() {
                Ok(CborValue::Integer(i128::from(value)))
            } else {
                Err(fail(
                    ErrorCode::ModelImageInvalid,
                    "extensions cannot contain floats",
                ))
            }
        }
        serde_json::Value::String(text) => {
            if text.len() > 1024 * 1024 {
                return Err(fail(
                    ErrorCode::ModelImageInvalid,
                    "extension string is too long",
                ));
            }
            Ok(CborValue::Text(text.clone()))
        }
        serde_json::Value::Array(items) => {
            if items.len() > 1_048_576 {
                return Err(fail(
                    ErrorCode::ModelImageInvalid,
                    "extension array is too large",
                ));
            }
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(json_to_cbor(item, depth + 1)?);
            }
            Ok(CborValue::Array(out))
        }
        serde_json::Value::Object(map) => {
            if map.len() > 1_048_576 {
                return Err(fail(
                    ErrorCode::ModelImageInvalid,
                    "extension object is too large",
                ));
            }
            let mut entries = BTreeMap::new();
            for (key, item) in map {
                entries.insert(key.clone(), json_to_cbor(item, depth + 1)?);
            }
            Ok(CborValue::map(entries))
        }
    }
}
