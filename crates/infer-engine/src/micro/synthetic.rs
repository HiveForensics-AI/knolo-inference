//! Checked-in synthetic weights. The generator is the source; the files are its bytes.

use std::fs;
use std::path::Path;

use infer_artifact::{compile_manifest, encode_safetensors, write_model_image};
use infer_contracts::{fail, ErrorCode, InferFailure};
use serde_json::{json, Value};

use super::weights::{
    micro_tensor_specs, synthetic_tensors, FAMILY, LOGIT_ABS_TOLERANCE_MILLIONTHS, VOCAB,
};
use crate::traits::VerifiedWeightSource;

/// Sixteen entries. Index is the token id. The newline is a JSON `\n` escape.
pub const TOKENIZER_JSON: &str = "{\"kind\":\"knolo.micro.tokens.v1\",\"tokens\":[\"<pad>\",\"<bos>\",\"<eos>\",\"\\n\",\" \",\"a\",\"e\",\"h\",\"i\",\"n\",\"o\",\"r\",\"s\",\"t\",\"u\",\":\"],\"version\":1}\n";
/// One message loop. A trailing newline after `endfor` is literal text.
pub const TEMPLATE_JINJA: &str =
    "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}\n";

#[derive(Debug, Clone)]
pub struct SyntheticModel {
    pub image_root: infer_contracts::DigestHex,
    pub artifact_root: infer_contracts::DigestHex,
    pub runtime_root: infer_contracts::DigestHex,
}

pub fn write_synthetic_model(dir: &Path) -> Result<SyntheticModel, InferFailure> {
    fs::create_dir_all(dir).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("cannot create the synthetic model directory: {err}"),
        )
    })?;
    let tensors = synthetic_tensors()?;
    let bytes = encode_safetensors(&tensors)?;
    fs::write(dir.join("weights.safetensors"), bytes).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("cannot write synthetic weights: {err}"),
        )
    })?;
    fs::write(dir.join("tokenizer.json"), TOKENIZER_JSON).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("cannot write the synthetic tokenizer: {err}"),
        )
    })?;
    fs::write(dir.join("template.jinja"), TEMPLATE_JINJA).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("cannot write the synthetic template: {err}"),
        )
    })?;
    let manifest = manifest_json()?;
    fs::write(dir.join("manifest.json"), manifest).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("cannot write the synthetic manifest: {err}"),
        )
    })?;
    let compiled = compile_manifest(&dir.join("manifest.json"))?;
    write_model_image(&dir.join("micro.kmodel"), &compiled.bytes)?;
    Ok(SyntheticModel {
        image_root: compiled.image_root,
        artifact_root: compiled.artifact_root,
        runtime_root: compiled.runtime_root,
    })
}

fn manifest_json() -> Result<String, InferFailure> {
    let inventory: Vec<Value> = micro_tensor_specs()
        .into_iter()
        .map(|spec| {
            json!({
                "dtype": spec.dtype,
                "name": spec.name,
                "shape": spec.shape,
            })
        })
        .collect();
    let value = json!({
        "architecture": {"adapter": super::ADAPTER_ID, "family": FAMILY},
        "capabilities": ["text-generation"],
        "generationDefaults": {
            "frequencyPenaltyMicros": 0,
            "maxOutputTokens": 4,
            "minPMillionths": 0,
            "presencePenaltyMicros": 0,
            "repetitionPenaltyMicros": 1000000,
            "temperatureMicros": 0,
            "topK": 1,
            "topPMillionths": 1000000
        },
        "kind": "knolo.infer.model-image",
        "license": {"acceptanceRequired": false, "id": "synthetic"},
        "name": "knolo/micro",
        "precisions": ["f32"],
        "requirements": {"minimumRamBytes": 1048576, "minimumVramBytes": 0},
        "sources": [],
        "specialTokens": {"additional": {}, "bos": 1, "eos": 2},
        "template": {"embedded": "template.jinja"},
        "tensorInventory": inventory,
        "tokenizer": {"embedded": "tokenizer.json"},
        "variant": "f32",
        "version": 1,
        "weights": {
            "files": [{"path": "weights.safetensors", "sizeBytes": 0}],
            "format": "safetensors"
        }
    });
    serde_json::to_string_pretty(&value)
        .map(|text| format!("{text}\n"))
        .map_err(|_| {
            fail(
                ErrorCode::ModelImageInvalid,
                "synthetic manifest could not be encoded",
            )
        })
}

#[derive(Debug, Clone)]
pub struct ConformanceCase {
    pub name: String,
    pub prompt: Vec<u32>,
    pub new_tokens: u32,
    pub prefill_logits: Vec<f32>,
    pub greedy_tokens: Vec<u32>,
    pub margin: f32,
}

pub fn render_conformance(
    source: &VerifiedWeightSource,
    cases: &[ConformanceCase],
) -> Result<String, InferFailure> {
    let rendered_cases: Vec<Value> = cases
        .iter()
        .map(|case| {
            json!({
                "greedyTokens": case.greedy_tokens,
                "marginHex": format!("{:08x}", case.margin.to_bits()),
                "name": case.name,
                "newTokens": case.new_tokens,
                "prefillLogitsHex": case.prefill_logits.iter().map(|logit| format!("{:08x}", logit.to_bits())).collect::<Vec<_>>(),
                "prompt": case.prompt,
            })
        })
        .collect();
    let value = json!({
        "adapter": super::ADAPTER_ID,
        "artifactRoot": source.artifact_root.as_str(),
        "blockSize": super::BLOCK_SIZE,
        "cases": rendered_cases,
        "family": FAMILY,
        "logitAbsTolerance": "1e-4",
        "logitAbsToleranceMillionths": LOGIT_ABS_TOLERANCE_MILLIONTHS,
        "modelImageRoot": source.image_root.as_str(),
        "referenceBackend": "reference-f32",
        "runtimeRoot": source.runtime_root.as_str(),
        "supportLevel": "experimental",
        "tieBreak": "lowest-index",
        "vocab": VOCAB,
        "weightSha256": source.image.files.first().map(|file| file.sha256.as_str()).unwrap_or(""),
    });
    serde_json::to_string_pretty(&value)
        .map(|text| format!("{text}\n"))
        .map_err(|_| {
            fail(
                ErrorCode::ContractInvalid,
                "micro conformance document could not be encoded",
            )
        })
}
