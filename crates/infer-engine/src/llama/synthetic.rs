//! Checked-in synthetic Llama weights. The generator is the source.

use std::fs;
use std::path::Path;

use infer_artifact::{
    compile_manifest, encode_gguf, encode_safetensors, write_model_image, GgufMetadata,
    GgufTensorDraft, GgufTensorType, GgufValue,
};
use infer_contracts::{fail, ErrorCode, InferFailure};
use serde_json::{json, Value};

use super::weights::{llama_tensor_specs, synthetic_tensors, FAMILY, VOCAB};
use crate::micro::ConformanceCase;

pub const TOKENIZER_JSON: &str = "{\"kind\":\"knolo.llama.tokens.v1\",\"tokens\":[\"<pad>\",\"<bos>\",\"<eos>\",\"\\n\",\" \",\"a\",\"e\",\"h\",\"i\",\"n\",\"o\",\"r\",\"s\",\"t\",\"u\",\":\",\"b\",\"c\",\"d\",\"f\",\"g\",\"j\",\"k\",\"l\",\"m\",\"p\",\"q\",\"v\",\"w\",\"x\",\"y\",\"z\"],\"version\":1}\n";

pub const TEMPLATE_JINJA: &str =
    "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}\n";

#[derive(Debug, Clone)]
pub struct LlamaModelFiles {
    pub image_root: infer_contracts::DigestHex,
    pub artifact_root: infer_contracts::DigestHex,
    pub runtime_root: infer_contracts::DigestHex,
}

pub fn write_llama_model(dir: &Path) -> Result<LlamaModelFiles, InferFailure> {
    write_precision(dir, "f32", "llama.kmodel")
}

pub fn write_llama_precision(
    dir: &Path,
    dtype: &str,
    image_name: &str,
) -> Result<LlamaModelFiles, InferFailure> {
    write_precision(dir, dtype, image_name)
}

/// Write one GGUF image of the synthetic fixture. The container architecture
/// string is not the adapter id. F32 and F16 payloads are the safetensors bytes.
pub fn write_llama_gguf(
    dir: &Path,
    precision: &str,
    image_name: &str,
) -> Result<LlamaModelFiles, InferFailure> {
    let tensor_type = match precision {
        "f32" => GgufTensorType::F32,
        "f16" => GgufTensorType::F16,
        _ => {
            return Err(fail(
                ErrorCode::UnsupportedQuantization,
                "the llama-tiny gguf fixture is f32 or f16",
            ))
        }
    };
    fs::create_dir_all(dir).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("cannot create the llama model directory: {err}"),
        )
    })?;
    let stored = synthetic_tensors(precision)?;
    let drafts = stored
        .into_iter()
        .map(|tensor| GgufTensorDraft {
            name: tensor.name,
            tensor_type,
            shape: tensor.shape.into_iter().map(u64::from).collect(),
            bytes: tensor.bytes,
        })
        .collect::<Vec<_>>();
    let bytes = encode_gguf(
        &[GgufMetadata {
            key: "general.architecture".into(),
            value: GgufValue::String("ignored".into()),
        }],
        &drafts,
    )?;
    fs::write(dir.join("weights.gguf"), bytes).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("cannot write llama gguf weights: {err}"),
        )
    })?;
    fs::write(dir.join("tokenizer.json"), TOKENIZER_JSON).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("cannot write the llama tokenizer: {err}"),
        )
    })?;
    fs::write(dir.join("template.jinja"), TEMPLATE_JINJA).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("cannot write the llama template: {err}"),
        )
    })?;
    let manifest = manifest_json(precision, "weights.gguf")?;
    let manifest = manifest.replace("\"format\": \"safetensors\"", "\"format\": \"gguf\"");
    fs::write(dir.join("manifest-gguf.json"), manifest).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("cannot write the llama manifest: {err}"),
        )
    })?;
    let compiled = compile_manifest(&dir.join("manifest-gguf.json"))?;
    write_model_image(&dir.join(image_name), &compiled.bytes)?;
    Ok(LlamaModelFiles {
        image_root: compiled.image_root,
        artifact_root: compiled.artifact_root,
        runtime_root: compiled.runtime_root,
    })
}

fn write_precision(
    dir: &Path,
    dtype: &str,
    image_name: &str,
) -> Result<LlamaModelFiles, InferFailure> {
    fs::create_dir_all(dir).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("cannot create the llama model directory: {err}"),
        )
    })?;
    let tensors = synthetic_tensors(dtype)?;
    let weight_name = match dtype {
        "f32" => "weights.safetensors",
        "f16" => "weights-f16.safetensors",
        "bf16" => "weights-bf16.safetensors",
        _ => {
            return Err(fail(
                ErrorCode::UnsupportedQuantization,
                "llama synthetic dtype must be f32, f16, or bf16",
            ))
        }
    };
    let bytes = encode_safetensors(&tensors)?;
    fs::write(dir.join(weight_name), bytes).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("cannot write llama weights: {err}"),
        )
    })?;
    fs::write(dir.join("tokenizer.json"), TOKENIZER_JSON).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("cannot write the llama tokenizer: {err}"),
        )
    })?;
    fs::write(dir.join("template.jinja"), TEMPLATE_JINJA).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("cannot write the llama template: {err}"),
        )
    })?;
    let manifest = manifest_json(dtype, weight_name)?;
    let manifest_name = match dtype {
        "f32" => "manifest.json",
        "f16" => "manifest-f16.json",
        "bf16" => "manifest-bf16.json",
        _ => "manifest.json",
    };
    fs::write(dir.join(manifest_name), manifest).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("cannot write the llama manifest: {err}"),
        )
    })?;
    let compiled = compile_manifest(&dir.join(manifest_name))?;
    write_model_image(&dir.join(image_name), &compiled.bytes)?;
    Ok(LlamaModelFiles {
        image_root: compiled.image_root,
        artifact_root: compiled.artifact_root,
        runtime_root: compiled.runtime_root,
    })
}

fn manifest_json(dtype: &str, weight_name: &str) -> Result<String, InferFailure> {
    let inventory: Vec<Value> = llama_tensor_specs(dtype)
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
        "architecture": {"adapter": super::weights::ADAPTER_ID, "family": FAMILY},
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
        "name": "knolo/llama-tiny",
        "precisions": [dtype],
        "requirements": {"minimumRamBytes": 1048576, "minimumVramBytes": 0},
        "sources": [],
        "specialTokens": {"additional": {}, "bos": 1, "eos": 2},
        "template": {"embedded": "template.jinja"},
        "tensorInventory": inventory,
        "tokenizer": {"embedded": "tokenizer.json"},
        "variant": dtype,
        "version": 1,
        "weights": {
            "files": [{"path": weight_name, "sizeBytes": 0}],
            "format": "safetensors"
        }
    });
    serde_json::to_string_pretty(&value)
        .map(|text| format!("{text}\n"))
        .map_err(|_| {
            fail(
                ErrorCode::ModelImageInvalid,
                "llama manifest could not be encoded",
            )
        })
}

pub fn render_llama_conformance(
    source: &crate::traits::VerifiedWeightSource,
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
        "adapter": super::weights::ADAPTER_ID,
        "artifactRoot": source.artifact_root.as_str(),
        "blockSize": super::weights::BLOCK_SIZE,
        "cases": rendered_cases,
        "family": FAMILY,
        "logitAbsTolerance": "1e-4",
        "modelImageRoot": source.image_root.as_str(),
        "referenceBackend": "reference-f32",
        "runtimeRoot": source.runtime_root.as_str(),
        "tieBreak": "lowest-index",
        "vocab": VOCAB,
        "weightSha256": source.image.files.first().map(|file| file.sha256.as_str()).unwrap_or(""),
    });
    serde_json::to_string_pretty(&value)
        .map(|text| format!("{text}\n"))
        .map_err(|_| {
            fail(
                ErrorCode::ContractInvalid,
                "llama conformance document could not be encoded",
            )
        })
}
