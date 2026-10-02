//! `knolo.llama.v1`: a two-layer transformer with a 32-token vocabulary.
//! Hidden size, heads, and the KV block match the micro fixture so the
//! existing page pool holds the cache. Weights may be stored as safetensors
//! f32, f16, or bf16, or as one allowlisted GGUF precision, and are decoded
//! to f32 before the forward.

mod oracle;
mod synthetic;
mod weights;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use infer_artifact::{read_gguf_tensors, read_verified_tensors, verify_image};
use infer_contracts::{
    fail, ErrorCode, InferFailure, ModelImageV1, PlacementPlanV1, TensorGroupV1,
};

use crate::micro::load_verified_micro;
use crate::traits::{KvLayout, VerifiedWeightSource};

pub use oracle::LlamaAdapter;
pub use synthetic::{
    render_llama_conformance, write_llama_gguf, write_llama_model, write_llama_precision,
    LlamaModelFiles, TEMPLATE_JINJA as LLAMA_TEMPLATE_JINJA,
    TOKENIZER_JSON as LLAMA_TOKENIZER_JSON,
};
pub use weights::{
    ADAPTER_ID as LLAMA_ADAPTER_ID, BLOCK_SIZE as LLAMA_BLOCK_SIZE, VOCAB as LLAMA_VOCAB,
};

static LLAMA_ADAPTER: LlamaAdapter = LlamaAdapter;

pub fn llama_adapter() -> &'static LlamaAdapter {
    &LLAMA_ADAPTER
}

pub const LLAMA_FIXTURE_CASES: &[crate::micro::MicroCase] = &[
    crate::micro::MicroCase {
        name: "three",
        prompt: &[1, 4, 7],
        new_tokens: 4,
    },
    crate::micro::MicroCase {
        name: "bos",
        prompt: &[1],
        new_tokens: 2,
    },
    crate::micro::MicroCase {
        name: "repeat",
        prompt: &[17, 19, 2, 3, 8],
        new_tokens: 3,
    },
];

pub fn adapter_vocab(id: &str) -> Result<u32, InferFailure> {
    if id == crate::micro::ADAPTER_ID {
        Ok(crate::micro::VOCAB as u32)
    } else if id == LLAMA_ADAPTER_ID {
        Ok(LLAMA_VOCAB as u32)
    } else {
        Err(fail(
            ErrorCode::UnsupportedArchitecture,
            format!("architecture adapter {id} is not compiled in"),
        ))
    }
}

pub fn llama_kv_layout() -> KvLayout {
    KvLayout {
        layers: weights::LAYERS as u32,
        kv_heads: weights::KV_HEADS as u32,
        head_dim: weights::HEAD_DIM as u32,
        block_size: weights::BLOCK_SIZE,
        dtype: "f32",
    }
}

pub fn validate_llama_image(image: &ModelImageV1) -> Result<(), InferFailure> {
    if image.architecture.adapter != LLAMA_ADAPTER_ID {
        return Err(fail(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "architecture adapter {} is not compiled in",
                image.architecture.adapter
            ),
        ));
    }
    if image.architecture.family != weights::FAMILY {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "knolo.llama.v1 requires architecture family llama",
        ));
    }
    let dtype = llama_declared_storage(image)?;
    if !image
        .capabilities
        .iter()
        .any(|capability| capability == "text-generation")
    {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "knolo.llama.v1 requires text-generation",
        ));
    }
    if image.tensor_inventory != weights::llama_tensor_specs(dtype) {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "tensor inventory does not match knolo.llama.v1",
        ));
    }
    check_token(image.special_tokens.bos)?;
    check_token(image.special_tokens.eos)?;
    check_token(image.special_tokens.pad)?;
    check_token(image.special_tokens.unk)?;
    for id in image.special_tokens.additional.values() {
        check_token(Some(*id))?;
    }
    Ok(())
}

fn check_token(id: Option<u32>) -> Result<(), InferFailure> {
    if let Some(id) = id {
        if id as usize >= weights::VOCAB {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                "special token id is outside the llama vocabulary",
            ));
        }
    }
    Ok(())
}

pub fn load_verified_model(
    kmodel: &Path,
    weights_dir: &Path,
) -> Result<VerifiedWeightSource, InferFailure> {
    let bytes = fs::read(kmodel).map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            fail(ErrorCode::ModelArtifactMissing, "model image is missing")
        } else {
            fail(
                ErrorCode::ModelImageInvalid,
                format!("model image could not be read: {err}"),
            )
        }
    })?;
    let verification = verify_image(&bytes)?;
    match verification.image.architecture.adapter.as_str() {
        crate::micro::ADAPTER_ID => load_verified_micro(kmodel, weights_dir),
        LLAMA_ADAPTER_ID => load_verified_llama_image(kmodel, weights_dir, verification),
        other => Err(fail(
            ErrorCode::UnsupportedArchitecture,
            format!("architecture adapter {other} is not compiled in"),
        )),
    }
}

pub fn load_verified_llama(
    kmodel: &Path,
    weights_dir: &Path,
) -> Result<VerifiedWeightSource, InferFailure> {
    let bytes = fs::read(kmodel).map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            fail(ErrorCode::ModelArtifactMissing, "model image is missing")
        } else {
            fail(
                ErrorCode::ModelImageInvalid,
                format!("model image could not be read: {err}"),
            )
        }
    })?;
    let verification = verify_image(&bytes)?;
    if verification.image.architecture.adapter != LLAMA_ADAPTER_ID {
        return Err(fail(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "architecture adapter {} is not compiled in",
                verification.image.architecture.adapter
            ),
        ));
    }
    load_verified_llama_image(kmodel, weights_dir, verification)
}

fn load_verified_llama_image(
    _kmodel: &Path,
    weights_dir: &Path,
    verification: infer_artifact::Verification,
) -> Result<VerifiedWeightSource, InferFailure> {
    validate_llama_image(&verification.image)?;
    let tensors = if verification.image.format == "gguf" {
        read_gguf_tensors(&verification.image, weights_dir)?
    } else {
        read_verified_tensors(&verification.image, weights_dir)?
    };
    let weights = weights::decode_llama_weights(&tensors)?;
    let mut weight_bytes = 0u64;
    for file in &verification.image.files {
        weight_bytes = weight_bytes
            .checked_add(file.size_bytes)
            .ok_or_else(|| fail(ErrorCode::ModelImageInvalid, "weight byte total overflows"))?;
    }
    Ok(VerifiedWeightSource {
        image_root: verification.image_root,
        artifact_root: verification.artifact_root,
        runtime_root: verification.runtime_root,
        image: verification.image,
        tensors,
        weights,
        weight_bytes,
    })
}

pub fn llama_cpu_placement(source: &VerifiedWeightSource) -> Result<PlacementPlanV1, InferFailure> {
    let storage = storage_of(source)?;
    llama_placement_bytes(
        source.runtime_root.clone(),
        source.weight_bytes,
        "cpu",
        storage,
    )
}

pub fn llama_cuda_placement(
    source: &VerifiedWeightSource,
) -> Result<PlacementPlanV1, InferFailure> {
    let storage = storage_of(source)?;
    llama_placement_bytes(
        source.runtime_root.clone(),
        source.weight_bytes,
        "slot-0",
        storage,
    )
}

pub fn llama_declared_storage(image: &ModelImageV1) -> Result<&str, InferFailure> {
    let allowed: &[&str] = match image.format.as_str() {
        "safetensors" => &["bf16", "f16", "f32"],
        "gguf" => &["f16", "f32", "q4_k_m", "q5_k_m", "q6_k", "q8_0"],
        _ => {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                "knolo.llama.v1 reads safetensors or gguf weights",
            ))
        }
    };
    match image.precisions.as_slice() {
        [one] if allowed.contains(&one.as_str()) => Ok(one.as_str()),
        _ => Err(fail(
            ErrorCode::UnsupportedQuantization,
            "knolo.llama.v1 weight precision is not allowlisted",
        )),
    }
}

fn storage_of(source: &VerifiedWeightSource) -> Result<&str, InferFailure> {
    llama_declared_storage(&source.image)
}

pub fn llama_placement_bytes(
    runtime_root: infer_contracts::DigestHex,
    weight_bytes: u64,
    device: &str,
    storage: &str,
) -> Result<PlacementPlanV1, InferFailure> {
    let workspace = 4096u64;
    let margin = 1024u64;
    let kv = crate::memory::planned_kv_bytes(&llama_kv_layout(), crate::paged::CPU_KV_PAGE_POOL)?;
    let total = weight_bytes
        .checked_add(kv)
        .and_then(|sum| sum.checked_add(workspace))
        .and_then(|sum| sum.checked_add(margin))
        .ok_or_else(|| {
            fail(
                ErrorCode::PlacementUnsatisfiable,
                "placement byte total overflows",
            )
        })?;
    Ok(PlacementPlanV1 {
        model_runtime_root: runtime_root,
        devices: vec![device.into()],
        tensor_groups: vec![TensorGroupV1 {
            name: "weights".into(),
            device: device.into(),
            compute_precision: "f32".into(),
            storage_precision: storage.into(),
        }],
        context_reservation_tokens: weights::MAX_CONTEXT,
        kv_block_size: weights::BLOCK_SIZE,
        kv_precision: "f32".into(),
        workspace_bytes: workspace,
        graph_capture_mode: "off".into(),
        safety_margin_bytes: margin,
        expected_weight_bytes: weight_bytes,
        expected_kv_bytes: kv,
        expected_workspace_bytes: workspace,
        expected_staging_bytes: 0,
        expected_overhead_bytes: 0,
        expected_total_bytes: total,
        rejection_reason: None,
        extensions: BTreeMap::new(),
    })
}

pub fn accept_llama_placement(
    source: &VerifiedWeightSource,
    placement: &PlacementPlanV1,
) -> Result<(), InferFailure> {
    placement.validate()?;
    if source.image.architecture.adapter != LLAMA_ADAPTER_ID {
        return Err(fail(
            ErrorCode::UnsupportedArchitecture,
            "placement is for knolo.llama.v1",
        ));
    }
    if placement.rejection_reason.is_some() {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "placement was already rejected",
        ));
    }
    let device = placement.devices.first().map(String::as_str).unwrap_or("");
    if placement.devices.len() != 1 || (device != "cpu" && device != "slot-0") {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "knolo.llama.v1 runs on cpu or slot-0",
        ));
    }
    if placement.model_runtime_root != source.runtime_root {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "placement runtime root does not match the model",
        ));
    }
    if placement.kv_precision != "f32" || placement.kv_block_size != weights::BLOCK_SIZE {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "knolo.llama.v1 requires one f32 KV block of 16 tokens",
        ));
    }
    if placement.context_reservation_tokens > weights::MAX_CONTEXT {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "context reservation exceeds the llama context",
        ));
    }
    if placement.graph_capture_mode != "off" {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "graph capture is off for the llama placement",
        ));
    }
    if placement.expected_weight_bytes != source.weight_bytes {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "placement weight bytes do not match the artifact",
        ));
    }
    if placement.expected_kv_bytes
        != crate::memory::planned_kv_bytes(&llama_kv_layout(), crate::paged::CPU_KV_PAGE_POOL)?
    {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "placement KV bytes do not match the page pool",
        ));
    }
    let storage = storage_of(source)?;
    if placement.tensor_groups.iter().any(|group| {
        group.device != device
            || group.compute_precision != "f32"
            || group.storage_precision != storage
    }) {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "llama tensor groups must use f32 compute and the weight storage precision",
        ));
    }
    Ok(())
}
