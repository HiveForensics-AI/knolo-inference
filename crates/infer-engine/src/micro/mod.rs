//! `knolo.micro.v1`: a two-layer f32 transformer with fixed shapes.

mod math;
mod oracle;
mod synthetic;
mod weights;

use std::fs;
use std::path::Path;

use infer_artifact::{read_verified_tensors, verify_image};
use infer_contracts::{fail, ErrorCode, InferFailure, PlacementPlanV1, TensorGroupV1};

pub use synthetic::{
    render_conformance, write_synthetic_model, ConformanceCase, SyntheticModel, TEMPLATE_JINJA,
    TOKENIZER_JSON,
};
pub use weights::{
    decode_micro_weights, kv_width, micro_kv_bytes, micro_tensor_specs, synthetic_tensors,
    LayerWeights, MicroWeights, ADAPTER_ID, BLOCK_SIZE, FAMILY, HEADS, HEAD_DIM, HIDDEN,
    INTERMEDIATE, KV_HEADS, LAYERS, LOGIT_ABS_TOLERANCE, LOGIT_ABS_TOLERANCE_MILLIONTHS,
    MAX_CONTEXT, MIN_GREEDY_MARGIN, RMS_EPS, ROPE_THETA, SYNTHETIC_SEED, VOCAB,
};

use crate::traits::{ArchitectureAdapter, KvLayout, VerifiedWeightSource};

pub use oracle::MicroAdapter;

static MICRO_ADAPTER: MicroAdapter = MicroAdapter;

#[derive(Debug, Clone, Copy)]
pub struct MicroCase {
    pub name: &'static str,
    pub prompt: &'static [u32],
    pub new_tokens: u32,
}

pub const FIXTURE_CASES: &[MicroCase] = &[
    MicroCase {
        name: "three",
        prompt: &[1, 4, 7],
        new_tokens: 4,
    },
    MicroCase {
        name: "bos",
        prompt: &[1],
        new_tokens: 2,
    },
    MicroCase {
        name: "repeat",
        prompt: &[15, 15, 2, 3, 8],
        new_tokens: 3,
    },
];

pub fn adapter_by_id(id: &str) -> Result<&'static MicroAdapter, InferFailure> {
    if id == ADAPTER_ID {
        Ok(&MICRO_ADAPTER)
    } else {
        Err(fail(
            ErrorCode::UnsupportedArchitecture,
            format!("architecture adapter {id} is not compiled in"),
        ))
    }
}

pub fn micro_kv_layout() -> KvLayout {
    KvLayout {
        layers: LAYERS as u32,
        kv_heads: KV_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        block_size: BLOCK_SIZE,
        dtype: "f32",
    }
}

pub fn validate_micro_image(image: &infer_contracts::ModelImageV1) -> Result<(), InferFailure> {
    if image.architecture.adapter != ADAPTER_ID {
        return Err(fail(
            ErrorCode::UnsupportedArchitecture,
            format!(
                "architecture adapter {} is not compiled in",
                image.architecture.adapter
            ),
        ));
    }
    if image.architecture.family != FAMILY {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "knolo.micro.v1 requires architecture family micro",
        ));
    }
    if image.format != "safetensors" {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "knolo.micro.v1 reads safetensors f32 weights",
        ));
    }
    if image.precisions != ["f32".to_string()] {
        return Err(fail(
            ErrorCode::UnsupportedQuantization,
            "knolo.micro.v1 accepts f32 weights only",
        ));
    }
    if !image
        .capabilities
        .iter()
        .any(|capability| capability == "text-generation")
    {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "knolo.micro.v1 requires text-generation",
        ));
    }
    let expected = micro_tensor_specs();
    if image.tensor_inventory != expected {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "tensor inventory does not match knolo.micro.v1",
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
        if id as usize >= VOCAB {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                "special token id is outside the micro vocabulary",
            ));
        }
    }
    Ok(())
}

pub fn load_verified_micro(
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
    let adapter = adapter_by_id(&verification.image.architecture.adapter)?;
    adapter.validate_config(&verification.image)?;
    let tensors = read_verified_tensors(&verification.image, weights_dir)?;
    let weights = decode_micro_weights(&tensors)?;
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

pub fn cpu_placement(source: &VerifiedWeightSource) -> Result<PlacementPlanV1, InferFailure> {
    placement_bytes(source.runtime_root.clone(), source.weight_bytes, "cpu")
}

/// Same byte plan as [`cpu_placement`], on device `slot-0`.
pub fn cuda_placement(source: &VerifiedWeightSource) -> Result<PlacementPlanV1, InferFailure> {
    placement_bytes(source.runtime_root.clone(), source.weight_bytes, "slot-0")
}

pub fn cpu_placement_bytes(
    runtime_root: infer_contracts::DigestHex,
    weight_bytes: u64,
) -> Result<PlacementPlanV1, InferFailure> {
    placement_bytes(runtime_root, weight_bytes, "cpu")
}

/// Same byte plan as [`cpu_placement_bytes`], on device `slot-0`.
pub fn cuda_placement_bytes(
    runtime_root: infer_contracts::DigestHex,
    weight_bytes: u64,
) -> Result<PlacementPlanV1, InferFailure> {
    placement_bytes(runtime_root, weight_bytes, "slot-0")
}

fn placement_bytes(
    runtime_root: infer_contracts::DigestHex,
    weight_bytes: u64,
    device: &str,
) -> Result<PlacementPlanV1, InferFailure> {
    let workspace = 4096u64;
    let margin = 1024u64;
    let kv = crate::memory::planned_kv_bytes(&micro_kv_layout(), crate::paged::CPU_KV_PAGE_POOL)?;
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
            storage_precision: "f32".into(),
        }],
        context_reservation_tokens: MAX_CONTEXT,
        kv_block_size: BLOCK_SIZE,
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
        extensions: std::collections::BTreeMap::new(),
    })
}

pub fn accept_cpu_placement(
    source: &VerifiedWeightSource,
    placement: &PlacementPlanV1,
) -> Result<(), InferFailure> {
    accept_micro_placement(source, placement, "cpu")
}

pub fn accept_cuda_placement(
    source: &VerifiedWeightSource,
    placement: &PlacementPlanV1,
) -> Result<(), InferFailure> {
    accept_micro_placement(source, placement, "slot-0")
}

fn accept_micro_placement(
    source: &VerifiedWeightSource,
    placement: &PlacementPlanV1,
    device: &str,
) -> Result<(), InferFailure> {
    placement.validate()?;
    if placement.rejection_reason.is_some() {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "placement was already rejected",
        ));
    }
    if placement.devices != [device.to_string()] {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            match device {
                "cpu" => "knolo.micro.v1 runs on cpu only",
                _ => "knolo.micro.v1 cuda placement uses slot-0",
            },
        ));
    }
    if placement.model_runtime_root != source.runtime_root {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "placement runtime root does not match the model",
        ));
    }
    if placement.kv_precision != "f32" || placement.kv_block_size != BLOCK_SIZE {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "knolo.micro.v1 requires one f32 KV block of 16 tokens",
        ));
    }
    if placement.context_reservation_tokens > MAX_CONTEXT {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "context reservation exceeds the micro context",
        ));
    }
    if placement.graph_capture_mode != "off" {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            match device {
                "cpu" => "graph capture is off for the cpu reference",
                _ => "graph capture is off for the cuda placement",
            },
        ));
    }
    if placement.expected_weight_bytes != source.weight_bytes {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "placement weight bytes do not match the artifact",
        ));
    }
    if placement.expected_kv_bytes
        != crate::memory::planned_kv_bytes(&micro_kv_layout(), crate::paged::CPU_KV_PAGE_POOL)?
    {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "placement KV bytes do not match the page pool",
        ));
    }
    if placement.tensor_groups.iter().any(|group| {
        group.device != device
            || group.compute_precision != "f32"
            || group.storage_precision != "f32"
    }) {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            match device {
                "cpu" => "micro tensor groups must be cpu f32",
                _ => "micro tensor groups must be slot-0 f32",
            },
        ));
    }
    Ok(())
}
