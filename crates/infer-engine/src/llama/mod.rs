//! `knolo.llama.v1`: a two-layer transformer with a 32-token vocabulary.
//! Hidden size, heads, and the KV block match the micro fixture so the
//! existing page pool holds the cache. Weights may be stored as safetensors
//! f32, f16, or bf16, or as one allowlisted GGUF precision, and are decoded
//! to f32 before the forward.

mod forward;
mod inventory;
mod oracle;
mod synthetic;
mod weights;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use infer_artifact::{read_llama_gguf, read_verified_tensors, verify_image};
use infer_contracts::{
    fail, CborValue, ErrorCode, InferFailure, ModelImageV1, PlacementPlanV1, TensorGroupV1,
};

use crate::llama_shape::{LlamaShape, RUN_KV_POOL_BYTES};
use crate::micro::{load_verified_micro, MicroWeights};
use crate::paged::{PagedKv, CPU_KV_PAGE_POOL};
use crate::traits::{KvLayout, VerifiedWeightSource};

pub use oracle::{LlamaAdapter, LlamaOracle};
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
    let _storage = llama_declared_storage(image)?;
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
    let vocab = if is_toy_inventory(image) {
        weights::VOCAB
    } else {
        inventory::parse_inventory(&image.tensor_inventory)?.vocab
    };
    check_token(image.special_tokens.bos, vocab)?;
    check_token(image.special_tokens.eos, vocab)?;
    check_token(image.special_tokens.pad, vocab)?;
    check_token(image.special_tokens.unk, vocab)?;
    for id in image.special_tokens.additional.values() {
        check_token(Some(*id), vocab)?;
    }
    Ok(())
}

fn is_toy_inventory(image: &ModelImageV1) -> bool {
    let Some(dtype) = single_storage(image) else {
        return false;
    };
    image.tensor_inventory == weights::llama_tensor_specs(dtype)
}

fn check_token(id: Option<u32>, vocab: usize) -> Result<(), InferFailure> {
    if let Some(id) = id {
        if id as usize >= vocab {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                "special token id is outside the llama vocabulary",
            ));
        }
    }
    Ok(())
}

/// Vocabulary and context the prompt compiler should use for this image.
///
/// The toy fixture stays at 32 and 16. A wider Llama reads its vocabulary
/// from the embedding and its context from `knolo.llama.context` when that
/// extension is present.
pub fn prompt_bounds(image: &ModelImageV1) -> Result<(u32, u32), InferFailure> {
    if image.architecture.adapter == crate::micro::ADAPTER_ID {
        return Ok((crate::micro::VOCAB as u32, crate::micro::MAX_CONTEXT));
    }
    if image.architecture.adapter != LLAMA_ADAPTER_ID {
        return adapter_vocab(&image.architecture.adapter).map(|vocab| (vocab, 0));
    }
    validate_llama_image(image)?;
    if is_toy_inventory(image) {
        return Ok((weights::VOCAB as u32, weights::MAX_CONTEXT));
    }
    let vocab = inventory::parse_inventory(&image.tensor_inventory)?.vocab as u32;
    let context = match image.extensions.get("knolo.llama.context") {
        Some(CborValue::Integer(value))
            if *value > 0 && *value <= i128::from(crate::llama_shape::LLAMA_RUN_CONTEXT) =>
        {
            let context = u32::try_from(*value).unwrap_or(0);
            if context < weights::BLOCK_SIZE || context % weights::BLOCK_SIZE != 0 {
                return Err(fail(
                    ErrorCode::ModelImageInvalid,
                    "knolo.llama.context must be a multiple of 16",
                ));
            }
            context
        }
        Some(_) => {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                "knolo.llama.context must be an integer",
            ))
        }
        None => crate::llama_shape::LLAMA_RUN_CONTEXT,
    };
    Ok((vocab, context))
}

pub fn open_paged_kv(
    source: &VerifiedWeightSource,
    layout: KvLayout,
) -> Result<PagedKv, InferFailure> {
    match source.llama {
        Some(shape) if !shape.is_toy() => {
            PagedKv::with_limit(layout, shape.page_count(), RUN_KV_POOL_BYTES)
        }
        _ => PagedKv::new(layout, CPU_KV_PAGE_POOL),
    }
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
    let (mut tensors, metadata) = if verification.image.format == "gguf" {
        let read = read_llama_gguf(&verification.image, weights_dir)?;
        (read.tensors, read.metadata)
    } else {
        (read_verified_tensors(&verification.image, weights_dir)?, Vec::new())
    };
    let toy = is_toy_inventory(&verification.image);
    let (weights, llama, llama_tensors) = if toy {
        let weights = weights::decode_llama_weights(&tensors)?;
        (weights, Some(LlamaShape::toy()), None)
    } else {
        if verification.image.format != "gguf" {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                "knolo.llama.v1 above the llama-tiny fixture needs GGUF attention metadata",
            ));
        }
        let partial = inventory::parse_inventory(&verification.image.tensor_inventory)?;
        let shape = inventory::shape_from_metadata(&partial, &metadata)?;
        if let Some(declared) = prompt_context_extension(&verification.image)? {
            if declared != shape.context {
                return Err(fail(
                    ErrorCode::ModelImageInvalid,
                    "knolo.llama.context does not match the GGUF context reservation",
                ));
            }
        }
        let decoded = weights::decode_wide_tensors(&tensors, &shape)?;
        for tensor in &mut tensors {
            tensor.bytes.clear();
            tensor.bytes.shrink_to_fit();
        }
        (empty_weights(), Some(shape), Some(decoded))
    };
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
        llama,
        llama_tensors,
    })
}

fn empty_weights() -> MicroWeights {
    MicroWeights {
        embed: Vec::new(),
        layers: Vec::new(),
        final_norm: Vec::new(),
        lm_head: Vec::new(),
    }
}

fn prompt_context_extension(image: &ModelImageV1) -> Result<Option<u32>, InferFailure> {
    match image.extensions.get("knolo.llama.context") {
        None => Ok(None),
        Some(CborValue::Integer(value)) if *value > 0 => {
            u32::try_from(*value).map(Some).map_err(|_| {
                fail(
                    ErrorCode::ModelImageInvalid,
                    "knolo.llama.context does not fit u32",
                )
            })
        }
        Some(_) => Err(fail(
            ErrorCode::ModelImageInvalid,
            "knolo.llama.context must be an integer",
        )),
    }
}

pub fn llama_cpu_placement(source: &VerifiedWeightSource) -> Result<PlacementPlanV1, InferFailure> {
    if source.llama.is_some_and(|shape| !shape.is_toy()) {
        return wide_placement(source, "cpu");
    }
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
    if source.llama.is_some_and(|shape| !shape.is_toy()) {
        return Err(fail(
            ErrorCode::UnsupportedKernel,
            "knolo.llama.v1 above the llama-tiny fixture has no CUDA kernel",
        ));
    }
    let storage = storage_of(source)?;
    llama_placement_bytes(
        source.runtime_root.clone(),
        source.weight_bytes,
        "slot-0",
        storage,
    )
}

fn wide_placement(
    source: &VerifiedWeightSource,
    device: &str,
) -> Result<PlacementPlanV1, InferFailure> {
    let shape = source.llama.ok_or_else(|| {
        fail(
            ErrorCode::ModelImageInvalid,
            "wide llama placement is missing its shape",
        )
    })?;
    let storage = storage_of(source)?;
    let workspace = 4096u64;
    let margin = 1024u64;
    let layout = KvLayout {
        layers: shape.layers as u32,
        kv_heads: shape.kv_heads as u32,
        head_dim: shape.head_dim as u32,
        block_size: shape.block_size(),
        dtype: "f32",
    };
    let kv = crate::memory::planned_kv_bytes_within(&layout, shape.page_count(), RUN_KV_POOL_BYTES)?;
    let total = source
        .weight_bytes
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
        model_runtime_root: source.runtime_root.clone(),
        devices: vec![device.into()],
        tensor_groups: vec![TensorGroupV1 {
            name: "weights".into(),
            device: device.into(),
            compute_precision: "f32".into(),
            storage_precision: storage.into(),
        }],
        context_reservation_tokens: shape.context,
        kv_block_size: shape.block_size(),
        kv_precision: "f32".into(),
        workspace_bytes: workspace,
        graph_capture_mode: "off".into(),
        safety_margin_bytes: margin,
        expected_weight_bytes: source.weight_bytes,
        expected_kv_bytes: kv,
        expected_workspace_bytes: workspace,
        expected_staging_bytes: 0,
        expected_overhead_bytes: 0,
        expected_total_bytes: total,
        rejection_reason: None,
        extensions: BTreeMap::new(),
    })
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
    if image.precisions.is_empty()
        || image
            .precisions
            .iter()
            .any(|precision| !allowed.contains(&precision.as_str()))
    {
        return Err(fail(
            ErrorCode::UnsupportedQuantization,
            "knolo.llama.v1 weight precision is not allowlisted",
        ));
    }
    if let Some(one) = single_storage(image) {
        return Ok(one);
    }
    for precision in &image.precisions {
        if image.tensor_inventory == weights::llama_tensor_specs(precision) {
            return Err(fail(
                ErrorCode::UnsupportedQuantization,
                "knolo.llama.v1 weight precision is not allowlisted",
            ));
        }
    }
    Ok(storage_label(&image.precisions))
}

fn single_storage(image: &ModelImageV1) -> Option<&str> {
    match image.precisions.as_slice() {
        [one] => Some(one.as_str()),
        _ => None,
    }
}

fn storage_label(precisions: &[String]) -> &str {
    const QUANTS: [&str; 4] = ["q4_k_m", "q5_k_m", "q6_k", "q8_0"];
    precisions
        .iter()
        .find(|precision| QUANTS.contains(&precision.as_str()))
        .map(String::as_str)
        .unwrap_or(precisions[0].as_str())
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
    if source.llama.is_some_and(|shape| !shape.is_toy()) {
        return accept_wide_placement(source, placement);
    }
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

fn accept_wide_placement(
    source: &VerifiedWeightSource,
    placement: &PlacementPlanV1,
) -> Result<(), InferFailure> {
    placement.validate()?;
    let shape = source.llama.ok_or_else(|| {
        fail(
            ErrorCode::ModelImageInvalid,
            "wide llama placement is missing its shape",
        )
    })?;
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
    if placement.devices.len() != 1 || device != "cpu" {
        return Err(fail(
            ErrorCode::UnsupportedKernel,
            "knolo.llama.v1 above the llama-tiny fixture has no CUDA kernel",
        ));
    }
    if placement.model_runtime_root != source.runtime_root {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "placement runtime root does not match the model",
        ));
    }
    if placement.kv_precision != "f32" || placement.kv_block_size != shape.block_size() {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "knolo.llama.v1 requires an f32 KV block of 16 tokens",
        ));
    }
    if placement.context_reservation_tokens > shape.context {
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
    let layout = KvLayout {
        layers: shape.layers as u32,
        kv_heads: shape.kv_heads as u32,
        head_dim: shape.head_dim as u32,
        block_size: shape.block_size(),
        dtype: "f32",
    };
    if placement.expected_kv_bytes
        != crate::memory::planned_kv_bytes_within(&layout, shape.page_count(), RUN_KV_POOL_BYTES)?
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
