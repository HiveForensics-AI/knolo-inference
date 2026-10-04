//! Inventory and GGUF metadata for a Llama that is not the toy fixture.

use std::collections::BTreeMap;

use infer_artifact::{GgufMetadata, GgufValue};
use infer_contracts::{fail, ErrorCode, InferFailure, TensorSpecV1};

use crate::llama_shape::{LlamaShape, LLAMA_BLOCK, LLAMA_RUN_CONTEXT};

#[derive(Debug, Clone)]
pub struct InventoryShape {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    pub q_rows: usize,
    pub k_rows: usize,
    pub intermediate: usize,
}

pub fn parse_inventory(specs: &[TensorSpecV1]) -> Result<InventoryShape, InferFailure> {
    let mut map = BTreeMap::new();
    for spec in specs {
        if map.insert(spec.name.as_str(), spec).is_some() {
            return Err(invalid(format!("duplicate tensor {}", spec.name)));
        }
    }
    let embed = required(&map, "embed.weight")?;
    let (vocab, hidden) = matrix(embed, "embed.weight")?;
    let final_norm = required(&map, "final_norm.weight")?;
    vector(final_norm, hidden, "final_norm.weight")?;
    let lm_head = required(&map, "lm_head.weight")?;
    let (lm_vocab, lm_hidden) = matrix(lm_head, "lm_head.weight")?;
    if lm_vocab != vocab || lm_hidden != hidden {
        return Err(invalid(
            "lm_head.weight does not match the embedding table",
        ));
    }
    let mut layers = 0usize;
    loop {
        let name = format!("layers.{layers}.attn_norm.weight");
        if !map.contains_key(name.as_str()) {
            break;
        }
        layers += 1;
        if layers > 512 {
            return Err(invalid("llama layer count is above 512"));
        }
    }
    if layers == 0 {
        return Err(invalid("tensor inventory does not match knolo.llama.v1"));
    }
    let mut q_rows = None;
    let mut k_rows = None;
    let mut intermediate = None;
    for layer in 0..layers {
        let prefix = format!("layers.{layer}");
        vector(
            required(&map, &format!("{prefix}.attn_norm.weight"))?,
            hidden,
            &format!("{prefix}.attn_norm.weight"),
        )?;
        let (q, q_hidden) = matrix(required(&map, &format!("{prefix}.attn.q.weight"))?, &format!("{prefix}.attn.q.weight"))?;
        let (k, k_hidden) = matrix(required(&map, &format!("{prefix}.attn.k.weight"))?, &format!("{prefix}.attn.k.weight"))?;
        let (v, v_hidden) = matrix(required(&map, &format!("{prefix}.attn.v.weight"))?, &format!("{prefix}.attn.v.weight"))?;
        let (o_rows, o_cols) = matrix(required(&map, &format!("{prefix}.attn.o.weight"))?, &format!("{prefix}.attn.o.weight"))?;
        vector(
            required(&map, &format!("{prefix}.mlp_norm.weight"))?,
            hidden,
            &format!("{prefix}.mlp_norm.weight"),
        )?;
        let (gate, gate_hidden) = matrix(
            required(&map, &format!("{prefix}.mlp.gate.weight"))?,
            &format!("{prefix}.mlp.gate.weight"),
        )?;
        let (up, up_hidden) = matrix(
            required(&map, &format!("{prefix}.mlp.up.weight"))?,
            &format!("{prefix}.mlp.up.weight"),
        )?;
        let (down_rows, down_cols) = matrix(
            required(&map, &format!("{prefix}.mlp.down.weight"))?,
            &format!("{prefix}.mlp.down.weight"),
        )?;
        if q_hidden != hidden || k_hidden != hidden || v_hidden != hidden || gate_hidden != hidden || up_hidden != hidden {
            return Err(invalid("a llama projection does not use the hidden size"));
        }
        if o_rows != hidden || o_cols != q || down_rows != hidden || down_cols != gate || v != k || up != gate {
            return Err(invalid("a llama projection shape is inconsistent"));
        }
        if q_rows.get_or_insert(q) != &q || k_rows.get_or_insert(k) != &k || intermediate.get_or_insert(gate) != &gate {
            return Err(invalid("llama layers do not share a shape"));
        }
    }
    let expected = 3 + layers * 9;
    if map.len() != expected {
        return Err(invalid("llama weights contain an unexpected tensor"));
    }
    Ok(InventoryShape {
        vocab,
        hidden,
        layers,
        q_rows: q_rows.unwrap_or(0),
        k_rows: k_rows.unwrap_or(0),
        intermediate: intermediate.unwrap_or(0),
    })
}

pub fn shape_from_metadata(
    inventory: &InventoryShape,
    metadata: &[GgufMetadata],
) -> Result<LlamaShape, InferFailure> {
    let heads = meta_usize(metadata, "llama.attention.head_count")?
        .ok_or_else(|| invalid("gguf metadata is missing llama.attention.head_count"))?;
    let kv_heads = meta_usize(metadata, "llama.attention.head_count_kv")?
        .ok_or_else(|| invalid("gguf metadata is missing llama.attention.head_count_kv"))?;
    if heads == 0 || kv_heads == 0 || heads < kv_heads || heads % kv_heads != 0 {
        return Err(invalid("llama attention head count is invalid"));
    }
    if inventory.q_rows % heads != 0 || inventory.k_rows % kv_heads != 0 {
        return Err(invalid(
            "llama attention head count does not match the tensor inventory",
        ));
    }
    let head_dim = inventory.q_rows / heads;
    if head_dim == 0 || head_dim % 2 != 0 || kv_heads * head_dim != inventory.k_rows {
        return Err(invalid(
            "llama attention head count does not match the tensor inventory",
        ));
    }
    if inventory.q_rows != inventory.hidden {
        return Err(invalid("llama query width does not match the hidden size"));
    }
    let declared = meta_u32(metadata, "llama.context_length")?.unwrap_or(LLAMA_RUN_CONTEXT);
    let capped = declared.min(LLAMA_RUN_CONTEXT);
    let context = capped / LLAMA_BLOCK * LLAMA_BLOCK;
    if context < LLAMA_BLOCK {
        return Err(invalid("llama context reservation is below 16 tokens"));
    }
    let rope_theta = meta_f32(metadata, "llama.rope.freq_base")?.unwrap_or(10_000.0);
    let rms_eps = match meta_f32(metadata, "llama.attention.layer_norm_rms_epsilon")? {
        Some(value) => value,
        None => meta_f32(metadata, "llama.attention.layer_norm_rms_eps")?.unwrap_or(1e-5),
    };
    if !rope_theta.is_finite() || rope_theta <= 0.0 || !rms_eps.is_finite() || rms_eps <= 0.0 {
        return Err(invalid("llama rope base or rms epsilon is invalid"));
    }
    Ok(LlamaShape {
        vocab: inventory.vocab,
        hidden: inventory.hidden,
        layers: inventory.layers,
        heads,
        kv_heads,
        head_dim,
        intermediate: inventory.intermediate,
        rms_eps,
        rope_theta,
        context,
    })
}

pub fn meta_u32(metadata: &[GgufMetadata], key: &str) -> Result<Option<u32>, InferFailure> {
    let Some(value) = metadata.iter().find(|item| item.key == key).map(|item| &item.value) else {
        return Ok(None);
    };
    let number = match value {
        GgufValue::Uint32(value) => u64::from(*value),
        GgufValue::Uint64(value) => *value,
        GgufValue::Int32(value) if *value >= 0 => *value as u64,
        GgufValue::Int64(value) if *value >= 0 => *value as u64,
        _ => return Err(invalid(format!("gguf metadata {key} is not an integer"))),
    };
    let number = u32::try_from(number).map_err(|_| invalid(format!("gguf metadata {key} does not fit u32")))?;
    Ok(Some(number))
}

fn meta_usize(metadata: &[GgufMetadata], key: &str) -> Result<Option<usize>, InferFailure> {
    Ok(meta_u32(metadata, key)?.map(|value| value as usize))
}

pub fn meta_f32(metadata: &[GgufMetadata], key: &str) -> Result<Option<f32>, InferFailure> {
    let Some(value) = metadata.iter().find(|item| item.key == key).map(|item| &item.value) else {
        return Ok(None);
    };
    let number = match value {
        GgufValue::Float32(bits) => f32::from_bits(*bits),
        GgufValue::Float64(bits) => f64::from_bits(*bits) as f32,
        _ => return Err(invalid(format!("gguf metadata {key} is not a float"))),
    };
    Ok(Some(number))
}

fn required<'a>(
    map: &BTreeMap<&str, &'a TensorSpecV1>,
    name: &str,
) -> Result<&'a TensorSpecV1, InferFailure> {
    map.get(name)
        .copied()
        .ok_or_else(|| invalid(format!("missing tensor {name}")))
}

fn matrix(spec: &TensorSpecV1, name: &str) -> Result<(usize, usize), InferFailure> {
    if spec.shape.len() != 2 {
        return Err(invalid(format!("tensor {name} is not a matrix")));
    }
    let rows = spec.shape[0] as usize;
    let cols = spec.shape[1] as usize;
    if rows == 0 || cols == 0 {
        return Err(invalid(format!("tensor {name} has a zero dimension")));
    }
    Ok((rows, cols))
}

fn vector(spec: &TensorSpecV1, hidden: usize, name: &str) -> Result<(), InferFailure> {
    if spec.shape != [hidden as u32] {
        return Err(invalid(format!("tensor {name} does not match knolo.llama.v1")));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> InferFailure {
    fail(ErrorCode::ModelImageInvalid, message)
}
