//! Fixed `knolo.micro.v1` shapes and the f32 tensors those shapes name.

use std::collections::BTreeMap;

use infer_artifact::TensorBytes;
use infer_contracts::{fail, ErrorCode, InferFailure, TensorSpecV1};

pub const ADAPTER_ID: &str = "knolo.micro.v1";
pub const FAMILY: &str = "micro";
pub const VOCAB: usize = 16;
pub const HIDDEN: usize = 8;
pub const LAYERS: usize = 2;
pub const HEADS: usize = 2;
pub const KV_HEADS: usize = 1;
pub const HEAD_DIM: usize = 4;
pub const INTERMEDIATE: usize = 16;
pub const MAX_CONTEXT: u32 = 16;
pub const BLOCK_SIZE: u32 = 16;
pub const RMS_EPS: f32 = 1e-5;
pub const ROPE_THETA: f32 = 10_000.0;
pub const LOGIT_ABS_TOLERANCE: f32 = 1e-4;
pub const LOGIT_ABS_TOLERANCE_MILLIONTHS: u32 = 100;
pub const MIN_GREEDY_MARGIN: f32 = 1e-2;
pub const SYNTHETIC_SEED: u64 = 0x6D69_6372_6F31_0001;

const _: () = {
    assert!(HEADS * HEAD_DIM == HIDDEN);
    assert!(KV_HEADS > 0 && HEADS >= KV_HEADS);
    assert!(HEAD_DIM % 2 == 0);
    assert!(BLOCK_SIZE == MAX_CONTEXT);
    assert!(BLOCK_SIZE == 16);
};

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub o: Vec<f32>,
    pub mlp_norm: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct MicroWeights {
    pub embed: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: Vec<f32>,
}

pub fn micro_tensor_specs() -> Vec<TensorSpecV1> {
    let mut specs = Vec::new();
    push(
        &mut specs,
        "embed.weight",
        vec![VOCAB as u32, HIDDEN as u32],
    );
    for layer in 0..LAYERS {
        let prefix = format!("layers.{layer}");
        push(
            &mut specs,
            &format!("{prefix}.attn_norm.weight"),
            vec![HIDDEN as u32],
        );
        push(
            &mut specs,
            &format!("{prefix}.attn.q.weight"),
            vec![(HEADS * HEAD_DIM) as u32, HIDDEN as u32],
        );
        push(
            &mut specs,
            &format!("{prefix}.attn.k.weight"),
            vec![(KV_HEADS * HEAD_DIM) as u32, HIDDEN as u32],
        );
        push(
            &mut specs,
            &format!("{prefix}.attn.v.weight"),
            vec![(KV_HEADS * HEAD_DIM) as u32, HIDDEN as u32],
        );
        push(
            &mut specs,
            &format!("{prefix}.attn.o.weight"),
            vec![HIDDEN as u32, (HEADS * HEAD_DIM) as u32],
        );
        push(
            &mut specs,
            &format!("{prefix}.mlp_norm.weight"),
            vec![HIDDEN as u32],
        );
        push(
            &mut specs,
            &format!("{prefix}.mlp.gate.weight"),
            vec![INTERMEDIATE as u32, HIDDEN as u32],
        );
        push(
            &mut specs,
            &format!("{prefix}.mlp.up.weight"),
            vec![INTERMEDIATE as u32, HIDDEN as u32],
        );
        push(
            &mut specs,
            &format!("{prefix}.mlp.down.weight"),
            vec![HIDDEN as u32, INTERMEDIATE as u32],
        );
    }
    push(&mut specs, "final_norm.weight", vec![HIDDEN as u32]);
    push(
        &mut specs,
        "lm_head.weight",
        vec![VOCAB as u32, HIDDEN as u32],
    );
    specs.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
    specs
}

fn push(specs: &mut Vec<TensorSpecV1>, name: &str, shape: Vec<u32>) {
    specs.push(TensorSpecV1 {
        name: name.to_string(),
        shape,
        dtype: "f32".into(),
    });
}

pub fn kv_width() -> usize {
    KV_HEADS * HEAD_DIM
}

pub fn micro_kv_bytes() -> u64 {
    LAYERS as u64 * u64::from(BLOCK_SIZE) * KV_HEADS as u64 * HEAD_DIM as u64 * 2 * 4
}

pub fn decode_micro_weights(tensors: &[TensorBytes]) -> Result<MicroWeights, InferFailure> {
    let mut map = BTreeMap::new();
    for tensor in tensors {
        if tensor.dtype != "f32" {
            return Err(fail(
                ErrorCode::UnsupportedQuantization,
                format!("tensor {} is not f32", tensor.name),
            ));
        }
        if map
            .insert(tensor.name.clone(), decode_f32(tensor)?)
            .is_some()
        {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                format!("duplicate tensor {}", tensor.name),
            ));
        }
    }
    let mut layers = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        let prefix = format!("layers.{layer}");
        layers.push(LayerWeights {
            attn_norm: take(&mut map, &format!("{prefix}.attn_norm.weight"), HIDDEN)?,
            q: take(
                &mut map,
                &format!("{prefix}.attn.q.weight"),
                HIDDEN * HIDDEN,
            )?,
            k: take(
                &mut map,
                &format!("{prefix}.attn.k.weight"),
                kv_width() * HIDDEN,
            )?,
            v: take(
                &mut map,
                &format!("{prefix}.attn.v.weight"),
                kv_width() * HIDDEN,
            )?,
            o: take(
                &mut map,
                &format!("{prefix}.attn.o.weight"),
                HIDDEN * HIDDEN,
            )?,
            mlp_norm: take(&mut map, &format!("{prefix}.mlp_norm.weight"), HIDDEN)?,
            gate: take(
                &mut map,
                &format!("{prefix}.mlp.gate.weight"),
                INTERMEDIATE * HIDDEN,
            )?,
            up: take(
                &mut map,
                &format!("{prefix}.mlp.up.weight"),
                INTERMEDIATE * HIDDEN,
            )?,
            down: take(
                &mut map,
                &format!("{prefix}.mlp.down.weight"),
                HIDDEN * INTERMEDIATE,
            )?,
        });
    }
    let weights = MicroWeights {
        embed: take(&mut map, "embed.weight", VOCAB * HIDDEN)?,
        layers,
        final_norm: take(&mut map, "final_norm.weight", HIDDEN)?,
        lm_head: take(&mut map, "lm_head.weight", VOCAB * HIDDEN)?,
    };
    if !map.is_empty() {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "micro weights contain an unexpected tensor",
        ));
    }
    Ok(weights)
}

fn take(
    map: &mut BTreeMap<String, Vec<f32>>,
    name: &str,
    len: usize,
) -> Result<Vec<f32>, InferFailure> {
    let values = map.remove(name).ok_or_else(|| {
        fail(
            ErrorCode::ModelArtifactMissing,
            format!("missing tensor {name}"),
        )
    })?;
    if values.len() != len {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            format!("tensor {name} does not match knolo.micro.v1"),
        ));
    }
    Ok(values)
}

fn decode_f32(tensor: &TensorBytes) -> Result<Vec<f32>, InferFailure> {
    if tensor.bytes.len() % 4 != 0 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            format!("tensor {} is not a sequence of f32 values", tensor.name),
        ));
    }
    let mut values = Vec::with_capacity(tensor.bytes.len() / 4);
    for chunk in tensor.bytes.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().expect("chunk is 4 bytes"));
        if !value.is_finite() {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                format!("tensor {} contains a non-finite value", tensor.name),
            ));
        }
        values.push(value);
    }
    Ok(values)
}

pub fn synthetic_tensors() -> Result<Vec<TensorBytes>, InferFailure> {
    let mut state = SYNTHETIC_SEED;
    let mut tensors = Vec::new();
    for spec in micro_tensor_specs() {
        let len = spec
            .shape
            .iter()
            .map(|dim| *dim as usize)
            .product::<usize>();
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(synthetic_value(&spec.name, next_signed(&mut state)));
        }
        let mut bytes = Vec::with_capacity(len * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        tensors.push(TensorBytes {
            name: spec.name,
            dtype: spec.dtype,
            shape: spec.shape,
            bytes,
        });
    }
    Ok(tensors)
}

fn synthetic_value(name: &str, signed: f32) -> f32 {
    if name.ends_with("norm.weight") {
        1.0 + 0.05 * signed
    } else if name == "lm_head.weight" {
        0.75 * signed
    } else {
        0.20 * signed
    }
}

fn next_signed(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let unit = ((*state >> 11) & 0x00ff_ffff) as f32 / 16_777_216.0;
    unit * 2.0 - 1.0
}
