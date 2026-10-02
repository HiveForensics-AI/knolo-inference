//! Fixed `knolo.llama.v1` shapes. Values are stored as f32, f16, or bf16
//! and decoded to the f32 oracle before the forward.

use std::collections::BTreeMap;

use infer_artifact::{GgufTensorType, TensorBytes};
use infer_contracts::{fail, ErrorCode, InferFailure, TensorSpecV1};

use crate::dequant::dequant_gguf;
use crate::micro::MicroWeights;

pub const ADAPTER_ID: &str = "knolo.llama.v1";
pub const FAMILY: &str = "llama";
pub const VOCAB: usize = 32;
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
pub const SYNTHETIC_SEED: u64 = 0x6C6C_616D_6132_0002;

const _: () = {
    assert!(HEADS * HEAD_DIM == HIDDEN);
    assert!(KV_HEADS > 0 && HEADS >= KV_HEADS);
    assert!(HEAD_DIM % 2 == 0);
    assert!(BLOCK_SIZE == MAX_CONTEXT);
    assert!(BLOCK_SIZE == 16);
    assert!(VOCAB == 32);
};

pub fn llama_tensor_specs(dtype: &str) -> Vec<TensorSpecV1> {
    let mut specs = Vec::new();
    push(
        &mut specs,
        "embed.weight",
        vec![VOCAB as u32, HIDDEN as u32],
        dtype,
    );
    for layer in 0..LAYERS {
        let prefix = format!("layers.{layer}");
        push(
            &mut specs,
            &format!("{prefix}.attn_norm.weight"),
            vec![HIDDEN as u32],
            dtype,
        );
        push(
            &mut specs,
            &format!("{prefix}.attn.q.weight"),
            vec![(HEADS * HEAD_DIM) as u32, HIDDEN as u32],
            dtype,
        );
        push(
            &mut specs,
            &format!("{prefix}.attn.k.weight"),
            vec![(KV_HEADS * HEAD_DIM) as u32, HIDDEN as u32],
            dtype,
        );
        push(
            &mut specs,
            &format!("{prefix}.attn.v.weight"),
            vec![(KV_HEADS * HEAD_DIM) as u32, HIDDEN as u32],
            dtype,
        );
        push(
            &mut specs,
            &format!("{prefix}.attn.o.weight"),
            vec![HIDDEN as u32, (HEADS * HEAD_DIM) as u32],
            dtype,
        );
        push(
            &mut specs,
            &format!("{prefix}.mlp_norm.weight"),
            vec![HIDDEN as u32],
            dtype,
        );
        push(
            &mut specs,
            &format!("{prefix}.mlp.gate.weight"),
            vec![INTERMEDIATE as u32, HIDDEN as u32],
            dtype,
        );
        push(
            &mut specs,
            &format!("{prefix}.mlp.up.weight"),
            vec![INTERMEDIATE as u32, HIDDEN as u32],
            dtype,
        );
        push(
            &mut specs,
            &format!("{prefix}.mlp.down.weight"),
            vec![HIDDEN as u32, INTERMEDIATE as u32],
            dtype,
        );
    }
    push(&mut specs, "final_norm.weight", vec![HIDDEN as u32], dtype);
    push(
        &mut specs,
        "lm_head.weight",
        vec![VOCAB as u32, HIDDEN as u32],
        dtype,
    );
    specs.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
    specs
}

fn push(specs: &mut Vec<TensorSpecV1>, name: &str, shape: Vec<u32>, dtype: &str) {
    specs.push(TensorSpecV1 {
        name: name.to_string(),
        shape,
        dtype: dtype.to_string(),
    });
}

pub fn kv_width() -> usize {
    KV_HEADS * HEAD_DIM
}

pub fn decode_llama_weights(tensors: &[TensorBytes]) -> Result<MicroWeights, InferFailure> {
    let mut map = BTreeMap::new();
    for tensor in tensors {
        let values = decode_tensor(tensor)?;
        if map.insert(tensor.name.clone(), values).is_some() {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                format!("duplicate tensor {}", tensor.name),
            ));
        }
    }
    let mut layers = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        let prefix = format!("layers.{layer}");
        layers.push(crate::micro::LayerWeights {
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
            "llama weights contain an unexpected tensor",
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
            format!("tensor {name} does not match knolo.llama.v1"),
        ));
    }
    Ok(values)
}

fn decode_tensor(tensor: &TensorBytes) -> Result<Vec<f32>, InferFailure> {
    let values = match tensor.dtype.as_str() {
        "f32" => decode_f32(&tensor.bytes, &tensor.name)?,
        "f16" => dequant_gguf(GgufTensorType::F16, &tensor.bytes)?,
        "bf16" => decode_bf16(&tensor.bytes, &tensor.name)?,
        "q8_0" => dequant_gguf(GgufTensorType::Q8_0, &tensor.bytes)?,
        "q4_k_m" => dequant_gguf(GgufTensorType::Q4_K, &tensor.bytes)?,
        "q5_k_m" => dequant_gguf(GgufTensorType::Q5_K, &tensor.bytes)?,
        "q6_k" => dequant_gguf(GgufTensorType::Q6_K, &tensor.bytes)?,
        _ => {
            return Err(fail(
                ErrorCode::UnsupportedQuantization,
                format!("tensor {} is not f32, f16, or bf16", tensor.name),
            ))
        }
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            format!("tensor {} contains a non-finite value", tensor.name),
        ));
    }
    Ok(values)
}

fn decode_f32(bytes: &[u8], name: &str) -> Result<Vec<f32>, InferFailure> {
    if bytes.len() % 4 != 0 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            format!("tensor {name} is not a sequence of f32 values"),
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes(
            chunk.try_into().expect("chunk is 4 bytes"),
        ));
    }
    Ok(values)
}

fn decode_bf16(bytes: &[u8], name: &str) -> Result<Vec<f32>, InferFailure> {
    if bytes.len() % 2 != 0 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            format!("tensor {name} is not a sequence of bf16 values"),
        ));
    }
    let mut values = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        values.push(f32::from_bits(u32::from(bits) << 16));
    }
    Ok(values)
}

pub fn synthetic_tensors(dtype: &str) -> Result<Vec<TensorBytes>, InferFailure> {
    let f32_tensors = synthetic_f32()?;
    if dtype == "f32" {
        return Ok(f32_tensors);
    }
    let mut converted = Vec::with_capacity(f32_tensors.len());
    for tensor in f32_tensors {
        let values = decode_f32(&tensor.bytes, &tensor.name)?;
        let bytes = match dtype {
            "f16" => encode_f16(&values)?,
            "bf16" => encode_bf16(&values),
            _ => {
                return Err(fail(
                    ErrorCode::UnsupportedQuantization,
                    "llama synthetic dtype must be f32, f16, or bf16",
                ))
            }
        };
        converted.push(TensorBytes {
            name: tensor.name,
            dtype: dtype.to_string(),
            shape: tensor.shape,
            bytes,
        });
    }
    Ok(converted)
}

pub fn synthetic_f32() -> Result<Vec<TensorBytes>, InferFailure> {
    let mut state = SYNTHETIC_SEED;
    let mut tensors = Vec::new();
    for spec in llama_tensor_specs("f32") {
        let len = spec
            .shape
            .iter()
            .map(|dim| *dim as usize)
            .product::<usize>();
        let mut bytes = Vec::with_capacity(len * 4);
        for _ in 0..len {
            let value = synthetic_value(&spec.name, next_signed(&mut state));
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
    let unit = (signed * 8.0).round() / 8.0;
    if name.ends_with("norm.weight") {
        1.0 + 0.125 * unit
    } else if name == "lm_head.weight" {
        0.5 * unit
    } else {
        0.25 * unit
    }
}

fn next_signed(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let unit = ((*state >> 11) & 0x00ff_ffff) as f32 / 16_777_216.0;
    unit * 2.0 - 1.0
}

fn encode_f16(values: &[f32]) -> Result<Vec<u8>, InferFailure> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&f32_to_f16_bits(*value).to_le_bytes());
    }
    Ok(bytes)
}

fn encode_bf16(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        let bits = (value.to_bits() >> 16) as u16;
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    bytes
}

/// Round-to-nearest-even binary32 to binary16. The synthetic weights are
/// multiples that survive this conversion unchanged.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 255 {
        let half_mant = if mant == 0 { 0 } else { 0x200 };
        return sign | 0x7c00 | half_mant;
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased < -14 {
        return sign;
    }
    let mut half_exp = (unbiased + 15) as u16;
    let mut half_mant = (mant >> 13) as u16;
    let round = (mant >> 12) & 1;
    let sticky = mant & 0xfff;
    if round == 1 && (sticky != 0 || half_mant & 1 == 1) {
        half_mant += 1;
        if half_mant == 0x400 {
            half_mant = 0;
            half_exp += 1;
            if half_exp >= 31 {
                return sign | 0x7c00;
            }
        }
    }
    sign | (half_exp << 10) | half_mant
}
