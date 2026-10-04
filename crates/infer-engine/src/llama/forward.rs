//! CPU forward for a Llama wider than the llama-tiny fixture.
//!
//! Quantized matrices stay in their GGUF payload and use `quant_gemm`.
//! Dense tensors are f32. The toy fixture does not call this path.

use infer_contracts::{fail, ErrorCode, InferFailure};

use crate::llama_shape::{LlamaShape, LlamaTensors};
use crate::micro::{add_residual, gemv, rmsnorm, rope, silu, softmax};
use crate::quant_gemm::quant_gemm;
use crate::traits::KvStore;

pub(crate) struct WideForward {
    pub shape: LlamaShape,
    pub tensors: LlamaTensors,
}

pub(crate) fn wide_step(
    wide: &WideForward,
    token: u32,
    kv: &mut dyn KvStore,
    sequence: u64,
) -> Result<Vec<f32>, InferFailure> {
    let shape = wide.shape;
    if token as usize >= shape.vocab {
        return Err(fail(
            ErrorCode::PromptCompilationFailed,
            "token id is outside the llama vocabulary",
        ));
    }
    let position = kv.token_len(sequence)?;
    let mut hidden = row(&wide.tensors, "embed.weight", shape.hidden, token as usize)?.to_vec();
    let kv_width = shape.kv_heads * shape.head_dim;
    for layer in 0..shape.layers {
        let prefix = format!("layers.{layer}");
        let normed = rmsnorm(
            &hidden,
            dense(&wide.tensors, &format!("{prefix}.attn_norm.weight"))?,
            shape.rms_eps,
        )?;
        let mut q = apply(
            &wide.tensors,
            &format!("{prefix}.attn.q.weight"),
            shape.hidden,
            shape.hidden,
            &normed,
        )?;
        let mut k = apply(
            &wide.tensors,
            &format!("{prefix}.attn.k.weight"),
            kv_width,
            shape.hidden,
            &normed,
        )?;
        let v = apply(
            &wide.tensors,
            &format!("{prefix}.attn.v.weight"),
            kv_width,
            shape.hidden,
            &normed,
        )?;
        for head in 0..shape.heads {
            let start = head * shape.head_dim;
            rope(
                &mut q[start..start + shape.head_dim],
                position,
                shape.rope_theta,
            )?;
        }
        for head in 0..shape.kv_heads {
            let start = head * shape.head_dim;
            rope(
                &mut k[start..start + shape.head_dim],
                position,
                shape.rope_theta,
            )?;
        }
        let layer_id = u32::try_from(layer).expect("layer index fits u32");
        kv.write_k(sequence, layer_id, position, &k)?;
        kv.write_v(sequence, layer_id, position, &v)?;
        let mixed = attend(wide, kv, sequence, layer_id, position, &q)?;
        let projected = apply(
            &wide.tensors,
            &format!("{prefix}.attn.o.weight"),
            shape.hidden,
            shape.hidden,
            &mixed,
        )?;
        add_residual(&mut hidden, &projected)?;
        let mlp_in = rmsnorm(
            &hidden,
            dense(&wide.tensors, &format!("{prefix}.mlp_norm.weight"))?,
            shape.rms_eps,
        )?;
        let gate = apply(
            &wide.tensors,
            &format!("{prefix}.mlp.gate.weight"),
            shape.intermediate,
            shape.hidden,
            &mlp_in,
        )?;
        let up = apply(
            &wide.tensors,
            &format!("{prefix}.mlp.up.weight"),
            shape.intermediate,
            shape.hidden,
            &mlp_in,
        )?;
        let mut activated = Vec::with_capacity(shape.intermediate);
        for (gate_v, up_v) in gate.iter().zip(&up) {
            activated.push(silu(*gate_v)? * up_v);
        }
        let down = apply(
            &wide.tensors,
            &format!("{prefix}.mlp.down.weight"),
            shape.hidden,
            shape.intermediate,
            &activated,
        )?;
        add_residual(&mut hidden, &down)?;
    }
    let final_hidden = rmsnorm(
        &hidden,
        dense(&wide.tensors, "final_norm.weight")?,
        shape.rms_eps,
    )?;
    let logits = apply(
        &wide.tensors,
        "lm_head.weight",
        shape.vocab,
        shape.hidden,
        &final_hidden,
    )?;
    kv.commit_token(sequence)?;
    Ok(logits)
}

fn attend(
    wide: &WideForward,
    kv: &dyn KvStore,
    sequence: u64,
    layer: u32,
    position: u32,
    q: &[f32],
) -> Result<Vec<f32>, InferFailure> {
    let shape = wide.shape;
    let scale = (shape.head_dim as f32).sqrt().recip();
    let kv_width = shape.kv_heads * shape.head_dim;
    let group = shape.heads / shape.kv_heads;
    let mut keys = Vec::with_capacity(position as usize + 1);
    let mut values = Vec::with_capacity(position as usize + 1);
    for token in 0..=position {
        let mut key = vec![0.0; kv_width];
        let mut value = vec![0.0; kv_width];
        kv.read_k(sequence, layer, token, &mut key)?;
        kv.read_v(sequence, layer, token, &mut value)?;
        keys.push(key);
        values.push(value);
    }
    let mut concat = vec![0.0; shape.hidden];
    for head in 0..shape.heads {
        let kv_head = head / group;
        let query = &q[head * shape.head_dim..(head + 1) * shape.head_dim];
        let mut scores = Vec::with_capacity(keys.len());
        for key in &keys {
            let key_head = &key[kv_head * shape.head_dim..(kv_head + 1) * shape.head_dim];
            let mut dot = 0.0f32;
            for (q_i, k_i) in query.iter().zip(key_head) {
                dot += q_i * k_i;
            }
            scores.push(dot * scale);
        }
        softmax(&mut scores)?;
        let mut mixed = vec![0.0; shape.head_dim];
        for (score, value) in scores.iter().zip(&values) {
            let value_head = &value[kv_head * shape.head_dim..(kv_head + 1) * shape.head_dim];
            for (slot, component) in mixed.iter_mut().zip(value_head) {
                *slot += score * component;
            }
        }
        concat[head * shape.head_dim..(head + 1) * shape.head_dim].copy_from_slice(&mixed);
    }
    Ok(concat)
}

fn apply(
    tensors: &LlamaTensors,
    name: &str,
    rows: usize,
    cols: usize,
    x: &[f32],
) -> Result<Vec<f32>, InferFailure> {
    if let Some(packed) = tensors.packed.iter().find(|tensor| tensor.name == name) {
        if packed.rows != rows || packed.cols != cols {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                format!("tensor {name} does not match knolo.llama.v1"),
            ));
        }
        return quant_gemm(
            packed.tensor_type,
            &packed.bytes,
            rows,
            cols,
            x,
            1,
        );
    }
    gemv(dense(tensors, name)?, rows, cols, x)
}

fn dense<'a>(tensors: &'a LlamaTensors, name: &str) -> Result<&'a [f32], InferFailure> {
    tensors.dense.get(name).map(AsRef::as_ref).ok_or_else(|| {
        fail(
            ErrorCode::ModelArtifactMissing,
            format!("missing tensor {name}"),
        )
    })
}

fn row<'a>(
    tensors: &'a LlamaTensors,
    name: &str,
    cols: usize,
    index: usize,
) -> Result<&'a [f32], InferFailure> {
    let matrix = dense(tensors, name)?;
    let start = index * cols;
    matrix.get(start..start + cols).ok_or_else(|| {
        fail(
            ErrorCode::ContractInvalid,
            "embedding row is outside the table",
        )
    })
}
