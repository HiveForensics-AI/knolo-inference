//! CUDA forward for a Llama wider than the llama-tiny fixture.
//!
//! Quantized matrices stay in their GGUF payload and use Candle's CUDA
//! quantized matmul. A quantized token embedding stays packed. Each step
//! dequantizes that row on the host and uploads the hidden vector.
//! Norms and a dense `f32`, `f16`, or `bf16` embedding are f32 on `slot-0`.
//! Key and value caches for the reserved context stay on the device, and
//! attention reads them there. The host page pool still receives each
//! committed vector so the scheduler's accounting is unchanged.
//!
//! The CPU `wide_step` oracle remains the numerical authority. Full f32
//! GEMM is required; TF32 is turned off for this model.

use std::collections::HashMap;

use candle_core::quantized::ggml_file::qtensor_from_ggml;
use candle_core::quantized::{GgmlDType, QMatMul};
use candle_core::{DType, Device, Module, Tensor, D};
use infer_artifact::GgufTensorType;
use infer_contracts::{fail, ErrorCode, InferFailure};
use infer_engine::{
    dequant_packed_row, DecodeBatch, DecodeOutput, ExecutableModel, KvLayout, KvStore, LlamaShape,
    LlamaTensors, ModelCapabilities, ModelRuntimeIdentity, PackedTensor, PrefillBatch,
    PrefillOutput, VerifiedWeightSource,
};

struct DevKv {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
}

enum Linear {
    Dense(Tensor),
    Quant(QMatMul),
}

impl Linear {
    fn apply(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let cols = x.dim(0)?;
        let x = x.reshape((1, cols))?;
        let y = match self {
            Linear::Dense(weight) => x.matmul(&weight.transpose(0, 1)?)?,
            Linear::Quant(weight) => weight.forward(&x)?,
        };
        y.squeeze(0)
    }
}

struct Layer {
    attn_norm: Tensor,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    mlp_norm: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
}

enum Embed {
    Dense(Tensor),
    Packed(PackedTensor),
}

pub struct CudaWideLlama {
    identity: ModelRuntimeIdentity,
    capabilities: ModelCapabilities,
    layout: KvLayout,
    context_limit: u32,
    shape: LlamaShape,
    kv_width: usize,
    embed: Embed,
    layers: Vec<Layer>,
    final_norm: Tensor,
    lm_head: Linear,
    freq_factors: Vec<f32>,
    device: Device,
    cache: HashMap<u64, DevKv>,
}

impl CudaWideLlama {
    pub fn new(
        source: &VerifiedWeightSource,
        context_limit: u32,
        device: Device,
    ) -> Result<Self, InferFailure> {
        let shape = source.llama.ok_or_else(|| {
            fail(
                ErrorCode::ModelImageInvalid,
                "wide llama forward is missing its shape",
            )
        })?;
        if shape.is_toy() {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                "the llama-tiny fixture does not use the wide CUDA forward",
            ));
        }
        if context_limit == 0 || context_limit > shape.context {
            return Err(fail(
                ErrorCode::PlacementUnsatisfiable,
                "llama context does not fit the reserved context",
            ));
        }
        let tensors = source.llama_tensors.clone().ok_or_else(|| {
            fail(
                ErrorCode::ModelImageInvalid,
                "wide llama forward is missing its tensors",
            )
        })?;
        candle_core::cuda::set_gemm_reduced_precision_f32(false);
        let embed = load_embed(&tensors, &shape, &device)?;
        let mut layers = Vec::with_capacity(shape.layers);
        for layer in 0..shape.layers {
            layers.push(load_layer(&tensors, &shape, layer, &device)?);
        }
        let final_norm = upload_vector(&tensors, "final_norm.weight", shape.hidden, &device)?;
        let lm_head = upload_linear(
            &tensors,
            "lm_head.weight",
            shape.vocab,
            shape.hidden,
            &device,
        )?;
        Ok(Self {
            identity: ModelRuntimeIdentity {
                adapter_id: infer_engine::LLAMA_ADAPTER_ID,
                model_image_root: source.image_root.clone(),
                artifact_root: source.artifact_root.clone(),
                runtime_root: source.runtime_root.clone(),
            },
            capabilities: ModelCapabilities {
                max_context_tokens: context_limit,
                vocab_size: shape.vocab as u32,
                text_generation: true,
            },
            layout: KvLayout {
                layers: shape.layers as u32,
                kv_heads: shape.kv_heads as u32,
                head_dim: shape.head_dim as u32,
                block_size: shape.block_size(),
                dtype: "f32",
            },
            context_limit,
            kv_width: shape.kv_heads * shape.head_dim,
            shape,
            embed,
            layers,
            final_norm,
            lm_head,
            freq_factors: tensors.rope_freq_factors.clone(),
            device,
            cache: HashMap::new(),
        })
    }

    fn reap(&mut self, kv: &dyn KvStore) {
        self.cache.retain(|id, _| kv.token_len(*id).is_ok());
    }

    fn ensure_cache(&mut self, sequence: u64) -> Result<(), InferFailure> {
        if self.cache.contains_key(&sequence) {
            return Ok(());
        }
        let context = self.context_limit as usize;
        let mut k = Vec::with_capacity(self.layers.len());
        let mut v = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            k.push(c(Tensor::zeros(
                (context, self.kv_width),
                DType::F32,
                &self.device,
            ))?);
            v.push(c(Tensor::zeros(
                (context, self.kv_width),
                DType::F32,
                &self.device,
            ))?);
        }
        self.cache.insert(sequence, DevKv { k, v });
        Ok(())
    }

    fn ensure_room(&self, len: u32, extra: u32) -> Result<(), InferFailure> {
        let end = len.checked_add(extra).ok_or_else(|| {
            fail(
                ErrorCode::ContextLimitExceeded,
                "llama context length overflows",
            )
        })?;
        if end > self.context_limit {
            return Err(fail(
                ErrorCode::ContextLimitExceeded,
                "llama context does not fit the reserved context",
            ));
        }
        Ok(())
    }

    fn step(
        &mut self,
        token: u32,
        kv: &mut dyn KvStore,
        sequence: u64,
    ) -> Result<Vec<f32>, InferFailure> {
        if token as usize >= self.shape.vocab {
            return Err(fail(
                ErrorCode::PromptCompilationFailed,
                "token id is outside the llama vocabulary",
            ));
        }
        let position = kv.token_len(sequence)?;
        self.ensure_room(position, 1)?;
        self.ensure_cache(sequence)?;
        let pos = position as usize;
        let mut hidden = match &self.embed {
            Embed::Dense(table) => c(table
                .narrow(0, token as usize, 1)
                .and_then(|row| row.squeeze(0)))?,
            Embed::Packed(packed) => {
                let row = dequant_packed_row(packed, token as usize)?;
                c(Tensor::from_vec(row, self.shape.hidden, &self.device))?
            }
        };
        for layer_id in 0..self.layers.len() {
            let normed = c(rmsnorm(
                &hidden,
                &self.layers[layer_id].attn_norm,
                self.shape.rms_eps,
            ))?;
            let q = c(rope(
                &c(self.layers[layer_id].q.apply(&normed))?,
                self.shape.heads,
                self.shape.head_dim,
                position,
                self.shape.rope_theta,
                &self.freq_factors,
            ))?;
            let k = c(rope(
                &c(self.layers[layer_id].k.apply(&normed))?,
                self.shape.kv_heads,
                self.shape.head_dim,
                position,
                self.shape.rope_theta,
                &self.freq_factors,
            ))?;
            let v = c(self.layers[layer_id].v.apply(&normed))?;
            let cache = self.cache.get(&sequence).expect("cache was inserted");
            c(cache.k[layer_id].slice_set(&c(k.reshape((1, self.kv_width)))?, 0, pos))?;
            c(cache.v[layer_id].slice_set(&c(v.reshape((1, self.kv_width)))?, 0, pos))?;
            let seq = pos + 1;
            let k_prefix = c(cache.k[layer_id].narrow(0, 0, seq))?;
            let v_prefix = c(cache.v[layer_id].narrow(0, 0, seq))?;
            let mixed = c(attend(
                &q,
                &k_prefix,
                &v_prefix,
                self.shape.heads,
                self.shape.kv_heads,
                self.shape.head_dim,
            ))?;
            let projected = c(self.layers[layer_id].o.apply(&mixed))?;
            hidden = c(hidden.broadcast_add(&projected))?;
            let mlp_in = c(rmsnorm(
                &hidden,
                &self.layers[layer_id].mlp_norm,
                self.shape.rms_eps,
            ))?;
            let gate = c(self.layers[layer_id].gate.apply(&mlp_in))?;
            let up = c(self.layers[layer_id].up.apply(&mlp_in))?;
            let activated = c(silu(&gate).and_then(|gate| gate.broadcast_mul(&up)))?;
            let down = c(self.layers[layer_id].down.apply(&activated))?;
            hidden = c(hidden.broadcast_add(&down))?;
        }
        let final_hidden = c(rmsnorm(&hidden, &self.final_norm, self.shape.rms_eps))?;
        let logits = c(self
            .lm_head
            .apply(&final_hidden)
            .and_then(|row| row.to_vec1::<f32>()))?;
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "llama logits are non-finite",
            ));
        }
        let cache = self.cache.get(&sequence).expect("cache was inserted");
        for layer_id in 0..self.layers.len() {
            let width = self.kv_width;
            let k_host = c(cache.k[layer_id]
                .narrow(0, pos, 1)
                .and_then(|row| row.reshape(width))
                .and_then(|row| row.to_vec1::<f32>()))?;
            let v_host = c(cache.v[layer_id]
                .narrow(0, pos, 1)
                .and_then(|row| row.reshape(width))
                .and_then(|row| row.to_vec1::<f32>()))?;
            kv.write_k(sequence, layer_id as u32, position, &k_host)?;
            kv.write_v(sequence, layer_id as u32, position, &v_host)?;
        }
        kv.commit_token(sequence)?;
        Ok(logits)
    }
}

impl ExecutableModel for CudaWideLlama {
    fn identity(&self) -> &ModelRuntimeIdentity {
        &self.identity
    }

    fn capabilities(&self) -> &ModelCapabilities {
        &self.capabilities
    }

    fn kv_layout(&self) -> KvLayout {
        self.layout
    }

    fn prefill(
        &mut self,
        batch: &PrefillBatch,
        kv: &mut dyn KvStore,
    ) -> Result<PrefillOutput, InferFailure> {
        if kv.layout() != &self.layout {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv layout does not match knolo.llama.v1",
            ));
        }
        if batch.token_ids.is_empty() {
            return Err(fail(
                ErrorCode::PromptCompilationFailed,
                "llama prefill prompt is empty",
            ));
        }
        self.reap(kv);
        let len = kv.token_len(batch.sequence_id)?;
        self.ensure_room(len, batch.token_ids.len() as u32)?;
        let mut logits = Vec::new();
        for token in &batch.token_ids {
            logits = self.step(*token, kv, batch.sequence_id)?;
        }
        Ok(PrefillOutput {
            tokens_written: batch.token_ids.len() as u32,
            logits,
        })
    }

    fn decode(
        &mut self,
        batch: &DecodeBatch,
        kv: &mut dyn KvStore,
    ) -> Result<DecodeOutput, InferFailure> {
        if kv.layout() != &self.layout {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv layout does not match knolo.llama.v1",
            ));
        }
        if batch.token_ids.len() != 1 || batch.positions.len() != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "llama decode accepts one token",
            ));
        }
        self.reap(kv);
        let len = kv.token_len(batch.sequence_id)?;
        if batch.positions[0] != len {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "decode position must equal the cache length",
            ));
        }
        Ok(DecodeOutput {
            logits: self.step(batch.token_ids[0], kv, batch.sequence_id)?,
        })
    }
}

fn load_layer(
    tensors: &LlamaTensors,
    shape: &LlamaShape,
    layer: usize,
    device: &Device,
) -> Result<Layer, InferFailure> {
    let prefix = format!("layers.{layer}");
    let kv_rows = shape.kv_heads * shape.head_dim;
    Ok(Layer {
        attn_norm: upload_vector(
            tensors,
            &format!("{prefix}.attn_norm.weight"),
            shape.hidden,
            device,
        )?,
        q: upload_linear(
            tensors,
            &format!("{prefix}.attn.q.weight"),
            shape.hidden,
            shape.hidden,
            device,
        )?,
        k: upload_linear(
            tensors,
            &format!("{prefix}.attn.k.weight"),
            kv_rows,
            shape.hidden,
            device,
        )?,
        v: upload_linear(
            tensors,
            &format!("{prefix}.attn.v.weight"),
            kv_rows,
            shape.hidden,
            device,
        )?,
        o: upload_linear(
            tensors,
            &format!("{prefix}.attn.o.weight"),
            shape.hidden,
            shape.hidden,
            device,
        )?,
        mlp_norm: upload_vector(
            tensors,
            &format!("{prefix}.mlp_norm.weight"),
            shape.hidden,
            device,
        )?,
        gate: upload_linear(
            tensors,
            &format!("{prefix}.mlp.gate.weight"),
            shape.intermediate,
            shape.hidden,
            device,
        )?,
        up: upload_linear(
            tensors,
            &format!("{prefix}.mlp.up.weight"),
            shape.intermediate,
            shape.hidden,
            device,
        )?,
        down: upload_linear(
            tensors,
            &format!("{prefix}.mlp.down.weight"),
            shape.hidden,
            shape.intermediate,
            device,
        )?,
    })
}

fn load_embed(
    tensors: &LlamaTensors,
    shape: &LlamaShape,
    device: &Device,
) -> Result<Embed, InferFailure> {
    if let Some(packed) = tensors
        .packed
        .iter()
        .find(|tensor| tensor.name == "embed.weight")
    {
        if packed.rows != shape.vocab || packed.cols != shape.hidden {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                "tensor embed.weight does not match knolo.llama.v1",
            ));
        }
        return Ok(Embed::Packed(packed.clone()));
    }
    Ok(Embed::Dense(upload_dense(
        tensors,
        "embed.weight",
        shape.vocab,
        shape.hidden,
        device,
    )?))
}

fn upload_vector(
    tensors: &LlamaTensors,
    name: &str,
    len: usize,
    device: &Device,
) -> Result<Tensor, InferFailure> {
    let values = dense(tensors, name, len)?;
    c(Tensor::from_vec(values, len, device))
}

fn upload_dense(
    tensors: &LlamaTensors,
    name: &str,
    rows: usize,
    cols: usize,
    device: &Device,
) -> Result<Tensor, InferFailure> {
    let values = dense(tensors, name, rows * cols)?;
    c(Tensor::from_vec(values, (rows, cols), device))
}

fn upload_linear(
    tensors: &LlamaTensors,
    name: &str,
    rows: usize,
    cols: usize,
    device: &Device,
) -> Result<Linear, InferFailure> {
    if let Some(packed) = tensors.packed.iter().find(|tensor| tensor.name == name) {
        if packed.rows != rows || packed.cols != cols {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                format!("tensor {name} does not match knolo.llama.v1"),
            ));
        }
        return upload_quant(packed, device);
    }
    Ok(Linear::Dense(upload_dense(
        tensors, name, rows, cols, device,
    )?))
}

fn upload_quant(packed: &PackedTensor, device: &Device) -> Result<Linear, InferFailure> {
    let dtype = ggml_dtype(packed.tensor_type)?;
    let tensor = qtensor_from_ggml(
        dtype,
        packed.bytes.as_ref(),
        vec![packed.rows, packed.cols],
        device,
    )
    .map_err(|err| {
        fail(
            ErrorCode::UnsupportedQuantization,
            format!(
                "tensor {} could not be placed on slot-0: {err}",
                packed.name
            ),
        )
    })?;
    let matmul = QMatMul::from_qtensor(tensor).map_err(|err| {
        fail(
            ErrorCode::UnsupportedQuantization,
            format!(
                "tensor {} could not be placed on slot-0: {err}",
                packed.name
            ),
        )
    })?;
    Ok(Linear::Quant(matmul))
}

fn ggml_dtype(tensor_type: GgufTensorType) -> Result<GgmlDType, InferFailure> {
    Ok(match tensor_type {
        GgufTensorType::Q8_0 => GgmlDType::Q8_0,
        GgufTensorType::Q4_K => GgmlDType::Q4K,
        GgufTensorType::Q5_K => GgmlDType::Q5K,
        GgufTensorType::Q6_K => GgmlDType::Q6K,
        GgufTensorType::F32 | GgufTensorType::F16 => {
            return Err(fail(
                ErrorCode::UnsupportedQuantization,
                "a dense llama matrix is not a quantized payload",
            ))
        }
    })
}

fn dense(tensors: &LlamaTensors, name: &str, len: usize) -> Result<Vec<f32>, InferFailure> {
    let values = tensors.dense.get(name).ok_or_else(|| {
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
    Ok(values.as_ref().to_vec())
}

fn c<T>(result: Result<T, candle_core::Error>) -> Result<T, InferFailure> {
    result.map_err(|err| {
        fail(
            ErrorCode::ContractInvalid,
            format!("candle forward failed: {err}"),
        )
    })
}

fn rmsnorm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor, candle_core::Error> {
    let mean = x.sqr()?.mean_all()?;
    let scale = (mean + f64::from(eps))?.sqrt()?.recip()?;
    x.broadcast_mul(&scale)?.broadcast_mul(weight)
}

/// llama.cpp `LLAMA_ROPE_TYPE_NORM` on the device. Adjacent pairs, same
/// angles as the CPU wide forward. `factors` are the Llama 3 per-pair
/// divisors. Empty means the unscaled base.
fn rope(
    flat: &Tensor,
    n_heads: usize,
    head_dim: usize,
    position: u32,
    theta: f32,
    factors: &[f32],
) -> Result<Tensor, candle_core::Error> {
    let half = head_dim / 2;
    if !factors.is_empty() && factors.len() != half {
        return Err(candle_core::Error::Msg(
            "rope frequency factors do not match the head dimension".into(),
        ));
    }
    let x = flat.reshape((n_heads, half, 2))?;
    let x0 = x.narrow(2, 0, 1)?;
    let x1 = x.narrow(2, 1, 1)?;
    let dim = head_dim as f32;
    let mut cos_v = Vec::with_capacity(half);
    let mut sin_v = Vec::with_capacity(half);
    for index in 0..half {
        let mut inv = theta.powf(-((2 * index) as f32) / dim);
        if !factors.is_empty() {
            let factor = factors[index];
            if !factor.is_finite() || factor <= 0.0 {
                return Err(candle_core::Error::Msg(
                    "rope frequency factor is not positive".into(),
                ));
            }
            inv /= factor;
        }
        let (sin, cos) = (position as f32 * inv).sin_cos();
        cos_v.push(cos);
        sin_v.push(sin);
    }
    let cos = Tensor::from_vec(cos_v, (1, half, 1), flat.device())?;
    let sin = Tensor::from_vec(sin_v, (1, half, 1), flat.device())?;
    let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let y1 = (x1.broadcast_mul(&cos)? + x0.broadcast_mul(&sin)?)?;
    Tensor::cat(&[&y0, &y1], 2)?.reshape((n_heads * head_dim,))
}

fn silu(x: &Tensor) -> Result<Tensor, candle_core::Error> {
    let zero = Tensor::zeros(x.shape(), x.dtype(), x.device())?;
    let ones = Tensor::ones(x.shape(), x.dtype(), x.device())?;
    let positive = (&ones / (&ones + x.neg()?.exp()?)?)?;
    let exp_x = x.exp()?;
    let negative = (&exp_x / (&exp_x + &ones)?)?;
    let mask = x.broadcast_lt(&zero)?;
    x.broadcast_mul(&mask.where_cond(&negative, &positive)?)
}

fn attend(
    q: &Tensor,
    k_prefix: &Tensor,
    v_prefix: &Tensor,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor, candle_core::Error> {
    let group = heads / kv_heads;
    let seq = k_prefix.dim(0)?;
    let q = q.reshape((kv_heads, group, head_dim))?.contiguous()?;
    let k = k_prefix
        .reshape((seq, kv_heads, head_dim))?
        .transpose(0, 1)?
        .contiguous()?;
    let v = v_prefix
        .reshape((seq, kv_heads, head_dim))?
        .transpose(0, 1)?
        .contiguous()?;
    let scale = (head_dim as f64).sqrt().recip();
    let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
    let scores = (scores * scale)?;
    let mixed = softmax_last(&scores)?.matmul(&v)?;
    mixed.reshape((heads * head_dim,))
}

fn softmax_last(scores: &Tensor) -> Result<Tensor, candle_core::Error> {
    let max = scores.max_keepdim(D::Minus1)?;
    let exp = scores.broadcast_sub(&max)?.exp()?;
    let sum = exp.sum_keepdim(D::Minus1)?;
    exp.broadcast_div(&sum)
}
