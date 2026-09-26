//! Candle graph for the same micro-transformer the oracle defines.
//!
//! Linear maps, RMSNorm, SiLU, and softmax go through Candle. RoPE is a
//! separate f32 loop so a transcription difference fails the parity test.
//! With the `cuda` feature, those Candle ops run on device ordinal 0. The KV
//! block stays on the host.

use candle_core::{Device, Tensor};
use infer_contracts::{fail, ErrorCode, InferFailure, ModelImageV1, PlacementPlanV1, TensorSpecV1};
#[cfg(feature = "cuda")]
use infer_engine::accept_cuda_placement;
use infer_engine::{
    accept_cpu_placement, kv_width, micro_kv_layout, MicroAdapter, ADAPTER_ID, HEADS, HEAD_DIM,
    HIDDEN, INTERMEDIATE, KV_HEADS, RMS_EPS, ROPE_THETA, VOCAB,
};
use infer_engine::{
    ArchitectureAdapter, DecodeBatch, DecodeOutput, ExecutableModel, KvLayout, KvStore,
    ModelCapabilities, ModelRuntimeIdentity, PrefillBatch, PrefillOutput, TensorBackend,
    VerifiedWeightSource,
};

pub const CANDLE_CPU_VERSION: &str = "0.8.4";

pub struct CandleCpuBackend;

impl TensorBackend for CandleCpuBackend {
    fn id(&self) -> &'static str {
        "candle-cpu"
    }

    fn device(&self) -> &'static str {
        "cpu"
    }
}

#[cfg(feature = "cuda")]
pub struct CandleCudaBackend;

#[cfg(feature = "cuda")]
impl TensorBackend for CandleCudaBackend {
    fn id(&self) -> &'static str {
        "candle-cuda"
    }

    fn device(&self) -> &'static str {
        "slot-0"
    }
}

struct LayerTensors {
    attn_norm: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    mlp_norm: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

pub struct CandleMicroModel {
    identity: ModelRuntimeIdentity,
    capabilities: ModelCapabilities,
    layout: KvLayout,
    context_limit: u32,
    embed: Tensor,
    layers: Vec<LayerTensors>,
    final_norm: Tensor,
    lm_head: Tensor,
    device: Device,
}

impl CandleMicroModel {
    fn new(
        source: &VerifiedWeightSource,
        context_limit: u32,
        device: Device,
    ) -> Result<Self, InferFailure> {
        if context_limit == 0 || context_limit > infer_engine::BLOCK_SIZE {
            return Err(fail(
                ErrorCode::PlacementUnsatisfiable,
                "micro context does not fit in one KV block",
            ));
        }
        let weights = &source.weights;
        let mut layers = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            layers.push(LayerTensors {
                attn_norm: vector(&layer.attn_norm, &device)?,
                q: matrix(&layer.q, HIDDEN, HIDDEN, &device)?,
                k: matrix(&layer.k, kv_width(), HIDDEN, &device)?,
                v: matrix(&layer.v, kv_width(), HIDDEN, &device)?,
                o: matrix(&layer.o, HIDDEN, HIDDEN, &device)?,
                mlp_norm: vector(&layer.mlp_norm, &device)?,
                gate: matrix(&layer.gate, INTERMEDIATE, HIDDEN, &device)?,
                up: matrix(&layer.up, INTERMEDIATE, HIDDEN, &device)?,
                down: matrix(&layer.down, HIDDEN, INTERMEDIATE, &device)?,
            });
        }
        Ok(Self {
            identity: ModelRuntimeIdentity {
                adapter_id: ADAPTER_ID,
                model_image_root: source.image_root.clone(),
                artifact_root: source.artifact_root.clone(),
                runtime_root: source.runtime_root.clone(),
            },
            capabilities: ModelCapabilities {
                max_context_tokens: context_limit,
                vocab_size: VOCAB as u32,
                text_generation: true,
            },
            layout: micro_kv_layout(),
            context_limit,
            embed: matrix(&weights.embed, VOCAB, HIDDEN, &device)?,
            layers,
            final_norm: vector(&weights.final_norm, &device)?,
            lm_head: matrix(&weights.lm_head, VOCAB, HIDDEN, &device)?,
            device,
        })
    }

    fn cache_len(&self, kv: &dyn KvStore, sequence: u64) -> Result<u32, InferFailure> {
        if kv.layout() != &self.layout {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv layout does not match knolo.micro.v1",
            ));
        }
        kv.token_len(sequence)
    }

    fn ensure_room(&self, len: u32, extra: u32) -> Result<(), InferFailure> {
        let end = len.checked_add(extra).ok_or_else(|| {
            fail(
                ErrorCode::ContextLimitExceeded,
                "micro context length overflows",
            )
        })?;
        if end > self.context_limit || end > self.layout.block_size {
            return Err(fail(
                ErrorCode::ContextLimitExceeded,
                "micro prompt does not fit in the reserved context",
            ));
        }
        Ok(())
    }

    fn step(
        &self,
        token: u32,
        kv: &mut dyn KvStore,
        sequence: u64,
    ) -> Result<Vec<f32>, InferFailure> {
        if token as usize >= VOCAB {
            return Err(fail(
                ErrorCode::PromptCompilationFailed,
                "token id is outside the micro vocabulary",
            ));
        }
        let position = self.cache_len(kv, sequence)?;
        self.ensure_room(position, 1)?;
        let mut hidden = embedding_row(&self.embed, token as usize)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let normed = rmsnorm(&hidden, &layer.attn_norm)?;
            let normed_v = to_vec(&normed)?;
            let mut q = gemv(&layer.q, &normed_v)?;
            let mut k = gemv(&layer.k, &normed_v)?;
            let v = gemv(&layer.v, &normed_v)?;
            for head in 0..HEADS {
                rope_inplace(&mut q[head * HEAD_DIM..(head + 1) * HEAD_DIM], position)?;
            }
            for head in 0..KV_HEADS {
                rope_inplace(&mut k[head * HEAD_DIM..(head + 1) * HEAD_DIM], position)?;
            }
            let layer_id = u32::try_from(layer_index).expect("layer index fits u32");
            kv.write_k(sequence, layer_id, position, &k)?;
            kv.write_v(sequence, layer_id, position, &v)?;
            let mixed = attend(kv, sequence, layer_id, position, &q, &self.device)?;
            let projected = as_tensor(&gemv(&layer.o, &mixed)?, &self.device)?;
            hidden = hidden.broadcast_add(&projected).map_err(candle_err)?;
            let mlp_normed = rmsnorm(&hidden, &layer.mlp_norm)?;
            let mlp_v = to_vec(&mlp_normed)?;
            let gate = as_tensor(&gemv(&layer.gate, &mlp_v)?, &self.device)?;
            let up = as_tensor(&gemv(&layer.up, &mlp_v)?, &self.device)?;
            let activated = gate
                .silu()
                .map_err(candle_err)?
                .mul(&up)
                .map_err(candle_err)?;
            let down = gemv(&layer.down, &to_vec(&activated)?)?;
            hidden = hidden
                .broadcast_add(&as_tensor(&down, &self.device)?)
                .map_err(candle_err)?;
        }
        let final_hidden = rmsnorm(&hidden, &self.final_norm)?;
        let logits = gemv(&self.lm_head, &to_vec(&final_hidden)?)?;
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "micro logits are non-finite",
            ));
        }
        kv.commit_token(sequence)?;
        Ok(logits)
    }
}

fn attend(
    kv: &dyn KvStore,
    sequence: u64,
    layer: u32,
    position: u32,
    q: &[f32],
    device: &Device,
) -> Result<Vec<f32>, InferFailure> {
    let scale = (HEAD_DIM as f32).sqrt().recip();
    let width = kv_width();
    let steps = position as usize + 1;
    let mut keys = Vec::with_capacity(steps);
    let mut values = Vec::with_capacity(steps);
    for token in 0..=position {
        let mut key = vec![0.0; width];
        let mut value = vec![0.0; width];
        kv.read_k(sequence, layer, token, &mut key)?;
        kv.read_v(sequence, layer, token, &mut value)?;
        keys.push(key);
        values.push(value);
    }
    let group = HEADS / KV_HEADS;
    let mut concat = vec![0.0; HIDDEN];
    for head in 0..HEADS {
        let kv_head = head / group;
        let query = &q[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let mut score_values = Vec::with_capacity(steps);
        for key in &keys {
            let key_head = &key[kv_head * HEAD_DIM..(kv_head + 1) * HEAD_DIM];
            let mut dot = 0.0f32;
            for (left, right) in query.iter().zip(key_head) {
                dot += left * right;
            }
            score_values.push(dot * scale);
        }
        let scores = as_tensor(&score_values, device)?;
        let probs = softmax_1d(&scores)?;
        let mut packed = Vec::with_capacity(steps * HEAD_DIM);
        for value in &values {
            packed.extend_from_slice(&value[kv_head * HEAD_DIM..(kv_head + 1) * HEAD_DIM]);
        }
        let value_matrix =
            Tensor::from_slice(&packed, (steps, HEAD_DIM), device).map_err(candle_err)?;
        let row = probs
            .reshape((1, steps))
            .map_err(candle_err)?
            .matmul(&value_matrix)
            .map_err(candle_err)?;
        let mixed = row
            .reshape(HEAD_DIM)
            .map_err(candle_err)?
            .to_vec1::<f32>()
            .map_err(candle_err)?;
        concat[head * HEAD_DIM..(head + 1) * HEAD_DIM].copy_from_slice(&mixed);
    }
    Ok(concat)
}

/// Same rotate-half angles as the oracle: `theta^(-2i/dim) * position`.
fn rope_inplace(head: &mut [f32], position: u32) -> Result<(), InferFailure> {
    let half = head.len() / 2;
    let dim = head.len() as f32;
    for i in 0..half {
        let inv = ROPE_THETA.powf(-((2 * i) as f32) / dim);
        let angle = position as f32 * inv;
        let (sin, cos) = angle.sin_cos();
        let x1 = head[i];
        let x2 = head[half + i];
        head[i] = x1 * cos - x2 * sin;
        head[half + i] = x2 * cos + x1 * sin;
        if !head[i].is_finite() || !head[half + i].is_finite() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "micro forward produced a non-finite rope",
            ));
        }
    }
    Ok(())
}

fn gemv(weight: &Tensor, x: &[f32]) -> Result<Vec<f32>, InferFailure> {
    let (rows, cols) = weight.dims2().map_err(candle_err)?;
    if x.len() != cols {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "candle matrix shape does not match its vector",
        ));
    }
    let column = Tensor::from_slice(x, (cols, 1), weight.device()).map_err(candle_err)?;
    let product = weight.matmul(&column).map_err(candle_err)?;
    product
        .reshape(rows)
        .map_err(candle_err)?
        .to_vec1::<f32>()
        .map_err(candle_err)
}

fn rmsnorm(x: &Tensor, weight: &Tensor) -> Result<Tensor, InferFailure> {
    let hidden = x.dims1().map_err(candle_err)?;
    let sum = x.sqr().map_err(candle_err)?.sum(0).map_err(candle_err)?;
    let sum = sum.to_scalar::<f32>().map_err(candle_err)?;
    let scale = (sum / hidden as f32 + RMS_EPS).sqrt().recip();
    if !scale.is_finite() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "micro forward produced a non-finite rmsnorm",
        ));
    }
    let scaled = x.affine(f64::from(scale), 0.0).map_err(candle_err)?;
    scaled.broadcast_mul(weight).map_err(candle_err)
}

fn softmax_1d(scores: &Tensor) -> Result<Tensor, InferFailure> {
    let max = scores.max(0).map_err(candle_err)?;
    let shifted = scores.broadcast_sub(&max).map_err(candle_err)?;
    let exps = shifted.exp().map_err(candle_err)?;
    let sum = exps.sum(0).map_err(candle_err)?;
    exps.broadcast_div(&sum).map_err(candle_err)
}

fn embedding_row(table: &Tensor, index: usize) -> Result<Tensor, InferFailure> {
    table
        .narrow(0, index, 1)
        .map_err(candle_err)?
        .reshape(HIDDEN)
        .map_err(candle_err)
}

fn matrix(data: &[f32], rows: usize, cols: usize, device: &Device) -> Result<Tensor, InferFailure> {
    Tensor::from_slice(data, (rows, cols), device).map_err(candle_err)
}

fn vector(data: &[f32], device: &Device) -> Result<Tensor, InferFailure> {
    Tensor::from_slice(data, data.len(), device).map_err(candle_err)
}

fn as_tensor(data: &[f32], device: &Device) -> Result<Tensor, InferFailure> {
    vector(data, device)
}

fn to_vec(tensor: &Tensor) -> Result<Vec<f32>, InferFailure> {
    tensor.to_vec1::<f32>().map_err(candle_err)
}

fn candle_err(err: candle_core::Error) -> InferFailure {
    fail(
        ErrorCode::ContractInvalid,
        format!("candle forward failed: {err}"),
    )
}

impl ExecutableModel for CandleMicroModel {
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
        if batch.token_ids.is_empty() {
            return Err(fail(
                ErrorCode::PromptCompilationFailed,
                "micro prefill prompt is empty",
            ));
        }
        for token in &batch.token_ids {
            if *token as usize >= VOCAB {
                return Err(fail(
                    ErrorCode::PromptCompilationFailed,
                    "token id is outside the micro vocabulary",
                ));
            }
        }
        let len = self.cache_len(kv, batch.sequence_id)?;
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
        if batch.token_ids.len() != 1 || batch.positions.len() != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "micro decode accepts one token",
            ));
        }
        let len = self.cache_len(kv, batch.sequence_id)?;
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

pub struct CandleMicroAdapter;

impl ArchitectureAdapter for CandleMicroAdapter {
    fn id(&self) -> &'static str {
        ADAPTER_ID
    }

    fn validate_config(&self, image: &ModelImageV1) -> Result<(), InferFailure> {
        MicroAdapter.validate_config(image)
    }

    fn expected_tensors(&self, image: &ModelImageV1) -> Result<Vec<TensorSpecV1>, InferFailure> {
        MicroAdapter.expected_tensors(image)
    }

    fn build(
        &self,
        source: &VerifiedWeightSource,
        placement: &PlacementPlanV1,
        backend: &dyn TensorBackend,
    ) -> Result<Box<dyn ExecutableModel>, InferFailure> {
        match (backend.id(), backend.device()) {
            ("reference-f32", "cpu") => MicroAdapter.build(source, placement, backend),
            ("candle-cpu", "cpu") => {
                accept_cpu_placement(source, placement)?;
                Ok(Box::new(CandleMicroModel::new(
                    source,
                    placement.context_reservation_tokens,
                    Device::Cpu,
                )?))
            }
            #[cfg(feature = "cuda")]
            ("candle-cuda", "slot-0") => {
                accept_cuda_placement(source, placement)?;
                let device = Device::new_cuda(0).map_err(|err| {
                    fail(
                        ErrorCode::PlacementUnsatisfiable,
                        format!("cuda device slot-0 is not available: {err}"),
                    )
                })?;
                Ok(Box::new(CandleMicroModel::new(
                    source,
                    placement.context_reservation_tokens,
                    device,
                )?))
            }
            _ => Err(fail(
                ErrorCode::UnsupportedKernel,
                if cfg!(feature = "cuda") {
                    "knolo.micro.v1 candle build accepts reference-f32, candle-cpu, or candle-cuda"
                } else {
                    "knolo.micro.v1 candle build accepts reference-f32 or candle-cpu"
                },
            )),
        }
    }
}

static CANDLE_ADAPTER: CandleMicroAdapter = CandleMicroAdapter;

pub fn cpu_adapter_by_id(id: &str) -> Result<&'static CandleMicroAdapter, InferFailure> {
    if id == ADAPTER_ID {
        Ok(&CANDLE_ADAPTER)
    } else {
        Err(fail(
            ErrorCode::UnsupportedArchitecture,
            format!("architecture adapter {id} is not compiled in"),
        ))
    }
}
