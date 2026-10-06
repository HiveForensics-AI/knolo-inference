//! f32 forward for `knolo.llama.v1`. The loops are the micro primitives with
//! this adapter's vocabulary.

use infer_contracts::{fail, ErrorCode, InferFailure, ModelImageV1, PlacementPlanV1, TensorSpecV1};

use crate::llama::weights::{
    kv_width, ADAPTER_ID, BLOCK_SIZE, HEADS, HEAD_DIM, HIDDEN, INTERMEDIATE, KV_HEADS, LAYERS,
    RMS_EPS, ROPE_THETA, VOCAB,
};
use crate::micro::{add_residual, gemv, rmsnorm, rope, silu, softmax, MicroWeights};
use crate::traits::{
    ArchitectureAdapter, DecodeBatch, DecodeOutput, ExecutableModel, KvLayout, KvStore,
    ModelCapabilities, ModelRuntimeIdentity, PrefillBatch, PrefillOutput, TensorBackend,
    VerifiedWeightSource,
};

pub struct LlamaOracle {
    identity: ModelRuntimeIdentity,
    capabilities: ModelCapabilities,
    layout: KvLayout,
    context_limit: u32,
    weights: MicroWeights,
    wide: Option<super::forward::WideForward>,
}

impl LlamaOracle {
    pub fn new(source: &VerifiedWeightSource, context_limit: u32) -> Result<Self, InferFailure> {
        if context_limit == 0 || context_limit > BLOCK_SIZE {
            return Err(fail(
                ErrorCode::PlacementUnsatisfiable,
                "llama context does not fit in one KV block",
            ));
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
            layout: super::llama_kv_layout(),
            context_limit,
            weights: source.weights.clone(),
            wide: None,
        })
    }

    pub fn wide(source: &VerifiedWeightSource, context_limit: u32) -> Result<Self, InferFailure> {
        let shape = source.llama.ok_or_else(|| {
            fail(
                ErrorCode::ModelImageInvalid,
                "wide llama forward is missing its shape",
            )
        })?;
        if shape.is_toy() {
            return Self::new(source, context_limit);
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
        Ok(Self {
            identity: ModelRuntimeIdentity {
                adapter_id: ADAPTER_ID,
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
            weights: source.weights.clone(),
            wide: Some(super::forward::WideForward { shape, tensors }),
        })
    }

    fn cache_len(&self, kv: &dyn KvStore, sequence: u64) -> Result<u32, InferFailure> {
        if kv.layout() != &self.layout {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv layout does not match knolo.llama.v1",
            ));
        }
        kv.token_len(sequence)
    }

    fn ensure_room(&self, len: u32, extra: u32) -> Result<(), InferFailure> {
        let end = len.checked_add(extra).ok_or_else(|| {
            fail(
                ErrorCode::ContextLimitExceeded,
                "llama context length overflows",
            )
        })?;
        if self.wide.is_some() {
            if end > self.context_limit {
                return Err(fail(
                    ErrorCode::ContextLimitExceeded,
                    "llama prompt does not fit in the reserved context",
                ));
            }
            return Ok(());
        }
        if end > self.context_limit || end > self.layout.block_size {
            return Err(fail(
                ErrorCode::ContextLimitExceeded,
                "llama prompt does not fit in the reserved context",
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
        if let Some(wide) = &self.wide {
            return super::forward::wide_step(wide, token, kv, sequence);
        }
        if token as usize >= VOCAB {
            return Err(fail(
                ErrorCode::PromptCompilationFailed,
                "token id is outside the llama vocabulary",
            ));
        }
        let position = self.cache_len(kv, sequence)?;
        self.ensure_room(position, 1)?;
        let mut hidden = row(&self.weights.embed, HIDDEN, token as usize)?.to_vec();
        for (layer_index, layer) in self.weights.layers.iter().enumerate() {
            let normed = rmsnorm(&hidden, &layer.attn_norm, RMS_EPS)?;
            let mut q = gemv(&layer.q, HIDDEN, HIDDEN, &normed)?;
            let mut k = gemv(&layer.k, kv_width(), HIDDEN, &normed)?;
            let v = gemv(&layer.v, kv_width(), HIDDEN, &normed)?;
            for head in 0..HEADS {
                rope(
                    &mut q[head * HEAD_DIM..(head + 1) * HEAD_DIM],
                    position,
                    ROPE_THETA,
                )?;
            }
            for head in 0..KV_HEADS {
                rope(
                    &mut k[head * HEAD_DIM..(head + 1) * HEAD_DIM],
                    position,
                    ROPE_THETA,
                )?;
            }
            let layer_id = u32::try_from(layer_index).expect("layer index fits u32");
            kv.write_k(sequence, layer_id, position, &k)?;
            kv.write_v(sequence, layer_id, position, &v)?;
            let mixed = attend(kv, sequence, layer_id, position, &q)?;
            let projected = gemv(&layer.o, HIDDEN, HIDDEN, &mixed)?;
            add_residual(&mut hidden, &projected)?;
            let mlp_in = rmsnorm(&hidden, &layer.mlp_norm, RMS_EPS)?;
            let gate = gemv(&layer.gate, INTERMEDIATE, HIDDEN, &mlp_in)?;
            let up = gemv(&layer.up, INTERMEDIATE, HIDDEN, &mlp_in)?;
            let mut activated = Vec::with_capacity(INTERMEDIATE);
            for (gate_v, up_v) in gate.iter().zip(&up) {
                activated.push(silu(*gate_v)? * up_v);
            }
            let down = gemv(&layer.down, HIDDEN, INTERMEDIATE, &activated)?;
            add_residual(&mut hidden, &down)?;
        }
        let final_hidden = rmsnorm(&hidden, &self.weights.final_norm, RMS_EPS)?;
        let logits = gemv(&self.weights.lm_head, VOCAB, HIDDEN, &final_hidden)?;
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
) -> Result<Vec<f32>, InferFailure> {
    let scale = (HEAD_DIM as f32).sqrt().recip();
    let mut concat = vec![0.0; HIDDEN];
    let group = HEADS / KV_HEADS;
    let mut keys = Vec::with_capacity(position as usize + 1);
    let mut values = Vec::with_capacity(position as usize + 1);
    for token in 0..=position {
        let mut key = vec![0.0; kv_width()];
        let mut value = vec![0.0; kv_width()];
        kv.read_k(sequence, layer, token, &mut key)?;
        kv.read_v(sequence, layer, token, &mut value)?;
        keys.push(key);
        values.push(value);
    }
    for head in 0..HEADS {
        let kv_head = head / group;
        let query = &q[head * HEAD_DIM..(head + 1) * HEAD_DIM];
        let mut scores = Vec::with_capacity(keys.len());
        for key in &keys {
            let key_head = &key[kv_head * HEAD_DIM..(kv_head + 1) * HEAD_DIM];
            let mut dot = 0.0f32;
            for (q_i, k_i) in query.iter().zip(key_head) {
                dot += q_i * k_i;
            }
            scores.push(dot * scale);
        }
        softmax(&mut scores)?;
        let mut mixed = vec![0.0; HEAD_DIM];
        for (score, value) in scores.iter().zip(&values) {
            let value_head = &value[kv_head * HEAD_DIM..(kv_head + 1) * HEAD_DIM];
            for (slot, component) in mixed.iter_mut().zip(value_head) {
                *slot += score * component;
            }
        }
        concat[head * HEAD_DIM..(head + 1) * HEAD_DIM].copy_from_slice(&mixed);
    }
    Ok(concat)
}

fn row(matrix: &[f32], cols: usize, index: usize) -> Result<&[f32], InferFailure> {
    let start = index * cols;
    matrix.get(start..start + cols).ok_or_else(|| {
        fail(
            ErrorCode::ContractInvalid,
            "embedding row is outside the table",
        )
    })
}

impl ExecutableModel for LlamaOracle {
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
                "llama prefill prompt is empty",
            ));
        }
        let vocab = self
            .wide
            .as_ref()
            .map(|wide| wide.shape.vocab)
            .unwrap_or(VOCAB);
        for token in &batch.token_ids {
            if *token as usize >= vocab {
                return Err(fail(
                    ErrorCode::PromptCompilationFailed,
                    "token id is outside the llama vocabulary",
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
                "llama decode accepts one token",
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

pub struct LlamaAdapter;

impl ArchitectureAdapter for LlamaAdapter {
    fn id(&self) -> &'static str {
        ADAPTER_ID
    }

    fn validate_config(&self, image: &ModelImageV1) -> Result<(), InferFailure> {
        super::validate_llama_image(image)
    }

    fn expected_tensors(&self, image: &ModelImageV1) -> Result<Vec<TensorSpecV1>, InferFailure> {
        self.validate_config(image)?;
        Ok(image.tensor_inventory.clone())
    }

    fn build(
        &self,
        source: &VerifiedWeightSource,
        placement: &PlacementPlanV1,
        backend: &dyn TensorBackend,
    ) -> Result<Box<dyn ExecutableModel>, InferFailure> {
        if backend.id() != "reference-f32" || backend.device() != "cpu" {
            return Err(fail(
                ErrorCode::UnsupportedKernel,
                "knolo.llama.v1 reference build accepts the reference-f32 cpu backend",
            ));
        }
        super::accept_llama_placement(source, placement)?;
        if source.llama.is_some_and(|shape| !shape.is_toy()) {
            return Ok(Box::new(LlamaOracle::wide(
                source,
                placement.context_reservation_tokens,
            )?));
        }
        Ok(Box::new(LlamaOracle::new(
            source,
            placement.context_reservation_tokens,
        )?))
    }
}

const _: () = {
    assert!(LAYERS > 0);
};
