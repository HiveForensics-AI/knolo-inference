//! Engine traits from the native-engine design. Candle types do not appear here.

use infer_artifact::TensorBytes;
use infer_contracts::{DigestHex, InferFailure, ModelImageV1, PlacementPlanV1, TensorSpecV1};

use crate::micro::MicroWeights;

#[derive(Debug, Clone)]
pub struct VerifiedWeightSource {
    pub image: ModelImageV1,
    pub image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub runtime_root: DigestHex,
    pub tensors: Vec<TensorBytes>,
    pub weights: MicroWeights,
    pub weight_bytes: u64,
}

pub type EngineError = InferFailure;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRuntimeIdentity {
    pub adapter_id: &'static str,
    pub model_image_root: infer_contracts::DigestHex,
    pub artifact_root: infer_contracts::DigestHex,
    pub runtime_root: infer_contracts::DigestHex,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub max_context_tokens: u32,
    pub vocab_size: u32,
    pub text_generation: bool,
}

/// One KV page holds `block_size` tokens for every layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayout {
    pub layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub dtype: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvSnapshot {
    pub len: u32,
    pub block_table: Vec<u32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct PrefillBatch {
    pub sequence_id: u64,
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct PrefillOutput {
    pub logits: Vec<f32>,
    pub tokens_written: u32,
}

#[derive(Debug, Clone)]
pub struct DecodeBatch {
    pub sequence_id: u64,
    pub token_ids: Vec<u32>,
    pub positions: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct DecodeOutput {
    pub logits: Vec<f32>,
}

pub trait TensorBackend: Send + Sync {
    fn id(&self) -> &'static str;
    fn device(&self) -> &'static str;
}

/// Reference f32 loops. `infer-native` supplies the Candle CPU backend.
pub struct ReferenceF32Backend;

impl TensorBackend for ReferenceF32Backend {
    fn id(&self) -> &'static str {
        "reference-f32"
    }

    fn device(&self) -> &'static str {
        "cpu"
    }
}

pub trait KvStore: Send {
    fn layout(&self) -> &KvLayout;
    fn begin_sequence(&mut self, sequence: u64) -> Result<(), EngineError>;
    fn release_sequence(&mut self, sequence: u64) -> Result<(), EngineError>;
    fn token_len(&self, sequence: u64) -> Result<u32, EngineError>;
    fn block_table(&self, sequence: u64) -> Result<Vec<u32>, EngineError>;
    fn write_k(
        &mut self,
        sequence: u64,
        layer: u32,
        token: u32,
        values: &[f32],
    ) -> Result<(), EngineError>;
    fn write_v(
        &mut self,
        sequence: u64,
        layer: u32,
        token: u32,
        values: &[f32],
    ) -> Result<(), EngineError>;
    fn read_k(
        &self,
        sequence: u64,
        layer: u32,
        token: u32,
        out: &mut [f32],
    ) -> Result<(), EngineError>;
    fn read_v(
        &self,
        sequence: u64,
        layer: u32,
        token: u32,
        out: &mut [f32],
    ) -> Result<(), EngineError>;
    fn commit_token(&mut self, sequence: u64) -> Result<u32, EngineError>;

    /// Drop a page reserved for the open token. The committed length stays.
    fn abort_pending(&mut self, _sequence: u64) -> Result<(), EngineError> {
        Ok(())
    }
}

pub trait ExecutableModel: Send {
    fn identity(&self) -> &ModelRuntimeIdentity;
    fn capabilities(&self) -> &ModelCapabilities;
    fn kv_layout(&self) -> KvLayout;
    fn prefill(
        &mut self,
        batch: &PrefillBatch,
        kv: &mut dyn KvStore,
    ) -> Result<PrefillOutput, EngineError>;
    fn decode(
        &mut self,
        batch: &DecodeBatch,
        kv: &mut dyn KvStore,
    ) -> Result<DecodeOutput, EngineError>;
}

pub trait ArchitectureAdapter: Send + Sync {
    fn id(&self) -> &'static str;
    fn validate_config(&self, image: &ModelImageV1) -> Result<(), EngineError>;
    fn expected_tensors(&self, image: &ModelImageV1) -> Result<Vec<TensorSpecV1>, EngineError>;
    fn build(
        &self,
        source: &VerifiedWeightSource,
        placement: &PlacementPlanV1,
        backend: &dyn TensorBackend,
    ) -> Result<Box<dyn ExecutableModel>, EngineError>;
}
