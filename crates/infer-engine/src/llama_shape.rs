//! Geometry for `knolo.llama.v1`. The toy fixture is the constant shape.
//! A wider inventory carries its own counts. RoPE stays rotate-half.

use std::collections::BTreeMap;
use std::sync::Arc;

use infer_artifact::GgufTensorType;

/// Tokens reserved for a Llama that is not the llama-tiny fixture.
pub const LLAMA_RUN_CONTEXT: u32 = 2048;

/// KV block used for every Llama placement. The toy context equals this.
pub const LLAMA_BLOCK: u32 = 16;

/// Host KV pool cap for a Llama whose context is wider than one block.
pub const RUN_KV_POOL_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// One decoded Llama. The toy numbers match `llama/weights.rs`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LlamaShape {
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub context: u32,
}

impl LlamaShape {
    pub fn toy() -> Self {
        Self {
            vocab: 32,
            hidden: 8,
            layers: 2,
            heads: 2,
            kv_heads: 1,
            head_dim: 4,
            intermediate: 16,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            context: 16,
        }
    }

    pub fn is_toy(self) -> bool {
        self == Self::toy()
    }

    pub fn block_size(self) -> u32 {
        LLAMA_BLOCK
    }

    /// Toy and single-block models keep the eight-page pool. A wider context
    /// takes one page per block.
    pub fn page_count(self) -> u32 {
        if self.is_toy() || self.context <= LLAMA_BLOCK {
            8
        } else {
            self.context / LLAMA_BLOCK
        }
    }
}

/// Quantized matrix kept in its GGUF payload. The f32 matrix is not stored.
#[derive(Debug, Clone)]
pub struct PackedTensor {
    pub name: String,
    pub tensor_type: GgufTensorType,
    pub rows: usize,
    pub cols: usize,
    pub bytes: Arc<[u8]>,
}

/// Weights for a Llama that is not the toy fixture.
#[derive(Debug, Clone)]
pub struct LlamaTensors {
    pub dense: BTreeMap<String, Arc<[f32]>>,
    pub packed: Vec<PackedTensor>,
}
