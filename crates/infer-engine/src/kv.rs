//! One contiguous KV block per sequence. The run command uses `PagedKv`.

use std::collections::BTreeMap;

use infer_contracts::{fail, ErrorCode, InferFailure};

use crate::traits::{KvLayout, KvSnapshot, KvStore};

const MAX_SEQUENCES: usize = 8;

struct SequenceState {
    block_id: u32,
    len: u32,
    k: Vec<f32>,
    v: Vec<f32>,
}

pub struct SingleBlockKv {
    layout: KvLayout,
    next_block: u32,
    sequences: BTreeMap<u64, SequenceState>,
}

impl SingleBlockKv {
    pub fn new(layout: KvLayout) -> Result<Self, InferFailure> {
        if layout.layers == 0
            || layout.kv_heads == 0
            || layout.head_dim == 0
            || layout.block_size == 0
            || layout.dtype != "f32"
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv layout is not a single f32 block",
            ));
        }
        Ok(Self {
            layout,
            next_block: 0,
            sequences: BTreeMap::new(),
        })
    }

    pub fn snapshot(&self, sequence: u64) -> Result<KvSnapshot, InferFailure> {
        let state = self.sequence(sequence)?;
        Ok(KvSnapshot {
            len: state.len,
            block_table: vec![state.block_id],
            k: state.k.clone(),
            v: state.v.clone(),
        })
    }

    fn sequence(&self, sequence: u64) -> Result<&SequenceState, InferFailure> {
        self.sequences
            .get(&sequence)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "sequence has no KV block"))
    }

    fn sequence_mut(&mut self, sequence: u64) -> Result<&mut SequenceState, InferFailure> {
        self.sequences
            .get_mut(&sequence)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "sequence has no KV block"))
    }

    fn kv_width(&self) -> usize {
        self.layout.kv_heads as usize * self.layout.head_dim as usize
    }

    fn offset(&self, layer: u32, token: u32) -> Result<usize, InferFailure> {
        if layer >= self.layout.layers {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv layer is outside the layout",
            ));
        }
        if token >= self.layout.block_size {
            return Err(fail(
                ErrorCode::ContextLimitExceeded,
                "kv token is outside the single block",
            ));
        }
        Ok((layer as usize * self.layout.block_size as usize + token as usize) * self.kv_width())
    }
}

impl KvStore for SingleBlockKv {
    fn layout(&self) -> &KvLayout {
        &self.layout
    }

    fn begin_sequence(&mut self, sequence: u64) -> Result<(), InferFailure> {
        if self.sequences.contains_key(&sequence) {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "sequence already has a KV block",
            ));
        }
        if self.sequences.len() >= MAX_SEQUENCES {
            return Err(fail(
                ErrorCode::InsufficientMemory,
                "single-block KV store is full",
            ));
        }
        let slots = self.layout.layers as usize * self.layout.block_size as usize * self.kv_width();
        let block_id = self.next_block;
        self.next_block = self.next_block.saturating_add(1);
        self.sequences.insert(
            sequence,
            SequenceState {
                block_id,
                len: 0,
                k: vec![0.0; slots],
                v: vec![0.0; slots],
            },
        );
        Ok(())
    }

    fn release_sequence(&mut self, sequence: u64) -> Result<(), InferFailure> {
        if self.sequences.remove(&sequence).is_none() {
            return Err(fail(ErrorCode::ContractInvalid, "sequence has no KV block"));
        }
        Ok(())
    }

    fn token_len(&self, sequence: u64) -> Result<u32, InferFailure> {
        Ok(self.sequence(sequence)?.len)
    }

    fn block_table(&self, sequence: u64) -> Result<Vec<u32>, InferFailure> {
        Ok(vec![self.sequence(sequence)?.block_id])
    }

    fn write_k(
        &mut self,
        sequence: u64,
        layer: u32,
        token: u32,
        values: &[f32],
    ) -> Result<(), InferFailure> {
        self.write(sequence, layer, token, values, true)
    }

    fn write_v(
        &mut self,
        sequence: u64,
        layer: u32,
        token: u32,
        values: &[f32],
    ) -> Result<(), InferFailure> {
        self.write(sequence, layer, token, values, false)
    }

    fn read_k(
        &self,
        sequence: u64,
        layer: u32,
        token: u32,
        out: &mut [f32],
    ) -> Result<(), InferFailure> {
        self.read(sequence, layer, token, out, true)
    }

    fn read_v(
        &self,
        sequence: u64,
        layer: u32,
        token: u32,
        out: &mut [f32],
    ) -> Result<(), InferFailure> {
        self.read(sequence, layer, token, out, false)
    }

    fn commit_token(&mut self, sequence: u64) -> Result<u32, InferFailure> {
        let block_size = self.layout.block_size;
        let state = self.sequence_mut(sequence)?;
        if state.len >= block_size {
            return Err(fail(ErrorCode::ContextLimitExceeded, "kv block is full"));
        }
        state.len += 1;
        Ok(state.len)
    }
}

impl SingleBlockKv {
    fn write(
        &mut self,
        sequence: u64,
        layer: u32,
        token: u32,
        values: &[f32],
        key: bool,
    ) -> Result<(), InferFailure> {
        let width = self.kv_width();
        if values.len() != width {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv vector width does not match the layout",
            ));
        }
        let offset = self.offset(layer, token)?;
        let len = self.sequence(sequence)?.len;
        if token != len {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv write must fill the open token",
            ));
        }
        let slab = if key {
            &mut self.sequence_mut(sequence)?.k
        } else {
            &mut self.sequence_mut(sequence)?.v
        };
        slab[offset..offset + width].copy_from_slice(values);
        Ok(())
    }

    fn read(
        &self,
        sequence: u64,
        layer: u32,
        token: u32,
        out: &mut [f32],
        key: bool,
    ) -> Result<(), InferFailure> {
        let width = self.kv_width();
        if out.len() != width {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv vector width does not match the layout",
            ));
        }
        let offset = self.offset(layer, token)?;
        let state = self.sequence(sequence)?;
        if token > state.len {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv read is past the open token",
            ));
        }
        let slab = if key { &state.k } else { &state.v };
        out.copy_from_slice(&slab[offset..offset + width]);
        Ok(())
    }
}
