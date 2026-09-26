//! Fixed-size KV pages for the CPU path.
//!
//! A page is reserved for the open token and becomes part of the block table
//! only when that token is committed. Released pages are zeroed. Prefix cache
//! is not allocated.

use std::collections::{BTreeMap, BTreeSet};

use infer_contracts::{fail, ErrorCode, InferFailure};

use crate::traits::{KvLayout, KvSnapshot, KvStore};

/// Pages kept for `knolo-infer run` on the micro-model. One sequence uses one page.
pub const CPU_KV_PAGE_POOL: u32 = 8;

struct Page {
    k: Vec<f32>,
    v: Vec<f32>,
}

struct SequencePages {
    committed: Vec<u32>,
    pending: Option<u32>,
    len: u32,
    active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCensus {
    pub total: u32,
    pub free: u32,
    /// Pages held by active sequences. An inactive sequence can be evicted,
    /// so its pages are not pinned. Prefix cache is not allocated.
    pub pinned: u32,
}

pub struct PagedKv {
    layout: KvLayout,
    pages: Vec<Page>,
    free: BTreeSet<u32>,
    sequences: BTreeMap<u64, SequencePages>,
}

impl PagedKv {
    pub fn new(layout: KvLayout, page_count: u32) -> Result<Self, InferFailure> {
        let pool = crate::memory::planned_kv_pool(&layout, page_count)?;
        let slots = usize::try_from(pool.slots)
            .map_err(|_| fail(ErrorCode::ContractInvalid, "kv page size overflows"))?;
        let pages = (0..page_count)
            .map(|_| Page {
                k: vec![0.0; slots],
                v: vec![0.0; slots],
            })
            .collect();
        Ok(Self {
            layout,
            pages,
            free: (0..page_count).collect(),
            sequences: BTreeMap::new(),
        })
    }

    /// Drop every sequence and zero every page. The pool stays allocated until
    /// the owner drops it. Model unload calls this before the worker exits.
    pub fn invalidate(&mut self) {
        let ids: Vec<u64> = self.sequences.keys().copied().collect();
        for id in ids {
            let _ = self.release_sequence(id);
        }
        for page in &mut self.pages {
            page.k.fill(0.0);
            page.v.fill(0.0);
        }
        self.free = (0..self.pages.len() as u32).collect();
        self.sequences.clear();
    }

    /// Key and value bytes allocated for the pool. The count is fixed at
    /// construction. Admitting and releasing sequences does not change it.
    pub fn resident_bytes(&self) -> u64 {
        self.pages.iter().fold(0u64, |total, page| {
            total
                .saturating_add((page.k.len() as u64).saturating_mul(4))
                .saturating_add((page.v.len() as u64).saturating_mul(4))
        })
    }

    pub fn census(&self) -> KvCensus {
        let mut pinned = 0u32;
        for state in self.sequences.values() {
            if !state.active {
                continue;
            }
            pinned = pinned.saturating_add(state.committed.len() as u32);
            if state.pending.is_some() {
                pinned = pinned.saturating_add(1);
            }
        }
        KvCensus {
            total: self.pages.len() as u32,
            free: self.free.len() as u32,
            pinned,
        }
    }

    pub fn deactivate(&mut self, sequence: u64) -> Result<(), InferFailure> {
        let state = self.sequence_mut(sequence)?;
        state.active = false;
        Ok(())
    }

    pub fn snapshot(&self, sequence: u64) -> Result<KvSnapshot, InferFailure> {
        let state = self.sequence(sequence)?;
        let mut k = Vec::new();
        let mut v = Vec::new();
        for id in &state.committed {
            let page = self.page(*id)?;
            k.extend_from_slice(&page.k);
            v.extend_from_slice(&page.v);
        }
        if let Some(id) = state.pending {
            let page = self.page(id)?;
            k.extend_from_slice(&page.k);
            v.extend_from_slice(&page.v);
        }
        Ok(KvSnapshot {
            len: state.len,
            block_table: self.table_of(state),
            k,
            v,
        })
    }

    fn width(&self) -> usize {
        self.layout.kv_heads as usize * self.layout.head_dim as usize
    }

    fn sequence(&self, sequence: u64) -> Result<&SequencePages, InferFailure> {
        self.sequences
            .get(&sequence)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "sequence has no KV block"))
    }

    fn sequence_mut(&mut self, sequence: u64) -> Result<&mut SequencePages, InferFailure> {
        self.sequences
            .get_mut(&sequence)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "sequence has no KV block"))
    }

    fn page(&self, id: u32) -> Result<&Page, InferFailure> {
        self.pages
            .get(id as usize)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "kv page is outside the pool"))
    }

    fn table_of(&self, state: &SequencePages) -> Vec<u32> {
        let mut table = state.committed.clone();
        if let Some(id) = state.pending {
            table.push(id);
        }
        table
    }

    fn free_page(&mut self, id: u32) {
        if let Some(page) = self.pages.get_mut(id as usize) {
            page.k.fill(0.0);
            page.v.fill(0.0);
            self.free.insert(id);
        }
    }

    fn reserve(&mut self, sequence: u64) -> Result<u32, InferFailure> {
        loop {
            if let Some(id) = self.free.pop_first() {
                return Ok(id);
            }
            let victim = self
                .sequences
                .iter()
                .find(|(id, state)| **id != sequence && !state.active)
                .map(|(id, _)| *id);
            let Some(victim) = victim else {
                return Err(fail(
                    ErrorCode::InsufficientMemory,
                    "kv page pool is exhausted",
                ));
            };
            self.release_sequence(victim)?;
        }
    }

    fn page_for_write(&mut self, sequence: u64, token: u32) -> Result<u32, InferFailure> {
        let block = self.layout.block_size;
        let page_index = token / block;
        let committed = self.sequence(sequence)?.committed.len() as u32;
        if page_index == committed {
            if self.sequence(sequence)?.pending.is_none() {
                let id = self.reserve(sequence)?;
                self.sequence_mut(sequence)?.pending = Some(id);
            }
            return self
                .sequence(sequence)?
                .pending
                .ok_or_else(|| fail(ErrorCode::ContractInvalid, "kv page is not reserved"));
        }
        if page_index + 1 == committed {
            if self.sequence(sequence)?.pending.is_some() {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "kv page was reserved mid-block",
                ));
            }
            return Ok(self.sequence(sequence)?.committed[page_index as usize]);
        }
        Err(fail(
            ErrorCode::ContractInvalid,
            "kv write must fill the open token",
        ))
    }

    fn page_for_read(&self, state: &SequencePages, token: u32) -> Result<u32, InferFailure> {
        if token > state.len {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv read is past the open token",
            ));
        }
        let page_index = token / self.layout.block_size;
        if page_index == state.committed.len() as u32 {
            return state
                .pending
                .ok_or_else(|| fail(ErrorCode::ContractInvalid, "kv page is not reserved"));
        }
        state
            .committed
            .get(page_index as usize)
            .copied()
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "kv page is not reserved"))
    }

    fn slot_offset(&self, layer: u32, token: u32) -> Result<usize, InferFailure> {
        if layer >= self.layout.layers {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv layer is outside the layout",
            ));
        }
        let slot = (token % self.layout.block_size) as usize;
        Ok((layer as usize * self.layout.block_size as usize + slot) * self.width())
    }
}

impl KvStore for PagedKv {
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
        self.sequences.insert(
            sequence,
            SequencePages {
                committed: Vec::new(),
                pending: None,
                len: 0,
                active: true,
            },
        );
        Ok(())
    }

    fn release_sequence(&mut self, sequence: u64) -> Result<(), InferFailure> {
        let state = self
            .sequences
            .remove(&sequence)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "sequence has no KV block"))?;
        for id in state.committed {
            self.free_page(id);
        }
        if let Some(id) = state.pending {
            self.free_page(id);
        }
        Ok(())
    }

    fn token_len(&self, sequence: u64) -> Result<u32, InferFailure> {
        Ok(self.sequence(sequence)?.len)
    }

    fn block_table(&self, sequence: u64) -> Result<Vec<u32>, InferFailure> {
        Ok(self.table_of(self.sequence(sequence)?))
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
        let block = self.layout.block_size;
        let state = self.sequence_mut(sequence)?;
        if state.len % block == 0 {
            let page = state.pending.take().ok_or_else(|| {
                fail(ErrorCode::ContractInvalid, "kv commit has no reserved page")
            })?;
            state.committed.push(page);
        } else if state.pending.is_some() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv page was reserved mid-block",
            ));
        }
        state.len = state
            .len
            .checked_add(1)
            .ok_or_else(|| fail(ErrorCode::ContextLimitExceeded, "kv token length overflows"))?;
        Ok(state.len)
    }

    fn abort_pending(&mut self, sequence: u64) -> Result<(), InferFailure> {
        let pending = self.sequence_mut(sequence)?.pending.take();
        if let Some(id) = pending {
            self.free_page(id);
        }
        Ok(())
    }
}

impl PagedKv {
    fn write(
        &mut self,
        sequence: u64,
        layer: u32,
        token: u32,
        values: &[f32],
        key: bool,
    ) -> Result<(), InferFailure> {
        if values.len() != self.width() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv vector width does not match the layout",
            ));
        }
        let len = self.sequence(sequence)?.len;
        if token != len {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv write must fill the open token",
            ));
        }
        let offset = self.slot_offset(layer, token)?;
        let page_id = self.page_for_write(sequence, token)?;
        let page = self
            .pages
            .get_mut(page_id as usize)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "kv page is outside the pool"))?;
        let slab = if key { &mut page.k } else { &mut page.v };
        slab[offset..offset + values.len()].copy_from_slice(values);
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
        if out.len() != self.width() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv vector width does not match the layout",
            ));
        }
        let offset = self.slot_offset(layer, token)?;
        let state = self.sequence(sequence)?;
        let page_id = self.page_for_read(state, token)?;
        let page = self.page(page_id)?;
        let slab = if key { &page.k } else { &page.v };
        out.copy_from_slice(&slab[offset..offset + out.len()]);
        Ok(())
    }
}
