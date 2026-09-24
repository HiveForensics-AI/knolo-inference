//! Continuous scheduler for one CPU model and one KV pool.
//!
//! An iteration is either one prefill chunk or one token for each selected
//! decode sequence. The micro-model forward stays single-sequence. Prefix
//! cache is not allocated.

use std::collections::BTreeMap;
use std::time::Duration;

use infer_contracts::{fail, ErrorCode, InferFailure, SamplerPlanV1};

use crate::sample::sample_token;
use crate::traits::{DecodeBatch, ExecutableModel, KvStore, PrefillBatch};

const CLASS_SCALE: u32 = 8;
const MAX_ITERS: u32 = 1_048_576;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceClass {
    Interactive,
    Standard,
    Batch,
    Background,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerConfig {
    pub runtime_root: String,
    pub max_context_tokens: u32,
    pub block_size: u32,
    pub page_capacity: u32,
    pub prefill_chunk_tokens: u32,
}

impl SchedulerConfig {
    pub fn new(
        runtime_root: impl Into<String>,
        max_context_tokens: u32,
        block_size: u32,
        page_capacity: u32,
        prefill_chunk_tokens: u32,
    ) -> Result<Self, InferFailure> {
        let runtime_root = runtime_root.into();
        if runtime_root.is_empty() || runtime_root.len() > 80 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "scheduler runtime root is invalid",
            ));
        }
        if max_context_tokens == 0 || max_context_tokens > MAX_ITERS {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "scheduler context limit is invalid",
            ));
        }
        if !(16..=65_536).contains(&block_size) || !block_size.is_power_of_two() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "KV block size must be a power of two from 16 to 65536",
            ));
        }
        if page_capacity == 0 || page_capacity > MAX_ITERS {
            return Err(fail(ErrorCode::ContractInvalid, "kv page pool is invalid"));
        }
        if prefill_chunk_tokens == 0 || prefill_chunk_tokens > MAX_ITERS {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "prefill chunk size is invalid",
            ));
        }
        Ok(Self {
            runtime_root,
            max_context_tokens,
            block_size,
            page_capacity,
            prefill_chunk_tokens,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleRequest {
    pub request_id: String,
    pub runtime_root: String,
    pub prompt: Vec<u32>,
    pub sampler: SamplerPlanV1,
    pub class: ServiceClass,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleOp {
    Prefill { sequence_id: u64, tokens: u32 },
    Decode { sequence_id: u64, token_id: u32 },
    Cancel { sequence_id: u64 },
    Fail { sequence_id: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleStep {
    pub batch: Vec<u64>,
    pub ops: Vec<ScheduleOp>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleResult {
    pub sequence_id: u64,
    pub request_id: String,
    pub tokens: Vec<u32>,
    pub finish_reason: String,
    pub prefill_logits: Vec<f32>,
    pub error_code: Option<ErrorCode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Queued,
    Prefilling,
    Decoding,
    Done,
}

struct Slot {
    request_id: String,
    sequence_id: u64,
    prompt: Vec<u32>,
    prompt_cursor: u32,
    sampler: SamplerPlanV1,
    class: ServiceClass,
    vtime: u64,
    pages: u32,
    holding_pages: bool,
    phase: Phase,
    cancel: bool,
    kv_open: bool,
    needs_forward: bool,
    logits: Vec<f32>,
    prefill_logits: Vec<f32>,
    tokens: Vec<u32>,
    finish_reason: String,
    error_code: Option<ErrorCode>,
}

/// Non-cumulative scheduler-iteration bounds, in nanoseconds. The last bound
/// is the open-ended bucket. Prefix cache is not allocated, so its counters
/// stay zero.
pub const ITERATION_BUCKETS: usize = 11;
pub const ITERATION_BOUND_NANOS: [u64; ITERATION_BUCKETS] = [
    100_000,
    500_000,
    1_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
    500_000_000,
    1_000_000_000,
    5_000_000_000,
    u64::MAX,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SchedulerCounters {
    pub queue_interactive: u32,
    pub queue_standard: u32,
    pub queue_batch: u32,
    pub queue_background: u32,
    pub active_sequences: u32,
    pub rejected_contract: u64,
    pub rejected_context: u64,
    pub rejected_memory: u64,
    pub rejected_prompt: u64,
    pub rejected_other: u64,
    pub cancellations: u64,
    pub oom: u64,
    pub prefill_tokens: u64,
    pub prefill_nanos: u64,
    pub decode_tokens: u64,
    pub decode_nanos: u64,
    pub iterations: u64,
    pub iteration_nanos: u64,
    pub iteration_buckets: [u64; ITERATION_BUCKETS],
    pub prefix_lookups: u64,
    pub prefix_hits: u64,
    pub prefix_reused_tokens: u64,
}

pub struct CpuScheduler {
    config: SchedulerConfig,
    slots: BTreeMap<u64, Slot>,
    ids: BTreeMap<String, u64>,
    next_sequence: u64,
    reserved_pages: u32,
    counters: SchedulerCounters,
}

impl CpuScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            slots: BTreeMap::new(),
            ids: BTreeMap::new(),
            next_sequence: 0,
            reserved_pages: 0,
            counters: SchedulerCounters::default(),
        }
    }

    pub fn submit(&mut self, request: ScheduleRequest) -> Result<u64, InferFailure> {
        match self.admit(request) {
            Ok(id) => Ok(id),
            Err(err) => {
                self.note_rejection(err.code);
                Err(err)
            }
        }
    }

    pub fn counters(&self) -> SchedulerCounters {
        let mut counters = self.counters;
        counters.queue_interactive = 0;
        counters.queue_standard = 0;
        counters.queue_batch = 0;
        counters.queue_background = 0;
        counters.active_sequences = 0;
        for slot in self.slots.values() {
            if slot.phase == Phase::Done {
                continue;
            }
            counters.active_sequences = counters.active_sequences.saturating_add(1);
            if slot.phase == Phase::Queued {
                match slot.class {
                    ServiceClass::Interactive => {
                        counters.queue_interactive = counters.queue_interactive.saturating_add(1);
                    }
                    ServiceClass::Standard => {
                        counters.queue_standard = counters.queue_standard.saturating_add(1);
                    }
                    ServiceClass::Batch => {
                        counters.queue_batch = counters.queue_batch.saturating_add(1);
                    }
                    ServiceClass::Background => {
                        counters.queue_background = counters.queue_background.saturating_add(1);
                    }
                }
            }
        }
        counters
    }

    fn admit(&mut self, request: ScheduleRequest) -> Result<u64, InferFailure> {
        request.sampler.validate()?;
        if !valid_request_id(&request.request_id) {
            return Err(fail(ErrorCode::ContractInvalid, "request id is invalid"));
        }
        if self.ids.contains_key(&request.request_id) {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "request id is already admitted",
            ));
        }
        if request.runtime_root != self.config.runtime_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "request runtime does not match the scheduler",
            ));
        }
        if request.prompt.is_empty() {
            return Err(fail(
                ErrorCode::PromptCompilationFailed,
                "prompt has no tokens",
            ));
        }
        let prompt_len = u32::try_from(request.prompt.len()).map_err(|_| {
            fail(
                ErrorCode::ContextLimitExceeded,
                "prompt does not fit in the reserved context",
            )
        })?;
        let total = prompt_len
            .checked_add(request.sampler.settings.max_output_tokens)
            .ok_or_else(|| {
                fail(
                    ErrorCode::ContextLimitExceeded,
                    "prompt plus reserved generation tokens overflow",
                )
            })?;
        if total > self.config.max_context_tokens {
            return Err(fail(
                ErrorCode::ContextLimitExceeded,
                "prompt plus reserved generation tokens exceed the context",
            ));
        }
        let pages = page_count(total, self.config.block_size)?;
        let reserved = self.reserved_pages.checked_add(pages).ok_or_else(|| {
            fail(
                ErrorCode::InsufficientMemory,
                "kv page pool cannot admit the request",
            )
        })?;
        if pages > self.config.page_capacity || reserved > self.config.page_capacity {
            return Err(fail(
                ErrorCode::InsufficientMemory,
                "kv page pool cannot admit the request",
            ));
        }
        let sequence_id = self.next_sequence.checked_add(1).ok_or_else(|| {
            fail(
                ErrorCode::ContractInvalid,
                "scheduler sequence id overflows",
            )
        })?;
        let vtime = self
            .slots
            .values()
            .filter(|slot| slot.is_runnable())
            .map(|slot| slot.vtime)
            .min()
            .unwrap_or(0);
        self.next_sequence = sequence_id;
        self.reserved_pages = reserved;
        self.ids.insert(request.request_id.clone(), sequence_id);
        self.slots.insert(
            sequence_id,
            Slot {
                request_id: request.request_id,
                sequence_id,
                prompt: request.prompt,
                prompt_cursor: 0,
                sampler: request.sampler,
                class: request.class,
                vtime,
                pages,
                holding_pages: true,
                phase: Phase::Queued,
                cancel: false,
                kv_open: false,
                needs_forward: false,
                logits: Vec::new(),
                prefill_logits: Vec::new(),
                tokens: Vec::new(),
                finish_reason: String::new(),
                error_code: None,
            },
        );
        Ok(sequence_id)
    }

    pub fn cancel(&mut self, request_id: &str) -> Result<(), InferFailure> {
        let sequence_id = *self
            .ids
            .get(request_id)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "request is not admitted"))?;
        let slot = self.slot_mut(sequence_id)?;
        if slot.phase == Phase::Done {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "request is already finished",
            ));
        }
        if slot.cancel {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "request is already cancelled",
            ));
        }
        slot.cancel = true;
        Ok(())
    }

    pub fn step(
        &mut self,
        model: &mut dyn ExecutableModel,
        kv: &mut dyn KvStore,
    ) -> Result<Option<ScheduleStep>, InferFailure> {
        let started = std::time::Instant::now();
        let result = self.step_once(model, kv);
        self.note_iteration(started.elapsed());
        result
    }

    fn step_once(
        &mut self,
        model: &mut dyn ExecutableModel,
        kv: &mut dyn KvStore,
    ) -> Result<Option<ScheduleStep>, InferFailure> {
        self.check_bound(model, kv)?;
        if let Some(sequence_id) = self.pick_cancel() {
            return self.serve_cancel(kv, sequence_id).map(Some);
        }
        let Some(sequence_id) = self.pick_runnable() else {
            return Ok(None);
        };
        if self.slot(sequence_id)?.phase == Phase::Decoding {
            return self.serve_decode_batch(model, kv).map(Some);
        }
        self.serve_prefill(model, kv, sequence_id).map(Some)
    }

    pub fn run_until_idle(
        &mut self,
        model: &mut dyn ExecutableModel,
        kv: &mut dyn KvStore,
    ) -> Result<Vec<ScheduleResult>, InferFailure> {
        for _ in 0..MAX_ITERS {
            if self.step(model, kv)?.is_none() {
                return Ok(self.results());
            }
        }
        Err(fail(
            ErrorCode::ContractInvalid,
            "scheduler iteration did not finish",
        ))
    }

    pub fn results(&self) -> Vec<ScheduleResult> {
        self.slots
            .values()
            .filter(|slot| slot.phase == Phase::Done)
            .map(|slot| ScheduleResult {
                sequence_id: slot.sequence_id,
                request_id: slot.request_id.clone(),
                tokens: slot.tokens.clone(),
                finish_reason: slot.finish_reason.clone(),
                prefill_logits: slot.prefill_logits.clone(),
                error_code: slot.error_code,
            })
            .collect()
    }

    /// Slots still held, including finished requests whose prompt and tokens
    /// have not been dropped yet.
    pub fn retained(&self) -> usize {
        self.slots.len()
    }

    /// Drop a finished slot so its prompt, tokens, and logits leave memory.
    /// The request id can be admitted again. A request that has not finished
    /// stays admitted.
    pub fn reap(&mut self, request_id: &str) -> Result<(), InferFailure> {
        let sequence_id = *self
            .ids
            .get(request_id)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "request is not admitted"))?;
        let slot = self.slot(sequence_id)?;
        if slot.phase != Phase::Done {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "request is still admitted",
            ));
        }
        if slot.holding_pages || slot.kv_open {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "finished request still holds kv",
            ));
        }
        self.slots.remove(&sequence_id);
        self.ids.remove(request_id);
        Ok(())
    }

    pub fn request_id(&self, sequence_id: u64) -> Result<String, InferFailure> {
        Ok(self.slot(sequence_id)?.request_id.clone())
    }

    /// Request id, output index, and token id of the latest sampled token.
    pub fn last_output(&self, sequence_id: u64) -> Result<(String, u32, u32), InferFailure> {
        let slot = self.slot(sequence_id)?;
        let len = slot.tokens.len();
        if len == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "scheduler decode reported no token",
            ));
        }
        let index = u32::try_from(len - 1).map_err(|_| {
            fail(
                ErrorCode::ContractInvalid,
                "scheduler token index overflows",
            )
        })?;
        Ok((slot.request_id.clone(), index, slot.tokens[len - 1]))
    }

    fn check_bound(
        &self,
        model: &dyn ExecutableModel,
        kv: &dyn KvStore,
    ) -> Result<(), InferFailure> {
        if model.identity().runtime_root.as_str() != self.config.runtime_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "model runtime does not match the scheduler",
            ));
        }
        if model.capabilities().max_context_tokens != self.config.max_context_tokens
            || model.kv_layout().block_size != self.config.block_size
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "model layout does not match the scheduler",
            ));
        }
        if kv.layout().block_size != self.config.block_size || kv.layout().dtype != "f32" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "kv layout does not match the scheduler",
            ));
        }
        Ok(())
    }

    fn pick_cancel(&self) -> Option<u64> {
        self.slots
            .values()
            .find(|slot| slot.cancel && slot.phase != Phase::Done)
            .map(|slot| slot.sequence_id)
    }

    fn pick_runnable(&self) -> Option<u64> {
        self.slots
            .values()
            .filter(|slot| slot.is_runnable())
            .min_by_key(|slot| (slot.vtime, slot.sequence_id))
            .map(|slot| slot.sequence_id)
    }

    fn decode_batch(&self) -> Vec<u64> {
        let min_prefill = self
            .slots
            .values()
            .filter(|slot| slot.awaits_prefill())
            .map(|slot| slot.vtime)
            .min();
        let mut rows = Vec::new();
        for slot in self.slots.values() {
            if slot.cancel || slot.phase != Phase::Decoding {
                continue;
            }
            if let Some(bound) = min_prefill {
                if slot.vtime > bound {
                    continue;
                }
            }
            rows.push((slot.vtime, slot.sequence_id));
        }
        rows.sort_unstable();
        rows.into_iter().map(|(_, id)| id).collect()
    }

    fn serve_cancel(
        &mut self,
        kv: &mut dyn KvStore,
        sequence_id: u64,
    ) -> Result<ScheduleStep, InferFailure> {
        self.retire(kv, sequence_id, "cancelled", None)?;
        self.counters.cancellations = self.counters.cancellations.saturating_add(1);
        Ok(ScheduleStep {
            batch: vec![sequence_id],
            ops: vec![ScheduleOp::Cancel { sequence_id }],
        })
    }

    fn serve_prefill(
        &mut self,
        model: &mut dyn ExecutableModel,
        kv: &mut dyn KvStore,
        sequence_id: u64,
    ) -> Result<ScheduleStep, InferFailure> {
        let cursor = self.slot(sequence_id)?.prompt_cursor;
        let prompt_len = u32::try_from(self.slot(sequence_id)?.prompt.len()).map_err(|_| {
            fail(
                ErrorCode::ContextLimitExceeded,
                "prompt does not fit in the reserved context",
            )
        })?;
        let remaining = prompt_len.checked_sub(cursor).ok_or_else(|| {
            fail(
                ErrorCode::ContractInvalid,
                "scheduler prefill cursor is past the prompt",
            )
        })?;
        let take = self.config.prefill_chunk_tokens.min(remaining);
        if take == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "scheduler prefill chunk is empty",
            ));
        }
        let token_ids =
            self.slot(sequence_id)?.prompt[cursor as usize..(cursor + take) as usize].to_vec();
        if !self.slot(sequence_id)?.kv_open {
            if let Err(err) = kv.begin_sequence(sequence_id) {
                self.retire(kv, sequence_id, "error", Some(err.code))?;
                return Ok(fail_step(sequence_id));
            }
            self.slot_mut(sequence_id)?.kv_open = true;
        }
        let started = std::time::Instant::now();
        match model.prefill(
            &PrefillBatch {
                sequence_id,
                token_ids,
            },
            kv,
        ) {
            Ok(output) => {
                self.note_prefill(started.elapsed(), u64::from(take));
                let slot = self.slot_mut(sequence_id)?;
                slot.prompt_cursor = cursor + take;
                slot.prefill_logits = output.logits.clone();
                slot.phase = Phase::Prefilling;
                if slot.prompt_cursor == prompt_len {
                    slot.logits = output.logits;
                    slot.needs_forward = false;
                    slot.phase = Phase::Decoding;
                }
                self.add_service(sequence_id, take)?;
                Ok(ScheduleStep {
                    batch: vec![sequence_id],
                    ops: vec![ScheduleOp::Prefill {
                        sequence_id,
                        tokens: take,
                    }],
                })
            }
            Err(err) => {
                self.note_prefill(started.elapsed(), 0);
                self.retire(kv, sequence_id, "error", Some(err.code))?;
                Ok(fail_step(sequence_id))
            }
        }
    }

    fn serve_decode_batch(
        &mut self,
        model: &mut dyn ExecutableModel,
        kv: &mut dyn KvStore,
    ) -> Result<ScheduleStep, InferFailure> {
        let batch = self.decode_batch();
        if batch.is_empty() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "scheduler decode batch is empty",
            ));
        }
        let mut ops = Vec::new();
        for sequence_id in &batch {
            let op = self.serve_decode_one(model, kv, *sequence_id)?;
            let failed = matches!(op, ScheduleOp::Fail { .. });
            ops.push(op);
            if failed {
                break;
            }
        }
        Ok(ScheduleStep { batch, ops })
    }

    fn serve_decode_one(
        &mut self,
        model: &mut dyn ExecutableModel,
        kv: &mut dyn KvStore,
        sequence_id: u64,
    ) -> Result<ScheduleOp, InferFailure> {
        let started = std::time::Instant::now();
        if self.slot(sequence_id)?.needs_forward {
            let token = self
                .slot(sequence_id)?
                .tokens
                .last()
                .copied()
                .ok_or_else(|| fail(ErrorCode::ContractInvalid, "scheduler decode has no token"))?;
            let position = kv.token_len(sequence_id)?;
            match model.decode(
                &DecodeBatch {
                    sequence_id,
                    token_ids: vec![token],
                    positions: vec![position],
                },
                kv,
            ) {
                Ok(output) => {
                    let slot = self.slot_mut(sequence_id)?;
                    slot.logits = output.logits;
                    slot.needs_forward = false;
                }
                Err(err) => {
                    self.note_decode(started.elapsed(), 0);
                    self.retire(kv, sequence_id, "error", Some(err.code))?;
                    return Ok(ScheduleOp::Fail { sequence_id });
                }
            }
        }
        if self.slot(sequence_id)?.logits.is_empty() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "scheduler decode has no logits",
            ));
        }
        let slot = self.slot(sequence_id)?;
        let mut history = slot.prompt.clone();
        history.extend_from_slice(&slot.tokens);
        let step = u32::try_from(slot.tokens.len())
            .map_err(|_| fail(ErrorCode::ContractInvalid, "scheduler token step overflows"))?;
        let logits = slot.logits.clone();
        let sampler = slot.sampler.clone();
        let sampled = match sample_token(&logits, &sampler, &history, step) {
            Ok(token) => token,
            Err(err) => {
                self.note_decode(started.elapsed(), 0);
                self.retire(kv, sequence_id, "error", Some(err.code))?;
                return Ok(ScheduleOp::Fail { sequence_id });
            }
        };
        let (stop, length) = {
            let slot = self.slot_mut(sequence_id)?;
            slot.tokens.push(sampled);
            let stop = slot.sampler.eos_token_ids.contains(&sampled);
            let length = slot.tokens.len() as u32 == slot.sampler.settings.max_output_tokens;
            (stop, length)
        };
        self.add_service(sequence_id, 1)?;
        if stop || length {
            let reason = if stop { "stop" } else { "length" };
            self.retire(kv, sequence_id, reason, None)?;
        } else {
            self.slot_mut(sequence_id)?.needs_forward = true;
        }
        self.note_decode(started.elapsed(), 1);
        Ok(ScheduleOp::Decode {
            sequence_id,
            token_id: sampled,
        })
    }

    fn add_service(&mut self, sequence_id: u64, tokens: u32) -> Result<(), InferFailure> {
        let class = self.slot(sequence_id)?.class;
        let per = u64::from(CLASS_SCALE / class_weight(class));
        let cost = u64::from(tokens).checked_mul(per).ok_or_else(|| {
            fail(
                ErrorCode::ContractInvalid,
                "scheduler virtual time overflows",
            )
        })?;
        let slot = self.slot_mut(sequence_id)?;
        slot.vtime = slot.vtime.checked_add(cost).ok_or_else(|| {
            fail(
                ErrorCode::ContractInvalid,
                "scheduler virtual time overflows",
            )
        })?;
        Ok(())
    }

    fn retire(
        &mut self,
        kv: &mut dyn KvStore,
        sequence_id: u64,
        finish: &str,
        error: Option<ErrorCode>,
    ) -> Result<(), InferFailure> {
        self.release_kv(kv, sequence_id)?;
        self.release_pages(sequence_id)?;
        let slot = self.slot_mut(sequence_id)?;
        slot.phase = Phase::Done;
        slot.finish_reason = finish.to_string();
        slot.error_code = error;
        slot.cancel = false;
        if error == Some(ErrorCode::InsufficientMemory) {
            self.counters.oom = self.counters.oom.saturating_add(1);
        }
        Ok(())
    }

    fn release_kv(&mut self, kv: &mut dyn KvStore, sequence_id: u64) -> Result<(), InferFailure> {
        if !self.slot(sequence_id)?.kv_open {
            return Ok(());
        }
        kv.abort_pending(sequence_id)?;
        kv.release_sequence(sequence_id)?;
        self.slot_mut(sequence_id)?.kv_open = false;
        Ok(())
    }

    fn release_pages(&mut self, sequence_id: u64) -> Result<(), InferFailure> {
        if !self.slot(sequence_id)?.holding_pages {
            return Ok(());
        }
        let pages = self.slot(sequence_id)?.pages;
        self.reserved_pages = self.reserved_pages.checked_sub(pages).ok_or_else(|| {
            fail(
                ErrorCode::ContractInvalid,
                "scheduler page reservation underflows",
            )
        })?;
        self.slot_mut(sequence_id)?.holding_pages = false;
        Ok(())
    }

    fn slot(&self, sequence_id: u64) -> Result<&Slot, InferFailure> {
        self.slots
            .get(&sequence_id)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "scheduler lost the sequence"))
    }

    fn slot_mut(&mut self, sequence_id: u64) -> Result<&mut Slot, InferFailure> {
        self.slots
            .get_mut(&sequence_id)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "scheduler lost the sequence"))
    }

    fn note_rejection(&mut self, code: ErrorCode) {
        let counter = match code {
            ErrorCode::ContractInvalid => &mut self.counters.rejected_contract,
            ErrorCode::ContextLimitExceeded => &mut self.counters.rejected_context,
            ErrorCode::InsufficientMemory => &mut self.counters.rejected_memory,
            ErrorCode::PromptCompilationFailed => &mut self.counters.rejected_prompt,
            _ => &mut self.counters.rejected_other,
        };
        *counter = counter.saturating_add(1);
        if code == ErrorCode::InsufficientMemory {
            self.counters.oom = self.counters.oom.saturating_add(1);
        }
    }

    fn note_prefill(&mut self, elapsed: Duration, tokens: u64) {
        self.counters.prefill_tokens = self.counters.prefill_tokens.saturating_add(tokens);
        self.counters.prefill_nanos = self
            .counters
            .prefill_nanos
            .saturating_add(duration_nanos(elapsed));
    }

    fn note_decode(&mut self, elapsed: Duration, tokens: u64) {
        self.counters.decode_tokens = self.counters.decode_tokens.saturating_add(tokens);
        self.counters.decode_nanos = self
            .counters
            .decode_nanos
            .saturating_add(duration_nanos(elapsed));
    }

    fn note_iteration(&mut self, elapsed: Duration) {
        let nanos = duration_nanos(elapsed);
        self.counters.iterations = self.counters.iterations.saturating_add(1);
        self.counters.iteration_nanos = self.counters.iteration_nanos.saturating_add(nanos);
        let index = ITERATION_BOUND_NANOS
            .iter()
            .position(|bound| nanos <= *bound)
            .unwrap_or(ITERATION_BUCKETS - 1);
        self.counters.iteration_buckets[index] =
            self.counters.iteration_buckets[index].saturating_add(1);
    }
}

impl Slot {
    fn is_runnable(&self) -> bool {
        self.phase != Phase::Done && !self.cancel
    }

    fn awaits_prefill(&self) -> bool {
        !self.cancel && matches!(self.phase, Phase::Queued | Phase::Prefilling)
    }
}

fn duration_nanos(elapsed: Duration) -> u64 {
    u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

fn class_weight(class: ServiceClass) -> u32 {
    match class {
        ServiceClass::Interactive => 8,
        ServiceClass::Standard => 4,
        ServiceClass::Batch => 2,
        ServiceClass::Background => 1,
    }
}

fn page_count(tokens: u32, block: u32) -> Result<u32, InferFailure> {
    let pages = u64::from(tokens).div_ceil(u64::from(block));
    u32::try_from(pages).map_err(|_| fail(ErrorCode::InsufficientMemory, "kv page count overflows"))
}

fn valid_request_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

fn fail_step(sequence_id: u64) -> ScheduleStep {
    ScheduleStep {
        batch: vec![sequence_id],
        ops: vec![ScheduleOp::Fail { sequence_id }],
    }
}
