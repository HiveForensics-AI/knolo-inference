//! Sampler version 1. Masks and penalties run before the draw.
//! Temperature 0 is greedy: no RNG, and ties keep the lowest token id.
//! Nucleus filters run only when temperature is non-zero.

use std::collections::HashMap;
use std::time::Instant;

use infer_contracts::{fail, ErrorCode, InferFailure, SamplerPlanV1};

use crate::philox::philox_unit;
use crate::traits::{DecodeBatch, ExecutableModel, KvStore, PrefillBatch};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampledOutput {
    pub tokens: Vec<u32>,
    pub finish_reason: String,
    pub prefill_micros: u64,
    pub decode_micros: u64,
}

pub fn sample_token(
    logits: &[f32],
    plan: &SamplerPlanV1,
    history: &[u32],
    step: u32,
) -> Result<u32, InferFailure> {
    plan.validate()?;
    if logits.is_empty() {
        return Err(fail(ErrorCode::ContractInvalid, "logit vector is empty"));
    }
    for value in logits {
        if !value.is_finite() {
            return Err(fail(ErrorCode::ContractInvalid, "logits are non-finite"));
        }
    }
    let mut adjusted = logits.to_vec();
    apply_penalties(&mut adjusted, plan, history)?;
    apply_top_k(&mut adjusted, plan.settings.top_k);
    if plan.settings.temperature_micros == 0 {
        return lowest_finite_argmax(&adjusted);
    }
    let temperature = plan.settings.temperature_micros as f32 / 1_000_000.0;
    for logit in &mut adjusted {
        if logit.is_finite() {
            *logit /= temperature;
        }
    }
    let mut probs = softmax(&adjusted)?;
    apply_top_p(&mut probs, plan.settings.top_p_millionths);
    apply_min_p(&mut probs, plan.settings.min_p_millionths);
    renormalize(&mut probs)?;
    let seed = plan.seed.ok_or_else(|| {
        fail(
            ErrorCode::ContractInvalid,
            "sampled generation requires a seed",
        )
    })?;
    let stream = plan.stream.ok_or_else(|| {
        fail(
            ErrorCode::ContractInvalid,
            "sampled generation requires a stream",
        )
    })?;
    let draw = philox_unit(seed, stream, step);
    Ok(pick(&probs, draw))
}

pub fn generate_samples(
    model: &mut dyn ExecutableModel,
    kv: &mut dyn KvStore,
    sequence: u64,
    prompt: &[u32],
    plan: &SamplerPlanV1,
) -> Result<SampledOutput, InferFailure> {
    plan.validate()?;
    if prompt.is_empty() {
        return Err(fail(
            ErrorCode::PromptCompilationFailed,
            "prompt has no tokens",
        ));
    }
    kv.begin_sequence(sequence)?;
    let started = Instant::now();
    let prefill = match model.prefill(
        &PrefillBatch {
            sequence_id: sequence,
            token_ids: prompt.to_vec(),
        },
        kv,
    ) {
        Ok(value) => value,
        Err(err) => {
            let _ = kv.abort_pending(sequence);
            return Err(err);
        }
    };
    let prefill_micros = elapsed_micros(started)?;
    let mut decode_micros = 0u64;
    let mut tokens = Vec::new();
    let mut logits = prefill.logits;
    let max_tokens = plan.settings.max_output_tokens;
    loop {
        let mut history = prompt.to_vec();
        history.extend_from_slice(&tokens);
        let next = sample_token(&logits, plan, &history, tokens.len() as u32)?;
        tokens.push(next);
        let stopped = plan.eos_token_ids.contains(&next);
        if stopped || tokens.len() == max_tokens as usize {
            let finish_reason = if stopped { "stop" } else { "length" };
            return Ok(SampledOutput {
                tokens,
                finish_reason: finish_reason.into(),
                prefill_micros,
                decode_micros,
            });
        }
        let position = kv.token_len(sequence)?;
        let decode_started = Instant::now();
        let decoded = match model.decode(
            &DecodeBatch {
                sequence_id: sequence,
                token_ids: vec![next],
                positions: vec![position],
            },
            kv,
        ) {
            Ok(value) => value,
            Err(err) => {
                let _ = kv.abort_pending(sequence);
                return Err(err);
            }
        };
        decode_micros = decode_micros
            .checked_add(elapsed_micros(decode_started)?)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "decode timing overflows"))?;
        logits = decoded.logits;
    }
}

fn apply_penalties(
    logits: &mut [f32],
    plan: &SamplerPlanV1,
    history: &[u32],
) -> Result<(), InferFailure> {
    let mut counts = HashMap::<u32, u32>::new();
    for id in history {
        if (*id as usize) < logits.len() {
            *counts.entry(*id).or_insert(0) += 1;
        }
    }
    let repetition = plan.settings.repetition_penalty_micros as f32 / 1_000_000.0;
    let presence = plan.settings.presence_penalty_micros as f32 / 1_000_000.0;
    let frequency = plan.settings.frequency_penalty_micros as f32 / 1_000_000.0;
    for (id, count) in counts {
        let logit = &mut logits[id as usize];
        if plan.settings.repetition_penalty_micros != 1_000_000 {
            if repetition == 0.0 {
                *logit = 0.0;
            } else if *logit > 0.0 {
                *logit /= repetition;
            } else {
                *logit *= repetition;
            }
        }
        if presence != 0.0 {
            *logit -= presence;
        }
        if frequency != 0.0 {
            *logit -= frequency * count as f32;
        }
        if !logit.is_finite() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "penalty produced a non-finite logit",
            ));
        }
    }
    Ok(())
}

fn apply_top_k(logits: &mut [f32], k: u32) {
    if k == 0 || k as usize >= logits.len() {
        return;
    }
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&left, &right| {
        logits[right]
            .total_cmp(&logits[left])
            .then(left.cmp(&right))
    });
    let mut keep = vec![false; logits.len()];
    for index in order.into_iter().take(k as usize) {
        keep[index] = true;
    }
    for (index, logit) in logits.iter_mut().enumerate() {
        if !keep[index] {
            *logit = f32::NEG_INFINITY;
        }
    }
}

fn softmax(logits: &[f32]) -> Result<Vec<f64>, InferFailure> {
    let mut max = f32::NEG_INFINITY;
    let mut finite = false;
    for value in logits {
        if value.is_finite() && *value > max {
            max = *value;
            finite = true;
        }
    }
    if !finite {
        return Err(fail(ErrorCode::ContractInvalid, "every logit was masked"));
    }
    let mut probs = Vec::with_capacity(logits.len());
    let mut sum = 0.0f64;
    for value in logits {
        let exponent = if value.is_finite() {
            f64::from(*value - max).exp()
        } else {
            0.0
        };
        probs.push(exponent);
        sum += exponent;
    }
    if !sum.is_finite() || sum == 0.0 {
        return Err(fail(ErrorCode::ContractInvalid, "softmax mass is empty"));
    }
    for prob in &mut probs {
        *prob /= sum;
    }
    Ok(probs)
}

fn apply_top_p(probs: &mut [f64], millionths: u32) {
    if millionths >= 1_000_000 {
        return;
    }
    let limit = f64::from(millionths) / 1_000_000.0;
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&left, &right| probs[right].total_cmp(&probs[left]).then(left.cmp(&right)));
    let mut cumulative = 0.0;
    let mut keep = vec![false; probs.len()];
    for index in order {
        if cumulative >= limit && keep.iter().any(|kept| *kept) {
            break;
        }
        keep[index] = true;
        cumulative += probs[index];
    }
    for (index, prob) in probs.iter_mut().enumerate() {
        if !keep[index] {
            *prob = 0.0;
        }
    }
}

fn apply_min_p(probs: &mut [f64], millionths: u32) {
    if millionths == 0 {
        return;
    }
    let floor = f64::from(millionths) / 1_000_000.0;
    let max = probs.iter().copied().fold(0.0, f64::max);
    let threshold = floor * max;
    for prob in probs.iter_mut() {
        if *prob < threshold {
            *prob = 0.0;
        }
    }
}

fn renormalize(probs: &mut [f64]) -> Result<(), InferFailure> {
    let sum: f64 = probs.iter().sum();
    if !sum.is_finite() || sum == 0.0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "sampling distribution is empty",
        ));
    }
    for prob in probs {
        *prob /= sum;
    }
    Ok(())
}

fn pick(probs: &[f64], draw: f64) -> u32 {
    let mut cumulative = 0.0;
    let mut last = 0u32;
    for (index, prob) in probs.iter().enumerate() {
        if *prob <= 0.0 {
            continue;
        }
        last = index as u32;
        cumulative += *prob;
        if draw < cumulative {
            return last;
        }
    }
    last
}

fn lowest_finite_argmax(logits: &[f32]) -> Result<u32, InferFailure> {
    let mut best_index = None;
    let mut best = f32::NEG_INFINITY;
    for (index, value) in logits.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        if best_index.is_none() || *value > best {
            best = *value;
            best_index = Some(index);
        }
    }
    let Some(index) = best_index else {
        return Err(fail(ErrorCode::ContractInvalid, "every logit was masked"));
    };
    u32::try_from(index).map_err(|_| fail(ErrorCode::ContractInvalid, "logit index overflows"))
}

fn elapsed_micros(started: Instant) -> Result<u64, InferFailure> {
    u64::try_from(started.elapsed().as_micros())
        .map_err(|_| fail(ErrorCode::ContractInvalid, "timing does not fit in u64"))
}
