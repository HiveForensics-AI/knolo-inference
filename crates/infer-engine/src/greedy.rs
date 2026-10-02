//! Greedy argmax. Ties keep the lowest token id. Sampler penalties are later.

use infer_contracts::{fail, ErrorCode, InferFailure};

use crate::traits::{DecodeBatch, ExecutableModel, KvStore, PrefillBatch};

#[derive(Debug, Clone, PartialEq)]
pub struct GreedyOutput {
    pub prefill_logits: Vec<f32>,
    pub tokens: Vec<u32>,
}

pub fn argmax(logits: &[f32]) -> Result<u32, InferFailure> {
    if logits.is_empty() {
        return Err(fail(ErrorCode::ContractInvalid, "logit vector is empty"));
    }
    let mut best_index = 0usize;
    let mut best = logits[0];
    if !best.is_finite() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "micro logits are non-finite",
        ));
    }
    for (index, value) in logits.iter().enumerate().skip(1) {
        if !value.is_finite() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "micro logits are non-finite",
            ));
        }
        if *value > best {
            best = *value;
            best_index = index;
        }
    }
    u32::try_from(best_index).map_err(|_| fail(ErrorCode::ContractInvalid, "logit index overflows"))
}

pub fn logit_margin(logits: &[f32]) -> Result<f32, InferFailure> {
    let best = argmax(logits)? as usize;
    if logits.len() == 1 {
        return Ok(f32::INFINITY);
    }
    let mut second = f32::NEG_INFINITY;
    for (index, value) in logits.iter().enumerate() {
        if index != best {
            second = second.max(*value);
        }
    }
    Ok(logits[best] - second)
}

pub fn greedy_generate(
    model: &mut dyn ExecutableModel,
    kv: &mut dyn KvStore,
    sequence: u64,
    prompt: &[u32],
    new_tokens: u32,
) -> Result<GreedyOutput, InferFailure> {
    kv.begin_sequence(sequence)?;
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
    let mut tokens = Vec::new();
    if new_tokens == 0 {
        return Ok(GreedyOutput {
            prefill_logits: prefill.logits,
            tokens,
        });
    }
    let mut next = argmax(&prefill.logits)?;
    tokens.push(next);
    for _ in 1..new_tokens {
        let position = kv.token_len(sequence)?;
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
        next = argmax(&decoded.logits)?;
        tokens.push(next);
    }
    Ok(GreedyOutput {
        prefill_logits: prefill.logits,
        tokens,
    })
}
