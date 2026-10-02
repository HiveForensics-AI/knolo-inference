//! Versioned supervisor/worker messages.
//!
//! The bytes are canonical CBOR. They are not a rooted Infer contract. Unknown
//! fields and unknown kinds are rejected.

use std::collections::BTreeMap;

use infer_contracts::{fail, CborValue, ErrorCode, InferFailure, SamplerPlanV1};
use infer_engine::{ServiceClass, ITERATION_BUCKETS};

use crate::frame::{read_frame, write_frame};

pub const PROTOCOL_VERSION: i128 = 1;
const MAX_PROMPT: usize = 4096;
const MAX_MESSAGE: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToWorker {
    Submit {
        request_id: String,
        prompt: Vec<u32>,
        sampler: Box<SamplerPlanV1>,
        class: ServiceClass,
    },
    Cancel {
        request_id: String,
    },
    Stats,
    Shutdown,
    Release,
}

/// Worker-local counters. Integers only. No request id, prompt, or alias.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WorkerStats {
    pub queue_interactive: u64,
    pub queue_standard: u64,
    pub queue_batch: u64,
    pub queue_background: u64,
    pub active_sequences: u64,
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
    pub kv_total: u64,
    pub kv_free: u64,
    pub kv_pinned: u64,
    /// Slots the scheduler still holds, including finished sequences.
    pub retained_sequences: u64,
    pub prefix_lookups: u64,
    pub prefix_hits: u64,
    pub prefix_reused_tokens: u64,
    pub model_load_nanos: u64,
    pub verified_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FromWorker {
    Ready {
        runtime_root: String,
    },
    Result {
        request_id: String,
        tokens: Vec<u32>,
        finish_reason: String,
        error_code: Option<String>,
    },
    Reject {
        request_id: String,
        code: String,
        message: String,
    },
    Ack {
        request_id: String,
    },
    Admitted {
        request_id: String,
    },
    Delta {
        request_id: String,
        index: u32,
        token_id: u32,
    },
    /// One prefill chunk. `tokens` is a count, not token ids.
    Prefill {
        request_id: String,
        chunk: u32,
        tokens: u32,
    },
    Stats(Box<WorkerStats>),
}

pub fn write_message(writer: impl std::io::Write, value: &CborValue) -> Result<(), InferFailure> {
    write_frame(writer, value)
}

pub fn read_message(reader: &mut impl std::io::Read) -> Result<CborValue, InferFailure> {
    read_frame(reader)
}

pub fn encode_to_worker(message: &ToWorker) -> Result<CborValue, InferFailure> {
    match message {
        ToWorker::Submit {
            request_id,
            prompt,
            sampler,
            class,
        } => {
            check_request_id(request_id)?;
            check_prompt(prompt)?;
            let mut fields = BTreeMap::new();
            fields.insert("class".into(), CborValue::Text(class_name(*class).into()));
            fields.insert("prompt".into(), u32_array(prompt));
            fields.insert("requestId".into(), CborValue::Text(request_id.clone()));
            fields.insert("sampler".into(), sampler.to_cbor()?);
            Ok(envelope("submit", fields))
        }
        ToWorker::Cancel { request_id } => {
            check_request_id(request_id)?;
            let mut fields = BTreeMap::new();
            fields.insert("requestId".into(), CborValue::Text(request_id.clone()));
            Ok(envelope("cancel", fields))
        }
        ToWorker::Stats => Ok(envelope("stats", BTreeMap::new())),
        ToWorker::Shutdown => Ok(envelope("shutdown", BTreeMap::new())),
        ToWorker::Release => Ok(envelope("release", BTreeMap::new())),
    }
}

pub fn encode_from_worker(message: &FromWorker) -> Result<CborValue, InferFailure> {
    match message {
        FromWorker::Ready { runtime_root } => {
            check_runtime_root(runtime_root)?;
            let mut fields = BTreeMap::new();
            fields.insert("runtimeRoot".into(), CborValue::Text(runtime_root.clone()));
            Ok(envelope("ready", fields))
        }
        FromWorker::Result {
            request_id,
            tokens,
            finish_reason,
            error_code,
        } => {
            check_request_id(request_id)?;
            check_prompt(tokens)?;
            check_finish(finish_reason)?;
            if let Some(code) = error_code {
                parse_code(code)?;
            }
            let mut fields = BTreeMap::new();
            fields.insert(
                "errorCode".into(),
                match error_code {
                    Some(code) => CborValue::Text(code.clone()),
                    None => CborValue::Null,
                },
            );
            fields.insert(
                "finishReason".into(),
                CborValue::Text(finish_reason.clone()),
            );
            fields.insert("requestId".into(), CborValue::Text(request_id.clone()));
            fields.insert("tokens".into(), u32_array(tokens));
            Ok(envelope("result", fields))
        }
        FromWorker::Reject {
            request_id,
            code,
            message,
        } => {
            check_request_id(request_id)?;
            parse_code(code)?;
            check_message(message)?;
            let mut fields = BTreeMap::new();
            fields.insert("errorCode".into(), CborValue::Text(code.clone()));
            fields.insert("message".into(), CborValue::Text(message.clone()));
            fields.insert("requestId".into(), CborValue::Text(request_id.clone()));
            Ok(envelope("reject", fields))
        }
        FromWorker::Ack { request_id } => {
            check_request_id(request_id)?;
            let mut fields = BTreeMap::new();
            fields.insert("requestId".into(), CborValue::Text(request_id.clone()));
            Ok(envelope("ack", fields))
        }
        FromWorker::Admitted { request_id } => {
            check_request_id(request_id)?;
            let mut fields = BTreeMap::new();
            fields.insert("requestId".into(), CborValue::Text(request_id.clone()));
            Ok(envelope("admitted", fields))
        }
        FromWorker::Delta {
            request_id,
            index,
            token_id,
        } => {
            check_request_id(request_id)?;
            let mut fields = BTreeMap::new();
            fields.insert("index".into(), CborValue::Integer(i128::from(*index)));
            fields.insert("requestId".into(), CborValue::Text(request_id.clone()));
            fields.insert("tokenId".into(), CborValue::Integer(i128::from(*token_id)));
            Ok(envelope("delta", fields))
        }
        FromWorker::Prefill {
            request_id,
            chunk,
            tokens,
        } => {
            check_request_id(request_id)?;
            let mut fields = BTreeMap::new();
            fields.insert("chunk".into(), CborValue::Integer(i128::from(*chunk)));
            fields.insert("requestId".into(), CborValue::Text(request_id.clone()));
            fields.insert("tokens".into(), CborValue::Integer(i128::from(*tokens)));
            Ok(envelope("prefill", fields))
        }
        FromWorker::Stats(stats) => Ok(envelope("stats", stats_fields(stats))),
    }
}

fn stats_fields(stats: &WorkerStats) -> BTreeMap<String, CborValue> {
    let mut fields = BTreeMap::new();
    fields.insert("activeSequences".into(), u64_value(stats.active_sequences));
    fields.insert("cancellations".into(), u64_value(stats.cancellations));
    fields.insert("decodeNanos".into(), u64_value(stats.decode_nanos));
    fields.insert("decodeTokens".into(), u64_value(stats.decode_tokens));
    fields.insert(
        "iterationBuckets".into(),
        u64_array(&stats.iteration_buckets),
    );
    fields.insert("iterationNanos".into(), u64_value(stats.iteration_nanos));
    fields.insert("iterations".into(), u64_value(stats.iterations));
    fields.insert("kvFree".into(), u64_value(stats.kv_free));
    fields.insert("kvPinned".into(), u64_value(stats.kv_pinned));
    fields.insert("kvTotal".into(), u64_value(stats.kv_total));
    fields.insert("modelLoadNanos".into(), u64_value(stats.model_load_nanos));
    fields.insert("oom".into(), u64_value(stats.oom));
    fields.insert("prefillNanos".into(), u64_value(stats.prefill_nanos));
    fields.insert("prefillTokens".into(), u64_value(stats.prefill_tokens));
    fields.insert("prefixHits".into(), u64_value(stats.prefix_hits));
    fields.insert("prefixLookups".into(), u64_value(stats.prefix_lookups));
    fields.insert(
        "prefixReusedTokens".into(),
        u64_value(stats.prefix_reused_tokens),
    );
    fields.insert("queueBackground".into(), u64_value(stats.queue_background));
    fields.insert("queueBatch".into(), u64_value(stats.queue_batch));
    fields.insert(
        "queueInteractive".into(),
        u64_value(stats.queue_interactive),
    );
    fields.insert("queueStandard".into(), u64_value(stats.queue_standard));
    fields.insert("rejectedContext".into(), u64_value(stats.rejected_context));
    fields.insert(
        "rejectedContract".into(),
        u64_value(stats.rejected_contract),
    );
    fields.insert("rejectedMemory".into(), u64_value(stats.rejected_memory));
    fields.insert("rejectedOther".into(), u64_value(stats.rejected_other));
    fields.insert("rejectedPrompt".into(), u64_value(stats.rejected_prompt));
    fields.insert(
        "retainedSequences".into(),
        u64_value(stats.retained_sequences),
    );
    fields.insert("verifiedBytes".into(), u64_value(stats.verified_bytes));
    fields
}

fn u64_value(value: u64) -> CborValue {
    CborValue::Integer(i128::from(value))
}

fn u64_array(values: &[u64]) -> CborValue {
    CborValue::Array(values.iter().copied().map(u64_value).collect())
}

pub fn decode_to_worker(value: &CborValue) -> Result<ToWorker, InferFailure> {
    let mut bag = Bag::parse(value)?;
    let version = bag.take_i128("version")?;
    if version != PROTOCOL_VERSION {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "ipc protocol version is not 1",
        ));
    }
    let kind = bag.take_text("kind")?;
    let message = match kind.as_str() {
        "submit" => {
            let class = parse_class(&bag.take_text("class")?)?;
            let prompt = bag.take_u32_array("prompt")?;
            let request_id = bag.take_text("requestId")?;
            let sampler = Box::new(SamplerPlanV1::from_cbor(&bag.take("sampler")?)?);
            check_request_id(&request_id)?;
            check_prompt(&prompt)?;
            ToWorker::Submit {
                request_id,
                prompt,
                sampler,
                class,
            }
        }
        "cancel" => {
            let request_id = bag.take_text("requestId")?;
            check_request_id(&request_id)?;
            ToWorker::Cancel { request_id }
        }
        "stats" => ToWorker::Stats,
        "shutdown" => ToWorker::Shutdown,
        "release" => ToWorker::Release,
        _ => {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "ipc message kind is unknown",
            ))
        }
    };
    bag.end()?;
    Ok(message)
}

pub fn decode_from_worker(value: &CborValue) -> Result<FromWorker, InferFailure> {
    let mut bag = Bag::parse(value)?;
    let version = bag.take_i128("version")?;
    if version != PROTOCOL_VERSION {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "ipc protocol version is not 1",
        ));
    }
    let kind = bag.take_text("kind")?;
    let message = match kind.as_str() {
        "ready" => {
            let runtime_root = bag.take_text("runtimeRoot")?;
            check_runtime_root(&runtime_root)?;
            FromWorker::Ready { runtime_root }
        }
        "result" => {
            let error_code = bag.take_optional_text("errorCode")?;
            let finish_reason = bag.take_text("finishReason")?;
            let request_id = bag.take_text("requestId")?;
            let tokens = bag.take_u32_array("tokens")?;
            check_request_id(&request_id)?;
            check_prompt(&tokens)?;
            check_finish(&finish_reason)?;
            if let Some(code) = &error_code {
                parse_code(code)?;
            }
            FromWorker::Result {
                request_id,
                tokens,
                finish_reason,
                error_code,
            }
        }
        "reject" => {
            let code = bag.take_text("errorCode")?;
            let message = bag.take_text("message")?;
            let request_id = bag.take_text("requestId")?;
            check_request_id(&request_id)?;
            parse_code(&code)?;
            check_message(&message)?;
            FromWorker::Reject {
                request_id,
                code,
                message,
            }
        }
        "ack" => {
            let request_id = bag.take_text("requestId")?;
            check_request_id(&request_id)?;
            FromWorker::Ack { request_id }
        }
        "admitted" => {
            let request_id = bag.take_text("requestId")?;
            check_request_id(&request_id)?;
            FromWorker::Admitted { request_id }
        }
        "delta" => {
            let index = bag.take_u32("index")?;
            let request_id = bag.take_text("requestId")?;
            let token_id = bag.take_u32("tokenId")?;
            check_request_id(&request_id)?;
            FromWorker::Delta {
                request_id,
                index,
                token_id,
            }
        }
        "prefill" => {
            let chunk = bag.take_u32("chunk")?;
            let request_id = bag.take_text("requestId")?;
            let tokens = bag.take_u32("tokens")?;
            check_request_id(&request_id)?;
            FromWorker::Prefill {
                request_id,
                chunk,
                tokens,
            }
        }
        "stats" => FromWorker::Stats(Box::new(take_stats(&mut bag)?)),
        _ => {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "ipc message kind is unknown",
            ))
        }
    };
    bag.end()?;
    Ok(message)
}

pub fn class_name(class: ServiceClass) -> &'static str {
    match class {
        ServiceClass::Interactive => "interactive",
        ServiceClass::Standard => "standard",
        ServiceClass::Batch => "batch",
        ServiceClass::Background => "background",
    }
}

pub fn parse_class(value: &str) -> Result<ServiceClass, InferFailure> {
    match value {
        "interactive" => Ok(ServiceClass::Interactive),
        "standard" => Ok(ServiceClass::Standard),
        "batch" => Ok(ServiceClass::Batch),
        "background" => Ok(ServiceClass::Background),
        _ => Err(fail(ErrorCode::ContractInvalid, "service class is unknown")),
    }
}

pub fn check_request_id(id: &str) -> Result<(), InferFailure> {
    if !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, "request id is invalid"))
    }
}

fn check_runtime_root(value: &str) -> Result<(), InferFailure> {
    if value.starts_with("sha256-") && value.len() == "sha256-".len() + 64 {
        Ok(())
    } else {
        Err(fail(
            ErrorCode::ContractInvalid,
            "worker runtime root is invalid",
        ))
    }
}

fn check_prompt(tokens: &[u32]) -> Result<(), InferFailure> {
    if tokens.len() > MAX_PROMPT {
        Err(fail(
            ErrorCode::ContractInvalid,
            "ipc token list is too long",
        ))
    } else {
        Ok(())
    }
}

fn check_finish(value: &str) -> Result<(), InferFailure> {
    match value {
        "stop" | "length" | "cancelled" | "error" => Ok(()),
        _ => Err(fail(ErrorCode::ContractInvalid, "finish reason is unknown")),
    }
}

fn check_message(value: &str) -> Result<(), InferFailure> {
    if value.is_empty() || value.len() > MAX_MESSAGE {
        Err(fail(
            ErrorCode::ContractInvalid,
            "ipc error message length is invalid",
        ))
    } else {
        Ok(())
    }
}

fn parse_code(value: &str) -> Result<ErrorCode, InferFailure> {
    ErrorCode::parse(value).ok_or_else(|| fail(ErrorCode::ContractInvalid, "error code is unknown"))
}

fn take_stats(bag: &mut Bag) -> Result<WorkerStats, InferFailure> {
    let buckets = bag.take_u64_array("iterationBuckets", ITERATION_BUCKETS)?;
    let mut iteration_buckets = [0u64; ITERATION_BUCKETS];
    iteration_buckets.copy_from_slice(&buckets);
    Ok(WorkerStats {
        queue_interactive: bag.take_u64("queueInteractive")?,
        queue_standard: bag.take_u64("queueStandard")?,
        queue_batch: bag.take_u64("queueBatch")?,
        queue_background: bag.take_u64("queueBackground")?,
        active_sequences: bag.take_u64("activeSequences")?,
        rejected_contract: bag.take_u64("rejectedContract")?,
        rejected_context: bag.take_u64("rejectedContext")?,
        rejected_memory: bag.take_u64("rejectedMemory")?,
        rejected_prompt: bag.take_u64("rejectedPrompt")?,
        rejected_other: bag.take_u64("rejectedOther")?,
        cancellations: bag.take_u64("cancellations")?,
        oom: bag.take_u64("oom")?,
        prefill_tokens: bag.take_u64("prefillTokens")?,
        prefill_nanos: bag.take_u64("prefillNanos")?,
        decode_tokens: bag.take_u64("decodeTokens")?,
        decode_nanos: bag.take_u64("decodeNanos")?,
        iterations: bag.take_u64("iterations")?,
        iteration_nanos: bag.take_u64("iterationNanos")?,
        iteration_buckets,
        kv_total: bag.take_u64("kvTotal")?,
        kv_free: bag.take_u64("kvFree")?,
        kv_pinned: bag.take_u64("kvPinned")?,
        retained_sequences: bag.take_u64("retainedSequences")?,
        prefix_lookups: bag.take_u64("prefixLookups")?,
        prefix_hits: bag.take_u64("prefixHits")?,
        prefix_reused_tokens: bag.take_u64("prefixReusedTokens")?,
        model_load_nanos: bag.take_u64("modelLoadNanos")?,
        verified_bytes: bag.take_u64("verifiedBytes")?,
    })
}

fn envelope(kind: &str, mut fields: BTreeMap<String, CborValue>) -> CborValue {
    fields.insert("kind".into(), CborValue::Text(kind.into()));
    fields.insert("version".into(), CborValue::Integer(PROTOCOL_VERSION));
    CborValue::map(fields)
}

fn u32_array(values: &[u32]) -> CborValue {
    CborValue::Array(
        values
            .iter()
            .copied()
            .map(|value| CborValue::Integer(i128::from(value)))
            .collect(),
    )
}

struct Bag {
    map: BTreeMap<String, CborValue>,
}

impl Bag {
    fn parse(value: &CborValue) -> Result<Self, InferFailure> {
        let mut map = BTreeMap::new();
        for (key, item) in value.as_map()? {
            if map.insert(key.clone(), item.clone()).is_some() {
                return Err(fail(
                    ErrorCode::CanonicalCborInvalid,
                    "duplicate CBOR map key",
                ));
            }
        }
        Ok(Self { map })
    }

    fn take(&mut self, key: &str) -> Result<CborValue, InferFailure> {
        self.map
            .remove(key)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, format!("missing field: {key}")))
    }

    fn take_text(&mut self, key: &str) -> Result<String, InferFailure> {
        match self.take(key)? {
            CborValue::Text(value) => Ok(value),
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                format!("field {key} must be text"),
            )),
        }
    }

    fn take_u64(&mut self, key: &str) -> Result<u64, InferFailure> {
        let value = self.take_i128(key)?;
        u64::try_from(value).map_err(|_| {
            fail(
                ErrorCode::ContractInvalid,
                format!("field {key} contains an integer outside u64"),
            )
        })
    }

    fn take_u64_array(&mut self, key: &str, len: usize) -> Result<Vec<u64>, InferFailure> {
        let CborValue::Array(items) = self.take(key)? else {
            return Err(fail(
                ErrorCode::ContractInvalid,
                format!("field {key} must be an array"),
            ));
        };
        if items.len() != len {
            return Err(fail(
                ErrorCode::ContractInvalid,
                format!("field {key} has the wrong length"),
            ));
        }
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            let CborValue::Integer(value) = item else {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    format!("field {key} must contain integers"),
                ));
            };
            let value = u64::try_from(value).map_err(|_| {
                fail(
                    ErrorCode::ContractInvalid,
                    format!("field {key} contains an integer outside u64"),
                )
            })?;
            out.push(value);
        }
        Ok(out)
    }

    fn take_u32(&mut self, key: &str) -> Result<u32, InferFailure> {
        let value = self.take_i128(key)?;
        u32::try_from(value).map_err(|_| {
            fail(
                ErrorCode::ContractInvalid,
                format!("field {key} contains an integer outside u32"),
            )
        })
    }

    fn take_i128(&mut self, key: &str) -> Result<i128, InferFailure> {
        match self.take(key)? {
            CborValue::Integer(value) => Ok(value),
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                format!("field {key} must be an integer"),
            )),
        }
    }

    fn take_optional_text(&mut self, key: &str) -> Result<Option<String>, InferFailure> {
        match self.take(key)? {
            CborValue::Null => Ok(None),
            CborValue::Text(value) => Ok(Some(value)),
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                format!("field {key} must be text or null"),
            )),
        }
    }

    fn take_u32_array(&mut self, key: &str) -> Result<Vec<u32>, InferFailure> {
        let CborValue::Array(items) = self.take(key)? else {
            return Err(fail(
                ErrorCode::ContractInvalid,
                format!("field {key} must be an array"),
            ));
        };
        if items.len() > MAX_PROMPT {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "ipc token list is too long",
            ));
        }
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            let CborValue::Integer(value) = item else {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    format!("field {key} must contain integers"),
                ));
            };
            let value = u32::try_from(value).map_err(|_| {
                fail(
                    ErrorCode::ContractInvalid,
                    format!("field {key} contains an integer outside u32"),
                )
            })?;
            out.push(value);
        }
        Ok(out)
    }

    fn end(self) -> Result<(), InferFailure> {
        if let Some(key) = self.map.keys().next() {
            Err(fail(
                ErrorCode::ContractInvalid,
                format!("unknown field: {key}"),
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use infer_contracts::FixedPointSamplerV1;

    fn sampler() -> SamplerPlanV1 {
        SamplerPlanV1 {
            settings: FixedPointSamplerV1 {
                temperature_micros: 0,
                top_p_millionths: 1_000_000,
                min_p_millionths: 0,
                repetition_penalty_micros: 1_000_000,
                presence_penalty_micros: 0,
                frequency_penalty_micros: 0,
                top_k: 0,
                max_output_tokens: 2,
            },
            rng: "none".into(),
            seed: None,
            stream: None,
            tie_break: "lowest-token-id".into(),
            eos_token_ids: vec![2],
            stop_string_roots: Vec::new(),
            extensions: BTreeMap::new(),
        }
    }

    #[test]
    fn submit_and_result_round_trip() {
        let message = ToWorker::Submit {
            request_id: "job-1".into(),
            prompt: vec![1, 4],
            sampler: Box::new(sampler()),
            class: ServiceClass::Interactive,
        };
        let decoded = decode_to_worker(&encode_to_worker(&message).unwrap()).unwrap();
        assert_eq!(decoded, message);
        let result = FromWorker::Result {
            request_id: "job-1".into(),
            tokens: vec![5],
            finish_reason: "stop".into(),
            error_code: None,
        };
        let decoded = decode_from_worker(&encode_from_worker(&result).unwrap()).unwrap();
        assert_eq!(decoded, result);
        let admitted = FromWorker::Admitted {
            request_id: "job-1".into(),
        };
        let delta = FromWorker::Delta {
            request_id: "job-1".into(),
            index: 0,
            token_id: 5,
        };
        assert_eq!(
            decode_from_worker(&encode_from_worker(&admitted).unwrap()).unwrap(),
            admitted
        );
        assert_eq!(
            decode_from_worker(&encode_from_worker(&delta).unwrap()).unwrap(),
            delta
        );
        let prefill = FromWorker::Prefill {
            request_id: "job-1".into(),
            chunk: 0,
            tokens: 4,
        };
        let encoded = encode_from_worker(&prefill).unwrap();
        assert_eq!(decode_from_worker(&encoded).unwrap(), prefill);
        let CborValue::Map(entries) = &encoded else {
            panic!("map");
        };
        assert!(entries.iter().all(|(key, value)| {
            key != "tokenId" && key != "prompt" && !matches!(value, CborValue::Array(_))
        }));
    }

    #[test]
    fn an_unknown_field_is_rejected() {
        let mut value = encode_to_worker(&ToWorker::Shutdown).unwrap();
        let CborValue::Map(entries) = &mut value else {
            panic!("map");
        };
        entries.push(("extra".into(), CborValue::Bool(true)));
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        let err = decode_to_worker(&value).unwrap_err();
        assert_eq!(err.code, ErrorCode::ContractInvalid);
    }

    #[test]
    fn stats_round_trip_has_no_text_fields() {
        let mut iteration_buckets = [0; infer_engine::ITERATION_BUCKETS];
        iteration_buckets[10] = 4;
        let stats = WorkerStats {
            queue_background: 2,
            iteration_buckets,
            verified_bytes: 70,
            ..WorkerStats::default()
        };
        let decoded = decode_from_worker(
            &encode_from_worker(&FromWorker::Stats(Box::new(stats.clone()))).unwrap(),
        )
        .unwrap();
        assert_eq!(decoded, FromWorker::Stats(Box::new(stats)));
        let decoded = decode_to_worker(&encode_to_worker(&ToWorker::Stats).unwrap()).unwrap();
        assert_eq!(decoded, ToWorker::Stats);
    }
}
