//! `knolo.micro.tokens.v1` and `knolo.llama.tokens.v1`: the token list index
//! is the id. Longest match wins. Equal lengths keep the lower id. Bytes
//! that match no token fail closed.
//!
//! `huggingface.tokenizers.v1` wraps a Hugging Face `tokenizer.json`. The
//! wrapper does not load remote code.

use std::str::FromStr;

use serde::Deserialize;
use tokenizers::Tokenizer;

use infer_artifact::{parse_strict_json, parse_strict_json_allowing_null};
use infer_contracts::{fail, ErrorCode, InferFailure};

#[derive(Clone)]
pub struct MicroTokenizer {
    pub tokens: Vec<String>,
    hf: Option<Tokenizer>,
}

impl PartialEq for MicroTokenizer {
    fn eq(&self, other: &Self) -> bool {
        self.tokens == other.tokens && self.hf.is_some() == other.hf.is_some()
    }
}

impl Eq for MicroTokenizer {}

impl std::fmt::Debug for MicroTokenizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MicroTokenizer")
            .field("tokens", &self.tokens)
            .field("huggingface", &self.hf.is_some())
            .finish()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizerFile {
    kind: String,
    version: u32,
    tokens: Vec<String>,
}

pub fn parse_tokenizer(bytes: &[u8], vocab: u32) -> Result<MicroTokenizer, InferFailure> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| fail(ErrorCode::TokenizerInvalid, "tokenizer bytes are not UTF-8"))?;
    let value = match parse_strict_json(text, ErrorCode::TokenizerInvalid) {
        Ok(value) => value,
        Err(err) if err.message == "JSON null is not allowed" => {
            // A real Hugging Face tokenizer.json stores unused stages as null.
            // Null stays banned for the list tokenizers and for the wrapper
            // fields around the inner object.
            let relaxed =
                parse_strict_json_allowing_null(text, ErrorCode::TokenizerInvalid)?;
            if relaxed.get("kind").and_then(|item| item.as_str())
                != Some("huggingface.tokenizers.v1")
            {
                return Err(err);
            }
            reject_null_outside_tokenizer(&relaxed)?;
            relaxed
        }
        Err(err) => return Err(err),
    };
    if value.get("kind").and_then(|item| item.as_str()) == Some("huggingface.tokenizers.v1") {
        return parse_huggingface(&value, vocab);
    }
    let file: TokenizerFile = serde_json::from_value(value).map_err(|_| {
        fail(
            ErrorCode::TokenizerInvalid,
            "tokenizer JSON is not an allowlisted token list",
        )
    })?;
    if (file.kind != "knolo.micro.tokens.v1" && file.kind != "knolo.llama.tokens.v1")
        || file.version != 1
    {
        return Err(fail(
            ErrorCode::TokenizerInvalid,
            "tokenizer kind and version must be knolo.micro.tokens.v1 or knolo.llama.tokens.v1",
        ));
    }
    if vocab == 0 || file.tokens.len() != vocab as usize {
        return Err(fail(
            ErrorCode::TokenizerInvalid,
            "tokenizer length does not match the model vocabulary",
        ));
    }
    let mut seen = Vec::with_capacity(file.tokens.len());
    for token in &file.tokens {
        if token.is_empty() || token.chars().count() > 64 {
            return Err(fail(
                ErrorCode::TokenizerInvalid,
                "a tokenizer entry is empty or longer than 64 characters",
            ));
        }
        if seen.iter().any(|existing: &String| existing == token) {
            return Err(fail(
                ErrorCode::TokenizerInvalid,
                "tokenizer entries must be unique",
            ));
        }
        seen.push(token.clone());
    }
    Ok(MicroTokenizer {
        tokens: file.tokens,
        hf: None,
    })
}

fn reject_null_outside_tokenizer(value: &serde_json::Value) -> Result<(), InferFailure> {
    let Some(object) = value.as_object() else {
        return Err(fail(
            ErrorCode::TokenizerInvalid,
            "JSON null is not allowed",
        ));
    };
    for (key, child) in object {
        if key == "tokenizer" {
            continue;
        }
        if contains_null(child) {
            return Err(fail(
                ErrorCode::TokenizerInvalid,
                "JSON null is not allowed",
            ));
        }
    }
    Ok(())
}

fn contains_null(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => true,
        serde_json::Value::Array(items) => items.iter().any(contains_null),
        serde_json::Value::Object(map) => map.values().any(contains_null),
        _ => false,
    }
}

fn parse_huggingface(value: &serde_json::Value, vocab: u32) -> Result<MicroTokenizer, InferFailure> {
    let object = value.as_object().ok_or_else(|| {
        fail(
            ErrorCode::TokenizerInvalid,
            "huggingface tokenizer wrapper must be an object",
        )
    })?;
    for key in object.keys() {
        if key != "kind" && key != "version" && key != "tokenizer" {
            return Err(fail(
                ErrorCode::TokenizerInvalid,
                "huggingface tokenizer wrapper has an unknown field",
            ));
        }
    }
    if object.get("version").and_then(|item| item.as_u64()) != Some(1) {
        return Err(fail(
            ErrorCode::TokenizerInvalid,
            "huggingface tokenizer version must be 1",
        ));
    }
    let inner = object.get("tokenizer").filter(|item| item.is_object()).ok_or_else(|| {
        fail(
            ErrorCode::TokenizerInvalid,
            "huggingface tokenizer is missing its model",
        )
    })?;
    let text = serde_json::to_string(inner).map_err(|_| {
        fail(
            ErrorCode::TokenizerInvalid,
            "huggingface tokenizer could not be encoded",
        )
    })?;
    let tokenizer = Tokenizer::from_str(&text).map_err(|err| {
        fail(
            ErrorCode::TokenizerInvalid,
            format!("huggingface tokenizer JSON is not an allowlisted token model: {err}"),
        )
    })?;
    let size = tokenizer.get_vocab_size(true);
    if vocab == 0 || size != vocab as usize {
        return Err(fail(
            ErrorCode::TokenizerInvalid,
            "tokenizer length does not match the model vocabulary",
        ));
    }
    let mut tokens = Vec::with_capacity(size);
    for id in 0..size {
        let id = u32::try_from(id).map_err(|_| {
            fail(
                ErrorCode::TokenizerInvalid,
                "huggingface tokenizer id does not fit u32",
            )
        })?;
        let token = tokenizer.id_to_token(id).ok_or_else(|| {
            fail(
                ErrorCode::TokenizerInvalid,
                "huggingface tokenizer id has no token",
            )
        })?;
        tokens.push(token);
    }
    Ok(MicroTokenizer {
        tokens,
        hf: Some(tokenizer),
    })
}

pub fn encode_text(tokenizer: &MicroTokenizer, text: &str) -> Result<Vec<u32>, InferFailure> {
    if let Some(hf) = &tokenizer.hf {
        let encoding = hf.encode(text, false).map_err(|_| {
            fail(
                ErrorCode::TokenizerInvalid,
                "huggingface tokenizer could not encode the prompt",
            )
        })?;
        return Ok(encoding.get_ids().to_vec());
    }
    let mut order: Vec<usize> = (0..tokenizer.tokens.len()).collect();
    order.sort_by(|&left, &right| {
        tokenizer.tokens[right]
            .len()
            .cmp(&tokenizer.tokens[left].len())
            .then(left.cmp(&right))
    });
    let mut ids = Vec::new();
    let mut rest = text;
    let mut offset = 0usize;
    while !rest.is_empty() {
        let mut matched = None;
        for index in &order {
            let token = &tokenizer.tokens[*index];
            if rest.starts_with(token.as_str()) {
                matched = Some((*index, token.len()));
                break;
            }
        }
        let Some((index, length)) = matched else {
            return Err(fail(
                ErrorCode::TokenizerInvalid,
                format!("tokenizer has no match at byte {offset}"),
            ));
        };
        ids.push(u32::try_from(index).map_err(|_| {
            fail(
                ErrorCode::TokenizerInvalid,
                "tokenizer id does not fit in u32",
            )
        })?);
        rest = &rest[length..];
        offset += length;
    }
    Ok(ids)
}

pub fn decode_tokens(tokenizer: &MicroTokenizer, ids: &[u32]) -> Result<String, InferFailure> {
    if let Some(hf) = &tokenizer.hf {
        return hf.decode(ids, false).map_err(|_| {
            fail(
                ErrorCode::TokenizerInvalid,
                "huggingface tokenizer could not decode the tokens",
            )
        });
    }
    let mut text = String::new();
    for id in ids {
        let Some(token) = tokenizer.tokens.get(*id as usize) else {
            return Err(fail(
                ErrorCode::TokenizerInvalid,
                "output token id is outside the tokenizer",
            ));
        };
        text.push_str(token);
    }
    Ok(text)
}

pub fn expect_special(
    tokenizer: &MicroTokenizer,
    id: Option<u32>,
    spelling: &str,
) -> Result<(), InferFailure> {
    let Some(id) = id else {
        return Ok(());
    };
    match tokenizer.tokens.get(id as usize) {
        Some(token) if token == spelling => Ok(()),
        _ => Err(fail(
            ErrorCode::TokenizerInvalid,
            format!("special token {spelling} is not at its declared id"),
        )),
    }
}
