//! `knolo.micro.tokens.v1`: the token list index is the id. Longest match wins.
//! Equal lengths keep the lower id. Bytes that match no token fail closed.

use serde::Deserialize;

use infer_artifact::parse_strict_json;
use infer_contracts::{fail, ErrorCode, InferFailure};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroTokenizer {
    pub tokens: Vec<String>,
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
    let value = parse_strict_json(text, ErrorCode::TokenizerInvalid)?;
    let file: TokenizerFile = serde_json::from_value(value).map_err(|_| {
        fail(
            ErrorCode::TokenizerInvalid,
            "tokenizer JSON is not knolo.micro.tokens.v1",
        )
    })?;
    if file.kind != "knolo.micro.tokens.v1" || file.version != 1 {
        return Err(fail(
            ErrorCode::TokenizerInvalid,
            "tokenizer kind and version must be knolo.micro.tokens.v1",
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
    })
}

pub fn encode_text(tokenizer: &MicroTokenizer, text: &str) -> Result<Vec<u32>, InferFailure> {
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
