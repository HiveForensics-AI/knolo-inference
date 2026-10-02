//! OpenAI chat subset on the loopback supervisor.
//!
//! Decimals are accepted only where the route maps them onto fixed-point
//! sampler fields. A decimal has one to six fractional digits and no exponent.
//! Unknown fields fail.

use std::collections::BTreeMap;

use infer_contracts::{fail, ErrorCode, InferFailure};
use serde_json::{json, Value};

const MAX_BYTES: usize = 64 * 1024;
const MAX_DEPTH: usize = 8;
const MAX_VALUES: usize = 4_096;
const MAX_STRING: usize = 32 * 1024;
const SCALE: u64 = 1_000_000;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ChatRequest {
    pub model: String,
    pub messages: Vec<Turn>,
    pub stream: bool,
    pub max_tokens: Option<u32>,
    pub temperature_micros: Option<u32>,
    pub top_p_millionths: Option<u32>,
    pub seed: Option<u64>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Turn {
    pub role: String,
    pub content: String,
}

pub(crate) fn parse_chat(input: &str) -> Result<ChatRequest, InferFailure> {
    let value = parse_json(input)?;
    let J::Object(mut object) = value else {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "JSON value must be an object",
        ));
    };
    if let Some(key) = object.keys().find(|key| {
        !matches!(
            key.as_str(),
            "max_tokens" | "messages" | "model" | "n" | "seed" | "stream" | "temperature" | "top_p"
        )
    }) {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("unknown field: {key}"),
        ));
    }
    if let Some(n) = object.remove("n") {
        require_n(&n)?;
    }
    let stream = match object.remove("stream") {
        None => false,
        Some(J::Bool(value)) => value,
        Some(_) => return Err(fail(ErrorCode::ContractInvalid, "stream must be a boolean")),
    };
    let model = expect_string(object.remove("model"), "model")?;
    let messages = expect_messages(object.remove("messages"))?;
    let max_tokens = match object.remove("max_tokens") {
        None => None,
        Some(value) => Some(expect_bounded_int(&value, "max_tokens", 1, 1_048_576)?),
    };
    let temperature_micros = match object.remove("temperature") {
        None => None,
        Some(value) => Some(expect_temperature(&value)?),
    };
    let top_p_millionths = match object.remove("top_p") {
        None => None,
        Some(value) => Some(expect_top_p(&value)?),
    };
    let seed = match object.remove("seed") {
        None => None,
        Some(J::Int(value)) => Some(value),
        Some(_) => return Err(fail(ErrorCode::ContractInvalid, "seed must be an integer")),
    };
    if !object.is_empty() {
        return Err(fail(ErrorCode::ContractInvalid, "unknown field"));
    }
    Ok(ChatRequest {
        model,
        messages,
        stream,
        max_tokens,
        temperature_micros,
        top_p_millionths,
        seed,
    })
}

pub(crate) fn models_document(alias: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "data": [{
            "id": alias,
            "object": "model",
            "owned_by": "knolo",
        }],
        "object": "list",
    }))
    .expect("models json")
}

pub(crate) fn completion_document(
    id: &str,
    model: &str,
    text: &str,
    finish: &str,
    prompt_tokens: u32,
    output_tokens: u32,
    receipt: Option<&str>,
) -> Vec<u8> {
    let total = prompt_tokens.saturating_add(output_tokens);
    let mut value = json!({
        "choices": [{
            "finish_reason": finish,
            "index": 0,
            "message": {"content": text, "role": "assistant"},
        }],
        "id": id,
        "model": model,
        "object": "chat.completion",
        "usage": {
            "completion_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total,
        },
    });
    if let Some(receipt) = receipt {
        value["knolo_receipt"] = json!(receipt);
    }
    serde_json::to_vec(&value).expect("completion json")
}

pub(crate) fn chunk_document(
    id: &str,
    model: &str,
    delta: Value,
    finish: Option<&str>,
    receipt: Option<&str>,
) -> String {
    let finish_reason = match finish {
        Some(value) => Value::String(value.to_string()),
        None => Value::Null,
    };
    let mut value = json!({
        "choices": [{
            "delta": delta,
            "finish_reason": finish_reason,
            "index": 0,
        }],
        "id": id,
        "model": model,
        "object": "chat.completion.chunk",
    });
    if let Some(receipt) = receipt {
        value["knolo_receipt"] = json!(receipt);
    }
    serde_json::to_string(&value).expect("chunk json")
}

pub(crate) fn error_document(code: &str, message: &str) -> Vec<u8> {
    let message: String = message.chars().take(1024).collect();
    let kind = match code {
        "WORKER_LOST"
        | "WORKER_START_FAILED"
        | "REQUEST_TIMEOUT"
        | "SERVICE_DRAINING"
        | "SERVICE_UNLOADED" => "server_error",
        _ => "invalid_request_error",
    };
    serde_json::to_vec(&json!({
        "error": {"code": code, "message": message, "type": kind}
    }))
    .unwrap_or_else(|_| {
        br#"{"error":{"code":"CONTRACT_INVALID","message":"response encoding failed","type":"invalid_request_error"}}"#
            .to_vec()
    })
}

fn require_n(value: &J) -> Result<(), InferFailure> {
    match value {
        J::Int(1) => Ok(()),
        _ => Err(fail(ErrorCode::ContractInvalid, "n must be 1")),
    }
}

fn expect_temperature(value: &J) -> Result<u32, InferFailure> {
    let micros = match value {
        J::Int(whole) => whole.checked_mul(SCALE).ok_or_else(|| {
            fail(
                ErrorCode::ContractInvalid,
                "temperature is outside 0 through 2",
            )
        })?,
        J::Decimal(micros) => u64::from(*micros),
        _ => {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "temperature must be a number",
            ))
        }
    };
    if micros > 2 * SCALE {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "temperature must be from 0 through 2",
        ));
    }
    u32::try_from(micros).map_err(|_| {
        fail(
            ErrorCode::ContractInvalid,
            "temperature must be from 0 through 2",
        )
    })
}

fn expect_top_p(value: &J) -> Result<u32, InferFailure> {
    let millionths = match value {
        J::Int(1) => SCALE,
        J::Int(_) => {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "top_p must be greater than 0 and at most 1",
            ))
        }
        J::Decimal(micros) => u64::from(*micros),
        _ => return Err(fail(ErrorCode::ContractInvalid, "top_p must be a number")),
    };
    if millionths == 0 || millionths > SCALE {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "top_p must be greater than 0 and at most 1",
        ));
    }
    u32::try_from(millionths).map_err(|_| {
        fail(
            ErrorCode::ContractInvalid,
            "top_p must be greater than 0 and at most 1",
        )
    })
}

fn expect_bounded_int(value: &J, key: &str, low: u32, high: u32) -> Result<u32, InferFailure> {
    let J::Int(raw) = value else {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("field {key} must be an integer"),
        ));
    };
    let value = u32::try_from(*raw).unwrap_or(u32::MAX);
    if value < low || value > high {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("field {key} is outside {low} through {high}"),
        ));
    }
    Ok(value)
}

fn expect_string(value: Option<J>, key: &str) -> Result<String, InferFailure> {
    match value {
        Some(J::String(text)) => Ok(text),
        Some(_) => Err(fail(
            ErrorCode::ContractInvalid,
            format!("field {key} must be text"),
        )),
        None => Err(fail(
            ErrorCode::ContractInvalid,
            format!("missing field: {key}"),
        )),
    }
}

fn expect_messages(value: Option<J>) -> Result<Vec<Turn>, InferFailure> {
    let Some(J::Array(items)) = value else {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "messages must be an array",
        ));
    };
    if items.is_empty() || items.len() > 32 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "messages must contain 1 to 32 entries",
        ));
    }
    let mut messages = Vec::with_capacity(items.len());
    for item in items {
        let J::Object(mut message) = item else {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "JSON value must be an object",
            ));
        };
        if let Some(key) = message
            .keys()
            .find(|key| key.as_str() != "content" && key.as_str() != "role")
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                format!("unknown field: {key}"),
            ));
        }
        messages.push(Turn {
            role: expect_string(message.remove("role"), "role")?,
            content: expect_string(message.remove("content"), "content")?,
        });
    }
    Ok(messages)
}

enum J {
    Bool(bool),
    String(String),
    Int(u64),
    Decimal(u32),
    Array(Vec<J>),
    Object(BTreeMap<String, J>),
}

fn parse_json(input: &str) -> Result<J, InferFailure> {
    if input.len() > MAX_BYTES {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "JSON document exceeds 64 KiB",
        ));
    }
    if input.as_bytes().starts_with(&[0xef, 0xbb, 0xbf]) {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "JSON document must not start with a BOM",
        ));
    }
    let mut parser = Parser {
        input,
        index: 0,
        depth: 0,
        values: 0,
    };
    parser.skip_ws();
    let value = parser.parse_value()?;
    parser.skip_ws();
    if parser.index != parser.input.len() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "JSON document has trailing data",
        ));
    }
    Ok(value)
}

struct Parser<'a> {
    input: &'a str,
    index: usize,
    depth: usize,
    values: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, message: impl Into<String>) -> InferFailure {
        fail(ErrorCode::ContractInvalid, message)
    }

    fn bump(&mut self) -> Result<(), InferFailure> {
        self.values += 1;
        if self.values > MAX_VALUES {
            Err(self.err("JSON document has too many values"))
        } else {
            Ok(())
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.as_bytes().get(self.index).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.index += 1;
        }
    }

    fn parse_value(&mut self) -> Result<J, InferFailure> {
        self.skip_ws();
        self.bump()?;
        match self.peek() {
            Some(b'n') => {
                self.consume(b"null")?;
                Err(self.err("JSON null is not allowed"))
            }
            Some(b't') => {
                self.consume(b"true")?;
                self.reject_ident_tail()?;
                Ok(J::Bool(true))
            }
            Some(b'f') => {
                self.consume(b"false")?;
                self.reject_ident_tail()?;
                Ok(J::Bool(false))
            }
            Some(b'"') => Ok(J::String(self.parse_string()?)),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(b'0'..=b'9') => self.parse_number(),
            Some(b'-') => Err(self.err("JSON numbers must be unsigned")),
            _ => Err(self.err("invalid JSON value")),
        }
    }

    fn consume(&mut self, literal: &[u8]) -> Result<(), InferFailure> {
        let rest = self.input.as_bytes().get(self.index..).unwrap_or(&[]);
        if rest.starts_with(literal) {
            self.index += literal.len();
            Ok(())
        } else {
            Err(self.err("invalid JSON literal"))
        }
    }

    fn reject_ident_tail(&self) -> Result<(), InferFailure> {
        if matches!(
            self.peek(),
            Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
        ) {
            Err(self.err("invalid JSON literal"))
        } else {
            Ok(())
        }
    }

    fn parse_array(&mut self) -> Result<J, InferFailure> {
        self.index += 1;
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err("JSON nesting exceeds the depth limit"));
        }
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.index += 1;
            self.depth -= 1;
            return Ok(J::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.index += 1;
                    self.skip_ws();
                    if self.peek() == Some(b']') {
                        return Err(self.err("trailing comma in JSON array"));
                    }
                }
                Some(b']') => {
                    self.index += 1;
                    self.depth -= 1;
                    return Ok(J::Array(items));
                }
                _ => return Err(self.err("JSON array is not closed")),
            }
        }
    }

    fn parse_object(&mut self) -> Result<J, InferFailure> {
        self.index += 1;
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err("JSON nesting exceeds the depth limit"));
        }
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.index += 1;
            self.depth -= 1;
            return Ok(J::Object(map));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("JSON object key must be a string"));
            }
            let key = self.parse_string()?;
            if key.len() > 1024 {
                return Err(self.err("JSON object key is too long"));
            }
            if map.contains_key(&key) {
                return Err(self.err(format!("duplicate JSON key {key}")));
            }
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(self.err("JSON object is missing ':'"));
            }
            self.index += 1;
            map.insert(key, self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.index += 1;
                    self.skip_ws();
                    if self.peek() == Some(b'}') {
                        return Err(self.err("trailing comma in JSON object"));
                    }
                }
                Some(b'}') => {
                    self.index += 1;
                    self.depth -= 1;
                    return Ok(J::Object(map));
                }
                _ => return Err(self.err("JSON object is not closed")),
            }
        }
    }

    fn parse_number(&mut self) -> Result<J, InferFailure> {
        let start = self.index;
        if self.peek() == Some(b'0') {
            self.index += 1;
            if matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.err("JSON integers must not have leading zeros"));
            }
        } else {
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.index += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E' | b'+' | b'-')) {
            return Err(self.err("JSON exponents are not accepted"));
        }
        if self.peek() != Some(b'.') {
            let text = &self.input[start..self.index];
            let value: u64 = text
                .parse()
                .map_err(|_| self.err("JSON integer exceeds u64"))?;
            return Ok(J::Int(value));
        }
        self.index += 1;
        let frac_at = self.index;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.index += 1;
        }
        let frac = &self.input[frac_at..self.index];
        if frac.is_empty() || frac.len() > 6 {
            return Err(self.err("JSON decimal must have 1 to 6 fractional digits"));
        }
        if matches!(self.peek(), Some(b'.' | b'e' | b'E')) {
            return Err(self.err("JSON exponents are not accepted"));
        }
        let whole: u64 = self.input[start..frac_at - 1]
            .parse()
            .map_err(|_| self.err("JSON decimal exceeds the fixed-point range"))?;
        let mut padded = frac.to_string();
        while padded.len() < 6 {
            padded.push('0');
        }
        let fraction: u64 = padded
            .parse()
            .map_err(|_| self.err("JSON decimal exceeds the fixed-point range"))?;
        let micros = whole
            .checked_mul(SCALE)
            .and_then(|scaled| scaled.checked_add(fraction))
            .ok_or_else(|| self.err("JSON decimal exceeds the fixed-point range"))?;
        let micros = u32::try_from(micros)
            .map_err(|_| self.err("JSON decimal exceeds the fixed-point range"))?;
        Ok(J::Decimal(micros))
    }

    fn parse_string(&mut self) -> Result<String, InferFailure> {
        self.index += 1;
        let mut out = String::new();
        while let Some(byte) = self.peek() {
            self.index += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escaped = self
                        .peek()
                        .ok_or_else(|| self.err("JSON string is not closed"))?;
                    self.index += 1;
                    let ch = match escaped {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{0008}',
                        b'f' => '\u{000c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => self.parse_hex()?,
                        _ => return Err(self.err("JSON string escape is invalid")),
                    };
                    out.push(ch);
                }
                0x00..=0x1f => return Err(self.err("JSON string contains a control character")),
                byte => {
                    let width =
                        utf8_width(byte).ok_or_else(|| self.err("JSON string is not UTF-8"))?;
                    let start = self.index - 1;
                    let end = start + width;
                    if self.input.len() < end {
                        return Err(self.err("JSON string is not UTF-8"));
                    }
                    let text = self.input[start..end].to_string();
                    if text.chars().count() != 1 {
                        return Err(self.err("JSON string is not UTF-8"));
                    }
                    out.push_str(&text);
                    self.index = end;
                }
            }
            if out.len() > MAX_STRING {
                return Err(self.err("JSON string exceeds 32 KiB"));
            }
        }
        Err(self.err("JSON string is not closed"))
    }

    fn parse_hex(&mut self) -> Result<char, InferFailure> {
        let start = self.index;
        if self.input.len() < start + 4 {
            return Err(self.err("JSON unicode escape is invalid"));
        }
        let digits = &self.input[start..start + 4];
        if !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(self.err("JSON unicode escape is invalid"));
        }
        self.index += 4;
        let code = u32::from_str_radix(digits, 16)
            .map_err(|_| self.err("JSON unicode escape is invalid"))?;
        char::from_u32(code).ok_or_else(|| self.err("JSON unicode escape is invalid"))
    }
}

fn utf8_width(byte: u8) -> Option<usize> {
    if byte < 0x80 {
        Some(1)
    } else if byte & 0xe0 == 0xc0 {
        Some(2)
    } else if byte & 0xf0 == 0xe0 {
        Some(3)
    } else if byte & 0xf8 == 0xf0 {
        Some(4)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(extra: &str) -> String {
        format!(r#"{{"messages":[{{"content":"hi","role":"user"}}],"model":"micro"{extra}}}"#)
    }

    #[test]
    fn temperature_and_top_p_scale_by_one_million() {
        let parsed = parse_chat(&chat(
            r#","max_tokens":2,"seed":7,"temperature":0.5,"top_p":1.0"#,
        ))
        .unwrap();
        assert_eq!(parsed.temperature_micros, Some(500_000));
        assert_eq!(parsed.top_p_millionths, Some(1_000_000));
        assert_eq!(parsed.max_tokens, Some(2));
        assert_eq!(parsed.seed, Some(7));
        assert!(!parsed.stream);
        let whole = parse_chat(&chat(r#","temperature":2,"top_p":1"#)).unwrap();
        assert_eq!(whole.temperature_micros, Some(2_000_000));
        assert_eq!(whole.top_p_millionths, Some(1_000_000));
    }

    #[test]
    fn unsupported_numbers_and_fields_are_refused() {
        let cases = [
            chat(r#","temperature":1e-1"#),
            chat(r#","temperature":2.5"#),
            chat(r#","temperature":-1"#),
            chat(r#","top_p":0"#),
            chat(r#","max_tokens":1.5"#),
            chat(r#","n":2"#),
            chat(r#","tools":[]"#),
            chat(r#","temperature":0.0000001"#),
        ];
        for body in cases {
            let err = parse_chat(&body).unwrap_err();
            assert_eq!(err.code, ErrorCode::ContractInvalid, "{body}: {err}");
        }
    }
}
