//! YAML subset for model-image authoring.
//!
//! The subset is indentation-based maps and lists, JSON-style double quotes,
//! single quotes, booleans `true` and `false`, and unsigned integers.
//! Anchors, tags, tabs, nulls, and floats are rejected. Duplicate keys are
//! rejected. Inline flow is only `[]` and `{}`.

use serde_json::{Map, Number, Value};

use infer_contracts::{fail, ErrorCode, InferFailure};

use crate::json::parse_strict_json;

const MAX_DEPTH: usize = 32;

pub fn parse_yaml_subset(input: &str) -> Result<Value, InferFailure> {
    if input.len() > 32 * 1024 * 1024 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "YAML document exceeds 32 MiB",
        ));
    }
    if input.contains('\0') {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "YAML document contains a null byte",
        ));
    }
    let lines = collect_lines(input)?;
    if lines.is_empty() {
        return Err(fail(ErrorCode::ModelImageInvalid, "YAML document is empty"));
    }
    let mut parser = Parser {
        lines,
        index: 0,
        depth: 0,
    };
    let value = parser.parse_node()?;
    if parser.index != parser.lines.len() {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "YAML document has trailing data",
        ));
    }
    Ok(value)
}

struct Parser<'a> {
    lines: Vec<(usize, &'a str)>,
    index: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, message: impl Into<String>) -> InferFailure {
        fail(ErrorCode::ModelImageInvalid, message)
    }

    fn current(&self) -> Result<(usize, &'a str), InferFailure> {
        self.lines
            .get(self.index)
            .copied()
            .ok_or_else(|| self.err("unexpected end of YAML"))
    }

    fn parse_node(&mut self) -> Result<Value, InferFailure> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err("YAML nesting exceeds the depth limit"));
        }
        let (indent, text) = self.current()?;
        let value = if text == "[]" {
            self.index += 1;
            Value::Array(Vec::new())
        } else if text == "{}" {
            self.index += 1;
            Value::Object(Map::new())
        } else if text == "-" || text.starts_with("- ") {
            self.parse_sequence(indent)?
        } else {
            self.parse_mapping(indent, None)?
        };
        self.depth -= 1;
        Ok(value)
    }

    fn parse_mapping(
        &mut self,
        indent: usize,
        mut pending: Option<(String, String)>,
    ) -> Result<Value, InferFailure> {
        let mut map = Map::new();
        loop {
            let (key, rest) = if let Some(pair) = pending.take() {
                pair
            } else {
                if self.index >= self.lines.len() || self.lines[self.index].0 != indent {
                    break;
                }
                let text = self.lines[self.index].1;
                if text.starts_with('-') {
                    return Err(self.err("expected a YAML key"));
                }
                let Some(pair) = match_key(text)? else {
                    return Err(self.err("expected a YAML key"));
                };
                self.index += 1;
                pair
            };
            if map.contains_key(&key) {
                return Err(self.err(format!("duplicate YAML key {key}")));
            }
            let value = self.value_after_key(indent, &rest)?;
            map.insert(key, value);
        }
        if map.is_empty() {
            return Err(self.err("YAML mapping is empty"));
        }
        Ok(Value::Object(map))
    }

    fn value_after_key(&mut self, key_indent: usize, rest: &str) -> Result<Value, InferFailure> {
        if rest.is_empty() {
            if self.index >= self.lines.len() || self.lines[self.index].0 <= key_indent {
                return Err(self.err("missing nested YAML value"));
            }
            return self.parse_node();
        }
        if rest == "[]" {
            return Ok(Value::Array(Vec::new()));
        }
        if rest == "{}" {
            return Ok(Value::Object(Map::new()));
        }
        parse_scalar(rest)
    }

    fn parse_sequence(&mut self, indent: usize) -> Result<Value, InferFailure> {
        let mut items = Vec::new();
        while self.index < self.lines.len() && self.lines[self.index].0 == indent {
            let text = self.lines[self.index].1;
            let Some(rest) = text.strip_prefix("- ") else {
                if text == "-" {
                    self.index += 1;
                    if self.index >= self.lines.len() || self.lines[self.index].0 <= indent {
                        return Err(self.err("empty YAML sequence item"));
                    }
                    items.push(self.parse_node()?);
                    continue;
                }
                break;
            };
            self.index += 1;
            if rest.is_empty() {
                return Err(self.err("empty YAML sequence item"));
            }
            if rest == "[]" {
                items.push(Value::Array(Vec::new()));
                continue;
            }
            if rest == "{}" {
                items.push(Value::Object(Map::new()));
                continue;
            }
            if let Some(pair) = match_key(rest)? {
                items.push(self.parse_mapping(indent + 2, Some(pair))?);
            } else {
                items.push(parse_scalar(rest)?);
            }
        }
        if items.is_empty() {
            return Err(self.err("YAML sequence is empty"));
        }
        Ok(Value::Array(items))
    }
}

fn collect_lines(input: &str) -> Result<Vec<(usize, &str)>, InferFailure> {
    if input.contains('\t') {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "YAML tabs are not allowed",
        ));
    }
    let mut lines = Vec::new();
    for raw in input.lines() {
        let indent = raw.bytes().take_while(|byte| *byte == b' ').count();
        let body = strip_comment(&raw[indent..]);
        if body.is_empty() {
            continue;
        }
        if body == "---" || body == "..." {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                "YAML document markers are not allowed",
            ));
        }
        lines.push((indent, body));
    }
    Ok(lines)
}

fn strip_comment(text: &str) -> &str {
    let bytes = text.as_bytes();
    let mut quote = None;
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if let Some(active) = quote {
            if byte == active {
                quote = None;
            }
            index += 1;
            continue;
        }
        if byte == b'"' || byte == b'\'' {
            quote = Some(byte);
            index += 1;
            continue;
        }
        if byte == b'#' && (index == 0 || bytes[index - 1] == b' ') {
            return text[..index].trim_end_matches(' ');
        }
        index += 1;
    }
    text.trim_end_matches(' ')
}

fn match_key(text: &str) -> Result<Option<(String, String)>, InferFailure> {
    let bytes = text.as_bytes();
    if bytes.is_empty() || !is_key_start(bytes[0]) {
        return Ok(None);
    }
    let mut index = 1;
    while index < bytes.len() && is_key_char(bytes[index]) {
        index += 1;
    }
    if index >= bytes.len() || bytes[index] != b':' {
        return Ok(None);
    }
    let key = text[..index].to_string();
    let rest = &text[index + 1..];
    if rest.is_empty() {
        return Ok(Some((key, String::new())));
    }
    if !rest.starts_with(' ') {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "YAML key colon must be followed by a space or a nested value",
        ));
    }
    Ok(Some((key, rest.trim_start_matches(' ').to_string())))
}

fn is_key_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn is_key_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' || byte == b'.'
}

fn parse_scalar(text: &str) -> Result<Value, InferFailure> {
    if text.is_empty() {
        return Err(fail(ErrorCode::ModelImageInvalid, "empty YAML scalar"));
    }
    if text.starts_with('"') {
        let value = parse_strict_json(text, ErrorCode::ModelImageInvalid)?;
        return match value {
            Value::String(text) => Ok(Value::String(text)),
            _ => Err(fail(
                ErrorCode::ModelImageInvalid,
                "quoted YAML scalar must be a string",
            )),
        };
    }
    if text.starts_with('\'') {
        return Ok(Value::String(parse_single(text)?));
    }
    if text.starts_with(['[', '{', '&', '*', '!', '|', '>']) {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "unsupported YAML syntax",
        ));
    }
    if text.contains([':', '&', '*', '!', '|', '>', '#']) {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "unsupported YAML scalar",
        ));
    }
    classify_plain(text)
}

fn parse_single(text: &str) -> Result<String, InferFailure> {
    if text.len() < 2 || !text.starts_with('\'') || !text.ends_with('\'') {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "unterminated YAML string",
        ));
    }
    let inner = &text[1..text.len() - 1];
    let mut out = String::new();
    let mut chars = inner.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\'' {
            if chars.next() == Some('\'') {
                out.push('\'');
            } else {
                return Err(fail(
                    ErrorCode::ModelImageInvalid,
                    "invalid YAML single-quoted string",
                ));
            }
        } else {
            out.push(ch);
        }
    }
    Ok(out)
}

fn classify_plain(text: &str) -> Result<Value, InferFailure> {
    if text == "true" {
        return Ok(Value::Bool(true));
    }
    if text == "false" {
        return Ok(Value::Bool(false));
    }
    if text == "null" || text == "~" || text.eq_ignore_ascii_case("null") {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "YAML null is not allowed",
        ));
    }
    if text.bytes().all(|byte| byte.is_ascii_digit()) {
        if text != "0" && text.starts_with('0') {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                "YAML integers must not have leading zeros",
            ));
        }
        let number: u64 = text
            .parse()
            .map_err(|_| fail(ErrorCode::ModelImageInvalid, "YAML integer exceeds u64"))?;
        return Ok(Value::Number(Number::from(number)));
    }
    if looks_like_float(text) {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "YAML floats are not allowed",
        ));
    }
    if text.chars().any(char::is_control) {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "YAML scalar contains a control character",
        ));
    }
    Ok(Value::String(text.to_string()))
}

fn looks_like_float(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.contains(&b'.')
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_digit() || *byte == b'.')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_manifest_shape_and_rejects_duplicates() {
        let text = r#"
kind: knolo.infer.model-image
version: 1
weights:
  format: safetensors
  files:
    - path: weights.safetensors
      sizeBytes: 0
capabilities:
  - text-generation
sources: []
specialTokens:
  additional: {}
license:
  acceptanceRequired: false
"#;
        let value = parse_yaml_subset(text).unwrap();
        assert_eq!(value["version"], 1);
        assert_eq!(value["weights"]["files"][0]["path"], "weights.safetensors");
        assert_eq!(value["weights"]["files"][0]["sizeBytes"], 0);
        assert!(value["sources"].as_array().unwrap().is_empty());
        assert!(value["specialTokens"]["additional"]
            .as_object()
            .unwrap()
            .is_empty());
        assert_eq!(value["license"]["acceptanceRequired"], false);
        assert!(parse_yaml_subset("kind: a\nkind: b\n").is_err());
        assert!(parse_yaml_subset("kind:\ttab\n").is_err());
        assert!(parse_yaml_subset("temp: 0.7\n").is_err());
        assert!(parse_yaml_subset("alias: &a\n").is_err());
    }
}
