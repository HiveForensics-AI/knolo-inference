//! JSON with the same fail-closed habits as canonical CBOR: no floats, no
//! duplicate keys, no trailing commas, and hard size and depth limits.

use serde_json::{Map, Number, Value};

use infer_contracts::{fail, ErrorCode, InferFailure};

const MAX_BYTES: usize = 32 * 1024 * 1024;
const MAX_DEPTH: usize = 32;
const MAX_VALUES: usize = 1_048_576;
const MAX_STRING: usize = 1024 * 1024;

pub fn parse_strict_json(input: &str, code: ErrorCode) -> Result<Value, InferFailure> {
    if input.len() > MAX_BYTES {
        return Err(fail(code, "JSON document exceeds 32 MiB"));
    }
    if input.as_bytes().starts_with(&[0xef, 0xbb, 0xbf]) {
        return Err(fail(code, "JSON document must not start with a BOM"));
    }
    let mut parser = Parser {
        input,
        index: 0,
        depth: 0,
        values: 0,
        code,
    };
    parser.skip_ws();
    let value = parser.parse_value()?;
    parser.skip_ws();
    if parser.index != parser.input.len() {
        return Err(fail(code, "JSON document has trailing data"));
    }
    Ok(value)
}

struct Parser<'a> {
    input: &'a str,
    index: usize,
    depth: usize,
    values: usize,
    code: ErrorCode,
}

impl<'a> Parser<'a> {
    fn err(&self, message: impl Into<String>) -> InferFailure {
        fail(self.code, message)
    }

    fn bump(&mut self) -> Result<(), InferFailure> {
        self.values += 1;
        if self.values > MAX_VALUES {
            Err(self.err("JSON document has too many values"))
        } else {
            Ok(())
        }
    }

    fn enter(&mut self) -> Result<(), InferFailure> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            Err(self.err("JSON nesting exceeds the depth limit"))
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

    fn parse_value(&mut self) -> Result<Value, InferFailure> {
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
                Ok(Value::Bool(true))
            }
            Some(b'f') => {
                self.consume(b"false")?;
                self.reject_ident_tail()?;
                Ok(Value::Bool(false))
            }
            Some(b'"') => Ok(Value::String(self.parse_string()?)),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(b'0'..=b'9') => self.parse_number(),
            Some(b'-') => Err(self.err("JSON numbers must be unsigned integers")),
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

    fn parse_array(&mut self) -> Result<Value, InferFailure> {
        self.index += 1;
        self.enter()?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.index += 1;
            self.depth -= 1;
            return Ok(Value::Array(items));
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
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.err("JSON array is not closed")),
            }
        }
    }

    fn parse_object(&mut self) -> Result<Value, InferFailure> {
        self.index += 1;
        self.enter()?;
        let mut map = Map::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.index += 1;
            self.depth -= 1;
            return Ok(Value::Object(map));
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
            let value = self.parse_value()?;
            map.insert(key, value);
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
                    return Ok(Value::Object(map));
                }
                _ => return Err(self.err("JSON object is not closed")),
            }
        }
    }

    fn parse_number(&mut self) -> Result<Value, InferFailure> {
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
        if matches!(self.peek(), Some(b'.' | b'e' | b'E' | b'+')) {
            return Err(self.err("JSON numbers must be integers"));
        }
        let text = &self.input[start..self.index];
        let value: u64 = text
            .parse()
            .map_err(|_| self.err("JSON integer exceeds u64"))?;
        Ok(Value::Number(Number::from(value)))
    }

    fn parse_string(&mut self) -> Result<String, InferFailure> {
        self.index += 1;
        let mut out = String::new();
        loop {
            let Some(ch) = self.peek_char()? else {
                return Err(self.err("unterminated JSON string"));
            };
            match ch {
                '"' => {
                    self.index += 1;
                    if out.len() > MAX_STRING {
                        return Err(self.err("JSON string is too long"));
                    }
                    return Ok(out);
                }
                '\\' => {
                    self.index += 1;
                    out.push(self.parse_escape()?);
                }
                ch if (ch as u32) < 0x20 => {
                    return Err(self.err("JSON string contains a control character"));
                }
                ch => {
                    self.index += ch.len_utf8();
                    out.push(ch);
                }
            }
        }
    }

    fn parse_escape(&mut self) -> Result<char, InferFailure> {
        let ch = self.next_char()?;
        match ch {
            '"' | '\\' | '/' => Ok(ch),
            'b' => Ok('\u{0008}'),
            'f' => Ok('\u{000c}'),
            'n' => Ok('\n'),
            'r' => Ok('\r'),
            't' => Ok('\t'),
            'u' => self.parse_unicode(),
            _ => Err(self.err("invalid JSON string escape")),
        }
    }

    fn parse_unicode(&mut self) -> Result<char, InferFailure> {
        let unit = self.parse_hex4()?;
        let scalar = if (0xD800..=0xDBFF).contains(&unit) {
            if self.peek_char()? != Some('\\') {
                return Err(self.err("lone UTF-16 surrogate in JSON string"));
            }
            self.index += 1;
            if self.next_char()? != 'u' {
                return Err(self.err("lone UTF-16 surrogate in JSON string"));
            }
            let low = self.parse_hex4()?;
            if !(0xDC00..=0xDFFF).contains(&low) {
                return Err(self.err("invalid UTF-16 surrogate pair in JSON string"));
            }
            0x10000 + (((unit - 0xD800) << 10) | (low - 0xDC00))
        } else if (0xDC00..=0xDFFF).contains(&unit) {
            return Err(self.err("lone UTF-16 surrogate in JSON string"));
        } else {
            unit
        };
        char::from_u32(scalar).ok_or_else(|| self.err("invalid Unicode scalar in JSON string"))
    }

    fn parse_hex4(&mut self) -> Result<u32, InferFailure> {
        let mut value = 0u32;
        for _ in 0..4 {
            let ch = self.next_char()?;
            let digit = match ch {
                '0'..='9' => u32::from(ch) - u32::from('0'),
                'a'..='f' => u32::from(ch) - u32::from('a') + 10,
                'A'..='F' => u32::from(ch) - u32::from('A') + 10,
                _ => return Err(self.err("invalid JSON unicode escape")),
            };
            value = (value << 4) | digit;
        }
        Ok(value)
    }

    fn peek_char(&self) -> Result<Option<char>, InferFailure> {
        let Some(rest) = self.input.get(self.index..) else {
            return Err(self.err("JSON is not on a character boundary"));
        };
        Ok(rest.chars().next())
    }

    fn next_char(&mut self) -> Result<char, InferFailure> {
        let ch = self
            .peek_char()?
            .ok_or_else(|| self.err("unexpected end of JSON"))?;
        self.index += ch.len_utf8();
        Ok(ch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_keys_floats_and_trailing_commas() {
        let code = ErrorCode::ModelImageInvalid;
        assert!(parse_strict_json(r#"{"a":1,"a":2}"#, code).is_err());
        assert!(parse_strict_json(r#"{"a":1.5}"#, code).is_err());
        assert!(parse_strict_json(r#"{"a":1,}"#, code).is_err());
        assert!(parse_strict_json(r#"{"a":null}"#, code).is_err());
        let value = parse_strict_json(r#"{"z":1,"a":{"b":"\u0041"}}"#, code).unwrap();
        assert_eq!(value["a"]["b"], "A");
    }
}
