//! Definite-length canonical CBOR.
//!
//! The subset matches KIP-0003: null, bool, integers, byte strings, UTF-8
//! text, arrays, and text-keyed maps. Floats, tags, indefinite lengths,
//! non-shortest integers, unsorted keys, and duplicate keys are rejected.

use std::collections::BTreeMap;

use crate::error::{fail, ErrorCode, InferFailure};

pub const MAX_DOCUMENT_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_DEPTH: usize = 32;
pub const MAX_ITEMS: usize = 1_048_576;

const I128_MIN_CBOR: i128 = -1i128 << 64;
const I128_MAX_CBOR: i128 = (1i128 << 64) - 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CborValue {
    Null,
    Bool(bool),
    Integer(i128),
    Bytes(Vec<u8>),
    Text(String),
    Array(Vec<CborValue>),
    Map(Vec<(String, CborValue)>),
}

impl CborValue {
    pub fn map(fields: BTreeMap<String, CborValue>) -> Self {
        Self::Map(fields.into_iter().collect())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        encode(self, &mut out);
        out
    }

    pub fn as_map(&self) -> Result<&[(String, CborValue)], InferFailure> {
        match self {
            Self::Map(entries) => Ok(entries),
            _ => Err(fail(ErrorCode::ContractInvalid, "expected a CBOR map")),
        }
    }

    pub fn as_array(&self) -> Result<&[CborValue], InferFailure> {
        match self {
            Self::Array(items) => Ok(items),
            _ => Err(fail(ErrorCode::ContractInvalid, "expected a CBOR array")),
        }
    }
}

pub fn encode_canonical(value: &CborValue) -> Vec<u8> {
    value.to_bytes()
}

pub fn decode_canonical(bytes: &[u8]) -> Result<CborValue, InferFailure> {
    if bytes.is_empty() {
        return Err(fail(ErrorCode::CanonicalCborInvalid, "empty CBOR input"));
    }
    if bytes.len() > MAX_DOCUMENT_BYTES {
        return Err(fail(
            ErrorCode::CanonicalCborInvalid,
            "CBOR input exceeds the size limit",
        ));
    }
    let mut decoder = Decoder {
        bytes,
        offset: 0,
        depth: 0,
    };
    let value = decoder.read()?;
    if decoder.offset != bytes.len() {
        return Err(fail(
            ErrorCode::CanonicalCborInvalid,
            "trailing bytes after CBOR value",
        ));
    }
    let recoded = value.to_bytes();
    if recoded != bytes {
        return Err(fail(
            ErrorCode::CanonicalCborInvalid,
            "CBOR value is not canonical",
        ));
    }
    Ok(value)
}

fn encode(value: &CborValue, out: &mut Vec<u8>) {
    match value {
        CborValue::Null => out.push(0xf6),
        CborValue::Bool(true) => out.push(0xf5),
        CborValue::Bool(false) => out.push(0xf4),
        CborValue::Integer(n) => encode_integer(*n, out),
        CborValue::Bytes(bytes) => {
            encode_length(2, bytes.len() as u128, out);
            out.extend_from_slice(bytes);
        }
        CborValue::Text(text) => {
            let bytes = text.as_bytes();
            encode_length(3, bytes.len() as u128, out);
            out.extend_from_slice(bytes);
        }
        CborValue::Array(items) => {
            encode_length(4, items.len() as u128, out);
            for item in items {
                encode(item, out);
            }
        }
        CborValue::Map(entries) => {
            encode_length(5, entries.len() as u128, out);
            for (key, item) in entries {
                encode(&CborValue::Text(key.clone()), out);
                encode(item, out);
            }
        }
    }
}

fn encode_integer(n: i128, out: &mut Vec<u8>) {
    debug_assert!(
        integer_in_cbor_range(n),
        "integer outside canonical CBOR range"
    );
    if n >= 0 {
        encode_length(0, n as u128, out);
    } else {
        let argument = (-1 - n) as u128;
        encode_length(1, argument, out);
    }
}

fn encode_length(major: u8, length: u128, out: &mut Vec<u8>) {
    let head = major << 5;
    if length < 24 {
        out.push(head | length as u8);
    } else if length <= 0xff {
        out.push(head | 24);
        out.push(length as u8);
    } else if length <= 0xffff {
        out.push(head | 25);
        out.push((length >> 8) as u8);
        out.push(length as u8);
    } else if length <= 0xffff_ffff {
        out.push(head | 26);
        out.extend_from_slice(&(length as u32).to_be_bytes());
    } else {
        out.push(head | 27);
        out.extend_from_slice(&(length as u64).to_be_bytes());
    }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
    depth: usize,
}

impl<'a> Decoder<'a> {
    fn read(&mut self) -> Result<CborValue, InferFailure> {
        if self.depth > MAX_DEPTH {
            return Err(fail(
                ErrorCode::CanonicalCborInvalid,
                "CBOR nesting exceeds the depth limit",
            ));
        }
        let initial = self.read_byte()?;
        let major = initial >> 5;
        let ai = initial & 31;
        match major {
            0 => Ok(CborValue::Integer(self.read_argument(ai)? as i128)),
            1 => {
                let argument = self.read_argument(ai)?;
                Ok(CborValue::Integer(-1 - argument as i128))
            }
            2 => {
                let length = self.read_argument(ai)?;
                Ok(CborValue::Bytes(self.read_bytes(length)?.to_vec()))
            }
            3 => {
                let length = self.read_argument(ai)?;
                let bytes = self.read_bytes(length)?;
                let text = std::str::from_utf8(bytes)
                    .map_err(|_| fail(ErrorCode::CanonicalCborInvalid, "CBOR text is not UTF-8"))?;
                Ok(CborValue::Text(text.to_string()))
            }
            4 => {
                let length = self.collection_len(ai)?;
                self.depth += 1;
                let mut items = Vec::with_capacity(length);
                for _ in 0..length {
                    items.push(self.read()?);
                }
                self.depth -= 1;
                Ok(CborValue::Array(items))
            }
            5 => {
                let length = self.collection_len(ai)?;
                self.depth += 1;
                let mut entries = Vec::with_capacity(length);
                let mut previous: Option<String> = None;
                for _ in 0..length {
                    let key_value = self.read()?;
                    let key = match key_value {
                        CborValue::Text(text) => text,
                        _ => {
                            return Err(fail(
                                ErrorCode::CanonicalCborInvalid,
                                "CBOR map key must be text",
                            ))
                        }
                    };
                    if let Some(prev) = &previous {
                        if key.as_bytes() <= prev.as_bytes() {
                            return Err(fail(
                                ErrorCode::CanonicalCborInvalid,
                                "CBOR map keys must be strictly increasing",
                            ));
                        }
                    }
                    previous = Some(key.clone());
                    let value = self.read()?;
                    entries.push((key, value));
                }
                self.depth -= 1;
                Ok(CborValue::Map(entries))
            }
            7 => match ai {
                20 => Ok(CborValue::Bool(false)),
                21 => Ok(CborValue::Bool(true)),
                22 => Ok(CborValue::Null),
                31 => Err(fail(
                    ErrorCode::CanonicalCborInvalid,
                    "indefinite-length CBOR is rejected",
                )),
                _ => Err(fail(
                    ErrorCode::CanonicalCborInvalid,
                    "CBOR floats, tags, and extra simple values are rejected",
                )),
            },
            _ => Err(fail(
                ErrorCode::CanonicalCborInvalid,
                "CBOR tags and indefinite forms are rejected",
            )),
        }
    }

    fn collection_len(&mut self, ai: u8) -> Result<usize, InferFailure> {
        let length = self.read_argument(ai)?;
        if length > MAX_ITEMS as u128 {
            return Err(fail(
                ErrorCode::CanonicalCborInvalid,
                "CBOR collection exceeds the item limit",
            ));
        }
        let length = length as usize;
        if length > self.bytes.len().saturating_sub(self.offset) {
            return Err(fail(
                ErrorCode::CanonicalCborInvalid,
                "truncated CBOR collection",
            ));
        }
        Ok(length)
    }

    fn read_argument(&mut self, ai: u8) -> Result<u128, InferFailure> {
        let (value, minimal_at) = match ai {
            0..=23 => (ai as u128, 0u128),
            24 => (self.read_byte()? as u128, 24u128),
            25 => (self.read_be(2)?, 0x100u128),
            26 => (self.read_be(4)?, 0x1_0000u128),
            27 => (self.read_be(8)?, 0x1_0000_0000u128),
            28..=30 => {
                return Err(fail(
                    ErrorCode::CanonicalCborInvalid,
                    "reserved CBOR additional information",
                ))
            }
            31 => {
                return Err(fail(
                    ErrorCode::CanonicalCborInvalid,
                    "indefinite-length CBOR is rejected",
                ))
            }
            _ => {
                return Err(fail(
                    ErrorCode::CanonicalCborInvalid,
                    "invalid CBOR additional information",
                ))
            }
        };
        if ai >= 24 && value < minimal_at {
            return Err(fail(
                ErrorCode::CanonicalCborInvalid,
                "CBOR integer is not shortest form",
            ));
        }
        if value > I128_MAX_CBOR as u128 {
            return Err(fail(
                ErrorCode::CanonicalCborInvalid,
                "CBOR integer exceeds the supported range",
            ));
        }
        Ok(value)
    }

    fn read_be(&mut self, width: usize) -> Result<u128, InferFailure> {
        let slice = self.read_bytes(width as u128)?;
        let mut value = 0u128;
        for byte in slice {
            value = (value << 8) | *byte as u128;
        }
        Ok(value)
    }

    fn read_bytes(&mut self, length: u128) -> Result<&'a [u8], InferFailure> {
        let length = usize::try_from(length).map_err(|_| {
            fail(
                ErrorCode::CanonicalCborInvalid,
                "CBOR length exceeds memory",
            )
        })?;
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| fail(ErrorCode::CanonicalCborInvalid, "CBOR length overflows"))?;
        if end > self.bytes.len() {
            return Err(fail(
                ErrorCode::CanonicalCborInvalid,
                "truncated CBOR value",
            ));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn read_byte(&mut self) -> Result<u8, InferFailure> {
        let byte = *self
            .bytes
            .get(self.offset)
            .ok_or_else(|| fail(ErrorCode::CanonicalCborInvalid, "truncated CBOR value"))?;
        self.offset += 1;
        Ok(byte)
    }
}

pub fn integer_in_cbor_range(n: i128) -> bool {
    (I128_MIN_CBOR..=I128_MAX_CBOR).contains(&n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(value: CborValue) {
        let bytes = value.to_bytes();
        let decoded = decode_canonical(&bytes).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn primitives_roundtrip() {
        roundtrip(CborValue::Null);
        roundtrip(CborValue::Bool(false));
        roundtrip(CborValue::Bool(true));
        for n in [
            0,
            1,
            23,
            24,
            255,
            256,
            65535,
            65536,
            1i128 << 32,
            -1,
            -24,
            -25,
            -1i128 << 64,
        ] {
            roundtrip(CborValue::Integer(n));
        }
        roundtrip(CborValue::Text("knolo".into()));
        roundtrip(CborValue::Bytes(vec![0, 255]));
        roundtrip(CborValue::Array(vec![
            CborValue::Integer(1),
            CborValue::Text("a".into()),
        ]));
        let mut map = BTreeMap::new();
        map.insert("b".into(), CborValue::Integer(1));
        map.insert("a".into(), CborValue::Integer(2));
        let encoded = CborValue::map(map).to_bytes();
        assert_eq!(hex(&encoded), "a2616102616201");
        roundtrip(decode_canonical(&encoded).unwrap());
    }

    #[test]
    fn rejects_noncanonical_and_unsupported() {
        let cases = [
            "",
            "9fff",
            "f90000",
            "c0f6",
            "f6f6",
            "a2616201616102",
            "a2616101616102",
            "1800",
            "18",
            "61ff",
            "1c",
            "f7",
        ];
        for hex_value in cases {
            let bytes = decode_hex(hex_value);
            assert!(decode_canonical(&bytes).is_err(), "accepted {hex_value}");
        }
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn decode_hex(value: &str) -> Vec<u8> {
        if value.is_empty() {
            return Vec::new();
        }
        (0..value.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&value[i..i + 2], 16).unwrap())
            .collect()
    }
}
