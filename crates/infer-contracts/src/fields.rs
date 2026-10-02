use std::collections::{BTreeMap, BTreeSet};

use crate::cbor::CborValue;
use crate::digest::DigestHex;
use crate::error::{fail, ErrorCode, InferFailure};

pub struct Fields {
    map: BTreeMap<String, CborValue>,
    used: BTreeSet<String>,
}

impl Fields {
    pub fn parse(value: &CborValue) -> Result<Self, InferFailure> {
        let mut map = BTreeMap::new();
        for (key, item) in value.as_map()? {
            if map.insert(key.clone(), item.clone()).is_some() {
                return Err(fail(
                    ErrorCode::CanonicalCborInvalid,
                    "duplicate CBOR map key",
                ));
            }
        }
        Ok(Self {
            map,
            used: BTreeSet::new(),
        })
    }

    pub fn peek_text(&self, key: &str) -> Result<String, InferFailure> {
        match self.map.get(key) {
            Some(CborValue::Text(value)) => Ok(value.clone()),
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

    pub fn key_names(&self) -> Vec<String> {
        self.map.keys().cloned().collect()
    }

    pub fn finish(self) -> Result<(), InferFailure> {
        let unknown: Vec<_> = self
            .map
            .keys()
            .filter(|key| !self.used.contains(*key))
            .cloned()
            .collect();
        if unknown.is_empty() {
            Ok(())
        } else {
            Err(fail(
                ErrorCode::ContractInvalid,
                format!(
                    "unknown field: {}",
                    unknown[0].chars().take(80).collect::<String>()
                ),
            ))
        }
    }

    fn take(&mut self, key: &str) -> Option<CborValue> {
        if self.map.contains_key(key) {
            self.used.insert(key.to_string());
            self.map.get(key).cloned()
        } else {
            None
        }
    }

    pub fn require(&mut self, key: &str) -> Result<CborValue, InferFailure> {
        self.take(key)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, format!("missing field: {key}")))
    }

    pub fn optional(&mut self, key: &str) -> Result<Option<CborValue>, InferFailure> {
        Ok(self.take(key))
    }

    pub fn text(&mut self, key: &str) -> Result<String, InferFailure> {
        match self.require(key)? {
            CborValue::Text(value) => Ok(value),
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                format!("field {key} must be text"),
            )),
        }
    }

    pub fn opt_text(&mut self, key: &str) -> Result<Option<String>, InferFailure> {
        match self.optional(key)? {
            None => Ok(None),
            Some(CborValue::Text(value)) => Ok(Some(value)),
            Some(_) => Err(fail(
                ErrorCode::ContractInvalid,
                format!("field {key} must be text"),
            )),
        }
    }

    pub fn bool(&mut self, key: &str) -> Result<bool, InferFailure> {
        match self.require(key)? {
            CborValue::Bool(value) => Ok(value),
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                format!("field {key} must be a boolean"),
            )),
        }
    }

    pub fn u32(&mut self, key: &str) -> Result<u32, InferFailure> {
        let value = self.require(key)?;
        expect_u32(key, &value)
    }

    pub fn u64(&mut self, key: &str) -> Result<u64, InferFailure> {
        let value = self.require(key)?;
        expect_u64(key, &value)
    }

    pub fn i64(&mut self, key: &str) -> Result<i64, InferFailure> {
        let value = self.require(key)?;
        expect_i64(key, &value)
    }

    pub fn opt_u64(&mut self, key: &str) -> Result<Option<u64>, InferFailure> {
        match self.optional(key)? {
            None => Ok(None),
            Some(value) => Ok(Some(expect_u64(key, &value)?)),
        }
    }

    pub fn digest(&mut self, key: &str) -> Result<DigestHex, InferFailure> {
        DigestHex::parse(&self.text(key)?)
    }

    pub fn opt_digest(&mut self, key: &str) -> Result<Option<DigestHex>, InferFailure> {
        match self.opt_text(key)? {
            None => Ok(None),
            Some(value) => Ok(Some(DigestHex::parse(&value)?)),
        }
    }

    pub fn array(&mut self, key: &str) -> Result<Vec<CborValue>, InferFailure> {
        match self.require(key)? {
            CborValue::Array(items) => Ok(items),
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                format!("field {key} must be an array"),
            )),
        }
    }

    pub fn nested(&mut self, key: &str) -> Result<Fields, InferFailure> {
        let value = self.require(key)?;
        Fields::parse(&value)
    }

    pub fn extensions(&mut self) -> Result<BTreeMap<String, CborValue>, InferFailure> {
        let Some(value) = self.optional("extensions")? else {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "missing field: extensions",
            ));
        };
        let mut fields = Fields::parse(&value)?;
        let mut out = BTreeMap::new();
        let keys: Vec<_> = fields.map.keys().cloned().collect();
        if keys.len() > 32 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "extensions exceed 32 entries",
            ));
        }
        for key in keys {
            if !valid_extension_key(&key) {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "extension keys must be dotted lowercase namespaces",
                ));
            }
            out.insert(key.clone(), fields.require(&key)?);
        }
        fields.finish()?;
        Ok(out)
    }
}

pub fn expect_kind_version(fields: &mut Fields, kind: &str) -> Result<(), InferFailure> {
    let actual = fields.text("kind")?;
    if actual != kind {
        return Err(fail(ErrorCode::ContractInvalid, "unexpected contract kind"));
    }
    let version = fields.u32("version")?;
    if version != 1 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("unsupported contract version: {version}"),
        ));
    }
    Ok(())
}

pub fn expect_u32(field: &str, value: &CborValue) -> Result<u32, InferFailure> {
    let n = expect_i128(field, value)?;
    u32::try_from(n).map_err(|_| {
        fail(
            ErrorCode::ContractInvalid,
            format!("field {field} is outside u32"),
        )
    })
}

pub fn expect_u64(field: &str, value: &CborValue) -> Result<u64, InferFailure> {
    let n = expect_i128(field, value)?;
    u64::try_from(n).map_err(|_| {
        fail(
            ErrorCode::ContractInvalid,
            format!("field {field} is outside u64"),
        )
    })
}

pub fn expect_i64(field: &str, value: &CborValue) -> Result<i64, InferFailure> {
    let n = expect_i128(field, value)?;
    i64::try_from(n).map_err(|_| {
        fail(
            ErrorCode::ContractInvalid,
            format!("field {field} is outside i64"),
        )
    })
}

fn expect_i128(field: &str, value: &CborValue) -> Result<i128, InferFailure> {
    match value {
        CborValue::Integer(n) => Ok(*n),
        _ => Err(fail(
            ErrorCode::ContractInvalid,
            format!("field {field} must be an integer"),
        )),
    }
}

pub fn bounded_text(field: &str, value: &str, max: usize) -> Result<(), InferFailure> {
    if value.is_empty() || value.len() > max || value.chars().any(|c| c.is_control()) {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("field {field} is empty or outside its bounds"),
        ));
    }
    Ok(())
}

pub fn message_text(field: &str, value: &str, max: usize) -> Result<(), InferFailure> {
    if value.is_empty()
        || value.len() > max
        || value
            .chars()
            .any(|c| c != '\n' && c != '\t' && c.is_control())
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            format!("field {field} is empty or outside its bounds"),
        ));
    }
    Ok(())
}

pub fn sorted_unique(field: &str, values: &[String]) -> Result<(), InferFailure> {
    for window in values.windows(2) {
        if window[0].as_bytes() >= window[1].as_bytes() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                format!("field {field} must be strictly sorted and unique"),
            ));
        }
    }
    Ok(())
}

pub fn one_of(field: &str, value: &str, allowed: &[&str]) -> Result<(), InferFailure> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(fail(
            ErrorCode::ContractInvalid,
            format!("field {field} has an unsupported value"),
        ))
    }
}

pub fn valid_extension_key(key: &str) -> bool {
    if key.len() > 64 || !key.contains('.') {
        return false;
    }
    let mut parts = key.split('.');
    let Some(first) = parts.next() else {
        return false;
    };
    if !is_ident(first, false) {
        return false;
    }
    let mut rest = 0;
    for part in parts {
        if !is_ident(part, true) {
            return false;
        }
        rest += 1;
    }
    rest >= 1
}

fn is_ident(part: &str, allow_hyphen: bool) -> bool {
    let mut chars = part.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_lowercase() {
        return false;
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || (allow_hyphen && c == '-'))
}

pub fn text_array(items: &[CborValue], field: &str) -> Result<Vec<String>, InferFailure> {
    items
        .iter()
        .map(|item| match item {
            CborValue::Text(value) => Ok(value.clone()),
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                format!("field {field} must contain text"),
            )),
        })
        .collect()
}

pub fn u32_array(items: &[CborValue], field: &str) -> Result<Vec<u32>, InferFailure> {
    items.iter().map(|item| expect_u32(field, item)).collect()
}

pub fn digest_array(items: &[CborValue], field: &str) -> Result<Vec<DigestHex>, InferFailure> {
    items
        .iter()
        .map(|item| match item {
            CborValue::Text(value) => DigestHex::parse(value),
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                format!("field {field} must contain digests"),
            )),
        })
        .collect()
}

pub fn insert_extensions(
    fields: &mut BTreeMap<String, CborValue>,
    extensions: &BTreeMap<String, CborValue>,
) -> Result<(), InferFailure> {
    if extensions.len() > 32 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "extensions exceed 32 entries",
        ));
    }
    for key in extensions.keys() {
        if !valid_extension_key(key) {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "extension keys must be dotted lowercase namespaces",
            ));
        }
    }
    fields.insert("extensions".into(), CborValue::map(extensions.clone()));
    Ok(())
}

pub fn cbor_text(value: impl Into<String>) -> CborValue {
    CborValue::Text(value.into())
}

pub fn cbor_u32(value: u32) -> CborValue {
    CborValue::Integer(i128::from(value))
}

pub fn cbor_u64(value: u64) -> CborValue {
    CborValue::Integer(i128::from(value))
}

pub fn cbor_i64(value: i64) -> CborValue {
    CborValue::Integer(i128::from(value))
}

pub fn cbor_digest(value: &DigestHex) -> CborValue {
    CborValue::Text(value.as_str().to_string())
}
