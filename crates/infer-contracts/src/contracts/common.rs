use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    self, bounded_text, cbor_digest, cbor_text, cbor_u32, cbor_u64, expect_u32, one_of,
    sorted_unique, Fields,
};

pub const PRECISIONS: &[&str] = &[
    "bf16", "f16", "f32", "i32", "q4_k_m", "q5_k_m", "q6_k", "q8_0", "u8",
];
pub const KV_PRECISIONS: &[&str] = &["bf16", "f16", "f32"];
pub const MAX_EMBEDDED_BYTES: usize = 16 * 1024 * 1024;

pub struct Builder {
    fields: BTreeMap<String, CborValue>,
}

impl Builder {
    pub fn typed(kind: &str) -> Self {
        let mut fields = BTreeMap::new();
        fields.insert("kind".into(), cbor_text(kind));
        fields.insert("version".into(), cbor_u32(1));
        Self { fields }
    }

    pub fn bare() -> Self {
        Self {
            fields: BTreeMap::new(),
        }
    }

    pub fn put(&mut self, key: &str, value: CborValue) {
        self.fields.insert(key.to_string(), value);
    }

    pub fn put_opt(&mut self, key: &str, value: Option<CborValue>) {
        if let Some(value) = value {
            self.fields.insert(key.to_string(), value);
        }
    }

    pub fn finish(self) -> CborValue {
        CborValue::map(self.fields)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureV1 {
    pub algorithm: String,
    pub key_id: String,
    pub signature: Vec<u8>,
}

impl SignatureV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("algorithm", &self.algorithm, &["ed25519"])?;
        bounded_text("keyId", &self.key_id, 128)?;
        if self.signature.len() != 64 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "ed25519 signatures are 64 bytes",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put("algorithm", cbor_text(&self.algorithm));
        b.put("keyId", cbor_text(&self.key_id));
        b.put("signature", CborValue::Bytes(self.signature.clone()));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let signature = match fields.require("signature")? {
            CborValue::Bytes(bytes) => bytes,
            _ => return Err(fail(ErrorCode::ContractInvalid, "signature must be bytes")),
        };
        let out = Self {
            algorithm: fields.text("algorithm")?,
            key_id: fields.text("keyId")?,
            signature,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }
}

pub fn signatures_to_cbor(items: &[SignatureV1]) -> Result<CborValue, InferFailure> {
    if items.len() > 8 {
        return Err(fail(ErrorCode::ContractInvalid, "too many signatures"));
    }
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.push(item.to_cbor()?);
    }
    Ok(CborValue::Array(out))
}

pub fn signatures_from_cbor(items: &[CborValue]) -> Result<Vec<SignatureV1>, InferFailure> {
    if items.len() > 8 {
        return Err(fail(ErrorCode::ContractInvalid, "too many signatures"));
    }
    items.iter().map(SignatureV1::from_cbor).collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddedArtifactV1 {
    pub bytes: Vec<u8>,
    pub root: DigestHex,
}

impl EmbeddedArtifactV1 {
    pub fn new(domain: &str, bytes: Vec<u8>) -> Result<Self, InferFailure> {
        if bytes.is_empty() || bytes.len() > MAX_EMBEDDED_BYTES {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "embedded artifact is empty or too large",
            ));
        }
        let root = digest_value(domain, &CborValue::Bytes(bytes.clone()))?;
        Ok(Self { bytes, root })
    }

    pub fn validate(&self, domain: &str, mismatch: ErrorCode) -> Result<(), InferFailure> {
        if self.bytes.is_empty() || self.bytes.len() > MAX_EMBEDDED_BYTES {
            return Err(fail(mismatch, "embedded artifact is empty or too large"));
        }
        let root = digest_value(domain, &CborValue::Bytes(self.bytes.clone()))?;
        if root != self.root {
            return Err(fail(
                mismatch,
                "embedded artifact root does not match bytes",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> CborValue {
        let mut b = Builder::bare();
        b.put("bytes", CborValue::Bytes(self.bytes.clone()));
        b.put("root", cbor_digest(&self.root));
        b.finish()
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let bytes = match fields.require("bytes")? {
            CborValue::Bytes(bytes) => bytes,
            _ => return Err(fail(ErrorCode::ContractInvalid, "embedded bytes required")),
        };
        let out = Self {
            bytes,
            root: fields.digest("root")?,
        };
        fields.finish()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactFileV1 {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: DigestHex,
}

impl ArtifactFileV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        validate_relative_path(&self.path)?;
        if self.size_bytes == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "artifact file size must be non-zero",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put("path", cbor_text(&self.path));
        b.put("sha256", cbor_digest(&self.sha256));
        b.put("sizeBytes", cbor_u64(self.size_bytes));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            path: fields.text("path")?,
            sha256: fields.digest("sha256")?,
            size_bytes: fields.u64("sizeBytes")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }
}

pub fn artifact_files_cbor(files: &[ArtifactFileV1]) -> Result<CborValue, InferFailure> {
    validate_files(files)?;
    let mut out = Vec::with_capacity(files.len());
    for file in files {
        out.push(file.to_cbor()?);
    }
    Ok(CborValue::Array(out))
}

pub fn artifact_root(files: &[ArtifactFileV1]) -> Result<DigestHex, InferFailure> {
    let mut b = Builder::bare();
    b.put("files", artifact_files_cbor(files)?);
    digest_value("infer-model-artifact", &b.finish())
}

pub fn validate_files(files: &[ArtifactFileV1]) -> Result<(), InferFailure> {
    if files.is_empty() || files.len() > 1024 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "artifact file list is empty or too large",
        ));
    }
    let paths: Vec<_> = files.iter().map(|file| file.path.clone()).collect();
    sorted_unique("files", &paths)?;
    for file in files {
        file.validate()?;
    }
    Ok(())
}

pub fn validate_relative_path(path: &str) -> Result<(), InferFailure> {
    if path.is_empty()
        || path.len() > 512
        || path.starts_with('/')
        || path.ends_with('/')
        || path.contains('\\')
        || path.contains("//")
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "artifact path is not relative POSIX",
        ));
    }
    for segment in path.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "artifact path is not relative POSIX",
            ));
        }
        if !segment
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "artifact path is not relative POSIX",
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialTokensV1 {
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub pad: Option<u32>,
    pub unk: Option<u32>,
    pub additional: BTreeMap<String, u32>,
}

impl SpecialTokensV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        if self.additional.len() > 256 {
            return Err(fail(ErrorCode::ContractInvalid, "too many special tokens"));
        }
        for name in self.additional.keys() {
            bounded_text("specialTokens", name, 64)?;
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        let mut extra = BTreeMap::new();
        for (name, id) in &self.additional {
            extra.insert(name.clone(), cbor_u32(*id));
        }
        b.put("additional", CborValue::map(extra));
        b.put_opt("bos", self.bos.map(cbor_u32));
        b.put_opt("eos", self.eos.map(cbor_u32));
        b.put_opt("pad", self.pad.map(cbor_u32));
        b.put_opt("unk", self.unk.map(cbor_u32));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let additional_value = fields.require("additional")?;
        let mut additional_fields = Fields::parse(&additional_value)?;
        let mut additional = BTreeMap::new();
        let names = additional_fields.key_names();
        for name in names {
            bounded_text("specialTokens", &name, 64)?;
            additional.insert(
                name.clone(),
                expect_u32("specialTokens", &additional_fields.require(&name)?)?,
            );
        }
        additional_fields.finish()?;
        let out = Self {
            additional,
            bos: optional_u32(&mut fields, "bos")?,
            eos: optional_u32(&mut fields, "eos")?,
            pad: optional_u32(&mut fields, "pad")?,
            unk: optional_u32(&mut fields, "unk")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }
}

fn optional_u32(fields: &mut Fields, key: &str) -> Result<Option<u32>, InferFailure> {
    match fields.optional(key)? {
        None => Ok(None),
        Some(value) => Ok(Some(expect_u32(key, &value)?)),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedPointSamplerV1 {
    pub temperature_micros: u32,
    pub top_p_millionths: u32,
    pub min_p_millionths: u32,
    pub repetition_penalty_micros: u32,
    pub presence_penalty_micros: u32,
    pub frequency_penalty_micros: u32,
    pub top_k: u32,
    pub max_output_tokens: u32,
}

impl FixedPointSamplerV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        if self.temperature_micros > 10_000_000
            || self.top_p_millionths > 1_000_000
            || self.min_p_millionths > 1_000_000
            || self.repetition_penalty_micros > 10_000_000
            || self.presence_penalty_micros > 10_000_000
            || self.frequency_penalty_micros > 10_000_000
            || self.top_k > 1_048_576
            || self.max_output_tokens == 0
            || self.max_output_tokens > 1_048_576
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "sampler fixed-point field is outside its bounds",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put(
            "frequencyPenaltyMicros",
            cbor_u32(self.frequency_penalty_micros),
        );
        b.put("maxOutputTokens", cbor_u32(self.max_output_tokens));
        b.put("minPMillionths", cbor_u32(self.min_p_millionths));
        b.put(
            "presencePenaltyMicros",
            cbor_u32(self.presence_penalty_micros),
        );
        b.put(
            "repetitionPenaltyMicros",
            cbor_u32(self.repetition_penalty_micros),
        );
        b.put("temperatureMicros", cbor_u32(self.temperature_micros));
        b.put("topK", cbor_u32(self.top_k));
        b.put("topPMillionths", cbor_u32(self.top_p_millionths));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            frequency_penalty_micros: fields.u32("frequencyPenaltyMicros")?,
            max_output_tokens: fields.u32("maxOutputTokens")?,
            min_p_millionths: fields.u32("minPMillionths")?,
            presence_penalty_micros: fields.u32("presencePenaltyMicros")?,
            repetition_penalty_micros: fields.u32("repetitionPenaltyMicros")?,
            temperature_micros: fields.u32("temperatureMicros")?,
            top_k: fields.u32("topK")?,
            top_p_millionths: fields.u32("topPMillionths")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }
}

pub fn require_sorted_precisions(values: &[String], allowed: &[&str]) -> Result<(), InferFailure> {
    sorted_unique("precisions", values)?;
    for value in values {
        one_of("precisions", value, allowed)?;
    }
    Ok(())
}

pub fn device_id(value: &str) -> Result<(), InferFailure> {
    if value == "cpu" {
        return Ok(());
    }
    let Some(rest) = value.strip_prefix("slot-") else {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "device id must be cpu or slot-N",
        ));
    };
    if rest.is_empty()
        || rest.len() > 3
        || !rest.bytes().all(|b| b.is_ascii_digit())
        || rest.starts_with('0') && rest.len() > 1
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "device id must be cpu or slot-N",
        ));
    }
    Ok(())
}

pub fn u64_array_sum(values: &[u64]) -> Result<u64, InferFailure> {
    let mut total = 0u64;
    for value in values {
        total = total
            .checked_add(*value)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "byte total overflows u64"))?;
    }
    Ok(total)
}

pub fn put_extensions(
    b: &mut Builder,
    extensions: &BTreeMap<String, CborValue>,
) -> Result<(), InferFailure> {
    let mut map = BTreeMap::new();
    fields::insert_extensions(&mut map, extensions)?;
    if let Some(value) = map.remove("extensions") {
        b.put("extensions", value);
    }
    Ok(())
}

pub fn digest_field_array(items: &[DigestHex]) -> CborValue {
    CborValue::Array(items.iter().map(cbor_digest).collect())
}

pub fn string_field_array(items: &[String]) -> CborValue {
    CborValue::Array(items.iter().cloned().map(CborValue::Text).collect())
}
