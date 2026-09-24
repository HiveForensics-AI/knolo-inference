//! Bounded GGUF reader.
//!
//! The bytes are untrusted until the caller has checked the file digest.
//! `read_verified_gguf` does that check before it parses. Tensor bodies are
//! copied out. Chat-template text is not executed. This module does not write
//! a converted artifact.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure};

use crate::io::prefixed;

pub const GGUF_VERSION: u32 = 3;
pub const GGUF_DEFAULT_ALIGNMENT: u32 = 32;
pub const GGUF_QUANT_VERSION: u32 = 2;
pub const MAX_GGUF_BYTES: u64 = 32 * 1024 * 1024;
const MAX_GGUF_METADATA: usize = 4096;
const MAX_GGUF_ARRAY: usize = 1_048_576;
const MAX_GGUF_STRING: usize = 1024 * 1024;
const MAX_GGUF_TENSORS: usize = 100_000;
const MAX_GGUF_TENSOR_NAME: usize = 64;
const MAX_GGUF_DIMENSIONS: usize = 4;
const MAX_GGUF_ARRAY_DEPTH: usize = 4;
const MAX_KEY_BYTES: usize = 65535;

const ARCHITECTURE_KEY: &str = "general.architecture";
const ALIGNMENT_KEY: &str = "general.alignment";
const QUANT_VERSION_KEY: &str = "general.quantization_version";

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    fn from_u32(value: u32) -> Result<Self, InferFailure> {
        Ok(match value {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => return Err(image("gguf metadata type is invalid")),
        })
    }

    fn min_bytes(self) -> u64 {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => 1,
            Self::Uint16 | Self::Int16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Uint64 | Self::Int64 | Self::Float64 | Self::String => 8,
            Self::Array => 12,
        }
    }
}

/// Metadata value. Floats are stored as IEEE bits so equality does not depend
/// on NaN payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(u32),
    Bool(bool),
    String(String),
    Array(GgufArray),
    Uint64(u64),
    Int64(i64),
    Float64(u64),
}

impl GgufValue {
    fn value_type(&self) -> GgufValueType {
        match self {
            Self::Uint8(_) => GgufValueType::Uint8,
            Self::Int8(_) => GgufValueType::Int8,
            Self::Uint16(_) => GgufValueType::Uint16,
            Self::Int16(_) => GgufValueType::Int16,
            Self::Uint32(_) => GgufValueType::Uint32,
            Self::Int32(_) => GgufValueType::Int32,
            Self::Float32(_) => GgufValueType::Float32,
            Self::Bool(_) => GgufValueType::Bool,
            Self::String(_) => GgufValueType::String,
            Self::Array(_) => GgufValueType::Array,
            Self::Uint64(_) => GgufValueType::Uint64,
            Self::Int64(_) => GgufValueType::Int64,
            Self::Float64(_) => GgufValueType::Float64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufArray {
    pub element: GgufValueType,
    pub items: Vec<GgufValue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufMetadata {
    pub key: String,
    pub value: GgufValue,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    /// ggml type `Q4_K` (12). The format name keeps the underscore.
    #[allow(non_camel_case_types)]
    Q4_K = 12,
    /// ggml type `Q5_K` (13). The format name keeps the underscore.
    #[allow(non_camel_case_types)]
    Q5_K = 13,
    /// ggml type `Q6_K` (14). The format name keeps the underscore.
    #[allow(non_camel_case_types)]
    Q6_K = 14,
}

impl GgufTensorType {
    pub fn from_u32(value: u32) -> Result<Self, InferFailure> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            8 => Ok(Self::Q8_0),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            _ => Err(unsupported(format!(
                "gguf tensor type {value} is not in the allowlist"
            ))),
        }
    }

    pub fn type_size(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q8_0 => 34,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
        }
    }

    pub fn block_elements(self) -> u64 {
        match self {
            Self::Q8_0 => 32,
            Self::Q4_K | Self::Q5_K | Self::Q6_K => 256,
            Self::F32 | Self::F16 => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q8_0 => "Q8_0",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
        }
    }

    pub fn requires_quant_version(self) -> bool {
        matches!(self, Self::Q8_0 | Self::Q4_K | Self::Q5_K | Self::Q6_K)
    }

    pub fn elements(self, shape: &[u64]) -> Result<u64, InferFailure> {
        check_shape(shape)?;
        let mut count = 1u64;
        for dim in shape {
            count = count
                .checked_mul(*dim)
                .ok_or_else(|| image("gguf tensor byte length overflows"))?;
        }
        Ok(count)
    }

    pub fn nbytes(self, shape: &[u64]) -> Result<u64, InferFailure> {
        check_shape(shape)?;
        let block = self.block_elements();
        if shape[0] % block != 0 {
            return Err(image(
                "gguf tensor dimension is not a multiple of the block",
            ));
        }
        let mut nbytes = shape[0] / block;
        nbytes = nbytes
            .checked_mul(self.type_size())
            .ok_or_else(|| image("gguf tensor byte length overflows"))?;
        for dim in &shape[1..] {
            nbytes = nbytes
                .checked_mul(*dim)
                .ok_or_else(|| image("gguf tensor byte length overflows"))?;
        }
        Ok(nbytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensor {
    pub name: String,
    pub tensor_type: GgufTensorType,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorDraft {
    pub name: String,
    pub tensor_type: GgufTensorType,
    pub shape: Vec<u64>,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufFile {
    pub version: u32,
    pub alignment: u32,
    pub architecture: String,
    pub quantization_version: Option<u32>,
    pub metadata: Vec<GgufMetadata>,
    pub tensors: Vec<GgufTensor>,
    pub tensor_info_end: u64,
    pub data_start: u64,
}

struct HeaderFacts {
    architecture: String,
    alignment: u32,
    quantization_version: Option<u32>,
}

struct TensorInfo {
    name: String,
    tensor_type: GgufTensorType,
    shape: Vec<u64>,
    offset: u64,
    nbytes: u64,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], InferFailure> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| image("gguf file is truncated"))?;
        if end > self.bytes.len() {
            return Err(image("gguf file is truncated"));
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, InferFailure> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, InferFailure> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, InferFailure> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Result<u64, InferFailure> {
        let bytes = self.take(8)?;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(buf))
    }

    fn i16(&mut self) -> Result<i16, InferFailure> {
        Ok(i16::from_le_bytes(self.u16()?.to_le_bytes()))
    }

    fn i32(&mut self) -> Result<i32, InferFailure> {
        Ok(i32::from_le_bytes(self.u32()?.to_le_bytes()))
    }

    fn i64(&mut self) -> Result<i64, InferFailure> {
        Ok(i64::from_le_bytes(self.u64()?.to_le_bytes()))
    }

    fn string(&mut self, limit: usize) -> Result<String, InferFailure> {
        let len = self.u64()?;
        if len > limit as u64 {
            return Err(image("gguf string exceeds its limit"));
        }
        let len = usize::try_from(len).map_err(|_| image("gguf string exceeds its limit"))?;
        if len > self.remaining() {
            return Err(image("gguf file is truncated"));
        }
        let bytes = self.take(len)?;
        let text = std::str::from_utf8(bytes).map_err(|_| image("gguf string is not utf-8"))?;
        if text.bytes().any(|byte| byte == 0) {
            return Err(image("gguf string contains a nul"));
        }
        Ok(text.to_string())
    }

    fn value(&mut self, ty: GgufValueType, depth: usize) -> Result<GgufValue, InferFailure> {
        Ok(match ty {
            GgufValueType::Uint8 => GgufValue::Uint8(self.u8()?),
            GgufValueType::Int8 => GgufValue::Int8(i8::from_le_bytes([self.u8()?])),
            GgufValueType::Uint16 => GgufValue::Uint16(self.u16()?),
            GgufValueType::Int16 => GgufValue::Int16(self.i16()?),
            GgufValueType::Uint32 => GgufValue::Uint32(self.u32()?),
            GgufValueType::Int32 => GgufValue::Int32(self.i32()?),
            GgufValueType::Float32 => GgufValue::Float32(self.u32()?),
            GgufValueType::Bool => match self.u8()? {
                0 => GgufValue::Bool(false),
                1 => GgufValue::Bool(true),
                _ => return Err(image("gguf boolean is invalid")),
            },
            GgufValueType::String => GgufValue::String(self.string(MAX_GGUF_STRING)?),
            GgufValueType::Array => self.array(depth)?,
            GgufValueType::Uint64 => GgufValue::Uint64(self.u64()?),
            GgufValueType::Int64 => GgufValue::Int64(self.i64()?),
            GgufValueType::Float64 => GgufValue::Float64(self.u64()?),
        })
    }

    fn array(&mut self, depth: usize) -> Result<GgufValue, InferFailure> {
        if depth >= MAX_GGUF_ARRAY_DEPTH {
            return Err(image("gguf array is too deep"));
        }
        let element = GgufValueType::from_u32(self.u32()?)?;
        let len = self.u64()?;
        if len > MAX_GGUF_ARRAY as u64 {
            return Err(image("gguf array exceeds its limit"));
        }
        let remaining = self.remaining() as u64;
        if len > remaining / element.min_bytes() {
            return Err(image("gguf file is truncated"));
        }
        let count = usize::try_from(len).map_err(|_| image("gguf array exceeds its limit"))?;
        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            items.push(self.value(element, depth + 1)?);
        }
        Ok(GgufValue::Array(GgufArray { element, items }))
    }
}

/// Parse a buffer the caller has already limited and hashed.
pub fn parse_gguf_bytes(bytes: &[u8]) -> Result<GgufFile, InferFailure> {
    if bytes.len() as u64 > MAX_GGUF_BYTES {
        return Err(image("gguf file exceeds the 32 MiB in-memory read limit"));
    }
    let mut cursor = Cursor { bytes, pos: 0 };
    let magic = cursor.take(4)?;
    if magic != b"GGUF" {
        return Err(image("gguf magic is invalid"));
    }
    let version = cursor.u32()?;
    if version != GGUF_VERSION {
        return Err(image("gguf version is not 3"));
    }
    let tensor_count = cursor.u64()?;
    let kv_count = cursor.u64()?;
    if kv_count > MAX_GGUF_METADATA as u64 {
        return Err(image("gguf metadata count exceeds its limit"));
    }
    if tensor_count > MAX_GGUF_TENSORS as u64 {
        return Err(image("gguf tensor count exceeds its limit"));
    }
    if kv_count > cursor.remaining() as u64 || tensor_count > cursor.remaining() as u64 {
        return Err(image("gguf file is truncated"));
    }
    let mut metadata = Vec::with_capacity(usize::try_from(kv_count).unwrap_or(0));
    let mut seen_keys = BTreeSet::new();
    for _ in 0..kv_count {
        let key = cursor.string(MAX_KEY_BYTES)?;
        validate_key(&key)?;
        if !seen_keys.insert(key.clone()) {
            return Err(image("duplicate gguf metadata key"));
        }
        let value_type = GgufValueType::from_u32(cursor.u32()?)?;
        let value = cursor.value(value_type, 0)?;
        metadata.push(GgufMetadata { key, value });
    }
    validate_metadata(&metadata)?;

    let mut infos = Vec::with_capacity(usize::try_from(tensor_count).unwrap_or(0));
    let mut seen_names = BTreeSet::new();
    for _ in 0..tensor_count {
        let name = cursor.string(MAX_GGUF_TENSOR_NAME)?;
        validate_tensor_name(&name)?;
        if !seen_names.insert(name.clone()) {
            return Err(image("duplicate gguf tensor"));
        }
        let n_dims = cursor.u32()?;
        if n_dims == 0 || n_dims > MAX_GGUF_DIMENSIONS as u32 {
            return Err(image("gguf tensor dimension count is outside 1..=4"));
        }
        let n_dims = n_dims as usize;
        if n_dims > cursor.remaining() / 8 {
            return Err(image("gguf file is truncated"));
        }
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(cursor.u64()?);
        }
        let tensor_type = GgufTensorType::from_u32(cursor.u32()?)?;
        let offset = cursor.u64()?;
        let nbytes = tensor_type.nbytes(&shape)?;
        infos.push(TensorInfo {
            name,
            tensor_type,
            shape,
            offset,
            nbytes,
        });
    }
    let facts = interpret(&metadata, infos.iter().map(|info| info.tensor_type))?;
    let tensor_info_end = cursor.pos as u64;
    let data_start = align_up(tensor_info_end, u64::from(facts.alignment))?;
    if data_start > bytes.len() as u64 {
        return Err(image("gguf data section is outside the file"));
    }
    let data_at = usize::try_from(data_start).unwrap_or(bytes.len());
    if bytes[cursor.pos..data_at].iter().any(|byte| *byte != 0) {
        return Err(image("gguf alignment padding is not zero"));
    }
    let data = &bytes[data_at..];
    let mut ranges = Vec::with_capacity(infos.len());
    let mut tensors = Vec::with_capacity(infos.len());
    for info in infos {
        if info.offset % u64::from(facts.alignment) != 0 {
            return Err(image("gguf tensor offset is not aligned"));
        }
        let end = info
            .offset
            .checked_add(info.nbytes)
            .ok_or_else(|| image("gguf tensor byte length overflows"))?;
        if end > data.len() as u64 {
            return Err(image("gguf tensor extends outside the file"));
        }
        let start = usize::try_from(info.offset).unwrap_or(data.len());
        let stop = usize::try_from(end).unwrap_or(data.len());
        ranges.push((info.offset, end));
        tensors.push(GgufTensor {
            name: info.name,
            tensor_type: info.tensor_type,
            shape: info.shape,
            offset: info.offset,
            bytes: data[start..stop].to_vec(),
        });
    }
    check_data_padding(data, &ranges)?;
    Ok(GgufFile {
        version,
        alignment: facts.alignment,
        architecture: facts.architecture,
        quantization_version: facts.quantization_version,
        metadata,
        tensors,
        tensor_info_end,
        data_start,
    })
}

/// Hash the file, compare it, and only then parse.
pub fn read_verified_gguf(
    path: &Path,
    expected_size: u64,
    expected_digest: &DigestHex,
    display: &str,
) -> Result<GgufFile, InferFailure> {
    let meta = fs::symlink_metadata(path).map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            fail(
                ErrorCode::ModelArtifactMissing,
                format!("missing file {display}"),
            )
        } else {
            fail(
                ErrorCode::ModelDigestMismatch,
                format!("artifact size does not match {display}: {err}"),
            )
        }
    })?;
    if !meta.file_type().is_file() || meta.len() != expected_size {
        return Err(fail(
            ErrorCode::ModelDigestMismatch,
            format!("artifact size does not match {display}"),
        ));
    }
    if expected_size > MAX_GGUF_BYTES {
        return Err(image("gguf file exceeds the 32 MiB in-memory read limit"));
    }
    let bytes = fs::read(path).map_err(|err| {
        fail(
            ErrorCode::ModelDigestMismatch,
            format!("artifact read failed for {display}: {err}"),
        )
    })?;
    if bytes.len() as u64 != expected_size {
        return Err(fail(
            ErrorCode::ModelDigestMismatch,
            format!("artifact size does not match {display}"),
        ));
    }
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&hasher.finalize());
    if prefixed(digest) != *expected_digest {
        return Err(fail(
            ErrorCode::ModelDigestMismatch,
            format!("artifact digest does not match {display}"),
        ));
    }
    parse_gguf_bytes(&bytes)
}

/// Write a GGUF container around payloads the caller already quantized.
///
/// The payload bytes are copied. Their tensor type is not changed.
pub fn encode_gguf(
    metadata: &[GgufMetadata],
    tensors: &[GgufTensorDraft],
) -> Result<Vec<u8>, InferFailure> {
    if tensors.len() > MAX_GGUF_TENSORS {
        return Err(image("gguf tensor count exceeds its limit"));
    }
    validate_metadata(metadata)?;
    let mut names = BTreeSet::new();
    for tensor in tensors {
        validate_tensor_name(&tensor.name)?;
        if !names.insert(tensor.name.as_str()) {
            return Err(image("duplicate gguf tensor"));
        }
        let nbytes = tensor.tensor_type.nbytes(&tensor.shape)?;
        if tensor.bytes.len() as u64 != nbytes {
            return Err(image("gguf tensor payload length does not match its shape"));
        }
    }
    let facts = interpret(metadata, tensors.iter().map(|tensor| tensor.tensor_type))?;
    let (offsets, data) = layout_data(tensors, facts.alignment);
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    push_u32(&mut out, GGUF_VERSION);
    push_u64(&mut out, tensors.len() as u64);
    push_u64(&mut out, metadata.len() as u64);
    for item in metadata {
        push_str(&mut out, &item.key);
        push_u32(&mut out, item.value.value_type() as u32);
        write_value(&mut out, &item.value);
    }
    for (tensor, offset) in tensors.iter().zip(offsets) {
        push_str(&mut out, &tensor.name);
        push_u32(&mut out, tensor.shape.len() as u32);
        for dim in &tensor.shape {
            push_u64(&mut out, *dim);
        }
        push_u32(&mut out, tensor.tensor_type as u32);
        push_u64(&mut out, offset);
    }
    let data_start = align_up(out.len() as u64, u64::from(facts.alignment))?;
    let data_at = usize::try_from(data_start).unwrap_or(usize::MAX);
    if data_at < out.len() {
        return Err(image("gguf data section is outside the file"));
    }
    out.resize(data_at, 0);
    out.extend_from_slice(&data);
    if out.len() as u64 > MAX_GGUF_BYTES {
        return Err(image("gguf file exceeds the 32 MiB in-memory read limit"));
    }
    Ok(out)
}

fn layout_data(tensors: &[GgufTensorDraft], alignment: u32) -> (Vec<u64>, Vec<u8>) {
    let alignment = alignment as usize;
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let rem = data.len() % alignment;
        if rem != 0 {
            data.resize(data.len() + (alignment - rem), 0);
        }
        offsets.push(data.len() as u64);
        data.extend_from_slice(&tensor.bytes);
    }
    (offsets, data)
}

fn interpret(
    metadata: &[GgufMetadata],
    tensor_types: impl IntoIterator<Item = GgufTensorType>,
) -> Result<HeaderFacts, InferFailure> {
    let mut architecture = None;
    let mut alignment = None;
    let mut quantization_version = None;
    for item in metadata {
        match item.key.as_str() {
            ARCHITECTURE_KEY => {
                let GgufValue::String(value) = &item.value else {
                    return Err(image("general.architecture must be a string"));
                };
                if !valid_architecture(value) {
                    return Err(image("general.architecture is invalid"));
                }
                architecture = Some(value.clone());
            }
            ALIGNMENT_KEY => match &item.value {
                GgufValue::Uint32(value) if valid_alignment(*value) => {
                    alignment = Some(*value);
                }
                GgufValue::Uint32(_) => {
                    return Err(image(
                        "general.alignment must be a multiple of 8 from 8 to 4096",
                    ));
                }
                _ => return Err(image("general.alignment must be a uint32")),
            },
            QUANT_VERSION_KEY => match &item.value {
                GgufValue::Uint32(value) => quantization_version = Some(*value),
                _ => return Err(image("general.quantization_version must be a uint32")),
            },
            _ => {}
        }
    }
    let architecture =
        architecture.ok_or_else(|| image("gguf metadata is missing general.architecture"))?;
    let quantized = tensor_types
        .into_iter()
        .any(|tensor_type| tensor_type.requires_quant_version());
    if quantized {
        match quantization_version {
            Some(GGUF_QUANT_VERSION) => {}
            Some(_) => {
                return Err(unsupported("gguf quantization version is not 2"));
            }
            None => {
                return Err(image(
                    "quantized gguf tensor requires general.quantization_version",
                ));
            }
        }
    }
    Ok(HeaderFacts {
        architecture,
        alignment: alignment.unwrap_or(GGUF_DEFAULT_ALIGNMENT),
        quantization_version,
    })
}

fn validate_metadata(items: &[GgufMetadata]) -> Result<(), InferFailure> {
    if items.len() > MAX_GGUF_METADATA {
        return Err(image("gguf metadata count exceeds its limit"));
    }
    let mut seen = BTreeSet::new();
    for item in items {
        validate_key(&item.key)?;
        if !seen.insert(item.key.as_str()) {
            return Err(image("duplicate gguf metadata key"));
        }
        validate_value(&item.value, 0)?;
    }
    Ok(())
}

fn validate_key(key: &str) -> Result<(), InferFailure> {
    if key.len() > MAX_KEY_BYTES || !key.is_ascii() {
        return Err(image("gguf metadata key is invalid"));
    }
    let mut segments = 0usize;
    for segment in key.split('.') {
        segments += 1;
        if segment.is_empty()
            || !segment
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        {
            return Err(image("gguf metadata key is invalid"));
        }
    }
    if segments < 2 {
        return Err(image("gguf metadata key is invalid"));
    }
    Ok(())
}

fn validate_tensor_name(name: &str) -> Result<(), InferFailure> {
    if name.len() > MAX_GGUF_TENSOR_NAME {
        return Err(image("gguf tensor name exceeds 64 bytes"));
    }
    if name.is_empty() || name.bytes().any(|byte| byte == 0) {
        return Err(image("gguf tensor name is invalid"));
    }
    Ok(())
}

fn validate_value(value: &GgufValue, depth: usize) -> Result<(), InferFailure> {
    match value {
        GgufValue::String(text) => {
            if text.len() > MAX_GGUF_STRING {
                return Err(image("gguf string exceeds its limit"));
            }
            if text.bytes().any(|byte| byte == 0) {
                return Err(image("gguf string contains a nul"));
            }
            Ok(())
        }
        GgufValue::Array(array) => {
            if depth >= MAX_GGUF_ARRAY_DEPTH {
                return Err(image("gguf array is too deep"));
            }
            if array.items.len() > MAX_GGUF_ARRAY {
                return Err(image("gguf array exceeds its limit"));
            }
            for item in &array.items {
                if item.value_type() != array.element {
                    return Err(image("gguf array element type does not match"));
                }
                validate_value(item, depth + 1)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn valid_architecture(value: &str) -> bool {
    (1..=64).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
}

fn valid_alignment(value: u32) -> bool {
    (8..=4096).contains(&value) && value % 8 == 0
}

fn check_shape(shape: &[u64]) -> Result<(), InferFailure> {
    if shape.is_empty() || shape.len() > MAX_GGUF_DIMENSIONS {
        return Err(image("gguf tensor dimension count is outside 1..=4"));
    }
    if shape.contains(&0) {
        return Err(image("gguf tensor dimension is zero"));
    }
    Ok(())
}

fn check_data_padding(data: &[u8], ranges: &[(u64, u64)]) -> Result<(), InferFailure> {
    let mut ordered = ranges.to_vec();
    ordered.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
    let mut cursor = 0u64;
    for &(start, end) in &ordered {
        if start < cursor {
            return Err(image("gguf tensor ranges overlap"));
        }
        let from = usize::try_from(cursor).unwrap_or(data.len());
        let to = usize::try_from(start).unwrap_or(data.len());
        if data[from..to].iter().any(|byte| *byte != 0) {
            return Err(image("gguf tensor padding is not zero"));
        }
        cursor = end;
    }
    let from = usize::try_from(cursor).unwrap_or(data.len());
    if data[from..].iter().any(|byte| *byte != 0) {
        return Err(image("gguf tensor padding is not zero"));
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64, InferFailure> {
    if alignment == 0 {
        return Err(image(
            "general.alignment must be a multiple of 8 from 8 to 4096",
        ));
    }
    let rem = value % alignment;
    if rem == 0 {
        return Ok(value);
    }
    value
        .checked_add(alignment - rem)
        .ok_or_else(|| image("gguf data section is outside the file"))
}

fn write_value(out: &mut Vec<u8>, value: &GgufValue) {
    match value {
        GgufValue::Uint8(inner) => out.push(*inner),
        GgufValue::Int8(inner) => out.push(*inner as u8),
        GgufValue::Uint16(inner) => push_u16(out, *inner),
        GgufValue::Int16(inner) => push_u16(out, *inner as u16),
        GgufValue::Uint32(inner) => push_u32(out, *inner),
        GgufValue::Int32(inner) => push_u32(out, *inner as u32),
        GgufValue::Float32(inner) => push_u32(out, *inner),
        GgufValue::Bool(inner) => out.push(u8::from(*inner)),
        GgufValue::String(inner) => push_str(out, inner),
        GgufValue::Array(array) => {
            push_u32(out, array.element as u32);
            push_u64(out, array.items.len() as u64);
            for item in &array.items {
                write_value(out, item);
            }
        }
        GgufValue::Uint64(inner) => push_u64(out, *inner),
        GgufValue::Int64(inner) => push_u64(out, *inner as u64),
        GgufValue::Float64(inner) => push_u64(out, *inner),
    }
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_str(out: &mut Vec<u8>, text: &str) {
    push_u64(out, text.len() as u64);
    out.extend_from_slice(text.as_bytes());
}

fn image(message: impl Into<String>) -> InferFailure {
    fail(ErrorCode::ModelImageInvalid, message)
}

fn unsupported(message: impl Into<String>) -> InferFailure {
    fail(ErrorCode::UnsupportedQuantization, message)
}
