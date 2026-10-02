//! Bounded safetensors inventory.
//!
//! The header is parsed and the offsets are checked against the file length.
//! Tensor bodies are not read by the streaming inventory. Padding between
//! tensors is only the 0–7 bytes needed to align the next tensor to 8 bytes.
//! `parse_safetensors_bytes` refuses a non-zero byte in that padding.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use serde_json::{Map, Value};

use infer_contracts::{fail, ErrorCode, InferFailure, TensorSpecV1};

use crate::io::prefixed;
use crate::json::parse_strict_json;
use crate::map_image;

use infer_contracts::DigestHex;

pub const MAX_SAFETENSORS_HEADER: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorBytes {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u32>,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorView {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u32>,
    pub start: u64,
    pub end: u64,
    pub file: String,
}

/// Largest weight file this crate will load into memory. Inventory hashing
/// can stream a larger file; interpreting the body cannot.
pub const MAX_IN_MEMORY_WEIGHT: u64 = 32 * 1024 * 1024;

/// Read tensor bodies only after each file's size and SHA-256 match the model image.
pub fn read_verified_tensors(
    image: &infer_contracts::ModelImageV1,
    weights_dir: &Path,
) -> Result<Vec<TensorBytes>, InferFailure> {
    if image.format != "safetensors" {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "only safetensors weight files can be read",
        ));
    }
    let mut found = Vec::new();
    let mut bodies = std::collections::BTreeMap::new();
    for file in &image.files {
        let path = crate::paths::resolve_inside(
            weights_dir,
            &file.path,
            ErrorCode::ModelImageInvalid,
            ErrorCode::ModelArtifactMissing,
        )?;
        let bytes = read_bytes_after_hash(&path, file.size_bytes, &file.sha256, &file.path)?;
        let mut views = parse_safetensors_bytes(&bytes)?;
        let header_len =
            u64::from_le_bytes(bytes[..8].try_into().expect("header length is 8 bytes"));
        let data_start = 8usize
            .checked_add(usize::try_from(header_len).unwrap_or(usize::MAX))
            .ok_or_else(|| {
                fail(
                    ErrorCode::ModelImageInvalid,
                    "safetensors header length overflows",
                )
            })?;
        for view in &mut views {
            view.file = file.path.clone();
            if bodies.contains_key(&view.name) {
                return Err(fail(
                    ErrorCode::ModelImageInvalid,
                    format!("duplicate tensor {}", view.name),
                ));
            }
            let start = data_start
                .checked_add(usize::try_from(view.start).unwrap_or(usize::MAX))
                .ok_or_else(|| {
                    fail(
                        ErrorCode::ModelImageInvalid,
                        format!("tensor {} offset overflows", view.name),
                    )
                })?;
            let end = data_start
                .checked_add(usize::try_from(view.end).unwrap_or(usize::MAX))
                .ok_or_else(|| {
                    fail(
                        ErrorCode::ModelImageInvalid,
                        format!("tensor {} offset overflows", view.name),
                    )
                })?;
            if end > bytes.len() || start > end {
                return Err(fail(
                    ErrorCode::ModelImageInvalid,
                    format!("tensor {} offset is outside the file", view.name),
                ));
            }
            bodies.insert(view.name.clone(), bytes[start..end].to_vec());
        }
        found.extend(views);
    }
    require_inventory(&image.tensor_inventory, &image.precisions, &found)?;
    let mut tensors = Vec::with_capacity(image.tensor_inventory.len());
    for spec in &image.tensor_inventory {
        let Some(bytes) = bodies.remove(&spec.name) else {
            return Err(fail(
                ErrorCode::ModelArtifactMissing,
                format!("missing tensor {}", spec.name),
            ));
        };
        tensors.push(TensorBytes {
            name: spec.name.clone(),
            dtype: spec.dtype.clone(),
            shape: spec.shape.clone(),
            bytes,
        });
    }
    Ok(tensors)
}

fn read_bytes_after_hash(
    path: &Path,
    expected_size: u64,
    expected_digest: &DigestHex,
    display: &str,
) -> Result<Vec<u8>, InferFailure> {
    let meta = std::fs::metadata(path).map_err(|err| {
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
    if expected_size > MAX_IN_MEMORY_WEIGHT {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "weight file exceeds the 32 MiB in-memory read limit",
        ));
    }
    let bytes = std::fs::read(path).map_err(|err| {
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
    Ok(bytes)
}

pub fn encode_safetensors(tensors: &[TensorBytes]) -> Result<Vec<u8>, InferFailure> {
    if tensors.is_empty() || tensors.len() > 100_000 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors tensor list is empty or too large",
        ));
    }
    let mut ordered: Vec<&TensorBytes> = tensors.iter().collect();
    ordered.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
    let mut data = Vec::new();
    let mut header = Map::new();
    for tensor in ordered {
        if header.contains_key(&tensor.name) {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                format!("duplicate tensor {}", tensor.name),
            ));
        }
        let spec = TensorSpecV1 {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            dtype: tensor.dtype.clone(),
        };
        spec.validate().map_err(map_image)?;
        let nbytes = byte_length(&tensor.shape, &tensor.dtype)?;
        if tensor.bytes.len() as u64 != nbytes {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                format!(
                    "tensor {} payload length does not match its shape",
                    tensor.name
                ),
            ));
        }
        let start = align8(data.len() as u64)?;
        if start as usize > data.len() {
            data.resize(start as usize, 0);
        }
        let end = start
            .checked_add(nbytes)
            .ok_or_else(|| fail(ErrorCode::ModelImageInvalid, "tensor offset overflows"))?;
        data.extend_from_slice(&tensor.bytes);
        let mut entry = Map::new();
        entry.insert(
            "data_offsets".into(),
            Value::Array(vec![json_u64(start), json_u64(end)]),
        );
        entry.insert(
            "dtype".into(),
            Value::String(safetensors_dtype(&tensor.dtype)?.to_string()),
        );
        entry.insert(
            "shape".into(),
            Value::Array(
                tensor
                    .shape
                    .iter()
                    .copied()
                    .map(|dim| json_u64(u64::from(dim)))
                    .collect(),
            ),
        );
        header.insert(tensor.name.clone(), Value::Object(entry));
    }
    let header_bytes = serde_json::to_vec(&Value::Object(header)).map_err(|_| {
        fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header could not be encoded",
        )
    })?;
    if header_bytes.len() > MAX_SAFETENSORS_HEADER {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header exceeds 16 MiB",
        ));
    }
    let mut out = Vec::with_capacity(8 + header_bytes.len() + data.len());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&data);
    Ok(out)
}

pub fn parse_safetensors_bytes(bytes: &[u8]) -> Result<Vec<TensorView>, InferFailure> {
    if bytes.len() < 8 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header is truncated",
        ));
    }
    let mut len_buf = [0u8; 8];
    len_buf.copy_from_slice(&bytes[..8]);
    let header_len = u64::from_le_bytes(len_buf);
    if header_len > bytes.len() as u64 - 8 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header is truncated",
        ));
    }
    let header_end = 8 + header_len as usize;
    let views = parse_header_bytes(&bytes[8..header_end], header_len, bytes.len() as u64)?;
    reject_nonzero_slack(&views, &bytes[header_end..])?;
    Ok(views)
}

/// Alignment slack may be present. A non-zero slack byte is a corrupted file.
fn reject_nonzero_slack(views: &[TensorView], data: &[u8]) -> Result<(), InferFailure> {
    let mut order: Vec<&TensorView> = views.iter().collect();
    order.sort_by(|left, right| {
        left.start
            .cmp(&right.start)
            .then(left.name.as_bytes().cmp(right.name.as_bytes()))
    });
    let mut cursor = 0usize;
    for view in order {
        let start = usize::try_from(view.start).unwrap_or(usize::MAX);
        let end = usize::try_from(view.end).unwrap_or(usize::MAX);
        if start > data.len() || end > data.len() || start > end || start < cursor {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                "safetensors data region does not match the tensor offsets",
            ));
        }
        if data[cursor..start].iter().any(|byte| *byte != 0) {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                "safetensors alignment padding is not zero",
            ));
        }
        cursor = end;
    }
    if data[cursor..].iter().any(|byte| *byte != 0) {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors alignment padding is not zero",
        ));
    }
    Ok(())
}

/// Hash `expected_size` bytes and parse the header from that same prefix.
/// A digest mismatch is returned before a header error.
pub fn inventory_verified_file(
    path: &Path,
    expected_size: u64,
    expected_digest: &DigestHex,
    display: &str,
) -> Result<Vec<TensorView>, InferFailure> {
    let meta = std::fs::metadata(path).map_err(|err| {
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
    let mut file = File::open(path)
        .map_err(|err| fail(ErrorCode::ModelImageInvalid, format!("safetensors: {err}")))?;
    let mut hasher = Sha256::new();
    let mut counted = 0u64;
    let mut len_buf = [0u8; 8];
    let mut header = Vec::new();
    let mut header_len = 0u64;
    let mut header_ok = false;
    if expected_size >= 8 {
        read_exact_hashed(&mut file, &mut hasher, &mut len_buf, &mut counted)?;
        header_len = u64::from_le_bytes(len_buf);
        header_ok = header_len > 0
            && header_len <= MAX_SAFETENSORS_HEADER as u64
            && 8 + header_len <= expected_size;
        if header_ok {
            header.resize(header_len as usize, 0);
            read_exact_hashed(&mut file, &mut hasher, &mut header, &mut counted)?;
        }
    }
    let mut buffer = [0u8; 64 * 1024];
    while counted < expected_size {
        let want = usize::try_from(expected_size - counted).unwrap_or(buffer.len());
        let want = want.min(buffer.len());
        let read = file.read(&mut buffer[..want]).map_err(|err| {
            fail(
                ErrorCode::ModelDigestMismatch,
                format!("artifact read failed for {display}: {err}"),
            )
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        counted += read as u64;
    }
    if counted != expected_size {
        return Err(fail(
            ErrorCode::ModelDigestMismatch,
            format!("artifact size does not match {display}"),
        ));
    }
    let mut digest_bytes = [0u8; 32];
    digest_bytes.copy_from_slice(&hasher.finalize());
    if prefixed(digest_bytes) != *expected_digest {
        return Err(fail(
            ErrorCode::ModelDigestMismatch,
            format!("artifact digest does not match {display}"),
        ));
    }
    if !header_ok {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header length is outside its bounds",
        ));
    }
    parse_header_bytes(&header, header_len, expected_size)
}

pub fn read_safetensors_inventory(path: &Path) -> Result<Vec<TensorView>, InferFailure> {
    let mut file = File::open(path)
        .map_err(|err| fail(ErrorCode::ModelImageInvalid, format!("safetensors: {err}")))?;
    let file_len = file
        .metadata()
        .map_err(|err| fail(ErrorCode::ModelImageInvalid, format!("safetensors: {err}")))?
        .len();
    if file_len < 8 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header is truncated",
        ));
    }
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).map_err(|_| {
        fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header is truncated",
        )
    })?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER as u64 || header_len > file_len - 8 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header length is outside its bounds",
        ));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header).map_err(|_| {
        fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header is truncated",
        )
    })?;
    // The tensor body stays unread. Offsets are checked against file_len.
    drop(file);
    parse_header_bytes(&header, header_len, file_len)
}

pub fn require_inventory(
    expected: &[TensorSpecV1],
    precisions: &[String],
    found: &[TensorView],
) -> Result<(), InferFailure> {
    let mut seen: std::collections::BTreeMap<&str, &TensorView> = std::collections::BTreeMap::new();
    for tensor in found {
        if seen.insert(tensor.name.as_str(), tensor).is_some() {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                format!("duplicate tensor {}", tensor.name),
            ));
        }
    }
    let mut expected_names = std::collections::BTreeSet::new();
    for spec in expected {
        if !expected_names.insert(spec.name.as_str()) {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                format!("duplicate tensor {}", spec.name),
            ));
        }
        let Some(actual) = seen.get(spec.name.as_str()) else {
            return Err(fail(
                ErrorCode::ModelArtifactMissing,
                format!("missing tensor {}", spec.name),
            ));
        };
        if actual.shape != spec.shape || actual.dtype != spec.dtype {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                format!(
                    "tensor {} does not match the model image inventory",
                    spec.name
                ),
            ));
        }
        if !precisions.iter().any(|item| item == &spec.dtype) {
            return Err(fail(
                ErrorCode::UnsupportedQuantization,
                format!("tensor {} dtype is not a declared precision", spec.name),
            ));
        }
    }
    for name in seen.keys() {
        if !expected_names.contains(name) {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                format!("unexpected tensor {name}"),
            ));
        }
    }
    Ok(())
}

fn parse_header_bytes(
    header: &[u8],
    header_len: u64,
    file_len: u64,
) -> Result<Vec<TensorView>, InferFailure> {
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER as u64 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header length is outside its bounds",
        ));
    }
    if header.len() as u64 != header_len {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header is truncated",
        ));
    }
    let data_start = 8u64.checked_add(header_len).ok_or_else(|| {
        fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header length overflows",
        )
    })?;
    if data_start > file_len {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header is truncated",
        ));
    }
    let text = std::str::from_utf8(header).map_err(|_| {
        fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header is not UTF-8",
        )
    })?;
    let tensors = parse_header_json(text)?;
    let data_len = file_len - data_start;
    check_layout(&tensors, data_len)?;
    Ok(tensors)
}

fn parse_header_json(text: &str) -> Result<Vec<TensorView>, InferFailure> {
    let value = parse_strict_json(text, ErrorCode::ModelImageInvalid).map_err(|err| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("safetensors header: {}", err.message),
        )
    })?;
    let Some(object) = value.as_object() else {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header must be an object",
        ));
    };
    let mut tensors = Vec::new();
    for (key, value) in object {
        if key == "__metadata__" {
            check_metadata(value)?;
            continue;
        }
        tensors.push(parse_tensor(key, value)?);
    }
    if tensors.is_empty() {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors file has no tensors",
        ));
    }
    if tensors.len() > 100_000 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors tensor list is too large",
        ));
    }
    Ok(tensors)
}

fn check_metadata(value: &Value) -> Result<(), InferFailure> {
    let Some(object) = value.as_object() else {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors metadata must be an object",
        ));
    };
    if object.len() > 64 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors metadata has too many entries",
        ));
    }
    for (key, item) in object {
        if key.len() > 256 || item.as_str().is_none_or(|text| text.len() > 4096) {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                "safetensors metadata entries must be short strings",
            ));
        }
    }
    Ok(())
}

fn parse_tensor(name: &str, value: &Value) -> Result<TensorView, InferFailure> {
    let Some(object) = value.as_object() else {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            format!("safetensors tensor {name} must be an object"),
        ));
    };
    if object.len() != 3
        || !object.contains_key("dtype")
        || !object.contains_key("shape")
        || !object.contains_key("data_offsets")
    {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            format!("safetensors tensor {name} has an unexpected field"),
        ));
    }
    let dtype_raw = object.get("dtype").and_then(Value::as_str).ok_or_else(|| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("tensor {name} dtype is invalid"),
        )
    })?;
    let dtype = knolo_dtype(dtype_raw)?.to_string();
    let shape = parse_shape(name, object.get("shape").expect("shape key checked"))?;
    let (start, end) = parse_offsets(
        name,
        object.get("data_offsets").expect("offsets key checked"),
    )?;
    let spec = TensorSpecV1 {
        name: name.to_string(),
        shape: shape.clone(),
        dtype: dtype.clone(),
    };
    spec.validate().map_err(map_image)?;
    let nbytes = byte_length(&shape, &dtype)?;
    if end.checked_sub(start) != Some(nbytes) {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            format!("tensor {name} byte length does not match its shape"),
        ));
    }
    Ok(TensorView {
        name: name.to_string(),
        dtype,
        shape,
        start,
        end,
        file: String::new(),
    })
}

fn parse_shape(name: &str, value: &Value) -> Result<Vec<u32>, InferFailure> {
    let Some(items) = value.as_array() else {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            format!("tensor {name} shape is invalid"),
        ));
    };
    if items.is_empty() || items.len() > 8 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            format!("tensor {name} shape is invalid"),
        ));
    }
    let mut shape = Vec::with_capacity(items.len());
    for item in items {
        let Some(dim) = item.as_u64() else {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                format!("tensor {name} shape is invalid"),
            ));
        };
        if dim == 0 || dim > u64::from(u32::MAX) {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                format!("tensor {name} shape is invalid"),
            ));
        }
        shape.push(dim as u32);
    }
    Ok(shape)
}

fn parse_offsets(name: &str, value: &Value) -> Result<(u64, u64), InferFailure> {
    let Some(items) = value.as_array() else {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            format!("tensor {name} offsets are invalid"),
        ));
    };
    if items.len() != 2 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            format!("tensor {name} offsets are invalid"),
        ));
    }
    let start = items[0].as_u64().ok_or_else(|| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("tensor {name} offsets are invalid"),
        )
    })?;
    let end = items[1].as_u64().ok_or_else(|| {
        fail(
            ErrorCode::ModelImageInvalid,
            format!("tensor {name} offsets are invalid"),
        )
    })?;
    if end < start {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            format!("tensor {name} offsets are reversed"),
        ));
    }
    Ok((start, end))
}

fn check_layout(tensors: &[TensorView], data_len: u64) -> Result<(), InferFailure> {
    for tensor in tensors.iter() {
        if tensor.start > data_len || tensor.end > data_len || tensor.start % 8 != 0 {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                format!(
                    "tensor {} offset is outside the file or is not aligned",
                    tensor.name
                ),
            ));
        }
    }
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by(|&left, &right| {
        tensors[left].start.cmp(&tensors[right].start).then(
            tensors[left]
                .name
                .as_bytes()
                .cmp(tensors[right].name.as_bytes()),
        )
    });
    let mut cursor = 0u64;
    for index in order {
        let tensor = &tensors[index];
        if tensor.start != cursor {
            let aligned = align8(cursor)?;
            if tensor.start != aligned || tensor.start.saturating_sub(cursor) >= 8 {
                return Err(fail(
                    ErrorCode::ModelImageInvalid,
                    format!(
                        "tensor {} leaves a gap or overlaps another tensor",
                        tensor.name
                    ),
                ));
            }
        }
        cursor = tensor.end;
    }
    if cursor > data_len || data_len - cursor >= 8 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors data region does not match the tensor offsets",
        ));
    }
    Ok(())
}

fn byte_length(shape: &[u32], dtype: &str) -> Result<u64, InferFailure> {
    let width = dtype_width(dtype)?;
    let mut product = 1u64;
    for dim in shape {
        product = product
            .checked_mul(u64::from(*dim))
            .ok_or_else(|| fail(ErrorCode::ModelImageInvalid, "tensor shape overflows"))?;
    }
    product
        .checked_mul(width)
        .ok_or_else(|| fail(ErrorCode::ModelImageInvalid, "tensor byte length overflows"))
}

fn dtype_width(dtype: &str) -> Result<u64, InferFailure> {
    match dtype {
        "f32" | "i32" => Ok(4),
        "f16" | "bf16" => Ok(2),
        "u8" => Ok(1),
        _ => Err(fail(
            ErrorCode::UnsupportedQuantization,
            "dtype is not in the safetensors allowlist",
        )),
    }
}

fn safetensors_dtype(dtype: &str) -> Result<&'static str, InferFailure> {
    match dtype {
        "f32" => Ok("F32"),
        "f16" => Ok("F16"),
        "bf16" => Ok("BF16"),
        "i32" => Ok("I32"),
        "u8" => Ok("U8"),
        _ => Err(fail(
            ErrorCode::UnsupportedQuantization,
            "dtype is not in the safetensors allowlist",
        )),
    }
}

fn knolo_dtype(dtype: &str) -> Result<&'static str, InferFailure> {
    match dtype {
        "F32" => Ok("f32"),
        "F16" => Ok("f16"),
        "BF16" => Ok("bf16"),
        "I32" => Ok("i32"),
        "U8" => Ok("u8"),
        _ => Err(fail(
            ErrorCode::UnsupportedQuantization,
            format!("safetensors dtype {dtype} is not supported"),
        )),
    }
}

fn align8(value: u64) -> Result<u64, InferFailure> {
    let rem = value % 8;
    if rem == 0 {
        Ok(value)
    } else {
        value
            .checked_add(8 - rem)
            .ok_or_else(|| fail(ErrorCode::ModelImageInvalid, "tensor offset overflows"))
    }
}

fn json_u64(value: u64) -> Value {
    Value::Number(value.into())
}

fn read_exact_hashed(
    file: &mut File,
    hasher: &mut Sha256,
    dest: &mut [u8],
    counted: &mut u64,
) -> Result<(), InferFailure> {
    file.read_exact(dest).map_err(|_| {
        fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header is truncated",
        )
    })?;
    hasher.update(&*dest);
    *counted += dest.len() as u64;
    Ok(())
}
