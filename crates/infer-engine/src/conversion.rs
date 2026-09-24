//! Explicit GGUF-to-f32 conversion.
//!
//! The source payload is not rewritten. The destination is a new
//! little-endian `f32` artifact, and the receipt names both artifact roots.
//! `dequant_gguf` and `quant_gemm` do not call this path. The layout is
//! specified in `spec/KIP-INFER-0025-conversion-receipt.md`.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use infer_artifact::GgufTensorType;
use infer_contracts::{
    artifact_root, conversion_config_root, fail, sha256_prefixed, validate_relative_path,
    ArtifactFileV1, ConversionReceiptV1, DigestHex, ErrorCode, InferFailure,
};

use crate::dequant::dequant_gguf;

/// One converted tensor: the new `f32` bytes and the receipt that names them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufConversion {
    pub tensor_type: GgufTensorType,
    pub shape: Vec<u64>,
    pub payload: Vec<u8>,
    pub source: ArtifactFileV1,
    pub destination: ArtifactFileV1,
    pub destination_bytes: Vec<u8>,
    pub converter_build_root: DigestHex,
    pub receipt: ConversionReceiptV1,
}

/// Expand one allowlisted payload and build its conversion receipt.
///
/// No file is created. The payload slice is left unchanged.
pub fn convert_gguf_tensor(
    tensor_type: GgufTensorType,
    shape: &[u64],
    payload: &[u8],
    source_path: &str,
    destination_path: &str,
    converter_build_root: &DigestHex,
) -> Result<GgufConversion, InferFailure> {
    validate_relative_path(source_path)?;
    validate_relative_path(destination_path)?;
    if source_path == destination_path {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "conversion destination matches the source path",
        ));
    }
    let packed = tensor_type.nbytes(shape)?;
    if payload.len() as u64 != packed {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "gguf tensor payload length does not match its type",
        ));
    }
    let values = dequant_gguf(tensor_type, payload)?;
    let destination_bytes = f32_le_bytes(&values);
    let source = artifact_file(source_path, payload)?;
    let destination = artifact_file(destination_path, &destination_bytes)?;
    let receipt = ConversionReceiptV1 {
        source_artifact_root: artifact_root(std::slice::from_ref(&source))?,
        converter_build_root: converter_build_root.clone(),
        conversion_config_root: conversion_config_root(tensor_type.name(), shape)?,
        destination_artifact_root: artifact_root(std::slice::from_ref(&destination))?,
        validation_result: "matched".into(),
        extensions: Default::default(),
    };
    let produced = GgufConversion {
        tensor_type,
        shape: shape.to_vec(),
        payload: payload.to_vec(),
        source,
        destination,
        destination_bytes,
        converter_build_root: converter_build_root.clone(),
        receipt,
    };
    verify_gguf_conversion(&produced)?;
    Ok(produced)
}

/// Recompute the receipt roots and compare the destination to `dequant_gguf`.
pub fn verify_gguf_conversion(conversion: &GgufConversion) -> Result<(), InferFailure> {
    let receipt = &conversion.receipt;
    receipt.validate()?;
    if receipt.converter_build_root != conversion.converter_build_root {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "converter build root does not match",
        ));
    }
    if receipt.conversion_config_root
        != conversion_config_root(conversion.tensor_type.name(), &conversion.shape)?
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "conversion configuration root does not match",
        ));
    }
    let source = artifact_file(&conversion.source.path, &conversion.payload)?;
    if receipt.source_artifact_root != artifact_root(std::slice::from_ref(&source))? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "source artifact root does not match",
        ));
    }
    let destination = artifact_file(&conversion.destination.path, &conversion.destination_bytes)?;
    if receipt.destination_artifact_root != artifact_root(std::slice::from_ref(&destination))? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "destination artifact root does not match",
        ));
    }
    let values = dequant_gguf(conversion.tensor_type, &conversion.payload)?;
    if conversion.destination_bytes != f32_le_bytes(&values) {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "conversion validation did not match",
        ));
    }
    Ok(())
}

/// Write the destination and the receipt. The source path is not opened.
///
/// Both output paths must be absent. A failed receipt write removes the
/// destination file.
pub fn write_gguf_conversion(
    base: &Path,
    conversion: &GgufConversion,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_gguf_conversion(conversion)?;
    validate_relative_path(receipt_path)?;
    if receipt_path == conversion.source.path || receipt_path == conversion.destination.path {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "conversion receipt path matches an artifact path",
        ));
    }
    let receipt_bytes = conversion.receipt.to_bytes()?;
    let destination = output_path(base, &conversion.destination.path)?;
    let receipt = output_path(base, receipt_path)?;
    write_new(&destination, &conversion.destination_bytes)?;
    if let Err(err) = write_new(&receipt, &receipt_bytes) {
        match fs::remove_file(&destination) {
            Ok(()) => {
                sync_parent(&destination);
                Err(err)
            }
            Err(remove_err) => Err(fail(
                ErrorCode::ContractInvalid,
                format!("conversion destination was left without a receipt: {remove_err}"),
            )),
        }
    } else {
        Ok(())
    }
}

fn artifact_file(path: &str, bytes: &[u8]) -> Result<ArtifactFileV1, InferFailure> {
    let file = ArtifactFileV1 {
        path: path.to_string(),
        size_bytes: bytes.len() as u64,
        sha256: sha256_prefixed(bytes),
    };
    file.validate()?;
    Ok(file)
}

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn output_path(base: &Path, relative: &str) -> Result<PathBuf, InferFailure> {
    validate_relative_path(relative)?;
    let base = fs::canonicalize(base).map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            fail(
                ErrorCode::ContractInvalid,
                "conversion directory does not exist",
            )
        } else {
            fail(
                ErrorCode::ContractInvalid,
                format!("conversion directory: {err}"),
            )
        }
    })?;
    if !base.is_dir() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "conversion directory does not exist",
        ));
    }
    let mut parent = base.clone();
    let mut parts = relative.split('/');
    let file_name = parts.next_back().expect("relative path has a file name");
    for segment in parts {
        parent.push(segment);
        let meta = match fs::symlink_metadata(&parent) {
            Ok(meta) => meta,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "conversion directory does not exist",
                ));
            }
            Err(err) => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    format!("conversion directory: {err}"),
                ));
            }
        };
        if meta.file_type().is_symlink() || !meta.is_dir() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "conversion path leaves the directory",
            ));
        }
    }
    let parent = fs::canonicalize(&parent).map_err(|err| {
        fail(
            ErrorCode::ContractInvalid,
            format!("conversion directory: {err}"),
        )
    })?;
    if parent != base && !parent.starts_with(&base) {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "conversion path leaves the directory",
        ));
    }
    let path = parent.join(file_name);
    match fs::symlink_metadata(&path) {
        Ok(_) => Err(fail(
            ErrorCode::ContractInvalid,
            "conversion output already exists",
        )),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(path),
        Err(err) => Err(fail(
            ErrorCode::ContractInvalid,
            format!("conversion output: {err}"),
        )),
    }
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), InferFailure> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| {
            if err.kind() == std::io::ErrorKind::AlreadyExists {
                fail(
                    ErrorCode::ContractInvalid,
                    "conversion output already exists",
                )
            } else {
                fail(ErrorCode::ContractInvalid, format!("write: {err}"))
            }
        })?;
    file.write_all(bytes)
        .map_err(|err| fail(ErrorCode::ContractInvalid, format!("write: {err}")))?;
    file.sync_all()
        .map_err(|err| fail(ErrorCode::ContractInvalid, format!("write: {err}")))?;
    drop(file);
    sync_parent(path);
    Ok(())
}

fn sync_parent(path: &Path) {
    let Some(parent) = path.parent() else {
        return;
    };
    if let Ok(dir) = File::open(parent) {
        let _ = dir.sync_all();
    }
}
