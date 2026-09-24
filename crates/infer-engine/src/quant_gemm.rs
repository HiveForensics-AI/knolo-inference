//! CPU quantized matrix product.
//!
//! The weight payload stays in its GGUF block layout. Each block is expanded
//! with the dequant rules and consumed immediately. The f32 matrix is not
//! allocated, and the payload is not rewritten. The product layout and the
//! 32 MiB output cap are specified in `spec/KIP-INFER-0024-quant-gemm.md`.

use infer_artifact::GgufTensorType;
use infer_contracts::{fail, ErrorCode, InferFailure};

use crate::dequant::{dequant_block, MAX_DEQUANT_BYTES};

/// Byte length of the `f32` buffer `quant_gemm` will allocate.
///
/// `rows` and `n` are the output height and the activation count. A zero
/// dimension is a shape error. A product above 32 MiB is
/// `INSUFFICIENT_MEMORY`.
pub fn quant_gemm_output_bytes(rows: usize, n: usize) -> Result<u64, InferFailure> {
    if rows == 0 || n == 0 {
        return Err(shape());
    }
    let bytes = u64::try_from(rows)
        .ok()
        .and_then(|rows| u64::try_from(n).ok().and_then(|n| rows.checked_mul(n)))
        .and_then(|elems| elems.checked_mul(4))
        .ok_or_else(overflow)?;
    if bytes > MAX_DEQUANT_BYTES {
        return Err(too_big());
    }
    Ok(bytes)
}

/// `Y = W X` for one allowlisted GGUF payload.
///
/// `cols` is GGUF `ne[0]` and is the blocked dimension. `rows` is the product
/// of the later dimensions. `x` holds `n` contiguous activation vectors of
/// length `cols`. The result holds `n` contiguous vectors of length `rows`.
/// Column `c` of a row is multiplied in increasing `c`, the same order as the
/// micro-model `gemv`.
pub fn quant_gemm(
    tensor_type: GgufTensorType,
    payload: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    n: usize,
) -> Result<Vec<f32>, InferFailure> {
    if rows == 0 || cols == 0 || n == 0 {
        return Err(shape());
    }
    let packed = packed_payload_bytes(tensor_type, rows, cols)?;
    if payload.len() as u64 != packed {
        return Err(payload_len());
    }
    let out_bytes = quant_gemm_output_bytes(rows, n)?;
    let x_len = cols.checked_mul(n).ok_or_else(overflow)?;
    if x.len() != x_len {
        return Err(shape());
    }

    let block_elems = tensor_type.block_elements() as usize;
    let block_bytes = tensor_type.type_size() as usize;
    let row_bytes = cols / block_elems * block_bytes;
    let count = usize::try_from(out_bytes / 4).map_err(|_| overflow())?;
    let mut y = vec![0.0f32; count];
    let mut decoded = [0.0f32; 256];
    for (row, row_payload) in payload.chunks_exact(row_bytes).enumerate() {
        for (block_index, block) in row_payload.chunks_exact(block_bytes).enumerate() {
            dequant_block(tensor_type, block, &mut decoded[..block_elems])?;
            let col0 = block_index * block_elems;
            for (offset, weight) in decoded[..block_elems].iter().copied().enumerate() {
                let col = col0 + offset;
                for k in 0..n {
                    let slot = &mut y[k * rows + row];
                    *slot += weight * x[k * cols + col];
                    if !slot.is_finite() {
                        return Err(product_non_finite());
                    }
                }
            }
        }
    }
    Ok(y)
}

fn packed_payload_bytes(
    tensor_type: GgufTensorType,
    rows: usize,
    cols: usize,
) -> Result<u64, InferFailure> {
    let block = tensor_type.block_elements();
    let cols_u = u64::try_from(cols).map_err(|_| overflow())?;
    let rows_u = u64::try_from(rows).map_err(|_| overflow())?;
    if cols_u % block != 0 {
        return Err(bad_dim());
    }
    (cols_u / block)
        .checked_mul(tensor_type.type_size())
        .and_then(|row_bytes| row_bytes.checked_mul(rows_u))
        .ok_or_else(overflow)
}

fn shape() -> InferFailure {
    fail(
        ErrorCode::ContractInvalid,
        "quantized matrix shape does not match its input",
    )
}

fn overflow() -> InferFailure {
    fail(
        ErrorCode::ModelImageInvalid,
        "gguf tensor byte length overflows",
    )
}

fn bad_dim() -> InferFailure {
    fail(
        ErrorCode::ModelImageInvalid,
        "gguf tensor dimension is not a multiple of the block",
    )
}

fn payload_len() -> InferFailure {
    fail(
        ErrorCode::ModelImageInvalid,
        "gguf tensor payload length does not match its type",
    )
}

fn product_non_finite() -> InferFailure {
    fail(
        ErrorCode::ContractInvalid,
        "quantized matrix product is non-finite",
    )
}

fn too_big() -> InferFailure {
    fail(
        ErrorCode::InsufficientMemory,
        "quantized product exceeds 32 MiB",
    )
}
