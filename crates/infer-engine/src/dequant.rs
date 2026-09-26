//! CPU dequant for the GGUF tensor types this slice accepts.
//!
//! The result is a new `f32` buffer. The payload is not modified and no
//! weight file is written. `quant_gemm` uses the same block expansion and
//! does not store that buffer. `Q6_K` follows the 210-byte super-block in
//! `spec/KIP-INFER-0018-q6k-dequant.md`. `Q5_K` follows the 176-byte
//! super-block in `spec/KIP-INFER-0019-q5k-dequant.md`. `Q4_K` follows the
//! 144-byte super-block in `spec/KIP-INFER-0020-q4k-dequant.md`.

use infer_artifact::GgufTensorType;
use infer_contracts::{fail, ErrorCode, InferFailure};

pub const MAX_DEQUANT_BYTES: u64 = 32 * 1024 * 1024;

/// Byte length of the `f32` buffer `dequant_gguf` will allocate.
pub fn dequant_output_bytes(
    tensor_type: GgufTensorType,
    payload_len: u64,
) -> Result<u64, InferFailure> {
    let type_size = tensor_type.type_size();
    if payload_len == 0 || payload_len % type_size != 0 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "gguf tensor payload length does not match its type",
        ));
    }
    let elements = payload_len
        .checked_div(type_size)
        .and_then(|blocks| blocks.checked_mul(tensor_type.block_elements()))
        .ok_or_else(|| {
            fail(
                ErrorCode::ModelImageInvalid,
                "gguf tensor byte length overflows",
            )
        })?;
    let bytes = elements.checked_mul(4).ok_or_else(|| {
        fail(
            ErrorCode::ModelImageInvalid,
            "gguf tensor byte length overflows",
        )
    })?;
    if bytes > MAX_DEQUANT_BYTES {
        return Err(fail(
            ErrorCode::InsufficientMemory,
            "dequant output exceeds 32 MiB",
        ));
    }
    Ok(bytes)
}

/// Expand one tensor payload to finite `f32` values.
pub fn dequant_gguf(tensor_type: GgufTensorType, bytes: &[u8]) -> Result<Vec<f32>, InferFailure> {
    let out_bytes = dequant_output_bytes(tensor_type, bytes.len() as u64)?;
    let count = usize::try_from(out_bytes / 4).unwrap_or(0);
    let mut out = Vec::with_capacity(count);
    let width = tensor_type.type_size() as usize;
    let elems = tensor_type.block_elements() as usize;
    let mut decoded = [0.0f32; 256];
    for block in bytes.chunks_exact(width) {
        dequant_block(tensor_type, block, &mut decoded[..elems])?;
        out.extend_from_slice(&decoded[..elems]);
    }
    if out.len() != count {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "dequant output length does not match its estimate",
        ));
    }
    Ok(out)
}

/// Expand one block into `out`. `out` has `block_elements` slots.
pub(crate) fn dequant_block(
    tensor_type: GgufTensorType,
    block: &[u8],
    out: &mut [f32],
) -> Result<(), InferFailure> {
    if block.len() != tensor_type.type_size() as usize
        || out.len() != tensor_type.block_elements() as usize
    {
        return Err(fail(ErrorCode::ContractInvalid, "quantized block width"));
    }
    match tensor_type {
        GgufTensorType::F32 => {
            let value = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
            write_finite(out, 0, value)
        }
        GgufTensorType::F16 => {
            let bits = u16::from_le_bytes([block[0], block[1]]);
            write_finite(out, 0, f16_to_f32(bits))
        }
        GgufTensorType::Q8_0 => {
            let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            if !scale.is_finite() {
                return Err(non_finite());
            }
            for (index, byte) in block[2..].iter().enumerate() {
                let quant = i8::from_le_bytes([*byte]);
                write_finite(out, index, scale * f32::from(quant))?;
            }
            Ok(())
        }
        GgufTensorType::Q4_K => dequant_q4_k(block, out),
        GgufTensorType::Q5_K => dequant_q5_k(block, out),
        GgufTensorType::Q6_K => dequant_q6_k(block, out),
    }
}

/// One `Q4_K` super-block: `d`, `dmin`, 12 scale bytes, then 128 bytes of 4-bit codes.
fn dequant_q4_k(block: &[u8], out: &mut [f32]) -> Result<(), InferFailure> {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    if !d.is_finite() || !dmin.is_finite() {
        return Err(non_finite());
    }
    let scales = &block[4..16];
    let qs = &block[16..144];
    let mut row = [0f32; 256];
    for chunk in 0..4 {
        let (scale_lo, min_lo) = k4_scale_min(chunk * 2, scales);
        let (scale_hi, min_hi) = k4_scale_min(chunk * 2 + 1, scales);
        let d_lo = d * f32::from(scale_lo);
        let m_lo = dmin * f32::from(min_lo);
        let d_hi = d * f32::from(scale_hi);
        let m_hi = dmin * f32::from(min_hi);
        let ql = &qs[chunk * 32..chunk * 32 + 32];
        let base = chunk * 64;
        for l in 0..32 {
            row[base + l] = d_lo * f32::from(ql[l] & 0x0f) - m_lo;
            row[base + 32 + l] = d_hi * f32::from(ql[l] >> 4) - m_hi;
        }
    }
    write_row(out, &row)
}

/// One `Q5_K` super-block: `d`, `dmin`, 12 scale bytes, 32 high bits, 128 low nibbles.
fn dequant_q5_k(block: &[u8], out: &mut [f32]) -> Result<(), InferFailure> {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    if !d.is_finite() || !dmin.is_finite() {
        return Err(non_finite());
    }
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];
    let mut row = [0f32; 256];
    let mut low_bit = 1u8;
    let mut high_bit = 2u8;
    for chunk in 0..4 {
        let (scale_lo, min_lo) = k4_scale_min(chunk * 2, scales);
        let (scale_hi, min_hi) = k4_scale_min(chunk * 2 + 1, scales);
        let d_lo = d * f32::from(scale_lo);
        let m_lo = dmin * f32::from(min_lo);
        let d_hi = d * f32::from(scale_hi);
        let m_hi = dmin * f32::from(min_hi);
        let ql = &qs[chunk * 32..chunk * 32 + 32];
        let base = chunk * 64;
        for l in 0..32 {
            let q_lo = u16::from(ql[l] & 0x0f) + u16::from(qh[l] & low_bit != 0) * 16;
            let q_hi = u16::from(ql[l] >> 4) + u16::from(qh[l] & high_bit != 0) * 16;
            row[base + l] = d_lo * f32::from(q_lo) - m_lo;
            row[base + 32 + l] = d_hi * f32::from(q_hi) - m_hi;
        }
        low_bit <<= 2;
        high_bit <<= 2;
    }
    write_row(out, &row)
}

/// Six-bit scale and min shared by `Q4_K` and `Q5_K`. Group `j` is in `0..8`.
fn k4_scale_min(group: usize, scales: &[u8]) -> (u8, u8) {
    if group < 4 {
        (scales[group] & 63, scales[group + 4] & 63)
    } else {
        let scale = (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4);
        let min = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
        (scale, min)
    }
}

/// One `Q6_K` super-block: 128 low bytes, 64 high bytes, 16 `i8` scales, then `d`.
fn dequant_q6_k(block: &[u8], out: &mut [f32]) -> Result<(), InferFailure> {
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    if !d.is_finite() {
        return Err(non_finite());
    }
    let mut row = [0f32; 256];
    for half in 0..2 {
        let ql = &block[half * 64..half * 64 + 64];
        let qh = &block[128 + half * 32..128 + half * 32 + 32];
        let scales = &block[192 + half * 8..192 + half * 8 + 8];
        for l in 0..32 {
            let group = l / 16;
            let q1 = q6_k_quant(ql[l], qh[l]);
            let q2 = q6_k_quant(ql[l + 32], qh[l] >> 2);
            let q3 = q6_k_quant(ql[l] >> 4, qh[l] >> 4);
            let q4 = q6_k_quant(ql[l + 32] >> 4, qh[l] >> 6);
            let base = half * 128;
            row[base + l] = d * f32::from(scales[group] as i8) * f32::from(q1);
            row[base + l + 32] = d * f32::from(scales[group + 2] as i8) * f32::from(q2);
            row[base + l + 64] = d * f32::from(scales[group + 4] as i8) * f32::from(q3);
            row[base + l + 96] = d * f32::from(scales[group + 6] as i8) * f32::from(q4);
        }
    }
    write_row(out, &row)
}

/// Six-bit code `0..=63` packed as a nibble plus two high bits, then biased by 32.
fn q6_k_quant(low_nibble: u8, high_bits: u8) -> i8 {
    let code = i16::from(low_nibble & 0x0f) | (i16::from(high_bits & 0x03) << 4);
    (code - 32) as i8
}

fn write_row(out: &mut [f32], row: &[f32; 256]) -> Result<(), InferFailure> {
    for (slot, value) in out.iter_mut().zip(row) {
        if !value.is_finite() {
            return Err(non_finite());
        }
        *slot = *value;
    }
    Ok(())
}

fn write_finite(out: &mut [f32], index: usize, value: f32) -> Result<(), InferFailure> {
    if !value.is_finite() {
        return Err(non_finite());
    }
    out[index] = value;
    Ok(())
}

fn non_finite() -> InferFailure {
    fail(
        ErrorCode::ModelImageInvalid,
        "dequant produced a non-finite value",
    )
}

/// IEEE binary16 to binary32. Subnormals shift until bit 10 is set; the stored
/// exponent is then `113 - shift`.
fn f16_to_f32(half: u16) -> f32 {
    let sign = u32::from(half & 0x8000) << 16;
    let exp = u32::from((half & 0x7c00) >> 10);
    let mant = u32::from(half & 0x03ff);
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut frac = mant;
            let mut shift = 0u32;
            while frac & 0x0400 == 0 {
                frac <<= 1;
                shift += 1;
            }
            frac &= 0x03ff;
            sign | ((113 - shift) << 23) | (frac << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}
