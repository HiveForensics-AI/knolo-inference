use infer_artifact::{
    encode_gguf, parse_gguf_bytes, GgufMetadata, GgufTensorDraft, GgufTensorType, GgufValue,
};
use infer_contracts::ErrorCode;
use infer_engine::{
    dequant_gguf, dequant_output_bytes, quant_gemm, quant_gemm_output_bytes, MAX_DEQUANT_BYTES,
};

fn meta_str(key: &str, value: &str) -> GgufMetadata {
    GgufMetadata {
        key: key.into(),
        value: GgufValue::String(value.into()),
    }
}

fn meta_u32(key: &str, value: u32) -> GgufMetadata {
    GgufMetadata {
        key: key.into(),
        value: GgufValue::Uint32(value),
    }
}

fn f16_bytes(bits: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    for value in bits {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn q8_block(scale: u16, quant: &[i8]) -> Vec<u8> {
    let mut out = scale.to_le_bytes().to_vec();
    for value in quant {
        out.push(*value as u8);
    }
    out
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// One `Q4_K` super-block. `scales` is 12 bytes and `qs` is 128.
fn q4_block(d: u16, dmin: u16, scales: [u8; 12], qs: [u8; 128]) -> Vec<u8> {
    let mut out = Vec::with_capacity(144);
    out.extend_from_slice(&d.to_le_bytes());
    out.extend_from_slice(&dmin.to_le_bytes());
    out.extend_from_slice(&scales);
    out.extend_from_slice(&qs);
    out
}

/// One `Q5_K` super-block. `scales` is 12 bytes, `qh` is 32, `qs` is 128.
fn q5_block(d: u16, dmin: u16, scales: [u8; 12], qh: [u8; 32], qs: [u8; 128]) -> Vec<u8> {
    let mut out = Vec::with_capacity(176);
    out.extend_from_slice(&d.to_le_bytes());
    out.extend_from_slice(&dmin.to_le_bytes());
    out.extend_from_slice(&scales);
    out.extend_from_slice(&qh);
    out.extend_from_slice(&qs);
    out
}

/// Pack eight `(scale, min)` pairs. `Q4_K` and `Q5_K` share this scale layout.
fn pack_k4_scales(groups: [(u8, u8); 8]) -> [u8; 12] {
    let mut scales = [0u8; 12];
    for (group, (scale, min)) in groups.into_iter().enumerate() {
        if group < 4 {
            scales[group] = scale;
            scales[group + 4] = min;
        } else {
            scales[group + 4] = (scale & 0x0f) | ((min & 0x0f) << 4);
            scales[group - 4] |= (scale >> 4) << 6;
            scales[group] |= (min >> 4) << 6;
        }
    }
    scales
}

/// One `Q6_K` super-block. `ql` is 128 bytes, `qh` is 64, `scales` is 16 `i8`.
fn q6_block(d: u16, scales: [i8; 16], ql: [u8; 128], qh: [u8; 64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(210);
    out.extend_from_slice(&ql);
    out.extend_from_slice(&qh);
    for scale in scales {
        out.push(scale as u8);
    }
    out.extend_from_slice(&d.to_le_bytes());
    out
}

#[test]
fn f16_matches_the_binary16_table_and_rejects_non_finite() {
    let payload = f16_bytes(&[
        0x0000, 0x8000, 0x0001, 0x8001, 0x0200, 0x03ff, 0x0400, 0x3800, 0x3c00, 0xc000, 0x7bff,
    ]);
    let values = dequant_gguf(GgufTensorType::F16, &payload).unwrap();
    assert_eq!(
        dequant_output_bytes(GgufTensorType::F16, payload.len() as u64).unwrap(),
        values.len() as u64 * 4
    );
    assert_eq!(
        bits(&values),
        vec![
            0x0000_0000,
            0x8000_0000,
            0x3380_0000,
            0xb380_0000,
            0x3800_0000,
            0x387f_c000,
            0x3880_0000,
            0x3f00_0000,
            0x3f80_0000,
            0xc000_0000,
            0x477f_e000,
        ]
    );
    for bits in [0x7c00u16, 0xfc00, 0x7e00] {
        let err = dequant_gguf(GgufTensorType::F16, &f16_bytes(&[bits])).unwrap_err();
        assert_eq!(err.code, ErrorCode::ModelImageInvalid, "{err}");
        assert!(err.message.contains("non-finite"), "{err}");
    }
    let odd = dequant_gguf(GgufTensorType::F16, &[0, 0, 0]).unwrap_err();
    assert_eq!(odd.code, ErrorCode::ModelImageInvalid);
}

#[test]
fn q8_0_is_scale_times_i8_and_does_not_rewrite_the_payload() {
    let mut first = [0i8; 32];
    first[0] = 4;
    first[1] = -2;
    first[2] = 127;
    first[3] = -128;
    let mut second = [0i8; 32];
    second[0] = 3;
    let mut payload = q8_block(0x3800, &first);
    payload.extend_from_slice(&q8_block(0x3c00, &second));
    let original = payload.clone();
    let values = dequant_gguf(GgufTensorType::Q8_0, &payload).unwrap();
    assert_eq!(payload, original);
    assert_eq!(
        dequant_output_bytes(GgufTensorType::Q8_0, payload.len() as u64).unwrap(),
        64 * 4
    );
    assert_eq!(values.len(), 64);
    assert_eq!(bits(&values[..4]), bits(&[2.0, -1.0, 63.5, -64.0]));
    assert!(values[4..32].iter().all(|value| value.to_bits() == 0));
    assert_eq!(values[32].to_bits(), 3.0f32.to_bits());
    assert!(values[33..].iter().all(|value| value.to_bits() == 0));

    let mut infinite = q8_block(0x7c00, &[0i8; 32]);
    infinite[2] = 1;
    let err = dequant_gguf(GgufTensorType::Q8_0, &infinite).unwrap_err();
    assert_eq!(err.code, ErrorCode::ModelImageInvalid, "{err}");
    assert!(err.message.contains("non-finite"), "{err}");
}

#[test]
fn f32_copy_rejects_non_finite_and_the_output_cap_is_the_estimate() {
    let mut payload = Vec::new();
    for value in [1.0f32, -0.5, 0.0] {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let values = dequant_gguf(GgufTensorType::F32, &payload).unwrap();
    assert_eq!(bits(&values), bits(&[1.0, -0.5, 0.0]));
    let mut infinite = payload.clone();
    infinite.extend_from_slice(&f32::INFINITY.to_le_bytes());
    let err = dequant_gguf(GgufTensorType::F32, &infinite).unwrap_err();
    assert!(err.message.contains("non-finite"), "{err}");

    let max_blocks = (MAX_DEQUANT_BYTES / 4) / 32;
    assert_eq!(
        dequant_output_bytes(GgufTensorType::Q8_0, max_blocks * 34).unwrap(),
        MAX_DEQUANT_BYTES
    );
    let over = (max_blocks as usize + 1) * 34;
    let err = dequant_gguf(GgufTensorType::Q8_0, &vec![0u8; over]).unwrap_err();
    assert_eq!(err.code, ErrorCode::InsufficientMemory, "{err}");
    assert!(err.message.contains("32 MiB"), "{err}");
}

#[test]
fn q6_k_is_superblock_scale_times_i8_scale_times_biased_code() {
    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];
    // l = 0: codes 63, 17, 53, 34 → quants 31, -15, 21, 2.
    ql[0] = 0x5f;
    ql[32] = 0x21;
    qh[0] = 0xb7;
    // l = 16: code 16 → quant -16, scale group 1.
    qh[16] = 0x01;
    // Second half, l = 0: code 8 → quant -24, scale index 8.
    ql[64] = 0x08;
    let mut scales = [0i8; 16];
    scales[0] = 2;
    scales[1] = 5;
    scales[2] = -1;
    scales[4] = 3;
    scales[6] = 4;
    scales[7] = -128;
    scales[8] = -2;
    let mut payload = q6_block(0x3c00, scales, ql, qh);
    // Second super-block: every code is 33 (quant 1), every scale is 1, d is 2.
    let ones_ql = [0x11u8; 128];
    let ones_qh = [0xaau8; 64];
    payload.extend(q6_block(0x4000, [1i8; 16], ones_ql, ones_qh));
    let original = payload.clone();
    let values = dequant_gguf(GgufTensorType::Q6_K, &payload).unwrap();
    assert_eq!(payload, original);
    assert_eq!(values.len(), 512);
    assert_eq!(
        dequant_output_bytes(GgufTensorType::Q6_K, payload.len() as u64).unwrap(),
        512 * 4
    );
    assert_eq!(values[0].to_bits(), 62.0f32.to_bits());
    assert_eq!(values[1].to_bits(), (-64.0f32).to_bits());
    assert_eq!(values[16].to_bits(), (-80.0f32).to_bits());
    assert_eq!(values[17].to_bits(), (-160.0f32).to_bits());
    assert_eq!(values[32].to_bits(), 15.0f32.to_bits());
    assert_eq!(values[33].to_bits(), 32.0f32.to_bits());
    assert!(values[48..64].iter().all(|value| *value == 0.0));
    assert_eq!(values[64].to_bits(), 63.0f32.to_bits());
    assert_eq!(values[65].to_bits(), (-96.0f32).to_bits());
    assert_eq!(values[96].to_bits(), 8.0f32.to_bits());
    assert_eq!(values[97].to_bits(), (-128.0f32).to_bits());
    assert_eq!(values[112].to_bits(), 4096.0f32.to_bits());
    assert_eq!(values[113].to_bits(), 4096.0f32.to_bits());
    assert_eq!(values[128].to_bits(), 48.0f32.to_bits());
    assert_eq!(values[129].to_bits(), 64.0f32.to_bits());
    assert!(values[144..256].iter().all(|value| *value == 0.0));
    assert!(values[256..]
        .iter()
        .all(|value| value.to_bits() == 2.0f32.to_bits()));

    let mut sub = [0i8; 16];
    sub[0] = 1;
    let mut sub_ql = [0u8; 128];
    let mut sub_qh = [0u8; 64];
    sub_ql[0] = 0x01;
    sub_qh[0] = 0x02;
    let subnormal =
        dequant_gguf(GgufTensorType::Q6_K, &q6_block(0x0001, sub, sub_ql, sub_qh)).unwrap();
    assert_eq!(subnormal[0].to_bits(), 0x3380_0000);
    assert_eq!(subnormal[1].to_bits(), 0xb600_0000);

    let infinite = q6_block(0x7c00, [1i8; 16], [0u8; 128], [0u8; 64]);
    let err = dequant_gguf(GgufTensorType::Q6_K, &infinite).unwrap_err();
    assert_eq!(err.code, ErrorCode::ModelImageInvalid, "{err}");
    assert!(err.message.contains("non-finite"), "{err}");

    let short = dequant_gguf(GgufTensorType::Q6_K, &vec![0u8; 209]).unwrap_err();
    assert_eq!(short.code, ErrorCode::ModelImageInvalid);

    let q6_block_bytes = GgufTensorType::Q6_K.type_size();
    let q6_elements = GgufTensorType::Q6_K.block_elements();
    let max_q6 = (MAX_DEQUANT_BYTES / 4) / q6_elements;
    assert_eq!(
        dequant_output_bytes(GgufTensorType::Q6_K, max_q6 * q6_block_bytes).unwrap(),
        MAX_DEQUANT_BYTES
    );
    let over = ((max_q6 + 1) * q6_block_bytes) as usize;
    let capped = dequant_gguf(GgufTensorType::Q6_K, &vec![0u8; over]).unwrap_err();
    assert_eq!(capped.code, ErrorCode::InsufficientMemory, "{capped}");
    assert!(capped.message.contains("32 MiB"), "{capped}");
}

#[test]
fn q5_k_is_scale_times_code_minus_min() {
    let scales = pack_k4_scales([
        (2, 1),
        (3, 0),
        (1, 4),
        (0, 2),
        (17, 5),
        (20, 48),
        (63, 0),
        (0, 63),
    ]);
    assert_eq!(
        scales,
        [0x42, 0x43, 0xc1, 0x00, 0x01, 0xc0, 0x04, 0xc2, 0x51, 0x04, 0x0f, 0xf0]
    );
    let mut qh = [0u8; 32];
    let mut qs = [0u8; 128];
    // l = 0 shares one qh byte across the four chunks. Bits 0, 1, 3, 4, and 6 are set.
    qh[0] = 0x5b;
    qs[0] = 0x0f;
    qs[1] = 0x10;
    qs[32] = 0x05;
    qs[64] = 0x41;
    qs[96] = 0x0f;
    let mut payload = q5_block(0x3c00, 0x3800, scales, qh, qs);
    let ones_qs = [0x11u8; 128];
    payload.extend(q5_block(
        0x4000,
        0x0000,
        pack_k4_scales([(1, 0); 8]),
        [0u8; 32],
        ones_qs,
    ));
    let original = payload.clone();
    let values = dequant_gguf(GgufTensorType::Q5_K, &payload).unwrap();
    assert_eq!(payload, original);
    assert_eq!(values.len(), 512);
    assert_eq!(
        dequant_output_bytes(GgufTensorType::Q5_K, payload.len() as u64).unwrap(),
        512 * 4
    );
    assert_eq!(values[0].to_bits(), 61.5f32.to_bits());
    assert_eq!(values[1].to_bits(), (-0.5f32).to_bits());
    assert_eq!(values[2].to_bits(), (-0.5f32).to_bits());
    assert_eq!(values[32].to_bits(), 48.0f32.to_bits());
    assert_eq!(values[33].to_bits(), 3.0f32.to_bits());
    assert_eq!(values[34].to_bits(), 0.0f32.to_bits());
    assert_eq!(values[64].to_bits(), 3.0f32.to_bits());
    assert_eq!(values[66].to_bits(), (-2.0f32).to_bits());
    assert_eq!(values[96].to_bits(), (-1.0f32).to_bits());
    assert_eq!(values[98].to_bits(), (-1.0f32).to_bits());
    assert_eq!(values[128].to_bits(), 286.5f32.to_bits());
    assert_eq!(values[130].to_bits(), (-2.5f32).to_bits());
    assert_eq!(values[160].to_bits(), 56.0f32.to_bits());
    assert_eq!(values[162].to_bits(), (-24.0f32).to_bits());
    assert_eq!(values[192].to_bits(), 1953.0f32.to_bits());
    assert_eq!(values[194].to_bits(), 0.0f32.to_bits());
    assert_eq!(values[224].to_bits(), (-31.5f32).to_bits());
    assert_eq!(values[226].to_bits(), (-31.5f32).to_bits());
    assert!(values[256..]
        .iter()
        .all(|value| value.to_bits() == 2.0f32.to_bits()));

    let mut sub_qs = [0u8; 128];
    sub_qs[0] = 0x01;
    let subnormal = dequant_gguf(
        GgufTensorType::Q5_K,
        &q5_block(
            0x0001,
            0x0000,
            pack_k4_scales([
                (1, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
            ]),
            [0u8; 32],
            sub_qs,
        ),
    )
    .unwrap();
    assert_eq!(subnormal[0].to_bits(), 0x3380_0000);

    let mut neg_qs = [0u8; 128];
    neg_qs[0] = 0x01;
    let negative_zero = dequant_gguf(
        GgufTensorType::Q5_K,
        &q5_block(
            0x8000,
            0x0000,
            pack_k4_scales([
                (1, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
            ]),
            [0u8; 32],
            neg_qs,
        ),
    )
    .unwrap();
    assert_eq!(negative_zero[0].to_bits(), 0x8000_0000);

    let infinite_d = q5_block(0x7c00, 0x0000, [0u8; 12], [0u8; 32], [0u8; 128]);
    let err = dequant_gguf(GgufTensorType::Q5_K, &infinite_d).unwrap_err();
    assert_eq!(err.code, ErrorCode::ModelImageInvalid, "{err}");
    assert!(err.message.contains("non-finite"), "{err}");
    let infinite_min = q5_block(0x3c00, 0x7c00, [0u8; 12], [0u8; 32], [0u8; 128]);
    let err = dequant_gguf(GgufTensorType::Q5_K, &infinite_min).unwrap_err();
    assert!(err.message.contains("non-finite"), "{err}");

    let short = dequant_gguf(GgufTensorType::Q5_K, &[0u8; 175]).unwrap_err();
    assert_eq!(short.code, ErrorCode::ModelImageInvalid);

    let q5_block_bytes = GgufTensorType::Q5_K.type_size();
    let q5_elements = GgufTensorType::Q5_K.block_elements();
    let max_q5 = (MAX_DEQUANT_BYTES / 4) / q5_elements;
    assert_eq!(
        dequant_output_bytes(GgufTensorType::Q5_K, max_q5 * q5_block_bytes).unwrap(),
        MAX_DEQUANT_BYTES
    );
    let over = ((max_q5 + 1) * q5_block_bytes) as usize;
    let capped = dequant_gguf(GgufTensorType::Q5_K, &vec![0u8; over]).unwrap_err();
    assert_eq!(capped.code, ErrorCode::InsufficientMemory, "{capped}");
    assert!(capped.message.contains("32 MiB"), "{capped}");
}

#[test]
fn q4_k_is_scale_times_code_minus_min() {
    let scales = pack_k4_scales([
        (2, 1),
        (3, 0),
        (1, 4),
        (0, 2),
        (17, 5),
        (20, 48),
        (63, 0),
        (0, 63),
    ]);
    assert_eq!(
        scales,
        [0x42, 0x43, 0xc1, 0x00, 0x01, 0xc0, 0x04, 0xc2, 0x51, 0x04, 0x0f, 0xf0]
    );
    let mut qs = [0u8; 128];
    qs[0] = 0x0f;
    qs[1] = 0x10;
    qs[2] = 0xff;
    // Chunk 1, group scale 0: the high nibble is ignored and the min is kept.
    qs[32] = 0xf5;
    qs[64] = 0x41;
    qs[96] = 0x0f;
    let mut payload = q4_block(0x3c00, 0x3800, scales, qs);
    payload.extend(q4_block(
        0x4000,
        0x0000,
        pack_k4_scales([(1, 0); 8]),
        [0x11u8; 128],
    ));
    let original = payload.clone();
    let values = dequant_gguf(GgufTensorType::Q4_K, &payload).unwrap();
    assert_eq!(payload, original);
    assert_eq!(values.len(), 512);
    assert_eq!(
        dequant_output_bytes(GgufTensorType::Q4_K, payload.len() as u64).unwrap(),
        512 * 4
    );
    assert_eq!(values[0].to_bits(), 29.5f32.to_bits());
    assert_eq!(values[1].to_bits(), (-0.5f32).to_bits());
    assert_eq!(values[2].to_bits(), 29.5f32.to_bits());
    assert_eq!(values[3].to_bits(), (-0.5f32).to_bits());
    assert_eq!(values[32].to_bits(), 0.0f32.to_bits());
    assert_eq!(values[33].to_bits(), 3.0f32.to_bits());
    assert_eq!(values[34].to_bits(), 45.0f32.to_bits());
    assert_eq!(values[35].to_bits(), 0.0f32.to_bits());
    assert_eq!(values[64].to_bits(), 3.0f32.to_bits());
    assert_eq!(values[66].to_bits(), (-2.0f32).to_bits());
    assert_eq!(values[96].to_bits(), (-1.0f32).to_bits());
    assert_eq!(values[98].to_bits(), (-1.0f32).to_bits());
    assert_eq!(values[128].to_bits(), 14.5f32.to_bits());
    assert_eq!(values[130].to_bits(), (-2.5f32).to_bits());
    assert_eq!(values[160].to_bits(), 56.0f32.to_bits());
    assert_eq!(values[162].to_bits(), (-24.0f32).to_bits());
    assert_eq!(values[192].to_bits(), 945.0f32.to_bits());
    assert_eq!(values[194].to_bits(), 0.0f32.to_bits());
    assert_eq!(values[224].to_bits(), (-31.5f32).to_bits());
    assert_eq!(values[226].to_bits(), (-31.5f32).to_bits());
    assert!(values[256..]
        .iter()
        .all(|value| value.to_bits() == 2.0f32.to_bits()));

    let mut sub_qs = [0u8; 128];
    sub_qs[0] = 0x01;
    let subnormal = dequant_gguf(
        GgufTensorType::Q4_K,
        &q4_block(
            0x0001,
            0x0000,
            pack_k4_scales([
                (1, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
            ]),
            sub_qs,
        ),
    )
    .unwrap();
    assert_eq!(subnormal[0].to_bits(), 0x3380_0000);

    let mut neg_qs = [0u8; 128];
    neg_qs[0] = 0x01;
    let negative_zero = dequant_gguf(
        GgufTensorType::Q4_K,
        &q4_block(
            0x8000,
            0x0000,
            pack_k4_scales([
                (1, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
            ]),
            neg_qs,
        ),
    )
    .unwrap();
    assert_eq!(negative_zero[0].to_bits(), 0x8000_0000);

    let infinite_d = q4_block(0x7c00, 0x0000, [0u8; 12], [0u8; 128]);
    let err = dequant_gguf(GgufTensorType::Q4_K, &infinite_d).unwrap_err();
    assert_eq!(err.code, ErrorCode::ModelImageInvalid, "{err}");
    assert!(err.message.contains("non-finite"), "{err}");
    let infinite_min = q4_block(0x3c00, 0x7c00, [0u8; 12], [0u8; 128]);
    let err = dequant_gguf(GgufTensorType::Q4_K, &infinite_min).unwrap_err();
    assert!(err.message.contains("non-finite"), "{err}");

    let short = dequant_gguf(GgufTensorType::Q4_K, &[0u8; 143]).unwrap_err();
    assert_eq!(short.code, ErrorCode::ModelImageInvalid);

    let q4_block_bytes = GgufTensorType::Q4_K.type_size();
    let q4_elements = GgufTensorType::Q4_K.block_elements();
    let max_q4 = (MAX_DEQUANT_BYTES / 4) / q4_elements;
    assert_eq!(
        dequant_output_bytes(GgufTensorType::Q4_K, max_q4 * q4_block_bytes).unwrap(),
        MAX_DEQUANT_BYTES
    );
    let over = ((max_q4 + 1) * q4_block_bytes) as usize;
    let capped = dequant_gguf(GgufTensorType::Q4_K, &vec![0u8; over]).unwrap_err();
    assert_eq!(capped.code, ErrorCode::InsufficientMemory, "{capped}");
    assert!(capped.message.contains("32 MiB"), "{capped}");
}

#[test]
fn parsed_q8_and_f16_payloads_dequant_without_changing_type() {
    let mut quant = [0i8; 32];
    quant[0] = 4;
    quant[1] = -2;
    let mut q8 = 0x3800u16.to_le_bytes().to_vec();
    for value in quant {
        q8.push(value as u8);
    }
    let f16 = 0x3c00u16.to_le_bytes().to_vec();
    let metadata = vec![
        meta_str("general.architecture", "micro"),
        meta_u32("general.quantization_version", 2),
    ];
    let tensors = vec![
        GgufTensorDraft {
            name: "q".into(),
            tensor_type: GgufTensorType::Q8_0,
            shape: vec![32],
            bytes: q8.clone(),
        },
        GgufTensorDraft {
            name: "h".into(),
            tensor_type: GgufTensorType::F16,
            shape: vec![1],
            bytes: f16.clone(),
        },
    ];
    let parsed = parse_gguf_bytes(&encode_gguf(&metadata, &tensors).unwrap()).unwrap();
    assert_eq!(parsed.tensors[0].tensor_type, GgufTensorType::Q8_0);
    assert_eq!(parsed.tensors[0].bytes, q8);
    let values = dequant_gguf(parsed.tensors[0].tensor_type, &parsed.tensors[0].bytes).unwrap();
    assert_eq!(
        values.len() as u64,
        parsed.tensors[0]
            .tensor_type
            .elements(&parsed.tensors[0].shape)
            .unwrap()
    );
    assert_eq!(values[0].to_bits(), 2.0f32.to_bits());
    assert_eq!(values[1].to_bits(), (-1.0f32).to_bits());
    let half = dequant_gguf(parsed.tensors[1].tensor_type, &parsed.tensors[1].bytes).unwrap();
    assert_eq!(half[0].to_bits(), 1.0f32.to_bits());
    assert_eq!(parsed.tensors[1].bytes, f16);
}

#[test]
fn parsed_q6_k_keeps_its_type_and_dequants_from_the_copied_payload() {
    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];
    ql[0] = 0x21;
    qh[0] = 0x02;
    let mut scales = [0i8; 16];
    scales[0] = 4;
    let mut bytes = Vec::with_capacity(210);
    bytes.extend_from_slice(&ql);
    bytes.extend_from_slice(&qh);
    for scale in scales {
        bytes.push(scale as u8);
    }
    bytes.extend_from_slice(&0x3c00u16.to_le_bytes());
    let metadata = vec![
        meta_str("general.architecture", "micro"),
        meta_u32("general.quantization_version", 2),
        meta_u32("general.file_type", 14),
    ];
    let tensors = vec![GgufTensorDraft {
        name: "blk.0.attn_q.weight".into(),
        tensor_type: GgufTensorType::Q6_K,
        shape: vec![256],
        bytes: bytes.clone(),
    }];
    let parsed = parse_gguf_bytes(&encode_gguf(&metadata, &tensors).unwrap()).unwrap();
    assert_eq!(parsed.tensors[0].tensor_type, GgufTensorType::Q6_K);
    assert_eq!(parsed.tensors[0].bytes, bytes);
    assert_eq!(parsed.quantization_version, Some(2));
    let values = dequant_gguf(parsed.tensors[0].tensor_type, &parsed.tensors[0].bytes).unwrap();
    assert_eq!(values.len(), 256);
    // code 33 → quant 1, scale 4, d = 1.
    assert_eq!(values[0].to_bits(), 4.0f32.to_bits());
    assert_eq!(parsed.tensors[0].bytes, bytes);
}

#[test]
fn parsed_q5_k_keeps_its_type_and_dequants_from_the_copied_payload() {
    let mut qs = [0u8; 128];
    qs[0] = 0x01;
    let mut scales = [0u8; 12];
    scales[0] = 4;
    let mut bytes = Vec::with_capacity(176);
    bytes.extend_from_slice(&0x3c00u16.to_le_bytes());
    bytes.extend_from_slice(&0x0000u16.to_le_bytes());
    bytes.extend_from_slice(&scales);
    bytes.extend_from_slice(&[0u8; 32]);
    bytes.extend_from_slice(&qs);
    let metadata = vec![
        meta_str("general.architecture", "micro"),
        meta_u32("general.quantization_version", 2),
        meta_u32("general.file_type", 13),
    ];
    let tensors = vec![GgufTensorDraft {
        name: "blk.0.attn_q.weight".into(),
        tensor_type: GgufTensorType::Q5_K,
        shape: vec![256],
        bytes: bytes.clone(),
    }];
    let parsed = parse_gguf_bytes(&encode_gguf(&metadata, &tensors).unwrap()).unwrap();
    assert_eq!(parsed.tensors[0].tensor_type, GgufTensorType::Q5_K);
    assert_eq!(parsed.tensors[0].bytes, bytes);
    assert_eq!(parsed.quantization_version, Some(2));
    let values = dequant_gguf(parsed.tensors[0].tensor_type, &parsed.tensors[0].bytes).unwrap();
    assert_eq!(values.len(), 256);
    // code 1, scale 4, min 0, d = 1.
    assert_eq!(values[0].to_bits(), 4.0f32.to_bits());
    assert_eq!(parsed.tensors[0].bytes, bytes);
}

#[test]
fn parsed_q4_k_keeps_its_type_and_dequants_from_the_copied_payload() {
    let mut qs = [0u8; 128];
    qs[0] = 0x01;
    let mut scales = [0u8; 12];
    scales[0] = 4;
    let mut bytes = Vec::with_capacity(144);
    bytes.extend_from_slice(&0x3c00u16.to_le_bytes());
    bytes.extend_from_slice(&0x0000u16.to_le_bytes());
    bytes.extend_from_slice(&scales);
    bytes.extend_from_slice(&qs);
    let metadata = vec![
        meta_str("general.architecture", "micro"),
        meta_u32("general.quantization_version", 2),
        meta_u32("general.file_type", 12),
    ];
    let tensors = vec![GgufTensorDraft {
        name: "blk.0.attn_q.weight".into(),
        tensor_type: GgufTensorType::Q4_K,
        shape: vec![256],
        bytes: bytes.clone(),
    }];
    let parsed = parse_gguf_bytes(&encode_gguf(&metadata, &tensors).unwrap()).unwrap();
    assert_eq!(parsed.tensors[0].tensor_type, GgufTensorType::Q4_K);
    assert_eq!(parsed.tensors[0].bytes, bytes);
    assert_eq!(parsed.quantization_version, Some(2));
    let values = dequant_gguf(parsed.tensors[0].tensor_type, &parsed.tensors[0].bytes).unwrap();
    assert_eq!(values.len(), 256);
    // code 1, scale 4, min 0, d = 1.
    assert_eq!(values[0].to_bits(), 4.0f32.to_bits());
    assert_eq!(parsed.tensors[0].bytes, bytes);
}

fn reference_product(weight: &[f32], rows: usize, cols: usize, x: &[f32], n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * n];
    for k in 0..n {
        for row in 0..rows {
            let mut acc = 0.0f32;
            for col in 0..cols {
                acc += weight[row * cols + col] * x[k * cols + col];
            }
            y[k * rows + row] = acc;
        }
    }
    y
}

fn assert_product_matches(
    tensor_type: GgufTensorType,
    payload: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    n: usize,
) {
    let original = payload.to_vec();
    let weight = dequant_gguf(tensor_type, payload).unwrap();
    let expected = reference_product(&weight, rows, cols, x, n);
    let got = quant_gemm(tensor_type, payload, rows, cols, x, n).unwrap();
    assert_eq!(payload, &original[..]);
    assert_eq!(got.len(), rows * n);
    assert_eq!(
        quant_gemm_output_bytes(rows, n).unwrap(),
        got.len() as u64 * 4
    );
    assert_eq!(bits(&got), bits(&expected));
}

fn activation(cols: usize, n: usize) -> Vec<f32> {
    (0..cols * n)
        .map(|index| ((index % 7) as f32) - 3.0)
        .collect()
}

#[test]
fn quant_gemm_matches_dequant_then_gemv() {
    let mut f32_payload = Vec::new();
    for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        f32_payload.extend_from_slice(&value.to_le_bytes());
    }
    let f32_x = [1.0f32, 0.0, 0.0, 0.0, 1.0, -1.0];
    let f32_y = quant_gemm(GgufTensorType::F32, &f32_payload, 2, 3, &f32_x, 2).unwrap();
    assert_eq!(bits(&f32_y), bits(&[1.0, 4.0, -1.0, -1.0]));
    assert_product_matches(GgufTensorType::F32, &f32_payload, 2, 3, &f32_x, 2);

    let f16_payload = f16_bytes(&[
        0x3c00, 0xc000, 0x3800, 0x0001, 0x4000, 0x0000, 0xbc00, 0x4200,
    ]);
    assert_product_matches(
        GgufTensorType::F16,
        &f16_payload,
        2,
        4,
        &activation(4, 2),
        2,
    );

    let mut q8_first = [0i8; 32];
    q8_first[0] = 4;
    q8_first[1] = -2;
    q8_first[3] = -128;
    let mut q8_second = [0i8; 32];
    q8_second[0] = 3;
    q8_second[31] = 9;
    let mut q8_payload = q8_block(0x3800, &q8_first);
    q8_payload.extend_from_slice(&q8_block(0x3c00, &q8_second));
    assert_product_matches(
        GgufTensorType::Q8_0,
        &q8_payload,
        2,
        32,
        &activation(32, 2),
        2,
    );

    let q4_scales = pack_k4_scales([
        (2, 1),
        (3, 0),
        (1, 4),
        (0, 2),
        (17, 5),
        (20, 48),
        (63, 0),
        (0, 63),
    ]);
    let mut q4_qs = [0u8; 128];
    q4_qs[0] = 0x0f;
    q4_qs[1] = 0x10;
    q4_qs[2] = 0xff;
    q4_qs[32] = 0xf5;
    q4_qs[64] = 0x41;
    q4_qs[96] = 0x0f;
    let mut q4_payload = q4_block(0x3c00, 0x3800, q4_scales, q4_qs);
    q4_payload.extend(q4_block(
        0x4000,
        0x0000,
        pack_k4_scales([(1, 0); 8]),
        [0x11u8; 128],
    ));
    assert_product_matches(
        GgufTensorType::Q4_K,
        &q4_payload,
        2,
        256,
        &activation(256, 1),
        1,
    );

    let mut q5_payload = q5_block(
        0x3c00,
        0x3800,
        pack_k4_scales([
            (1, 2),
            (0, 1),
            (4, 0),
            (3, 5),
            (17, 0),
            (8, 9),
            (63, 1),
            (2, 63),
        ]),
        [0x01u8; 32],
        [0x1eu8; 128],
    );
    q5_payload.extend(q5_block(
        0x4000,
        0x0000,
        pack_k4_scales([(1, 0); 8]),
        [0u8; 32],
        [0x11u8; 128],
    ));
    assert_product_matches(
        GgufTensorType::Q5_K,
        &q5_payload,
        2,
        256,
        &activation(256, 1),
        1,
    );

    let mut q6_ql = [0u8; 128];
    let mut q6_qh = [0u8; 64];
    q6_ql[0] = 0x5f;
    q6_ql[32] = 0x21;
    q6_qh[0] = 0xb7;
    q6_qh[16] = 0x01;
    let mut q6_scales = [0i8; 16];
    q6_scales[0] = 2;
    q6_scales[1] = -1;
    q6_scales[8] = -2;
    let mut q6_payload = q6_block(0x3c00, q6_scales, q6_ql, q6_qh);
    q6_payload.extend(q6_block(0x4000, [0i8; 16], [0x11u8; 128], [0xaau8; 64]));
    assert_product_matches(
        GgufTensorType::Q6_K,
        &q6_payload,
        2,
        256,
        &activation(256, 1),
        1,
    );
}

#[test]
fn quant_gemm_rejects_a_bad_shape_and_a_non_finite_product() {
    let one = 1.0f32.to_le_bytes();
    let shape = quant_gemm(GgufTensorType::F32, &one, 1, 1, &[], 1).unwrap_err();
    assert_eq!(shape.code, ErrorCode::ContractInvalid, "{shape}");
    assert!(shape.message.contains("shape"), "{shape}");
    let zero_n = quant_gemm(GgufTensorType::F32, &one, 1, 1, &[1.0], 0).unwrap_err();
    assert_eq!(zero_n.code, ErrorCode::ContractInvalid, "{zero_n}");

    let short = quant_gemm(GgufTensorType::F32, &[0, 0, 0], 1, 1, &[1.0], 1).unwrap_err();
    assert_eq!(short.code, ErrorCode::ModelImageInvalid, "{short}");
    assert!(short.message.contains("payload length"), "{short}");

    let dimension = quant_gemm(GgufTensorType::Q8_0, &[0u8; 34], 1, 31, &[], 1).unwrap_err();
    assert_eq!(dimension.code, ErrorCode::ModelImageInvalid, "{dimension}");
    assert!(
        dimension.message.contains("not a multiple of the block"),
        "{dimension}"
    );

    let huge_n = (MAX_DEQUANT_BYTES / 4) as usize + 1;
    let capped = quant_gemm(GgufTensorType::F32, &one, 1, 1, &[], huge_n).unwrap_err();
    assert_eq!(capped.code, ErrorCode::InsufficientMemory, "{capped}");
    assert!(capped.message.contains("32 MiB"), "{capped}");
    let capped_bytes = quant_gemm_output_bytes(1, huge_n).unwrap_err();
    assert_eq!(
        capped_bytes.code,
        ErrorCode::InsufficientMemory,
        "{capped_bytes}"
    );

    let wide = 1.0e30f32.to_le_bytes();
    let overflowed = quant_gemm(GgufTensorType::F32, &wide, 1, 1, &[1.0e30], 1).unwrap_err();
    assert_eq!(overflowed.code, ErrorCode::ContractInvalid, "{overflowed}");
    assert!(overflowed.message.contains("non-finite"), "{overflowed}");
    let nan = quant_gemm(GgufTensorType::F32, &one, 1, 1, &[f32::NAN], 1).unwrap_err();
    assert!(nan.message.contains("non-finite"), "{nan}");

    let half = 0x7c00u16.to_le_bytes();
    let original = half;
    let infinite = quant_gemm(GgufTensorType::F16, &half, 1, 1, &[1.0], 1).unwrap_err();
    assert_eq!(infinite.code, ErrorCode::ModelImageInvalid, "{infinite}");
    assert!(infinite.message.contains("non-finite"), "{infinite}");
    assert_eq!(half, original);
}

#[test]
fn quant_gemm_multiplies_a_matrix_dequant_refuses() {
    let block = q4_block(0x4000, 0x0000, pack_k4_scales([(1, 0); 8]), [0x11u8; 128]);
    let rows = ((MAX_DEQUANT_BYTES / 4) / 256 + 1) as usize;
    assert_eq!(rows, 32_769);
    let mut payload = Vec::with_capacity(rows * block.len());
    for _ in 0..rows {
        payload.extend_from_slice(&block);
    }
    let original = payload.clone();
    let refused = dequant_gguf(GgufTensorType::Q4_K, &payload).unwrap_err();
    assert_eq!(refused.code, ErrorCode::InsufficientMemory, "{refused}");
    let x = vec![1.0f32; 256];
    let y = quant_gemm(GgufTensorType::Q4_K, &payload, rows, 256, &x, 1).unwrap();
    assert_eq!(payload, original);
    assert_eq!(y.len(), rows);
    assert!(y.iter().all(|value| value.to_bits() == 512.0f32.to_bits()));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn quant_gemm_rejects_a_packed_length_that_overflows() {
    let rows = (u64::MAX / 4) as usize + 1;
    let err = quant_gemm(GgufTensorType::F32, &[], rows, 1, &[], 1).unwrap_err();
    assert_eq!(err.code, ErrorCode::ModelImageInvalid, "{err}");
    assert!(err.message.contains("overflows"), "{err}");
}
