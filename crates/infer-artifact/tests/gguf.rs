use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use infer_artifact::{
    encode_gguf, hash_regular_file, parse_gguf_bytes, read_verified_gguf, GgufArray, GgufMetadata,
    GgufTensor, GgufTensorDraft, GgufTensorType, GgufValue, GgufValueType,
};
use infer_contracts::{DigestHex, ErrorCode, InferFailure};

static TEMP: AtomicU64 = AtomicU64::new(0);

fn scratch() -> PathBuf {
    let n = TEMP.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("knolo-gguf-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn code_of(err: InferFailure) -> ErrorCode {
    err.code
}

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

fn architecture() -> GgufMetadata {
    meta_str("general.architecture", "micro")
}

fn f16_payload(bits: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    for value in bits {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn f32_payload(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn q4_payload(blocks: usize) -> Vec<u8> {
    vec![0u8; blocks * 144]
}

fn q5_payload(blocks: usize) -> Vec<u8> {
    vec![0u8; blocks * 176]
}

fn q6_payload(blocks: usize) -> Vec<u8> {
    vec![0u8; blocks * 210]
}

fn q8_payload(blocks: &[(u16, [i8; 32])]) -> Vec<u8> {
    let mut out = Vec::new();
    for (scale, quant) in blocks {
        out.extend_from_slice(&scale.to_le_bytes());
        for value in quant {
            out.push(*value as u8);
        }
    }
    out
}

fn draft(
    name: &str,
    tensor_type: GgufTensorType,
    shape: &[u64],
    bytes: Vec<u8>,
) -> GgufTensorDraft {
    GgufTensorDraft {
        name: name.into(),
        tensor_type,
        shape: shape.to_vec(),
        bytes,
    }
}

fn push_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_str(buf: &mut Vec<u8>, text: &str) {
    push_u64(buf, text.len() as u64);
    buf.extend_from_slice(text.as_bytes());
}

fn header(buf: &mut Vec<u8>, tensors: u64, pairs: u64) {
    buf.extend_from_slice(b"GGUF");
    push_u32(buf, 3);
    push_u64(buf, tensors);
    push_u64(buf, pairs);
}

fn kv_str(buf: &mut Vec<u8>, key: &str, value: &str) {
    push_str(buf, key);
    push_u32(buf, 8);
    push_str(buf, value);
}

fn kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    push_str(buf, key);
    push_u32(buf, 4);
    push_u32(buf, value);
}

fn pad_align(buf: &mut Vec<u8>, alignment: usize) {
    while buf.len() % alignment != 0 {
        buf.push(0);
    }
}

fn locate_tensor(bytes: &[u8], name: &str) -> (usize, usize, usize) {
    let mut needle = (name.len() as u64).to_le_bytes().to_vec();
    needle.extend_from_slice(name.as_bytes());
    let at = bytes
        .windows(needle.len())
        .position(|window| window == needle.as_slice())
        .unwrap_or_else(|| panic!("missing tensor {name}"));
    let mut pos = at + needle.len();
    let n_dims = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let dim_at = pos;
    pos += n_dims * 8;
    (pos, pos + 4, dim_at)
}

fn value_offset(bytes: &[u8], key: &str) -> usize {
    let mut needle = (key.len() as u64).to_le_bytes().to_vec();
    needle.extend_from_slice(key.as_bytes());
    let at = bytes
        .windows(needle.len())
        .position(|window| window == needle.as_slice())
        .unwrap_or_else(|| panic!("missing key {key}"));
    at + needle.len() + 4
}

fn put_u32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], at: usize, value: u64) {
    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

fn expect_parse(bytes: &[u8], expected: ErrorCode, fragment: &str) {
    let err = parse_gguf_bytes(bytes).unwrap_err();
    assert_eq!(code_of(err.clone()), expected, "{err}");
    assert!(err.message.contains(fragment), "{err}");
}

fn expect_encode(
    metadata: &[GgufMetadata],
    tensors: &[GgufTensorDraft],
    expected: ErrorCode,
    fragment: &str,
) {
    let err = encode_gguf(metadata, tensors).unwrap_err();
    assert_eq!(code_of(err.clone()), expected, "{err}");
    assert!(err.message.contains(fragment), "{err}");
}

#[test]
fn roundtrip_keeps_types_payloads_and_unexecuted_template() {
    let mut quant = [0i8; 32];
    quant[0] = 4;
    quant[1] = -2;
    quant[31] = 127;
    let template = "{{ payload }} {% raw %}";
    let metadata = vec![
        architecture(),
        meta_u32("general.alignment", 64),
        meta_u32("general.quantization_version", 2),
        meta_u32("general.file_type", 15),
        meta_str("tokenizer.chat_template", template),
        meta_str("general.base_model.0.name", "parent"),
        GgufMetadata {
            key: "knolo.flag".into(),
            value: GgufValue::Bool(true),
        },
        GgufMetadata {
            key: "knolo.bias".into(),
            value: GgufValue::Int16(-2),
        },
        GgufMetadata {
            key: "llama.rope.freq_base".into(),
            value: GgufValue::Float32(10_000.0f32.to_bits()),
        },
        GgufMetadata {
            key: "knolo.tags".into(),
            value: GgufValue::Array(GgufArray {
                element: GgufValueType::String,
                items: Vec::new(),
            }),
        },
    ];
    let f16 = f16_payload(&[0x3c00, 0xc000, 0x3800, 0x0000]);
    let q8 = q8_payload(&[(0x3800, quant), (0x3c00, [0i8; 32])]);
    let f32s = f32_payload(&[1.0, -0.5]);
    let tensors = vec![
        draft(
            "blk.0.attn_q.weight",
            GgufTensorType::F16,
            &[4],
            f16.clone(),
        ),
        draft(
            "blk.0.ffn_down.weight",
            GgufTensorType::Q8_0,
            &[32, 2],
            q8.clone(),
        ),
        draft("token_embd.weight", GgufTensorType::F32, &[2], f32s.clone()),
    ];
    let bytes = encode_gguf(&metadata, &tensors).unwrap();
    let parsed = parse_gguf_bytes(&bytes).unwrap();
    assert_eq!(parsed.version, 3);
    assert_eq!(parsed.alignment, 64);
    assert_eq!(parsed.architecture, "micro");
    assert_eq!(parsed.quantization_version, Some(2));
    assert!(parsed.data_start > parsed.tensor_info_end);
    assert_eq!(parsed.data_start % 64, 0);
    assert_eq!(parsed.metadata, metadata);
    assert_eq!(
        parsed.tensors[0],
        GgufTensor {
            name: "blk.0.attn_q.weight".into(),
            tensor_type: GgufTensorType::F16,
            shape: vec![4],
            offset: parsed.tensors[0].offset,
            bytes: f16,
        }
    );
    assert_eq!(parsed.tensors[1].tensor_type, GgufTensorType::Q8_0);
    assert_eq!(parsed.tensors[1].bytes, q8);
    assert_eq!(parsed.tensors[1].shape, vec![32, 2]);
    assert_eq!(
        parsed.tensors[1]
            .tensor_type
            .elements(&parsed.tensors[1].shape)
            .unwrap(),
        64
    );
    assert_eq!(parsed.tensors[2].bytes, f32s);
    assert_eq!(parsed.tensors[2].tensor_type, GgufTensorType::F32);
    let absolute = parsed.data_start + parsed.tensors[1].offset;
    let start = usize::try_from(absolute).unwrap();
    assert_eq!(
        &bytes[start..start + parsed.tensors[1].bytes.len()],
        parsed.tensors[1].bytes.as_slice()
    );
}

#[test]
fn omitted_alignment_defaults_to_32_and_f16_needs_no_quant_version() {
    let bytes = encode_gguf(
        &[architecture()],
        &[draft(
            "weight",
            GgufTensorType::F16,
            &[2],
            f16_payload(&[0x3c00, 0x0000]),
        )],
    )
    .unwrap();
    let parsed = parse_gguf_bytes(&bytes).unwrap();
    assert_eq!(parsed.alignment, 32);
    assert_eq!(parsed.quantization_version, None);
    assert_eq!(parsed.tensors[0].tensor_type, GgufTensorType::F16);
}

#[test]
fn depth_four_array_roundtrips_and_depth_five_is_rejected() {
    let nested = nest(4);
    let bytes = encode_gguf(
        &[
            architecture(),
            GgufMetadata {
                key: "knolo.nested".into(),
                value: nested.clone(),
            },
        ],
        &[],
    )
    .unwrap();
    let parsed = parse_gguf_bytes(&bytes).unwrap();
    assert_eq!(parsed.metadata[1].value, nested);
    assert!(parsed.tensors.is_empty());
    expect_encode(
        &[
            architecture(),
            GgufMetadata {
                key: "knolo.nested".into(),
                value: nest(5),
            },
        ],
        &[],
        ErrorCode::ModelImageInvalid,
        "too deep",
    );
    expect_parse(&raw_nested(5), ErrorCode::ModelImageInvalid, "too deep");
    let parsed_raw = parse_gguf_bytes(&raw_nested(4)).unwrap();
    assert_eq!(parsed_raw.architecture, "micro");
}

fn nest(depth: usize) -> GgufValue {
    let mut value = GgufValue::Uint8(7);
    for _ in 0..depth {
        let element = match &value {
            GgufValue::Uint8(_) => GgufValueType::Uint8,
            GgufValue::Array(_) => GgufValueType::Array,
            _ => unreachable!("nest only wraps u8"),
        };
        value = GgufValue::Array(GgufArray {
            element,
            items: vec![value],
        });
    }
    value
}

fn raw_nested(depth: usize) -> Vec<u8> {
    let mut body = Vec::new();
    push_u32(&mut body, 0);
    push_u64(&mut body, 1);
    body.push(7);
    for _ in 1..depth {
        let mut wrapped = Vec::new();
        push_u32(&mut wrapped, 9);
        push_u64(&mut wrapped, 1);
        wrapped.extend_from_slice(&body);
        body = wrapped;
    }
    let mut buf = Vec::new();
    header(&mut buf, 0, 2);
    kv_str(&mut buf, "general.architecture", "micro");
    push_str(&mut buf, "knolo.nested");
    push_u32(&mut buf, 9);
    buf.extend_from_slice(&body);
    pad_align(&mut buf, 32);
    buf
}

#[test]
fn digest_is_checked_before_the_header() {
    let bytes = encode_gguf(
        &[architecture()],
        &[draft(
            "weight",
            GgufTensorType::F32,
            &[1],
            f32_payload(&[1.0]),
        )],
    )
    .unwrap();
    let dir = scratch();
    let path = dir.join("model.gguf");
    fs::write(&path, &bytes).unwrap();
    let hashed = hash_regular_file(&path, ErrorCode::ModelDigestMismatch).unwrap();
    let parsed = read_verified_gguf(&path, hashed.size, &hashed.sha256, "model.gguf").unwrap();
    assert_eq!(parsed.tensors[0].bytes, f32_payload(&[1.0]));

    let mut bad = bytes.clone();
    bad[0] = 0;
    let bad_path = dir.join("bad.gguf");
    fs::write(&bad_path, &bad).unwrap();
    let bad_hash = hash_regular_file(&bad_path, ErrorCode::ModelDigestMismatch).unwrap();
    let wrong = flip_digest(&bad_hash.sha256);
    let digest_err = read_verified_gguf(&bad_path, bad_hash.size, &wrong, "bad.gguf").unwrap_err();
    assert_eq!(code_of(digest_err.clone()), ErrorCode::ModelDigestMismatch);
    assert!(digest_err.message.contains("digest"), "{digest_err}");
    assert!(!digest_err.message.contains("magic"), "{digest_err}");
    let magic_err =
        read_verified_gguf(&bad_path, bad_hash.size, &bad_hash.sha256, "bad.gguf").unwrap_err();
    assert_eq!(code_of(magic_err.clone()), ErrorCode::ModelImageInvalid);
    assert!(magic_err.message.contains("magic"), "{magic_err}");

    let size_err =
        read_verified_gguf(&path, hashed.size + 1, &hashed.sha256, "model.gguf").unwrap_err();
    assert_eq!(code_of(size_err.clone()), ErrorCode::ModelDigestMismatch);
    assert!(size_err.message.contains("size"), "{size_err}");
    let missing = read_verified_gguf(
        &dir.join("nope.gguf"),
        hashed.size,
        &hashed.sha256,
        "nope.gguf",
    )
    .unwrap_err();
    assert_eq!(code_of(missing), ErrorCode::ModelArtifactMissing);

    let link = dir.join("link.gguf");
    std::os::unix::fs::symlink(&path, &link).unwrap();
    let link_err = read_verified_gguf(&link, hashed.size, &hashed.sha256, "link.gguf").unwrap_err();
    assert_eq!(code_of(link_err.clone()), ErrorCode::ModelDigestMismatch);
    assert!(link_err.message.contains("size"), "{link_err}");
    let _ = fs::remove_dir_all(&dir);
}

fn flip_digest(digest: &DigestHex) -> DigestHex {
    let mut chars = digest.as_str().as_bytes().to_vec();
    let last = chars.len() - 1;
    chars[last] = if chars[last] == b'a' { b'b' } else { b'a' };
    DigestHex::parse(std::str::from_utf8(&chars).unwrap()).unwrap()
}

#[test]
fn corruption_fails_closed() {
    let bytes = encode_gguf(
        &[architecture(), meta_u32("general.alignment", 64)],
        &[
            draft("wa", GgufTensorType::F32, &[1], f32_payload(&[1.0])),
            draft("wb", GgufTensorType::F32, &[1], f32_payload(&[2.0])),
        ],
    )
    .unwrap();
    let parsed = parse_gguf_bytes(&bytes).unwrap();
    assert!(parsed.data_start > parsed.tensor_info_end);

    let mut magic = bytes.clone();
    magic[0] = 0;
    expect_parse(&magic, ErrorCode::ModelImageInvalid, "magic");

    let mut version = bytes.clone();
    put_u32(&mut version, 4, 2);
    expect_parse(&version, ErrorCode::ModelImageInvalid, "version");
    expect_parse(&bytes[..10], ErrorCode::ModelImageInvalid, "truncated");

    let mut upper = bytes.clone();
    let key = b"general.architecture";
    let key_at = upper
        .windows(key.len())
        .position(|window| window == key)
        .unwrap();
    upper[key_at + key.len() - 1] = b'E';
    expect_parse(&upper, ErrorCode::ModelImageInvalid, "key is invalid");

    let mut utf8 = bytes.clone();
    let micro = utf8
        .windows(5)
        .position(|window| window == b"micro")
        .unwrap();
    utf8[micro + 4] = 0xff;
    expect_parse(&utf8, ErrorCode::ModelImageInvalid, "utf-8");

    let mut nul = bytes.clone();
    nul[micro] = 0;
    expect_parse(&nul, ErrorCode::ModelImageInvalid, "nul");

    let mut pad = bytes.clone();
    let at = usize::try_from(parsed.data_start).unwrap() - 1;
    assert_eq!(pad[at], 0);
    pad[at] = 1;
    expect_parse(&pad, ErrorCode::ModelImageInvalid, "alignment padding");

    let mut gap = bytes.clone();
    let gap_at = usize::try_from(parsed.data_start + parsed.tensors[0].bytes.len() as u64).unwrap();
    assert_eq!(gap[gap_at], 0);
    gap[gap_at] = 1;
    expect_parse(&gap, ErrorCode::ModelImageInvalid, "tensor padding");

    let mut trailing = bytes.clone();
    trailing.push(1);
    expect_parse(&trailing, ErrorCode::ModelImageInvalid, "tensor padding");
    let mut zero_tail = bytes.clone();
    zero_tail.push(0);
    parse_gguf_bytes(&zero_tail).unwrap();

    let mut short = bytes.clone();
    short.pop();
    expect_parse(&short, ErrorCode::ModelImageInvalid, "outside the file");

    let (type_at, offset_at, _) = locate_tensor(&bytes, "wb");
    for type_id in [2u32, 4, 10, 11, 15, 30, 39, 99] {
        let mut patched = bytes.clone();
        put_u32(&mut patched, type_at, type_id);
        expect_parse(
            &patched,
            ErrorCode::UnsupportedQuantization,
            "not in the allowlist",
        );
    }

    let mut unaligned = bytes.clone();
    put_u64(&mut unaligned, offset_at, 8);
    expect_parse(&unaligned, ErrorCode::ModelImageInvalid, "not aligned");

    let mut overlap = bytes.clone();
    put_u64(&mut overlap, offset_at, 0);
    expect_parse(&overlap, ErrorCode::ModelImageInvalid, "overlap");

    let mut names = bytes.clone();
    let mut needle = 2u64.to_le_bytes().to_vec();
    needle.extend_from_slice(b"wb");
    let name_at = names
        .windows(needle.len())
        .position(|window| window == needle.as_slice())
        .unwrap();
    names[name_at + needle.len() - 1] = b'a';
    expect_parse(
        &names,
        ErrorCode::ModelImageInvalid,
        "duplicate gguf tensor",
    );

    let mut flag = encode_gguf(
        &[
            architecture(),
            GgufMetadata {
                key: "knolo.flag".into(),
                value: GgufValue::Bool(true),
            },
        ],
        &[],
    )
    .unwrap();
    let flag_at = value_offset(&flag, "knolo.flag");
    flag[flag_at] = 2;
    expect_parse(&flag, ErrorCode::ModelImageInvalid, "boolean");

    let mut quant = encode_gguf(
        &[architecture(), meta_u32("general.quantization_version", 2)],
        &[draft(
            "weight",
            GgufTensorType::Q8_0,
            &[32],
            q8_payload(&[(0x3c00, [1i8; 32])]),
        )],
    )
    .unwrap();
    let quant_at = value_offset(&quant, "general.quantization_version");
    put_u32(&mut quant, quant_at, 1);
    expect_parse(
        &quant,
        ErrorCode::UnsupportedQuantization,
        "version is not 2",
    );
    let (_, _, dim_at) = locate_tensor(&quant, "weight");
    put_u32(&mut quant, quant_at, 2);
    put_u64(&mut quant, dim_at, 16);
    expect_parse(
        &quant,
        ErrorCode::ModelImageInvalid,
        "not a multiple of the block",
    );

    let mut huge = Vec::new();
    header(&mut huge, 0, 2);
    kv_str(&mut huge, "general.architecture", "micro");
    push_str(&mut huge, "knolo.values");
    push_u32(&mut huge, 9);
    push_u32(&mut huge, 0);
    push_u64(&mut huge, 5);
    huge.push(1);
    expect_parse(&huge, ErrorCode::ModelImageInvalid, "truncated");

    let mut duplicate = Vec::new();
    header(&mut duplicate, 0, 2);
    kv_str(&mut duplicate, "general.architecture", "micro");
    kv_str(&mut duplicate, "general.architecture", "other");
    pad_align(&mut duplicate, 32);
    expect_parse(
        &duplicate,
        ErrorCode::ModelImageInvalid,
        "duplicate gguf metadata key",
    );

    let mut align = Vec::new();
    header(&mut align, 0, 2);
    kv_str(&mut align, "general.architecture", "micro");
    kv_u32(&mut align, "general.alignment", 4);
    expect_parse(&align, ErrorCode::ModelImageInvalid, "multiple of 8");

    let mut missing_arch = Vec::new();
    header(&mut missing_arch, 0, 1);
    kv_u32(&mut missing_arch, "general.alignment", 32);
    expect_parse(
        &missing_arch,
        ErrorCode::ModelImageInvalid,
        "missing general.architecture",
    );

    let mut bare_q8 = Vec::new();
    header(&mut bare_q8, 1, 1);
    kv_str(&mut bare_q8, "general.architecture", "micro");
    push_str(&mut bare_q8, "weight");
    push_u32(&mut bare_q8, 1);
    push_u64(&mut bare_q8, 32);
    push_u32(&mut bare_q8, 8);
    push_u64(&mut bare_q8, 0);
    expect_parse(
        &bare_q8,
        ErrorCode::ModelImageInvalid,
        "requires general.quantization_version",
    );
}

#[test]
fn encoder_rejects_the_same_bounds() {
    expect_encode(
        &[architecture(), architecture()],
        &[],
        ErrorCode::ModelImageInvalid,
        "duplicate gguf metadata key",
    );
    expect_encode(
        &[],
        &[],
        ErrorCode::ModelImageInvalid,
        "missing general.architecture",
    );
    expect_encode(
        &[meta_str("general.architecture", "Llama")],
        &[],
        ErrorCode::ModelImageInvalid,
        "general.architecture is invalid",
    );
    let accepted = encode_gguf(&[architecture(), meta_u32("general.alignment", 24)], &[]).unwrap();
    assert_eq!(parse_gguf_bytes(&accepted).unwrap().alignment, 24);
    expect_encode(
        &[architecture(), meta_u32("general.alignment", 4)],
        &[],
        ErrorCode::ModelImageInvalid,
        "multiple of 8",
    );
    expect_encode(
        &[meta_str("Architecture", "micro")],
        &[],
        ErrorCode::ModelImageInvalid,
        "key is invalid",
    );
    expect_encode(
        &[meta_str("architecture", "micro")],
        &[],
        ErrorCode::ModelImageInvalid,
        "key is invalid",
    );
    let long_name = "n".repeat(65);
    expect_encode(
        &[architecture()],
        &[draft(
            &long_name,
            GgufTensorType::F32,
            &[1],
            f32_payload(&[1.0]),
        )],
        ErrorCode::ModelImageInvalid,
        "exceeds 64 bytes",
    );
    expect_encode(
        &[architecture()],
        &[
            draft("weight", GgufTensorType::F32, &[1], f32_payload(&[1.0])),
            draft("weight", GgufTensorType::F32, &[1], f32_payload(&[1.0])),
        ],
        ErrorCode::ModelImageInvalid,
        "duplicate gguf tensor",
    );
    expect_encode(
        &[architecture()],
        &[draft(
            "weight",
            GgufTensorType::Q8_0,
            &[32],
            q8_payload(&[(0x3c00, [0i8; 32])]),
        )],
        ErrorCode::ModelImageInvalid,
        "requires general.quantization_version",
    );
    expect_encode(
        &[architecture(), meta_u32("general.quantization_version", 1)],
        &[draft(
            "weight",
            GgufTensorType::Q8_0,
            &[32],
            q8_payload(&[(0x3c00, [0i8; 32])]),
        )],
        ErrorCode::UnsupportedQuantization,
        "version is not 2",
    );
    expect_encode(
        &[architecture(), meta_u32("general.quantization_version", 2)],
        &[draft(
            "weight",
            GgufTensorType::Q8_0,
            &[16, 2],
            q8_payload(&[(0x3c00, [0i8; 32])]),
        )],
        ErrorCode::ModelImageInvalid,
        "not a multiple of the block",
    );
    let q6 = q6_payload(1);
    let q6_file = encode_gguf(
        &[
            architecture(),
            meta_u32("general.quantization_version", 2),
            meta_u32("general.file_type", 14),
        ],
        &[
            draft("q6", GgufTensorType::Q6_K, &[256], q6.clone()),
            draft("bias", GgufTensorType::F32, &[1], f32_payload(&[1.0])),
        ],
    )
    .unwrap();
    let parsed_q6 = parse_gguf_bytes(&q6_file).unwrap();
    assert_eq!(parsed_q6.tensors[0].tensor_type, GgufTensorType::Q6_K);
    assert_eq!(parsed_q6.tensors[0].bytes, q6);
    assert_eq!(parsed_q6.tensors[0].offset, 0);
    assert_eq!(parsed_q6.tensors[1].offset, 224);
    assert_eq!(parsed_q6.tensors[1].tensor_type, GgufTensorType::F32);
    expect_encode(
        &[architecture()],
        &[draft("weight", GgufTensorType::Q6_K, &[256], q6_payload(1))],
        ErrorCode::ModelImageInvalid,
        "requires general.quantization_version",
    );
    expect_encode(
        &[architecture(), meta_u32("general.quantization_version", 1)],
        &[draft("weight", GgufTensorType::Q6_K, &[256], q6_payload(1))],
        ErrorCode::UnsupportedQuantization,
        "version is not 2",
    );
    expect_encode(
        &[architecture(), meta_u32("general.quantization_version", 2)],
        &[draft("weight", GgufTensorType::Q6_K, &[128], q6_payload(1))],
        ErrorCode::ModelImageInvalid,
        "not a multiple of the block",
    );
    let q5 = q5_payload(1);
    let q5_file = encode_gguf(
        &[
            architecture(),
            meta_u32("general.quantization_version", 2),
            meta_u32("general.file_type", 13),
        ],
        &[
            draft("q5", GgufTensorType::Q5_K, &[256], q5.clone()),
            draft("bias", GgufTensorType::F32, &[1], f32_payload(&[1.0])),
        ],
    )
    .unwrap();
    let parsed_q5 = parse_gguf_bytes(&q5_file).unwrap();
    assert_eq!(parsed_q5.tensors[0].tensor_type, GgufTensorType::Q5_K);
    assert_eq!(parsed_q5.tensors[0].bytes, q5);
    assert_eq!(parsed_q5.tensors[0].offset, 0);
    assert_eq!(parsed_q5.tensors[1].offset, 192);
    assert_eq!(parsed_q5.tensors[1].tensor_type, GgufTensorType::F32);
    expect_encode(
        &[architecture()],
        &[draft("weight", GgufTensorType::Q5_K, &[256], q5_payload(1))],
        ErrorCode::ModelImageInvalid,
        "requires general.quantization_version",
    );
    expect_encode(
        &[architecture(), meta_u32("general.quantization_version", 1)],
        &[draft("weight", GgufTensorType::Q5_K, &[256], q5_payload(1))],
        ErrorCode::UnsupportedQuantization,
        "version is not 2",
    );
    expect_encode(
        &[architecture(), meta_u32("general.quantization_version", 2)],
        &[draft("weight", GgufTensorType::Q5_K, &[128], q5_payload(1))],
        ErrorCode::ModelImageInvalid,
        "not a multiple of the block",
    );
    let q4 = q4_payload(1);
    let q4_file = encode_gguf(
        &[
            architecture(),
            meta_u32("general.quantization_version", 2),
            meta_u32("general.file_type", 12),
        ],
        &[
            draft("q4", GgufTensorType::Q4_K, &[256], q4.clone()),
            draft("bias", GgufTensorType::F32, &[1], f32_payload(&[1.0])),
        ],
    )
    .unwrap();
    let parsed_q4 = parse_gguf_bytes(&q4_file).unwrap();
    assert_eq!(parsed_q4.tensors[0].tensor_type, GgufTensorType::Q4_K);
    assert_eq!(parsed_q4.tensors[0].bytes, q4);
    assert_eq!(parsed_q4.tensors[0].offset, 0);
    assert_eq!(parsed_q4.tensors[1].offset, 160);
    assert_eq!(parsed_q4.tensors[1].tensor_type, GgufTensorType::F32);
    expect_encode(
        &[architecture()],
        &[draft("weight", GgufTensorType::Q4_K, &[256], q4_payload(1))],
        ErrorCode::ModelImageInvalid,
        "requires general.quantization_version",
    );
    expect_encode(
        &[architecture(), meta_u32("general.quantization_version", 1)],
        &[draft("weight", GgufTensorType::Q4_K, &[256], q4_payload(1))],
        ErrorCode::UnsupportedQuantization,
        "version is not 2",
    );
    expect_encode(
        &[architecture(), meta_u32("general.quantization_version", 2)],
        &[draft("weight", GgufTensorType::Q4_K, &[128], q4_payload(1))],
        ErrorCode::ModelImageInvalid,
        "not a multiple of the block",
    );
    expect_encode(
        &[architecture()],
        &[draft("weight", GgufTensorType::F32, &[1], vec![1, 2, 3])],
        ErrorCode::ModelImageInvalid,
        "payload length",
    );
    let mut wide = architecture();
    wide.key = format!("general.{}", "a".repeat(70_000));
    expect_encode(&[wide], &[], ErrorCode::ModelImageInvalid, "key is invalid");
}
