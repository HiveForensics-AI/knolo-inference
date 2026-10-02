use std::fs;
use std::path::PathBuf;

use infer_artifact::GgufTensorType;
use infer_contracts::{decode_contract, sha256_prefixed, DigestHex, ErrorCode};
use infer_engine::{
    convert_gguf_tensor, dequant_gguf, reference_engine_build, reference_kernel_bundle,
    verify_gguf_conversion, write_gguf_conversion,
};

fn converter_root() -> DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-convert"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-convert-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn le_f32(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::new();
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

#[test]
fn conversion_matches_dequant_and_does_not_rewrite_the_source() {
    let root = converter_root();
    let cases = [
        (GgufTensorType::F32, vec![2u64], le_f32(&[1.0, -2.0])),
        (
            GgufTensorType::F16,
            vec![1u64],
            0x3c00u16.to_le_bytes().to_vec(),
        ),
        (GgufTensorType::Q8_0, vec![32u64], {
            let mut block = 0x3c00u16.to_le_bytes().to_vec();
            block.extend_from_slice(&[1u8, 2, 255]);
            block.resize(34, 0);
            block
        }),
        (GgufTensorType::Q4_K, vec![256u64], vec![0u8; 144]),
        (GgufTensorType::Q5_K, vec![256u64], vec![0u8; 176]),
        (GgufTensorType::Q6_K, vec![256u64], vec![0u8; 210]),
    ];
    for (tensor_type, shape, payload) in cases {
        let before = payload.clone();
        let converted = convert_gguf_tensor(
            tensor_type,
            &shape,
            &payload,
            "source.gguf",
            "weights.f32",
            &root,
        )
        .unwrap();
        assert_eq!(payload, before);
        let expected = le_f32(&dequant_gguf(tensor_type, &payload).unwrap());
        assert_eq!(converted.destination_bytes, expected);
        assert_eq!(converted.receipt.validation_result, "matched");
        assert_ne!(
            converted.receipt.source_artifact_root,
            converted.receipt.destination_artifact_root
        );
        let again = convert_gguf_tensor(
            tensor_type,
            &shape,
            &payload,
            "source.gguf",
            "weights.f32",
            &root,
        )
        .unwrap();
        assert_eq!(
            again.receipt.to_bytes().unwrap(),
            converted.receipt.to_bytes().unwrap()
        );
        verify_gguf_conversion(&converted).unwrap();
    }
}

#[test]
fn conversion_writes_both_files_and_leaves_the_source_in_place() {
    let dir = scratch("write");
    let root = converter_root();
    let payload = 1.5f32.to_le_bytes();
    fs::write(dir.join("source.gguf"), payload).unwrap();
    let converted = convert_gguf_tensor(
        GgufTensorType::F32,
        &[1],
        &payload,
        "source.gguf",
        "weights.f32",
        &root,
    )
    .unwrap();
    assert!(!dir.join("weights.f32").exists());
    write_gguf_conversion(&dir, &converted, "weights.conversion.cbor").unwrap();
    assert_eq!(fs::read(dir.join("source.gguf")).unwrap(), payload);
    assert_eq!(
        fs::read(dir.join("weights.f32")).unwrap(),
        converted.destination_bytes
    );
    let stored = fs::read(dir.join("weights.conversion.cbor")).unwrap();
    assert_eq!(stored, converted.receipt.to_bytes().unwrap());
    let contract = decode_contract(&stored).unwrap();
    assert_eq!(contract.kind(), "knolo.infer.conversion-receipt");

    let again = write_gguf_conversion(&dir, &converted, "weights.conversion.cbor").unwrap_err();
    assert_eq!(again.code, ErrorCode::ContractInvalid, "{again}");
    assert!(again.message.contains("already exists"), "{again}");
    assert_eq!(
        fs::read(dir.join("weights.f32")).unwrap(),
        converted.destination_bytes
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn conversion_refuses_to_replace_an_existing_output_or_the_source() {
    let dir = scratch("exists");
    let root = converter_root();
    let payload = 1.0f32.to_le_bytes();
    fs::write(dir.join("source.gguf"), payload).unwrap();
    fs::write(dir.join("weights.f32"), [9u8, 9, 9, 9]).unwrap();
    let converted = convert_gguf_tensor(
        GgufTensorType::F32,
        &[1],
        &payload,
        "source.gguf",
        "weights.f32",
        &root,
    )
    .unwrap();
    let err = write_gguf_conversion(&dir, &converted, "weights.conversion.cbor").unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(err.message.contains("already exists"), "{err}");
    assert_eq!(fs::read(dir.join("weights.f32")).unwrap(), [9, 9, 9, 9]);
    assert!(!dir.join("weights.conversion.cbor").exists());
    assert_eq!(fs::read(dir.join("source.gguf")).unwrap(), payload);

    let same = convert_gguf_tensor(
        GgufTensorType::F32,
        &[1],
        &payload,
        "source.gguf",
        "source.gguf",
        &root,
    )
    .unwrap_err();
    assert_eq!(same.code, ErrorCode::ContractInvalid, "{same}");
    assert!(same.message.contains("matches the source path"), "{same}");

    let receipt_is_source = write_gguf_conversion(&dir, &converted, "source.gguf").unwrap_err();
    assert!(
        receipt_is_source
            .message
            .contains("matches an artifact path"),
        "{receipt_is_source}"
    );
    assert_eq!(fs::read(dir.join("source.gguf")).unwrap(), payload);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn conversion_does_not_follow_a_symlink_out_of_the_directory() {
    let dir = scratch("link");
    let root = converter_root();
    let payload = 1.0f32.to_le_bytes();
    let outside_name = format!("knolo-not-created-{}", std::process::id());
    let converted = convert_gguf_tensor(
        GgufTensorType::F32,
        &[1],
        &payload,
        "source.gguf",
        &format!("escape/{outside_name}"),
        &root,
    )
    .unwrap();
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_gguf_conversion(&dir, &converted, "out.conversion.cbor").unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    assert!(!dir.join("out.conversion.cbor").exists());

    std::os::unix::fs::symlink(&outside, dir.join("weights.f32")).unwrap();
    let linked = convert_gguf_tensor(
        GgufTensorType::F32,
        &[1],
        &payload,
        "source.gguf",
        "weights.f32",
        &root,
    )
    .unwrap();
    let err = write_gguf_conversion(&dir, &linked, "out.conversion.cbor").unwrap_err();
    assert!(err.message.contains("already exists"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn conversion_rejects_a_bad_shape_a_non_finite_value_and_a_large_output() {
    let root = converter_root();
    let same_dir =
        convert_gguf_tensor(GgufTensorType::F32, &[1], &[], "/tmp/a", "b.f32", &root).unwrap_err();
    assert!(same_dir.message.contains("relative POSIX"), "{same_dir}");

    let rank = convert_gguf_tensor(
        GgufTensorType::F32,
        &[1, 1, 1, 1, 1],
        &[0, 0, 0, 0],
        "a.gguf",
        "b.f32",
        &root,
    )
    .unwrap_err();
    assert_eq!(rank.code, ErrorCode::ModelImageInvalid, "{rank}");
    assert!(rank.message.contains("1..=4"), "{rank}");

    let block = convert_gguf_tensor(
        GgufTensorType::Q8_0,
        &[31],
        &[0u8; 34],
        "a.gguf",
        "b.f32",
        &root,
    )
    .unwrap_err();
    assert!(
        block.message.contains("not a multiple of the block"),
        "{block}"
    );

    let short = convert_gguf_tensor(
        GgufTensorType::F32,
        &[1],
        &[0, 0, 0],
        "a.gguf",
        "b.f32",
        &root,
    )
    .unwrap_err();
    assert!(short.message.contains("payload length"), "{short}");

    let nan = f32::NAN.to_le_bytes();
    let non_finite =
        convert_gguf_tensor(GgufTensorType::F32, &[1], &nan, "a.gguf", "b.f32", &root).unwrap_err();
    assert_eq!(
        non_finite.code,
        ErrorCode::ModelImageInvalid,
        "{non_finite}"
    );
    assert!(non_finite.message.contains("non-finite"), "{non_finite}");

    let rows = (32 * 1024 * 1024 / 1024) + 1;
    let payload = vec![0u8; rows * 144];
    let capped = convert_gguf_tensor(
        GgufTensorType::Q4_K,
        &[256, rows as u64],
        &payload,
        "a.gguf",
        "b.f32",
        &root,
    )
    .unwrap_err();
    assert_eq!(capped.code, ErrorCode::InsufficientMemory, "{capped}");
    assert!(capped.message.contains("32 MiB"), "{capped}");
}

#[test]
fn a_tampered_destination_is_not_written() {
    let dir = scratch("tamper");
    let root = converter_root();
    let payload = 1.0f32.to_le_bytes();
    let mut converted = convert_gguf_tensor(
        GgufTensorType::F32,
        &[1],
        &payload,
        "source.gguf",
        "weights.f32",
        &root,
    )
    .unwrap();
    converted.destination_bytes[0] ^= 0xff;
    let err = write_gguf_conversion(&dir, &converted, "weights.conversion.cbor").unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message.contains("destination artifact root") || err.message.contains("did not match"),
        "{err}"
    );
    assert!(!dir.join("weights.f32").exists());
    assert!(!dir.join("weights.conversion.cbor").exists());

    converted.destination_bytes[0] ^= 0xff;
    converted.receipt.validation_result = "rejected".into();
    let rejected = write_gguf_conversion(&dir, &converted, "weights.conversion.cbor").unwrap_err();
    assert!(rejected.message.contains("validationResult"), "{rejected}");
    assert!(fs::read_dir(&dir).unwrap().next().is_none());
    let _ = fs::remove_dir_all(&dir);
}
