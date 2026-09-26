use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use infer_artifact::{
    compile_manifest, encode_safetensors, parse_safetensors_bytes, pin_alias, read_lockfile,
    sha256_prefixed, unsupported_pull, verify_image, verify_weights, write_lockfile, ErrorCode,
    InferFailure, TensorBytes, MAX_DOCUMENT_BYTES,
};
use infer_contracts::{
    ArchitectureRefV1, ArtifactFileV1, EmbeddedArtifactV1, FixedPointSamplerV1, LicenseV1,
    ModelArtifactSetV1, ModelImageV1, ResourceRequirementsV1, SpecialTokensV1, TensorSpecV1,
};

static TEMP: AtomicU64 = AtomicU64::new(0);

fn scratch() -> PathBuf {
    let n = TEMP.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("knolo-infer-artifact-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn weight_bytes() -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&1.0f32.to_le_bytes());
    payload.extend_from_slice(&2.0f32.to_le_bytes());
    encode_safetensors(&[TensorBytes {
        name: "w".into(),
        dtype: "f32".into(),
        shape: vec![2],
        bytes: payload,
    }])
    .unwrap()
}

fn sample_image(files: Vec<ArtifactFileV1>, tensors: Vec<TensorSpecV1>) -> ModelImageV1 {
    ModelImageV1 {
        name: "knolo/micro".into(),
        variant: "f32".into(),
        architecture: ArchitectureRefV1 {
            family: "micro".into(),
            adapter: "knolo.micro.v1".into(),
        },
        format: "safetensors".into(),
        files,
        tokenizer: EmbeddedArtifactV1::new("infer-tokenizer", b"tok".to_vec()).unwrap(),
        template: EmbeddedArtifactV1::new("infer-template", b"tpl".to_vec()).unwrap(),
        special_tokens: SpecialTokensV1 {
            bos: Some(1),
            eos: Some(2),
            pad: None,
            unk: None,
            additional: std::collections::BTreeMap::new(),
        },
        generation_defaults: FixedPointSamplerV1 {
            temperature_micros: 0,
            top_p_millionths: 1_000_000,
            min_p_millionths: 0,
            repetition_penalty_micros: 1_000_000,
            presence_penalty_micros: 0,
            frequency_penalty_micros: 0,
            top_k: 1,
            max_output_tokens: 8,
        },
        capabilities: vec!["text-generation".into()],
        license: LicenseV1 {
            id: "synthetic".into(),
            acceptance_required: false,
        },
        sources: Vec::new(),
        tensor_inventory: tensors,
        precisions: vec!["f32".into()],
        requirements: ResourceRequirementsV1 {
            minimum_ram_bytes: 1024,
            minimum_vram_bytes: 0,
        },
        placement_hints: None,
        extensions: std::collections::BTreeMap::new(),
        signatures: Vec::new(),
    }
}

fn code(err: InferFailure) -> ErrorCode {
    err.code
}

#[test]
fn json_and_yaml_manifests_compile_to_the_same_image() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/model-image");
    fs::write(root.join("weights.safetensors"), weight_bytes()).unwrap();
    let from_json = compile_manifest(&root.join("manifest.json")).unwrap();
    let from_yaml = compile_manifest(&root.join("manifest.yaml")).unwrap();
    assert_eq!(from_json.bytes, from_yaml.bytes);
    let verified = verify_image(&from_json.bytes).unwrap();
    assert_eq!(verified.image_root, from_json.image_root);
    assert_eq!(verified.artifact_root, from_json.artifact_root);
    assert_eq!(verified.runtime_root, from_json.runtime_root);
    assert!(!verified.weights_checked);
    verify_weights(&verified.image, &root).unwrap();
    let loaded = infer_artifact::read_verified_tensors(&verified.image, &root).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].name, "w");
    assert_eq!(
        f32::from_le_bytes(loaded[0].bytes[0..4].try_into().unwrap()),
        1.0
    );
    assert_eq!(
        f32::from_le_bytes(loaded[0].bytes[4..8].try_into().unwrap()),
        2.0
    );
    fs::write(root.join("micro.kmodel"), &from_json.bytes).unwrap();
    let expected = format!(
        "{{\n  \"modelImageRoot\": \"{}\",\n  \"artifactRoot\": \"{}\",\n  \"runtimeRoot\": \"{}\",\n  \"weightSha256\": \"{}\",\n  \"kmodelHex\": \"{}\"\n}}\n",
        from_json.image_root,
        from_json.artifact_root,
        from_json.runtime_root,
        verified.image.files[0].sha256,
        infer_contracts::encode_hex(&from_json.bytes)
    );
    fs::write(root.join("expected.json"), expected).unwrap();
    let again = verify_image(&fs::read(root.join("micro.kmodel")).unwrap()).unwrap();
    assert_eq!(again.image_root, from_json.image_root);
}

#[test]
fn truncated_oversized_and_wrong_kind_images_fail() {
    let dir = private_fixture();
    let compiled = compile_manifest(&dir.join("manifest.json")).unwrap();
    assert_eq!(
        code(verify_image(&[]).unwrap_err()),
        ErrorCode::ModelImageInvalid
    );
    assert_eq!(
        code(verify_image(&compiled.bytes[..compiled.bytes.len() / 2]).unwrap_err()),
        ErrorCode::ModelImageInvalid
    );
    assert_eq!(
        code(verify_image(&[0xff]).unwrap_err()),
        ErrorCode::ModelImageInvalid
    );
    let oversized = vec![0u8; MAX_DOCUMENT_BYTES + 1];
    assert_eq!(
        code(verify_image(&oversized).unwrap_err()),
        ErrorCode::ModelImageInvalid
    );
    let other = ModelArtifactSetV1 {
        files: vec![ArtifactFileV1 {
            path: "a.bin".into(),
            size_bytes: 1,
            sha256: sha256_prefixed(b"a"),
        }],
        extensions: std::collections::BTreeMap::new(),
    };
    assert_eq!(
        code(verify_image(&other.to_bytes().unwrap()).unwrap_err()),
        ErrorCode::ModelImageInvalid
    );
    let _ = fs::remove_dir_all(&dir);
}

fn private_fixture() -> PathBuf {
    let dir = scratch();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/model-image");
    fs::write(dir.join("weights.safetensors"), weight_bytes()).unwrap();
    fs::write(
        dir.join("tokenizer.json"),
        fs::read(root.join("tokenizer.json")).unwrap(),
    )
    .unwrap();
    fs::write(
        dir.join("template.jinja"),
        fs::read(root.join("template.jinja")).unwrap(),
    )
    .unwrap();
    fs::write(
        dir.join("manifest.json"),
        fs::read(root.join("manifest.json")).unwrap(),
    )
    .unwrap();
    dir
}

#[test]
fn duplicate_paths_escapes_and_digest_mismatches_fail_closed() {
    let dir = scratch();
    fs::write(dir.join("tokenizer.json"), b"tok").unwrap();
    fs::write(dir.join("template.jinja"), b"tpl").unwrap();
    fs::write(dir.join("weights.safetensors"), weight_bytes()).unwrap();
    let manifest = r#"{
      "kind": "knolo.infer.model-image",
      "version": 1,
      "name": "knolo/micro",
      "variant": "f32",
      "architecture": {"family": "micro", "adapter": "knolo.micro.v1"},
      "weights": {"format": "safetensors", "files": [
        {"path": "weights.safetensors"},
        {"path": "weights.safetensors"}
      ]},
      "tokenizer": {"embedded": "tokenizer.json"},
      "template": {"embedded": "template.jinja"},
      "specialTokens": {"bos": 1, "eos": 2, "additional": {}},
      "generationDefaults": {
        "temperatureMicros": 0, "topPMillionths": 1000000, "minPMillionths": 0,
        "repetitionPenaltyMicros": 1000000, "presencePenaltyMicros": 0,
        "frequencyPenaltyMicros": 0, "topK": 1, "maxOutputTokens": 8
      },
      "capabilities": ["text-generation"],
      "license": {"id": "synthetic", "acceptanceRequired": false},
      "sources": [],
      "tensorInventory": [{"name": "w", "shape": [2], "dtype": "f32"}],
      "precisions": ["f32"],
      "requirements": {"minimumRamBytes": 1024, "minimumVramBytes": 0}
    }"#;
    let manifest_path = dir.join("manifest.json");
    fs::write(&manifest_path, manifest).unwrap();
    let err = compile_manifest(&manifest_path).unwrap_err();
    assert_eq!(err.code, ErrorCode::ModelImageInvalid);
    assert!(err.message.contains("duplicate weight path"));

    let escaped = manifest.replace(
        "{\"path\": \"weights.safetensors\"},\n        {\"path\": \"weights.safetensors\"}",
        "{\"path\": \"../weights.safetensors\"}",
    );
    fs::write(&manifest_path, escaped).unwrap();
    let err = compile_manifest(&manifest_path).unwrap_err();
    assert!(err.message.contains("relative POSIX"), "{err}");

    let good = manifest.replace(
        "{\"path\": \"weights.safetensors\"},\n        {\"path\": \"weights.safetensors\"}",
        "{\"path\": \"weights.safetensors\"}",
    );
    fs::write(&manifest_path, good).unwrap();
    let compiled = compile_manifest(&manifest_path).unwrap();
    let image = verify_image(&compiled.bytes).unwrap().image;
    let mut tampered = weight_bytes();
    let last = tampered.len() - 1;
    tampered[last] ^= 0xff;
    fs::write(dir.join("weights.safetensors"), tampered).unwrap();
    assert_eq!(
        code(verify_weights(&image, &dir).unwrap_err()),
        ErrorCode::ModelDigestMismatch
    );
    verify_image(&compiled.bytes).unwrap();

    let gguf = fs::read_to_string(&manifest_path)
        .unwrap()
        .replace("\"format\": \"safetensors\"", "\"format\": \"gguf\"");
    fs::write(&manifest_path, gguf).unwrap();
    assert_eq!(
        code(compile_manifest(&manifest_path).unwrap_err()),
        ErrorCode::ModelImageInvalid
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn safetensors_inventory_rejects_corruption() {
    let round = encode_safetensors(&[
        TensorBytes {
            name: "a".into(),
            dtype: "u8".into(),
            shape: vec![1],
            bytes: vec![7],
        },
        TensorBytes {
            name: "b".into(),
            dtype: "f32".into(),
            shape: vec![1],
            bytes: 1.0f32.to_le_bytes().to_vec(),
        },
    ])
    .unwrap();
    let parsed = parse_safetensors_bytes(&round).unwrap();
    assert_eq!(parsed[0].name, "a");
    assert_eq!(parsed[0].end - parsed[0].start, 1);
    assert_eq!(parsed[1].start, 8);

    assert_eq!(
        code(parse_safetensors_bytes(b"short").unwrap_err()),
        ErrorCode::ModelImageInvalid
    );
    let f64 = header_file(
        r#"{"w":{"data_offsets":[0,8],"dtype":"F64","shape":[1]}}"#,
        8,
    );
    assert_eq!(
        code(parse_safetensors_bytes(&f64).unwrap_err()),
        ErrorCode::UnsupportedQuantization
    );
    let overlap = header_file(
        r#"{"a":{"data_offsets":[0,4],"dtype":"F32","shape":[1]},"b":{"data_offsets":[0,4],"dtype":"F32","shape":[1]}}"#,
        4,
    );
    assert_eq!(
        code(parse_safetensors_bytes(&overlap).unwrap_err()),
        ErrorCode::ModelImageInvalid
    );
    let overflow = header_file(
        r#"{"w":{"data_offsets":[0,4],"dtype":"F32","shape":[4294967295,4294967295]}}"#,
        4,
    );
    assert_eq!(
        code(parse_safetensors_bytes(&overflow).unwrap_err()),
        ErrorCode::ModelImageInvalid
    );
    let duplicate = header_file(
        r#"{"w":{"data_offsets":[0,4],"dtype":"F32","shape":[1]},"w":{"data_offsets":[0,4],"dtype":"F32","shape":[1]}}"#,
        4,
    );
    assert_eq!(
        code(parse_safetensors_bytes(&duplicate).unwrap_err()),
        ErrorCode::ModelImageInvalid
    );

    let dir = scratch();
    let weights = weight_bytes();
    fs::write(dir.join("w.safetensors"), &weights).unwrap();
    let file = ArtifactFileV1 {
        path: "w.safetensors".into(),
        size_bytes: weights.len() as u64,
        sha256: sha256_prefixed(&weights),
    };
    let image = sample_image(
        vec![file],
        vec![
            TensorSpecV1 {
                name: "w".into(),
                shape: vec![2],
                dtype: "f32".into(),
            },
            TensorSpecV1 {
                name: "q".into(),
                shape: vec![1],
                dtype: "f32".into(),
            },
        ],
    );
    assert_eq!(
        code(verify_weights(&image, &dir).unwrap_err()),
        ErrorCode::ModelArtifactMissing
    );
    let extra = encode_safetensors(&[
        TensorBytes {
            name: "w".into(),
            dtype: "f32".into(),
            shape: vec![1],
            bytes: 1.0f32.to_le_bytes().to_vec(),
        },
        TensorBytes {
            name: "z".into(),
            dtype: "f32".into(),
            shape: vec![1],
            bytes: 1.0f32.to_le_bytes().to_vec(),
        },
    ])
    .unwrap();
    fs::write(dir.join("w.safetensors"), &extra).unwrap();
    let image = sample_image(
        vec![ArtifactFileV1 {
            path: "w.safetensors".into(),
            size_bytes: extra.len() as u64,
            sha256: sha256_prefixed(&extra),
        }],
        vec![TensorSpecV1 {
            name: "w".into(),
            shape: vec![1],
            dtype: "f32".into(),
        }],
    );
    let err = verify_weights(&image, &dir).unwrap_err();
    assert_eq!(err.code, ErrorCode::ModelImageInvalid);
    assert!(err.message.contains("unexpected tensor"));

    fs::write(dir.join("bad.safetensors"), b"not-a-header").unwrap();
    let bad = fs::read(dir.join("bad.safetensors")).unwrap();
    let image = sample_image(
        vec![ArtifactFileV1 {
            path: "bad.safetensors".into(),
            size_bytes: bad.len() as u64,
            sha256: sha256_prefixed(&bad),
        }],
        vec![TensorSpecV1 {
            name: "w".into(),
            shape: vec![1],
            dtype: "f32".into(),
        }],
    );
    assert_eq!(
        code(verify_weights(&image, &dir).unwrap_err()),
        ErrorCode::ModelImageInvalid
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn symlink_cannot_escape_the_weight_directory() {
    let dir = scratch();
    let outside = scratch();
    fs::write(outside.join("secret.safetensors"), weight_bytes()).unwrap();
    std::os::unix::fs::symlink(
        outside.join("secret.safetensors"),
        dir.join("weights.safetensors"),
    )
    .unwrap();
    fs::write(dir.join("tokenizer.json"), b"tok").unwrap();
    fs::write(dir.join("template.jinja"), b"tpl").unwrap();
    let manifest = r#"{
      "kind": "knolo.infer.model-image",
      "version": 1,
      "name": "knolo/micro",
      "variant": "f32",
      "architecture": {"family": "micro", "adapter": "knolo.micro.v1"},
      "weights": {"format": "safetensors", "files": [{"path": "weights.safetensors"}]},
      "tokenizer": {"embedded": "tokenizer.json"},
      "template": {"embedded": "template.jinja"},
      "specialTokens": {"bos": 1, "eos": 2, "additional": {}},
      "generationDefaults": {
        "temperatureMicros": 0, "topPMillionths": 1000000, "minPMillionths": 0,
        "repetitionPenaltyMicros": 1000000, "presencePenaltyMicros": 0,
        "frequencyPenaltyMicros": 0, "topK": 1, "maxOutputTokens": 8
      },
      "capabilities": ["text-generation"],
      "license": {"id": "synthetic", "acceptanceRequired": false},
      "sources": [],
      "tensorInventory": [{"name": "w", "shape": [2], "dtype": "f32"}],
      "precisions": ["f32"],
      "requirements": {"minimumRamBytes": 1024, "minimumVramBytes": 0},
      "trustRemoteCode": true
    }"#;
    let manifest_path = dir.join("manifest.json");
    fs::write(&manifest_path, manifest).unwrap();
    assert_eq!(
        code(compile_manifest(&manifest_path).unwrap_err()),
        ErrorCode::ModelImageInvalid
    );
    let manifest = manifest.replace(",\n      \"trustRemoteCode\": true", "");
    fs::write(&manifest_path, manifest).unwrap();
    let err = compile_manifest(&manifest_path).unwrap_err();
    assert!(
        err.message.contains("escapes") || err.message.contains("not a regular file"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&outside);
}

#[test]
fn lockfile_pin_is_atomic_and_pull_is_refused() {
    let dir = scratch();
    fs::write(dir.join("tokenizer.json"), b"tok").unwrap();
    fs::write(dir.join("template.jinja"), b"tpl").unwrap();
    fs::write(dir.join("weights.safetensors"), weight_bytes()).unwrap();
    let manifest = include_str!("../../../conformance/model-image/manifest.json");
    fs::write(dir.join("manifest.json"), manifest).unwrap();
    fs::write(
        dir.join("tokenizer.json"),
        fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../conformance/model-image/tokenizer.json"),
        )
        .unwrap(),
    )
    .unwrap();
    fs::write(
        dir.join("template.jinja"),
        fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../conformance/model-image/template.jinja"),
        )
        .unwrap(),
    )
    .unwrap();
    let compiled = compile_manifest(&dir.join("manifest.json")).unwrap();
    let image = verify_image(&compiled.bytes).unwrap().image;
    let root = sha256_prefixed(b"engine");
    let mut lock = infer_artifact::new_lockfile(&root);
    pin_alias(&mut lock, "daily", "models/daily.kmodel", &image).unwrap();
    pin_alias(&mut lock, "other", "models/other.kmodel", &image).unwrap();
    let lock_path = dir.join("knolo.infer.lock.json");
    write_lockfile(&lock_path, &lock).unwrap();
    assert!(dir.read_dir().unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".tmp")
    }));
    let loaded = read_lockfile(&lock_path).unwrap().unwrap();
    assert_eq!(loaded.models.len(), 2);
    assert_eq!(loaded.engine.channel, "native");
    assert_eq!(loaded.engine.build_root, root.as_str());
    pin_alias(&mut lock, "daily", "models/daily-v2.kmodel", &image).unwrap();
    write_lockfile(&lock_path, &lock).unwrap();
    let loaded = read_lockfile(&lock_path).unwrap().unwrap();
    assert_eq!(loaded.models.len(), 2);
    assert_eq!(
        loaded.models["daily"].model_image_path,
        "models/daily-v2.kmodel"
    );
    fs::write(
        &lock_path,
        "{\"kind\":\"knolo.infer.lock\",\"kind\":\"x\"}\n",
    )
    .unwrap();
    assert_eq!(
        code(read_lockfile(&lock_path).unwrap_err()),
        ErrorCode::ContractInvalid
    );
    assert!(pin_alias(&mut lock, "Daily", "models/daily.kmodel", &image).is_err());
    assert!(pin_alias(&mut lock, "daily", "/tmp/daily.kmodel", &image).is_err());
    assert_eq!(unsupported_pull().code, ErrorCode::ModelArtifactMissing);
    assert!(unsupported_pull().message.contains("unsupported"));
    let _ = fs::remove_dir_all(&dir);
}

fn header_file(header: &str, data_len: usize) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend(std::iter::repeat_n(0u8, data_len));
    out
}
