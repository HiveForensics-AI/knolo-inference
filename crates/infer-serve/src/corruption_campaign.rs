//! The nine parser probes for the corruption campaign.
//!
//! This module is tests only. `knolo-infer serve` does not call it.

use std::collections::BTreeMap;
use std::io::Cursor;

use infer_artifact::{
    encode_gguf, encode_safetensors, parse_gguf_bytes, parse_safetensors_bytes, verify_image,
    GgufMetadata, GgufTensorDraft, GgufTensorType, GgufValue, TensorBytes,
};
use infer_contracts::{
    fail, sha256_prefixed, ArchitectureRefV1, ArtifactFileV1, CborValue, DigestHex,
    EmbeddedArtifactV1, EngineReceiptBindingV1, ErrorCode, ExecutionReceiptBindingV1,
    FixedPointSamplerV1, HardwareReceiptBindingV1, InferFailure, InferenceReceiptV1, LicenseV1,
    ModelImageV1, ModelReceiptBindingV1, OutputReceiptBindingV1, PlacementReceiptBindingV1,
    PromptReceiptBindingV1, ResourceRequirementsV1, SamplerReceiptBindingV1, SpecialTokensV1,
    TensorSpecV1, TimingReceiptV1, FUZZ_TARGETS,
};
use infer_engine::{
    measure_corruption_fuzz, CorruptionObservation, CorruptionProbe, FuzzSeed, MAX_FUZZ_SEED_BYTES,
};
use infer_prompt::{parse_tokenizer, render_chat_template};
use infer_receipt::verify_receipt_bytes;

use crate::frame::{write_frame, FrameDecoder};
use crate::openai::parse_chat;

use infer_contracts::ChatMessageV1;

struct LiveProbe;

impl CorruptionProbe for LiveProbe {
    fn identify(&self, target: &str, bytes: &[u8]) -> Result<Vec<u8>, InferFailure> {
        match target {
            "cbor" => {
                let value = infer_contracts::decode_canonical(bytes)?;
                Ok(value.to_bytes())
            }
            "kmodel" => {
                let verification = verify_image(bytes)?;
                Ok(verification.image_root.as_str().as_bytes().to_vec())
            }
            "gguf" => identify_gguf(bytes),
            "safetensors" => identify_safetensors(bytes),
            "template" => identify_template(bytes),
            "tokenizer" => {
                let parsed = parse_tokenizer(bytes, 16)?;
                Ok(parsed.tokens.join("\0").into_bytes())
            }
            "api" => identify_api(bytes),
            "ipc" => identify_ipc(bytes),
            "receipt" => {
                let receipt = verify_receipt_bytes(bytes)?;
                Ok(receipt.receipt_id.as_str().as_bytes().to_vec())
            }
            _ => Err(fail(
                ErrorCode::ContractInvalid,
                "field target has an unsupported value",
            )),
        }
    }
}

fn identify_gguf(bytes: &[u8]) -> Result<Vec<u8>, InferFailure> {
    let file = parse_gguf_bytes(bytes)?;
    let mut identity = file.architecture.into_bytes();
    for tensor in file.tensors {
        identity.extend(tensor.name.as_bytes());
        identity.extend_from_slice(&(tensor.tensor_type as u32).to_le_bytes());
        for dim in tensor.shape {
            identity.extend_from_slice(&dim.to_le_bytes());
        }
        identity.extend(tensor.bytes);
    }
    Ok(identity)
}

fn identify_safetensors(bytes: &[u8]) -> Result<Vec<u8>, InferFailure> {
    let views = parse_safetensors_bytes(bytes)?;
    if bytes.len() < 8 {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header is truncated",
        ));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("eight length bytes"));
    let data_at = 8 + header_len as usize;
    if data_at > bytes.len() {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "safetensors header is truncated",
        ));
    }
    let data = &bytes[data_at..];
    let mut views = views;
    views.sort_by(|left, right| left.name.cmp(&right.name));
    let mut identity = Vec::new();
    for view in views {
        identity.extend(view.name.as_bytes());
        identity.extend(view.dtype.as_bytes());
        for dim in view.shape {
            identity.extend_from_slice(&dim.to_le_bytes());
        }
        let start = usize::try_from(view.start).unwrap_or(usize::MAX);
        let end = usize::try_from(view.end).unwrap_or(usize::MAX);
        if end > data.len() || start > end {
            return Err(fail(
                ErrorCode::ModelImageInvalid,
                "safetensors data region does not match the tensor offsets",
            ));
        }
        identity.extend_from_slice(&data[start..end]);
    }
    Ok(identity)
}

fn identify_template(bytes: &[u8]) -> Result<Vec<u8>, InferFailure> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| fail(ErrorCode::TemplateInvalid, "template bytes are not UTF-8"))?;
    let rendered = render_chat_template(
        text,
        &[ChatMessageV1 {
            role: "user".into(),
            content: "hi".into(),
        }],
    )?;
    Ok(rendered.into_bytes())
}

fn identify_api(bytes: &[u8]) -> Result<Vec<u8>, InferFailure> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| fail(ErrorCode::ContractInvalid, "api bytes are not UTF-8"))?;
    let chat = parse_chat(text)?;
    let mut identity = chat.model.into_bytes();
    identity.push(u8::from(chat.stream));
    for turn in chat.messages {
        identity.push(0);
        identity.extend(turn.role.as_bytes());
        identity.push(0);
        identity.extend(turn.content.as_bytes());
    }
    Ok(identity)
}

fn identify_ipc(bytes: &[u8]) -> Result<Vec<u8>, InferFailure> {
    let mut decoder = FrameDecoder::new();
    decoder.push(bytes)?;
    let Some(value) = decoder.pop()? else {
        return Err(fail(ErrorCode::ContractInvalid, "ipc frame is truncated"));
    };
    if decoder.has_unread() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "ipc frame has trailing bytes",
        ));
    }
    Ok(value.to_bytes())
}

fn raw(bytes: &[u8]) -> DigestHex {
    sha256_prefixed(bytes)
}

fn without_trailing_newline(mut bytes: Vec<u8>) -> Vec<u8> {
    if bytes.last() == Some(&b'\n') {
        bytes.pop();
    }
    bytes
}

fn workspace(relative: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative);
    std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn kmodel_seed() -> Vec<u8> {
    let image = ModelImageV1 {
        name: "micro".into(),
        variant: "f32".into(),
        architecture: ArchitectureRefV1 {
            family: "micro".into(),
            adapter: "knolo.micro.v1".into(),
        },
        format: "safetensors".into(),
        files: vec![ArtifactFileV1 {
            path: "weights.safetensors".into(),
            size_bytes: 16,
            sha256: raw(b"weights"),
        }],
        tokenizer: EmbeddedArtifactV1::new("infer-tokenizer", b"tok-v1".to_vec()).unwrap(),
        template: EmbeddedArtifactV1::new("infer-template", b"{{ m }}".to_vec()).unwrap(),
        special_tokens: SpecialTokensV1 {
            bos: Some(1),
            eos: Some(2),
            pad: None,
            unk: None,
            additional: BTreeMap::new(),
        },
        generation_defaults: FixedPointSamplerV1 {
            temperature_micros: 0,
            top_p_millionths: 1_000_000,
            min_p_millionths: 0,
            repetition_penalty_micros: 0,
            presence_penalty_micros: 0,
            frequency_penalty_micros: 0,
            top_k: 0,
            max_output_tokens: 16,
        },
        capabilities: vec!["text-generation".into()],
        license: LicenseV1 {
            id: "test".into(),
            acceptance_required: false,
        },
        sources: vec![],
        tensor_inventory: vec![TensorSpecV1 {
            name: "w".into(),
            shape: vec![2, 2],
            dtype: "f32".into(),
        }],
        precisions: vec!["f32".into()],
        requirements: ResourceRequirementsV1 {
            minimum_ram_bytes: 1024,
            minimum_vram_bytes: 0,
        },
        placement_hints: None,
        extensions: BTreeMap::new(),
        signatures: vec![],
    };
    image.to_bytes().unwrap()
}

fn gguf_seed() -> Vec<u8> {
    encode_gguf(
        &[GgufMetadata {
            key: "general.architecture".into(),
            value: GgufValue::String("micro".into()),
        }],
        &[GgufTensorDraft {
            name: "w".into(),
            tensor_type: GgufTensorType::F32,
            shape: vec![1],
            bytes: 1.0f32.to_le_bytes().to_vec(),
        }],
    )
    .unwrap()
}

fn safetensors_seed() -> Vec<u8> {
    encode_safetensors(&[TensorBytes {
        name: "w".into(),
        dtype: "f32".into(),
        shape: vec![1],
        bytes: 1.0f32.to_le_bytes().to_vec(),
    }])
    .unwrap()
}

fn ipc_seed() -> Vec<u8> {
    let mut bytes = Vec::new();
    write_frame(&mut bytes, &CborValue::Text("ipc".into())).unwrap();
    bytes
}

fn receipt_seed() -> Vec<u8> {
    let mut receipt = InferenceReceiptV1 {
        receipt_id: raw(b"placeholder"),
        intent_root: raw(b"intent"),
        model: ModelReceiptBindingV1 {
            model_image_root: raw(b"image"),
            artifact_root: raw(b"artifact"),
            model_runtime_root: raw(b"runtime"),
            architecture_adapter_id: "knolo.micro.v1".into(),
            architecture_adapter_root: raw(b"adapter"),
            config_root: raw(b"config"),
            tokenizer_root: raw(b"tokenizer"),
            template_root: raw(b"template"),
            storage_precision: "f32".into(),
            compute_precision: "f32".into(),
        },
        engine: EngineReceiptBindingV1 {
            engine_build_root: raw(b"engine"),
            backend: "native".into(),
            backend_version: "0.1.0".into(),
            binary_sha256: raw(b"binary"),
            kernel_bundle_root: raw(b"bundle"),
            kernel_plan_root: raw(b"plan"),
        },
        hardware: HardwareReceiptBindingV1 {
            hardware_root: raw(b"hardware"),
            gpu_model: None,
            compute_capability: None,
            device_slot: "cpu".into(),
        },
        placement: PlacementReceiptBindingV1 {
            placement_root: raw(b"placement"),
            kv_block_size: 16,
            kv_precision: "f32".into(),
        },
        prompt: PromptReceiptBindingV1 {
            messages_root: raw(b"messages"),
            tools_root: None,
            evidence_root: None,
            rendered_text_root: raw(b"rendered"),
            token_id_root: raw(b"tokens"),
            prompt_token_count: 2,
            truncation_root: raw(b"truncation"),
        },
        knowledge: None,
        sampler: SamplerReceiptBindingV1 {
            sampler_plan_root: raw(b"sampler"),
            temperature_micros: 0,
            seed: None,
            rng: "none".into(),
        },
        execution: ExecutionReceiptBindingV1 {
            execution_plan_root: raw(b"execution"),
            scheduling_mode: "isolated".into(),
            prefill_chunk_tokens: 4,
            kv_block_size: 16,
            prefix_cache_hit_tokens: 0,
            batch_trace_root: raw(b"batch"),
            speculative_plan_root: None,
            event_trace_root: raw(b"event"),
            attempt: 0,
        },
        output: OutputReceiptBindingV1 {
            output_token_root: raw(b"output-tokens"),
            output_text_root: raw(b"output-text"),
            token_count: 1,
            finish_reason: "stop".into(),
            structured_output_root: None,
        },
        timing: TimingReceiptV1 {
            queue_micros: 1,
            prefill_micros: 2,
            decode_micros: 3,
            total_micros: 6,
        },
        assurance: "compatibility".into(),
        previous_receipt_root: None,
        extensions: BTreeMap::new(),
        signatures: vec![],
    };
    receipt.receipt_id = receipt.computed_id().unwrap();
    receipt.to_bytes().unwrap()
}

fn campaign_seeds() -> Vec<FuzzSeed> {
    let bodies = [
        ("cbor", CborValue::Text("cbor-seed".into()).to_bytes()),
        ("kmodel", kmodel_seed()),
        ("gguf", gguf_seed()),
        ("safetensors", safetensors_seed()),
        (
            "template",
            workspace("models/micro-transformer/template.jinja"),
        ),
        (
            "tokenizer",
            without_trailing_newline(workspace("models/micro-transformer/tokenizer.json")),
        ),
        (
            "api",
            br#"{"messages":[{"content":"hi","role":"user"}],"model":"micro"}"#.to_vec(),
        ),
        ("ipc", ipc_seed()),
        ("receipt", receipt_seed()),
    ];
    assert_eq!(bodies.len(), FUZZ_TARGETS.len());
    bodies
        .into_iter()
        .zip(FUZZ_TARGETS)
        .map(|((name, bytes), expected)| {
            assert_eq!(name, expected);
            assert!(bytes.len() <= MAX_FUZZ_SEED_BYTES, "{name} is too large");
            FuzzSeed {
                target: name.to_string(),
                label: name.to_string(),
                bytes,
            }
        })
        .collect()
}

#[test]
fn the_nine_parsers_fail_closed_on_the_bounded_campaign() {
    let observed = CorruptionObservation {
        engine_build_root: raw(b"live-fuzz-engine"),
        seeds: campaign_seeds(),
    };
    let probe = LiveProbe;
    let measured = measure_corruption_fuzz(&observed, &probe).unwrap_or_else(|err| {
        panic!("campaign did not fail closed: {err}");
    });
    assert_eq!(measured.report.validation_result, "fail-closed");
    assert_eq!(measured.report.mutation_count, 54);
    assert_eq!(measured.report.accepted_count, 0);
    assert_eq!(
        measured.report.rejected_count + measured.report.distinguished_count,
        54
    );

    let mut trailing = safetensors_seed();
    trailing.push(0xff);
    let err = parse_safetensors_bytes(&trailing).unwrap_err();
    assert!(
        err.message.contains("alignment padding is not zero"),
        "{err}"
    );

    let mut frame = Cursor::new(ipc_seed());
    let value = crate::frame::read_frame(&mut frame).unwrap();
    assert_eq!(value, CborValue::Text("ipc".into()));
}
