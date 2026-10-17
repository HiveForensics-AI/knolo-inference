use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use infer_contracts::ErrorCode;
use infer_engine::{
    argmax, cpu_placement, greedy_generate, llama_cpu_placement, llama_kv_layout,
    load_verified_micro, load_verified_model, logit_margin, micro_kv_layout, write_llama_model,
    write_synthetic_model, ExecutableModel, KvSnapshot, KvStore, ReferenceF32Backend,
    SingleBlockKv, TensorBackend, ADAPTER_ID, FIXTURE_CASES, LLAMA_ADAPTER_ID, LLAMA_FIXTURE_CASES,
    LLAMA_VOCAB, LOGIT_ABS_TOLERANCE, MIN_GREEDY_MARGIN, VOCAB,
};
#[cfg(feature = "cuda")]
use infer_engine::{cuda_placement, llama_cuda_placement, open_paged_kv};
#[cfg(feature = "cuda")]
use infer_native::CandleCudaBackend;
use infer_native::{cpu_adapter_by_id, CandleCpuBackend, CANDLE_CPU_VERSION};

static TEMP: AtomicU64 = AtomicU64::new(0);

fn scratch() -> PathBuf {
    let n = TEMP.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("knolo-infer-candle-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

struct OtherBackend;

impl TensorBackend for OtherBackend {
    fn id(&self) -> &'static str {
        "candle-cuda"
    }

    fn device(&self) -> &'static str {
        "cuda"
    }
}

#[test]
fn candle_cpu_matches_oracle_logits_and_greedy_tokens() {
    assert_eq!(CANDLE_CPU_VERSION, "0.8.4");
    let dir = scratch();
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let placement = cpu_placement(&source).unwrap();
    let adapter = cpu_adapter_by_id(ADAPTER_ID).unwrap();
    let mut oracle = adapter
        .build(&source, &placement, &ReferenceF32Backend)
        .unwrap();
    let mut candle = adapter
        .build(&source, &placement, &CandleCpuBackend)
        .unwrap();
    for fixture in FIXTURE_CASES {
        let mut oracle_kv = SingleBlockKv::new(micro_kv_layout()).unwrap();
        let mut candle_kv = SingleBlockKv::new(micro_kv_layout()).unwrap();
        let oracle_out = greedy_generate(
            oracle.as_mut(),
            &mut oracle_kv,
            1,
            fixture.prompt,
            fixture.new_tokens,
        )
        .unwrap();
        let candle_out = greedy_generate(
            candle.as_mut(),
            &mut candle_kv,
            1,
            fixture.prompt,
            fixture.new_tokens,
        )
        .unwrap();
        let diff = max_abs(&oracle_out.prefill_logits, &candle_out.prefill_logits);
        assert!(
            diff <= LOGIT_ABS_TOLERANCE,
            "{} prefill logit diff {diff} exceeds {LOGIT_ABS_TOLERANCE}",
            fixture.name
        );
        assert_eq!(candle_out.tokens, oracle_out.tokens, "{}", fixture.name);
        assert_eq!(oracle_out.prefill_logits.len(), VOCAB);
        let margin = logit_margin(&oracle_out.prefill_logits).unwrap();
        assert!(margin >= MIN_GREEDY_MARGIN);
        assert_eq!(
            argmax(&candle_out.prefill_logits).unwrap(),
            oracle_out.tokens[0]
        );
        assert_snapshots_close(
            &oracle_kv.snapshot(1).unwrap(),
            &candle_kv.snapshot(1).unwrap(),
            fixture.name,
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn candle_cuda_matches_oracle_logits_and_greedy_tokens() {
    let dir = scratch();
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let placement = cuda_placement(&source).unwrap();
    let adapter = cpu_adapter_by_id(ADAPTER_ID).unwrap();
    let mut oracle = adapter
        .build(
            &source,
            &cpu_placement(&source).unwrap(),
            &ReferenceF32Backend,
        )
        .unwrap();
    let mut candle = adapter
        .build(&source, &placement, &CandleCudaBackend)
        .unwrap();
    for fixture in FIXTURE_CASES {
        let mut oracle_kv = SingleBlockKv::new(micro_kv_layout()).unwrap();
        let mut candle_kv = SingleBlockKv::new(micro_kv_layout()).unwrap();
        let oracle_out = greedy_generate(
            oracle.as_mut(),
            &mut oracle_kv,
            1,
            fixture.prompt,
            fixture.new_tokens,
        )
        .unwrap();
        let candle_out = greedy_generate(
            candle.as_mut(),
            &mut candle_kv,
            1,
            fixture.prompt,
            fixture.new_tokens,
        )
        .unwrap();
        let diff = max_abs(&oracle_out.prefill_logits, &candle_out.prefill_logits);
        assert!(
            diff <= LOGIT_ABS_TOLERANCE,
            "{} prefill logit diff {diff} exceeds {LOGIT_ABS_TOLERANCE}",
            fixture.name
        );
        assert_eq!(candle_out.tokens, oracle_out.tokens, "{}", fixture.name);
        assert_snapshots_close(
            &oracle_kv.snapshot(1).unwrap(),
            &candle_kv.snapshot(1).unwrap(),
            fixture.name,
        );
    }
}

fn assert_snapshots_close(oracle: &KvSnapshot, candle: &KvSnapshot, name: &str) {
    assert_eq!(oracle.len, candle.len);
    assert_eq!(oracle.block_table.len(), 1);
    assert_eq!(candle.block_table.len(), 1);
    let key_diff = max_abs(&oracle.k, &candle.k);
    let value_diff = max_abs(&oracle.v, &candle.v);
    assert!(
        key_diff <= LOGIT_ABS_TOLERANCE && value_diff <= LOGIT_ABS_TOLERANCE,
        "{name} kv diff key {key_diff} value {value_diff}"
    );
}

#[test]
fn candle_adapter_rejects_unknown_architecture_and_cuda() {
    let dir = scratch();
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    assert!(cpu_adapter_by_id("knolo.missing.v1").is_err());
    match cpu_adapter_by_id(ADAPTER_ID).unwrap().build(
        &source,
        &cpu_placement(&source).unwrap(),
        &OtherBackend,
    ) {
        Ok(_) => panic!("candle adapter accepted a cuda backend"),
        Err(err) => assert_eq!(err.code, ErrorCode::UnsupportedKernel),
    }
    let mut model = cpu_adapter_by_id(ADAPTER_ID)
        .unwrap()
        .build(&source, &cpu_placement(&source).unwrap(), &CandleCpuBackend)
        .unwrap();
    let mut kv = SingleBlockKv::new(micro_kv_layout()).unwrap();
    kv.begin_sequence(1).unwrap();
    let err = ExecutableModel::prefill(
        model.as_mut(),
        &infer_engine::PrefillBatch {
            sequence_id: 1,
            token_ids: vec![1; 17],
        },
        &mut kv,
    )
    .unwrap_err();
    assert_eq!(err.code, ErrorCode::ContextLimitExceeded);
    assert_eq!(kv.token_len(1).unwrap(), 0);
}

#[test]
fn candle_cpu_llama_matches_oracle_logits_and_greedy_tokens() {
    let dir = scratch();
    write_llama_model(&dir).unwrap();
    let source = load_verified_model(&dir.join("llama.kmodel"), &dir).unwrap();
    let placement = llama_cpu_placement(&source).unwrap();
    let adapter = cpu_adapter_by_id(LLAMA_ADAPTER_ID).unwrap();
    let mut oracle = adapter
        .build(&source, &placement, &ReferenceF32Backend)
        .unwrap();
    let mut candle = adapter
        .build(&source, &placement, &CandleCpuBackend)
        .unwrap();
    for fixture in LLAMA_FIXTURE_CASES {
        let mut oracle_kv = SingleBlockKv::new(llama_kv_layout()).unwrap();
        let mut candle_kv = SingleBlockKv::new(llama_kv_layout()).unwrap();
        let oracle_out = greedy_generate(
            oracle.as_mut(),
            &mut oracle_kv,
            1,
            fixture.prompt,
            fixture.new_tokens,
        )
        .unwrap();
        let candle_out = greedy_generate(
            candle.as_mut(),
            &mut candle_kv,
            1,
            fixture.prompt,
            fixture.new_tokens,
        )
        .unwrap();
        let diff = max_abs(&oracle_out.prefill_logits, &candle_out.prefill_logits);
        assert!(
            diff <= LOGIT_ABS_TOLERANCE,
            "{} prefill logit diff {diff} exceeds {LOGIT_ABS_TOLERANCE}",
            fixture.name
        );
        assert_eq!(candle_out.tokens, oracle_out.tokens, "{}", fixture.name);
        assert_eq!(oracle_out.prefill_logits.len(), LLAMA_VOCAB);
        assert_snapshots_close(
            &oracle_kv.snapshot(1).unwrap(),
            &candle_kv.snapshot(1).unwrap(),
            fixture.name,
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn wide_llama_on_slot0_matches_the_cpu_oracle() {
    let cases = [
        (
            "f32",
            f32_wide_drafts(),
            &["f32"][..],
            40usize,
            2u32,
            1u32,
            true,
        ),
        (
            "q8_0",
            q8_wide_drafts(),
            &["f32", "q8_0"][..],
            64,
            4,
            4,
            false,
        ),
        (
            "q4_k_m",
            q4_wide_drafts(),
            &["f32", "q4_k_m"][..],
            256,
            8,
            4,
            false,
        ),
    ];
    for (name, drafts, precisions, vocab, heads, kv_heads, check_logits) in cases {
        let dir = scratch();
        write_wide_image(&dir, &drafts, precisions, vocab, heads, kv_heads);
        let source = load_verified_model(&dir.join("wide.kmodel"), &dir).unwrap();
        assert!(!source.llama.unwrap().is_toy(), "{name}");
        let adapter = cpu_adapter_by_id(LLAMA_ADAPTER_ID).unwrap();
        let mut cpu = adapter
            .build(
                &source,
                &llama_cpu_placement(&source).unwrap(),
                &ReferenceF32Backend,
            )
            .unwrap();
        let mut gpu = adapter
            .build(
                &source,
                &llama_cuda_placement(&source).unwrap(),
                &CandleCudaBackend,
            )
            .unwrap();
        let prompt = [1u32, 4, 7, 9];
        let mut cpu_kv = open_paged_kv(&source, cpu.kv_layout()).unwrap();
        let mut gpu_kv = open_paged_kv(&source, gpu.kv_layout()).unwrap();
        let cpu_out = greedy_generate(cpu.as_mut(), &mut cpu_kv, 1, &prompt, 2).unwrap();
        let gpu_out = greedy_generate(gpu.as_mut(), &mut gpu_kv, 1, &prompt, 2).unwrap();
        assert_eq!(gpu_out.tokens, cpu_out.tokens, "{name}");
        if check_logits {
            let diff = max_abs(&cpu_out.prefill_logits, &gpu_out.prefill_logits);
            assert!(
                diff <= LOGIT_ABS_TOLERANCE,
                "{name} prefill logit diff {diff} exceeds {LOGIT_ABS_TOLERANCE}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
struct WideDraft {
    knolo: &'static str,
    ggml: &'static str,
    shape: Vec<u32>,
    dtype: &'static str,
}

#[cfg(feature = "cuda")]
fn f32_wide_drafts() -> Vec<WideDraft> {
    vec![
        WideDraft {
            knolo: "embed.weight",
            ggml: "token_embd.weight",
            shape: vec![40, 8],
            dtype: "f32",
        },
        WideDraft {
            knolo: "layers.0.attn_norm.weight",
            ggml: "blk.0.attn_norm.weight",
            shape: vec![8],
            dtype: "f32",
        },
        WideDraft {
            knolo: "layers.0.attn.q.weight",
            ggml: "blk.0.attn_q.weight",
            shape: vec![8, 8],
            dtype: "f32",
        },
        WideDraft {
            knolo: "layers.0.attn.k.weight",
            ggml: "blk.0.attn_k.weight",
            shape: vec![4, 8],
            dtype: "f32",
        },
        WideDraft {
            knolo: "layers.0.attn.v.weight",
            ggml: "blk.0.attn_v.weight",
            shape: vec![4, 8],
            dtype: "f32",
        },
        WideDraft {
            knolo: "layers.0.attn.o.weight",
            ggml: "blk.0.attn_output.weight",
            shape: vec![8, 8],
            dtype: "f32",
        },
        WideDraft {
            knolo: "layers.0.mlp_norm.weight",
            ggml: "blk.0.ffn_norm.weight",
            shape: vec![8],
            dtype: "f32",
        },
        WideDraft {
            knolo: "layers.0.mlp.gate.weight",
            ggml: "blk.0.ffn_gate.weight",
            shape: vec![16, 8],
            dtype: "f32",
        },
        WideDraft {
            knolo: "layers.0.mlp.up.weight",
            ggml: "blk.0.ffn_up.weight",
            shape: vec![16, 8],
            dtype: "f32",
        },
        WideDraft {
            knolo: "layers.0.mlp.down.weight",
            ggml: "blk.0.ffn_down.weight",
            shape: vec![8, 16],
            dtype: "f32",
        },
        WideDraft {
            knolo: "final_norm.weight",
            ggml: "output_norm.weight",
            shape: vec![8],
            dtype: "f32",
        },
        WideDraft {
            knolo: "lm_head.weight",
            ggml: "output.weight",
            shape: vec![40, 8],
            dtype: "f32",
        },
    ]
}

#[cfg(feature = "cuda")]
fn q8_wide_drafts() -> Vec<WideDraft> {
    let quant = |knolo, ggml, shape| WideDraft {
        knolo,
        ggml,
        shape,
        dtype: "q8_0",
    };
    let norm = |knolo, ggml, shape| WideDraft {
        knolo,
        ggml,
        shape,
        dtype: "f32",
    };
    vec![
        quant("embed.weight", "token_embd.weight", vec![64, 32]),
        norm(
            "layers.0.attn_norm.weight",
            "blk.0.attn_norm.weight",
            vec![32],
        ),
        quant(
            "layers.0.attn.q.weight",
            "blk.0.attn_q.weight",
            vec![32, 32],
        ),
        quant(
            "layers.0.attn.k.weight",
            "blk.0.attn_k.weight",
            vec![32, 32],
        ),
        quant(
            "layers.0.attn.v.weight",
            "blk.0.attn_v.weight",
            vec![32, 32],
        ),
        quant(
            "layers.0.attn.o.weight",
            "blk.0.attn_output.weight",
            vec![32, 32],
        ),
        norm(
            "layers.0.mlp_norm.weight",
            "blk.0.ffn_norm.weight",
            vec![32],
        ),
        quant(
            "layers.0.mlp.gate.weight",
            "blk.0.ffn_gate.weight",
            vec![64, 32],
        ),
        quant(
            "layers.0.mlp.up.weight",
            "blk.0.ffn_up.weight",
            vec![64, 32],
        ),
        quant(
            "layers.0.mlp.down.weight",
            "blk.0.ffn_down.weight",
            vec![32, 64],
        ),
        norm("final_norm.weight", "output_norm.weight", vec![32]),
        quant("lm_head.weight", "output.weight", vec![64, 32]),
    ]
}

#[cfg(feature = "cuda")]
fn q4_wide_drafts() -> Vec<WideDraft> {
    let quant = |knolo, ggml, shape| WideDraft {
        knolo,
        ggml,
        shape,
        dtype: "q4_k_m",
    };
    let norm = |knolo, ggml, shape| WideDraft {
        knolo,
        ggml,
        shape,
        dtype: "f32",
    };
    vec![
        norm("embed.weight", "token_embd.weight", vec![256, 256]),
        norm(
            "layers.0.attn_norm.weight",
            "blk.0.attn_norm.weight",
            vec![256],
        ),
        quant(
            "layers.0.attn.q.weight",
            "blk.0.attn_q.weight",
            vec![256, 256],
        ),
        quant(
            "layers.0.attn.k.weight",
            "blk.0.attn_k.weight",
            vec![128, 256],
        ),
        quant(
            "layers.0.attn.v.weight",
            "blk.0.attn_v.weight",
            vec![128, 256],
        ),
        quant(
            "layers.0.attn.o.weight",
            "blk.0.attn_output.weight",
            vec![256, 256],
        ),
        norm(
            "layers.0.mlp_norm.weight",
            "blk.0.ffn_norm.weight",
            vec![256],
        ),
        quant(
            "layers.0.mlp.gate.weight",
            "blk.0.ffn_gate.weight",
            vec![512, 256],
        ),
        quant(
            "layers.0.mlp.up.weight",
            "blk.0.ffn_up.weight",
            vec![512, 256],
        ),
        quant(
            "layers.0.mlp.down.weight",
            "blk.0.ffn_down.weight",
            vec![256, 512],
        ),
        norm("final_norm.weight", "output_norm.weight", vec![256]),
        quant("lm_head.weight", "output.weight", vec![256, 256]),
    ]
}

#[cfg(feature = "cuda")]
fn write_wide_image(
    dir: &std::path::Path,
    drafts: &[WideDraft],
    precisions: &[&str],
    vocab: usize,
    heads: u32,
    kv_heads: u32,
) {
    use infer_artifact::{
        compile_manifest, encode_gguf, write_model_image, GgufMetadata, GgufTensorDraft,
        GgufTensorType, GgufValue,
    };
    let mut gguf = Vec::new();
    for draft in drafts {
        let rows = draft.shape[0] as usize;
        let cols = draft.shape.get(1).copied().unwrap_or(1) as usize;
        let (tensor_type, bytes) = match draft.dtype {
            "q8_0" => (GgufTensorType::Q8_0, q8_payload(rows, cols)),
            "q4_k_m" => (GgufTensorType::Q4_K, q4_payload(rows, cols)),
            _ => {
                let len = draft
                    .shape
                    .iter()
                    .map(|dim| *dim as usize)
                    .product::<usize>();
                (GgufTensorType::F32, f32_payload(draft.knolo, len))
            }
        };
        let mut shape = draft.shape.clone();
        shape.reverse();
        gguf.push(GgufTensorDraft {
            name: draft.ggml.into(),
            tensor_type,
            shape: shape.into_iter().map(u64::from).collect(),
            bytes,
        });
    }
    let mut metadata = vec![
        GgufMetadata {
            key: "general.architecture".into(),
            value: GgufValue::String("llama".into()),
        },
        GgufMetadata {
            key: "llama.attention.head_count".into(),
            value: GgufValue::Uint32(heads),
        },
        GgufMetadata {
            key: "llama.attention.head_count_kv".into(),
            value: GgufValue::Uint32(kv_heads),
        },
        GgufMetadata {
            key: "llama.rope.freq_base".into(),
            value: GgufValue::Float32(500_000.0f32.to_bits()),
        },
        GgufMetadata {
            key: "llama.attention.layer_norm_rms_eps".into(),
            value: GgufValue::Float32(1e-5f32.to_bits()),
        },
        GgufMetadata {
            key: "llama.context_length".into(),
            value: GgufValue::Uint32(32),
        },
    ];
    if precisions.iter().any(|precision| *precision != "f32") {
        metadata.push(GgufMetadata {
            key: "general.quantization_version".into(),
            value: GgufValue::Uint32(2),
        });
    }
    let bytes = encode_gguf(&metadata, &gguf).unwrap();
    fs::write(dir.join("weights.gguf"), bytes).unwrap();
    let mut tokens = vec![
        "<pad>".to_string(),
        "<bos>".to_string(),
        "<eos>".to_string(),
        "\n".to_string(),
    ];
    while tokens.len() < vocab {
        tokens.push(format!("t{}", tokens.len()));
    }
    let tokenizer = serde_json::json!({
        "kind": "knolo.llama.tokens.v1",
        "tokens": tokens,
        "version": 1
    });
    fs::write(
        dir.join("tokenizer.json"),
        format!("{}\n", serde_json::to_string(&tokenizer).unwrap()),
    )
    .unwrap();
    fs::write(
        dir.join("template.jinja"),
        "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}\n",
    )
    .unwrap();
    let inventory: Vec<_> = drafts
        .iter()
        .map(|draft| serde_json::json!({"dtype": draft.dtype, "name": draft.knolo, "shape": draft.shape}))
        .collect();
    let manifest = serde_json::json!({
        "architecture": {"adapter": "knolo.llama.v1", "family": "llama"},
        "capabilities": ["text-generation"],
        "extensions": {"knolo.llama.context": 32},
        "generationDefaults": {
            "frequencyPenaltyMicros": 0,
            "maxOutputTokens": 4,
            "minPMillionths": 0,
            "presencePenaltyMicros": 0,
            "repetitionPenaltyMicros": 1000000,
            "temperatureMicros": 0,
            "topK": 1,
            "topPMillionths": 1000000
        },
        "kind": "knolo.infer.model-image",
        "license": {"acceptanceRequired": false, "id": "synthetic"},
        "name": "knolo/llama-wide",
        "precisions": precisions,
        "requirements": {"minimumRamBytes": 1048576, "minimumVramBytes": 0},
        "sources": [],
        "specialTokens": {"additional": {}, "bos": 1, "eos": 2},
        "template": {"embedded": "template.jinja"},
        "tensorInventory": inventory,
        "tokenizer": {"embedded": "tokenizer.json"},
        "variant": "wide",
        "version": 1,
        "weights": {"files": [{"path": "weights.gguf"}], "format": "gguf"}
    });
    fs::write(
        dir.join("manifest.json"),
        format!("{}\n", serde_json::to_string_pretty(&manifest).unwrap()),
    )
    .unwrap();
    let compiled = compile_manifest(&dir.join("manifest.json")).unwrap();
    write_model_image(&dir.join("wide.kmodel"), &compiled.bytes).unwrap();
}

#[cfg(feature = "cuda")]
fn f32_payload(name: &str, len: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(len * 4);
    for index in 0..len {
        let value = if name.ends_with("norm.weight") {
            1.0
        } else {
            ((index % 7) as f32 - 3.0) * 0.05
        };
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[cfg(feature = "cuda")]
fn q8_payload(rows: usize, cols: usize) -> Vec<u8> {
    assert!(cols % 32 == 0);
    let mut bytes = Vec::with_capacity(rows * (cols / 32) * 34);
    let scale = 0x3c00u16.to_le_bytes();
    for _row in 0..rows {
        for _block in 0..(cols / 32) {
            bytes.extend_from_slice(&scale);
            for lane in 0..32 {
                bytes.push(((lane % 3) as i8 - 1) as u8);
            }
        }
    }
    bytes
}

#[cfg(feature = "cuda")]
fn q4_payload(rows: usize, cols: usize) -> Vec<u8> {
    assert!(cols % 256 == 0);
    let mut block = [0u8; 144];
    block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    for slot in 0..4 {
        block[4 + slot] = 1;
        block[8 + slot] = 1;
    }
    for lane in 0..128 {
        block[16 + lane] = (lane as u8).wrapping_mul(3);
    }
    let blocks = rows * (cols / 256);
    let mut bytes = Vec::with_capacity(blocks * 144);
    for _ in 0..blocks {
        bytes.extend_from_slice(&block);
    }
    bytes
}

#[cfg(feature = "cuda")]
#[test]
fn candle_cuda_llama_matches_oracle_logits_and_greedy_tokens() {
    let dir = scratch();
    write_llama_model(&dir).unwrap();
    let source = load_verified_model(&dir.join("llama.kmodel"), &dir).unwrap();
    let adapter = cpu_adapter_by_id(LLAMA_ADAPTER_ID).unwrap();
    let mut oracle = adapter
        .build(
            &source,
            &llama_cpu_placement(&source).unwrap(),
            &ReferenceF32Backend,
        )
        .unwrap();
    let mut candle = adapter
        .build(
            &source,
            &llama_cuda_placement(&source).unwrap(),
            &CandleCudaBackend,
        )
        .unwrap();
    for fixture in LLAMA_FIXTURE_CASES {
        let mut oracle_kv = SingleBlockKv::new(llama_kv_layout()).unwrap();
        let mut candle_kv = SingleBlockKv::new(llama_kv_layout()).unwrap();
        let oracle_out = greedy_generate(
            oracle.as_mut(),
            &mut oracle_kv,
            1,
            fixture.prompt,
            fixture.new_tokens,
        )
        .unwrap();
        let candle_out = greedy_generate(
            candle.as_mut(),
            &mut candle_kv,
            1,
            fixture.prompt,
            fixture.new_tokens,
        )
        .unwrap();
        let diff = max_abs(&oracle_out.prefill_logits, &candle_out.prefill_logits);
        assert!(
            diff <= LOGIT_ABS_TOLERANCE,
            "{} prefill logit diff {diff} exceeds {LOGIT_ABS_TOLERANCE}",
            fixture.name
        );
        assert_eq!(candle_out.tokens, oracle_out.tokens, "{}", fixture.name);
        assert_snapshots_close(
            &oracle_kv.snapshot(1).unwrap(),
            &candle_kv.snapshot(1).unwrap(),
            fixture.name,
        );
    }
}
