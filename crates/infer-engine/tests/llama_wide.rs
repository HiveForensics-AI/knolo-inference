//! A Llama inventory wider than llama-tiny, loaded from GGUF metadata.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use infer_artifact::{
    compile_manifest, encode_gguf, write_model_image, GgufMetadata, GgufTensorDraft, GgufTensorType,
    GgufValue,
};
use infer_contracts::ErrorCode;
use infer_engine::{
    greedy_generate, llama_cpu_placement, llama_cuda_placement, load_verified_model,
    open_paged_kv, ArchitectureAdapter, LlamaAdapter, ReferenceF32Backend,
};

static TEMP: AtomicU64 = AtomicU64::new(0);

fn scratch() -> PathBuf {
    let n = TEMP.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("knolo-infer-wide-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
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

fn meta_f32(key: &str, value: f32) -> GgufMetadata {
    GgufMetadata {
        key: key.into(),
        value: GgufValue::Float32(value.to_bits()),
    }
}

fn f32_bytes(name: &str, len: usize) -> Vec<u8> {
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

fn q8_bytes(rows: usize, cols: usize) -> Vec<u8> {
    assert!(cols % 32 == 0);
    let mut bytes = Vec::with_capacity(rows * (cols / 32) * 34);
    let scale = 0x3c00u16.to_le_bytes();
    for _row in 0..rows {
        for _block in 0..(cols / 32) {
            bytes.extend_from_slice(&scale);
            for lane in 0..32 {
                let quant = (lane % 3) as i8 - 1;
                bytes.push(quant as u8);
            }
        }
    }
    bytes
}

struct TensorDraft {
    knolo: &'static str,
    ggml: &'static str,
    shape: Vec<u32>,
    dtype: &'static str,
}

fn write_image(dir: &Path, drafts: &[TensorDraft], precisions: &[&str], vocab: usize) {
    let mut gguf = Vec::new();
    for draft in drafts {
        let len = draft.shape.iter().map(|dim| *dim as usize).product::<usize>();
        let (tensor_type, bytes) = if draft.dtype == "q8_0" {
            (GgufTensorType::Q8_0, q8_bytes(draft.shape[0] as usize, draft.shape[1] as usize))
        } else {
            (GgufTensorType::F32, f32_bytes(draft.knolo, len))
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
        meta_str("general.architecture", "llama"),
        meta_u32("llama.attention.head_count", if drafts[0].shape[1] == 8 { 2 } else { 4 }),
        meta_u32("llama.attention.head_count_kv", if drafts[0].shape[1] == 8 { 1 } else { 4 }),
        meta_f32("llama.rope.freq_base", 500_000.0),
        meta_f32("llama.attention.layer_norm_rms_eps", 1e-5),
        meta_u32("llama.context_length", 32),
    ];
    if precisions.contains(&"q8_0") {
        metadata.push(meta_u32("general.quantization_version", 2));
    }
    let bytes = encode_gguf(&metadata, &gguf).unwrap();
    fs::write(dir.join("weights.gguf"), bytes).unwrap();
    let mut tokens = vec!["<pad>".to_string(), "<bos>".to_string(), "<eos>".to_string(), "\n".to_string()];
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
        .map(|draft| {
            serde_json::json!({
                "dtype": draft.dtype,
                "name": draft.knolo,
                "shape": draft.shape,
            })
        })
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

fn f32_drafts() -> Vec<TensorDraft> {
    vec![
        TensorDraft { knolo: "embed.weight", ggml: "token_embd.weight", shape: vec![40, 8], dtype: "f32" },
        TensorDraft { knolo: "layers.0.attn_norm.weight", ggml: "blk.0.attn_norm.weight", shape: vec![8], dtype: "f32" },
        TensorDraft { knolo: "layers.0.attn.q.weight", ggml: "blk.0.attn_q.weight", shape: vec![8, 8], dtype: "f32" },
        TensorDraft { knolo: "layers.0.attn.k.weight", ggml: "blk.0.attn_k.weight", shape: vec![4, 8], dtype: "f32" },
        TensorDraft { knolo: "layers.0.attn.v.weight", ggml: "blk.0.attn_v.weight", shape: vec![4, 8], dtype: "f32" },
        TensorDraft { knolo: "layers.0.attn.o.weight", ggml: "blk.0.attn_output.weight", shape: vec![8, 8], dtype: "f32" },
        TensorDraft { knolo: "layers.0.mlp_norm.weight", ggml: "blk.0.ffn_norm.weight", shape: vec![8], dtype: "f32" },
        TensorDraft { knolo: "layers.0.mlp.gate.weight", ggml: "blk.0.ffn_gate.weight", shape: vec![16, 8], dtype: "f32" },
        TensorDraft { knolo: "layers.0.mlp.up.weight", ggml: "blk.0.ffn_up.weight", shape: vec![16, 8], dtype: "f32" },
        TensorDraft { knolo: "layers.0.mlp.down.weight", ggml: "blk.0.ffn_down.weight", shape: vec![8, 16], dtype: "f32" },
        TensorDraft { knolo: "final_norm.weight", ggml: "output_norm.weight", shape: vec![8], dtype: "f32" },
        TensorDraft { knolo: "lm_head.weight", ggml: "output.weight", shape: vec![40, 8], dtype: "f32" },
    ]
}

fn q8_drafts() -> Vec<TensorDraft> {
    let quant = |knolo, ggml, shape| TensorDraft { knolo, ggml, shape, dtype: "q8_0" };
    let norm = |knolo, ggml, shape| TensorDraft { knolo, ggml, shape, dtype: "f32" };
    vec![
        quant("embed.weight", "token_embd.weight", vec![64, 32]),
        norm("layers.0.attn_norm.weight", "blk.0.attn_norm.weight", vec![32]),
        quant("layers.0.attn.q.weight", "blk.0.attn_q.weight", vec![32, 32]),
        quant("layers.0.attn.k.weight", "blk.0.attn_k.weight", vec![32, 32]),
        quant("layers.0.attn.v.weight", "blk.0.attn_v.weight", vec![32, 32]),
        quant("layers.0.attn.o.weight", "blk.0.attn_output.weight", vec![32, 32]),
        norm("layers.0.mlp_norm.weight", "blk.0.ffn_norm.weight", vec![32]),
        quant("layers.0.mlp.gate.weight", "blk.0.ffn_gate.weight", vec![64, 32]),
        quant("layers.0.mlp.up.weight", "blk.0.ffn_up.weight", vec![64, 32]),
        quant("layers.0.mlp.down.weight", "blk.0.ffn_down.weight", vec![32, 64]),
        norm("final_norm.weight", "output_norm.weight", vec![32]),
        quant("lm_head.weight", "output.weight", vec![64, 32]),
    ]
}

fn run_prompt(dir: &Path, prompt: &[u32], new_tokens: u32) -> Result<Vec<u32>, infer_contracts::InferFailure> {
    let source = load_verified_model(&dir.join("wide.kmodel"), dir)?;
    let placement = llama_cpu_placement(&source)?;
    let mut model = LlamaAdapter.build(&source, &placement, &ReferenceF32Backend)?;
    let mut kv = open_paged_kv(&source, model.kv_layout())?;
    let output = greedy_generate(model.as_mut(), &mut kv, 1, prompt, new_tokens)?;
    Ok(output.tokens)
}

#[test]
fn wide_f32_gguf_reads_metadata_and_crosses_a_kv_page() {
    let dir = scratch();
    write_image(&dir, &f32_drafts(), &["f32"], 40);
    let source = load_verified_model(&dir.join("wide.kmodel"), &dir).unwrap();
    let shape = source.llama.unwrap();
    assert!(!shape.is_toy());
    assert_eq!(shape.vocab, 40);
    assert_eq!(shape.hidden, 8);
    assert_eq!(shape.layers, 1);
    assert_eq!(shape.heads, 2);
    assert_eq!(shape.kv_heads, 1);
    assert_eq!(shape.head_dim, 4);
    assert_eq!(shape.context, 32);
    assert_eq!(shape.rope_theta, 500_000.0);
    assert_eq!(shape.page_count(), 2);
    let cuda = llama_cuda_placement(&source).unwrap_err();
    assert_eq!(cuda.code, ErrorCode::UnsupportedKernel);
    assert!(cuda.message.contains("no CUDA kernel"));

    let prompt: Vec<u32> = (1..21).collect();
    let first = run_prompt(&dir, &prompt, 2).unwrap();
    let second = run_prompt(&dir, &prompt, 2).unwrap();
    assert_eq!(first.len(), 2);
    assert_eq!(first, second);
    let err = run_prompt(&dir, &(0..33).map(|id| (id % 39) + 1).collect::<Vec<_>>(), 1).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContextLimitExceeded);
}

#[test]
fn wide_q8_gguf_is_deterministic_on_cpu() {
    let dir = scratch();
    write_image(&dir, &q8_drafts(), &["f32", "q8_0"], 64);
    let source = load_verified_model(&dir.join("wide.kmodel"), &dir).unwrap();
    let shape = source.llama.unwrap();
    assert_eq!(shape.heads, 4);
    assert_eq!(shape.kv_heads, 4);
    assert_eq!(shape.hidden, 32);
    assert!(!source.llama_tensors.as_ref().unwrap().packed.is_empty());
    let prompt = vec![1u32, 4, 7, 9];
    let first = run_prompt(&dir, &prompt, 2).unwrap();
    let second = run_prompt(&dir, &prompt, 2).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 2);
}
