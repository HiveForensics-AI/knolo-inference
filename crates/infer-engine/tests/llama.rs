use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use infer_artifact::{compile_manifest, write_model_image};
use infer_contracts::ErrorCode;
use infer_engine::{
    greedy_generate, llama_cpu_placement, llama_kv_layout, load_verified_micro,
    load_verified_model, logit_margin, render_llama_conformance, write_llama_gguf,
    write_llama_model, write_llama_precision, ArchitectureAdapter, ConformanceCase, LlamaAdapter,
    ReferenceF32Backend, SingleBlockKv, LLAMA_ADAPTER_ID, LLAMA_FIXTURE_CASES, LLAMA_VOCAB,
    LOGIT_ABS_TOLERANCE, MIN_GREEDY_MARGIN,
};

static TEMP: AtomicU64 = AtomicU64::new(0);

fn scratch() -> PathBuf {
    let n = TEMP.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("knolo-infer-llama-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn same_bits(name: &str, left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len(), "{name}");
    for (index, (left_value, right_value)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            left_value.to_bits(),
            right_value.to_bits(),
            "{name}[{index}]"
        );
    }
}

fn weights_match(left: &infer_engine::MicroWeights, right: &infer_engine::MicroWeights) {
    same_bits("embed", &left.embed, &right.embed);
    same_bits("final_norm", &left.final_norm, &right.final_norm);
    same_bits("lm_head", &left.lm_head, &right.lm_head);
    assert_eq!(left.layers.len(), right.layers.len());
    for (index, (left_layer, right_layer)) in left.layers.iter().zip(&right.layers).enumerate() {
        let prefix = format!("layers.{index}");
        same_bits(
            &format!("{prefix}.attn_norm"),
            &left_layer.attn_norm,
            &right_layer.attn_norm,
        );
        same_bits(&format!("{prefix}.q"), &left_layer.q, &right_layer.q);
        same_bits(&format!("{prefix}.k"), &left_layer.k, &right_layer.k);
        same_bits(&format!("{prefix}.v"), &left_layer.v, &right_layer.v);
        same_bits(&format!("{prefix}.o"), &left_layer.o, &right_layer.o);
        same_bits(
            &format!("{prefix}.mlp_norm"),
            &left_layer.mlp_norm,
            &right_layer.mlp_norm,
        );
        same_bits(
            &format!("{prefix}.gate"),
            &left_layer.gate,
            &right_layer.gate,
        );
        same_bits(&format!("{prefix}.up"), &left_layer.up, &right_layer.up);
        same_bits(
            &format!("{prefix}.down"),
            &left_layer.down,
            &right_layer.down,
        );
    }
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| (left_value - right_value).abs())
        .fold(0.0, f32::max)
}

fn generate(
    source: &infer_engine::VerifiedWeightSource,
    prompt: &[u32],
    new_tokens: u32,
) -> infer_engine::GreedyOutput {
    let placement = llama_cpu_placement(source).unwrap();
    let mut model = LlamaAdapter
        .build(source, &placement, &ReferenceF32Backend)
        .unwrap();
    let mut kv = SingleBlockKv::new(llama_kv_layout()).unwrap();
    greedy_generate(model.as_mut(), &mut kv, 1, prompt, new_tokens).unwrap()
}

#[test]
fn published_llama_fixture_matches_the_oracle() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = repo.join("models/llama-tiny");
    let conformance = repo.join("conformance/llama-tiny");
    fs::create_dir_all(&conformance).unwrap();
    let written = write_llama_model(&model_dir).unwrap();
    write_llama_precision(&model_dir, "f16", "llama-f16.kmodel").unwrap();
    write_llama_precision(&model_dir, "bf16", "llama-bf16.kmodel").unwrap();
    let source = load_verified_model(&model_dir.join("llama.kmodel"), &model_dir).unwrap();
    assert_eq!(source.image_root, written.image_root);
    assert_eq!(source.artifact_root, written.artifact_root);
    assert_eq!(source.runtime_root, written.runtime_root);
    assert_eq!(source.image.architecture.adapter, LLAMA_ADAPTER_ID);
    assert_eq!(source.image.precisions, ["f32".to_string()]);

    let mut cases = Vec::new();
    for fixture in LLAMA_FIXTURE_CASES {
        let output = generate(&source, fixture.prompt, fixture.new_tokens);
        assert_eq!(output.tokens.len(), fixture.new_tokens as usize);
        assert_eq!(output.prefill_logits.len(), LLAMA_VOCAB);
        let margin = logit_margin(&output.prefill_logits).unwrap();
        assert!(
            margin >= MIN_GREEDY_MARGIN,
            "{} margin {margin} is below {MIN_GREEDY_MARGIN}",
            fixture.name
        );
        let second = generate(&source, fixture.prompt, fixture.new_tokens);
        assert_eq!(second.tokens, output.tokens);
        same_bits("prefill", &second.prefill_logits, &output.prefill_logits);
        cases.push(ConformanceCase {
            name: fixture.name.to_string(),
            prompt: fixture.prompt.to_vec(),
            new_tokens: fixture.new_tokens,
            prefill_logits: output.prefill_logits,
            greedy_tokens: output.tokens,
            margin,
        });
    }
    let document = render_llama_conformance(&source, &cases).unwrap();
    let path = conformance.join("expected.json");
    fs::write(&path, &document).unwrap();
    assert_eq!(fs::read_to_string(&path).unwrap(), document);

    for (dtype, image) in [("f16", "llama-f16.kmodel"), ("bf16", "llama-bf16.kmodel")] {
        let stored = load_verified_model(&model_dir.join(image), &model_dir).unwrap();
        assert_eq!(stored.image.precisions, [dtype.to_string()]);
        weights_match(&source.weights, &stored.weights);
        for fixture in LLAMA_FIXTURE_CASES {
            let reference = generate(&source, fixture.prompt, fixture.new_tokens);
            let decoded = generate(&stored, fixture.prompt, fixture.new_tokens);
            assert_eq!(decoded.tokens, reference.tokens, "{dtype} {}", fixture.name);
            let diff = max_abs(&decoded.prefill_logits, &reference.prefill_logits);
            assert!(
                diff <= LOGIT_ABS_TOLERANCE,
                "{dtype} {} logit diff {diff}",
                fixture.name
            );
        }
    }
}

#[test]
fn unknown_adapter_fails_before_the_weight_file_is_opened() {
    let dir = scratch();
    write_llama_model(&dir).unwrap();
    let manifest = fs::read_to_string(dir.join("manifest.json")).unwrap();
    let rewritten = manifest.replace("knolo.llama.v1", "knolo.missing.v1");
    let manifest_path = dir.join("manifest-missing.json");
    fs::write(&manifest_path, rewritten).unwrap();
    let compiled = compile_manifest(&manifest_path).unwrap();
    write_model_image(&dir.join("missing.kmodel"), &compiled.bytes).unwrap();
    fs::remove_file(dir.join("weights.safetensors")).unwrap();

    let missing = load_verified_model(&dir.join("missing.kmodel"), &dir).unwrap_err();
    assert_eq!(missing.code, ErrorCode::UnsupportedArchitecture);
    assert!(missing.message.contains("knolo.missing.v1"));
    assert_ne!(missing.code, ErrorCode::ModelArtifactMissing);

    let refused = load_verified_micro(&dir.join("llama.kmodel"), &dir).unwrap_err();
    assert_eq!(refused.code, ErrorCode::UnsupportedArchitecture);
    assert!(refused
        .message
        .contains("knolo.micro.v1 loader does not open knolo.llama.v1"));
    assert_ne!(refused.code, ErrorCode::ModelArtifactMissing);
}

#[test]
fn llama_f16_gguf_matches_the_oracle_and_writes_no_conversion() {
    let dir = scratch();
    let oracle_files = write_llama_model(&dir).unwrap();
    let oracle = load_verified_model(&dir.join("llama.kmodel"), &dir).unwrap();
    let gguf_files = write_llama_gguf(&dir, "f16", "llama-f16.gguf.kmodel").unwrap();
    assert_ne!(gguf_files.artifact_root, oracle_files.artifact_root);
    let before = dir_names(&dir);
    let stored = load_verified_model(&dir.join("llama-f16.gguf.kmodel"), &dir).unwrap();
    assert_eq!(stored.image.format, "gguf");
    assert_eq!(stored.image.precisions, ["f16".to_string()]);
    assert_eq!(dir_names(&dir), before);
    weights_match(&oracle.weights, &stored.weights);
    for fixture in LLAMA_FIXTURE_CASES {
        let reference = generate(&oracle, fixture.prompt, fixture.new_tokens);
        let decoded = generate(&stored, fixture.prompt, fixture.new_tokens);
        assert_eq!(decoded.tokens, reference.tokens, "{}", fixture.name);
        let diff = max_abs(&decoded.prefill_logits, &reference.prefill_logits);
        assert!(diff <= LOGIT_ABS_TOLERANCE, "{}", fixture.name);
    }
}

#[test]
fn micro_gguf_stays_invalid_and_a_foreign_ggml_type_is_unsupported() {
    let dir = scratch();
    write_llama_model(&dir).unwrap();
    let manifest = fs::read_to_string(dir.join("manifest.json")).unwrap();
    let micro = manifest
        .replace("knolo.llama.v1", "knolo.micro.v1")
        .replace("\"family\": \"llama\"", "\"family\": \"micro\"")
        .replace("\"format\": \"safetensors\"", "\"format\": \"gguf\"");
    fs::write(dir.join("manifest-micro-gguf.json"), micro).unwrap();
    let err = compile_manifest(&dir.join("manifest-micro-gguf.json")).unwrap_err();
    assert_eq!(err.code, ErrorCode::ModelImageInvalid);

    write_llama_gguf(&dir, "f16", "llama-f16.gguf.kmodel").unwrap();
    let mut bytes = fs::read(dir.join("weights.gguf")).unwrap();
    let type_at = locate_tensor_type(&bytes, "embed.weight");
    bytes[type_at..type_at + 4].copy_from_slice(&2u32.to_le_bytes());
    fs::write(dir.join("weights.gguf"), bytes).unwrap();
    let err = compile_manifest(&dir.join("manifest-gguf.json")).unwrap_err();
    assert_eq!(err.code, ErrorCode::UnsupportedQuantization);
}

fn dir_names(dir: &std::path::Path) -> Vec<String> {
    let mut names = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn locate_tensor_type(bytes: &[u8], name: &str) -> usize {
    let mut needle = (name.len() as u64).to_le_bytes().to_vec();
    needle.extend_from_slice(name.as_bytes());
    let at = bytes
        .windows(needle.len())
        .position(|window| window == needle.as_slice())
        .unwrap_or_else(|| panic!("missing tensor {name}"));
    let pos = at + needle.len();
    let n_dims = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
    pos + 4 + n_dims * 8
}
