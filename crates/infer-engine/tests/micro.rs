use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use infer_contracts::{ErrorCode, InferFailure};
use infer_engine::{
    accept_cpu_placement, accept_cuda_placement, adapter_by_id, argmax, cpu_placement,
    cuda_placement, greedy_generate, load_verified_micro, logit_margin, micro_kv_layout,
    render_conformance, validate_micro_image, write_synthetic_model, ArchitectureAdapter,
    ConformanceCase, DecodeBatch, ExecutableModel, KvStore, MicroAdapter, PagedKv, PrefillBatch,
    ReferenceF32Backend, SingleBlockKv, TensorBackend, ADAPTER_ID, CPU_KV_PAGE_POOL, FIXTURE_CASES,
    MIN_GREEDY_MARGIN, VOCAB,
};

static TEMP: AtomicU64 = AtomicU64::new(0);

fn scratch() -> PathBuf {
    let n = TEMP.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("knolo-infer-micro-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn code(err: InferFailure) -> ErrorCode {
    err.code
}

fn loaded() -> (PathBuf, infer_engine::VerifiedWeightSource) {
    let dir = scratch();
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    (dir, source)
}

fn oracle(source: &infer_engine::VerifiedWeightSource) -> Box<dyn ExecutableModel> {
    let placement = cpu_placement(source).unwrap();
    MicroAdapter
        .build(source, &placement, &ReferenceF32Backend)
        .unwrap()
}

struct ForeignBackend;

impl TensorBackend for ForeignBackend {
    fn id(&self) -> &'static str {
        "candle-cpu"
    }

    fn device(&self) -> &'static str {
        "cpu"
    }
}

#[test]
fn published_fixture_matches_the_oracle() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = repo.join("models/micro-transformer");
    let conformance = repo.join("conformance/micro-model");
    fs::create_dir_all(&conformance).unwrap();
    let written = write_synthetic_model(&model_dir).unwrap();
    let source = load_verified_micro(&model_dir.join("micro.kmodel"), &model_dir).unwrap();
    assert_eq!(source.image_root, written.image_root);
    assert_eq!(source.artifact_root, written.artifact_root);
    assert_eq!(source.runtime_root, written.runtime_root);
    assert_eq!(source.image.architecture.adapter, ADAPTER_ID);

    let mut model = oracle(&source);
    let mut cases = Vec::new();
    for fixture in FIXTURE_CASES {
        let mut kv = SingleBlockKv::new(micro_kv_layout()).unwrap();
        let output = greedy_generate(
            model.as_mut(),
            &mut kv,
            1,
            fixture.prompt,
            fixture.new_tokens,
        )
        .unwrap();
        assert_eq!(output.tokens.len(), fixture.new_tokens as usize);
        assert_eq!(output.prefill_logits.len(), VOCAB);
        let margin = logit_margin(&output.prefill_logits).unwrap();
        assert!(
            margin >= MIN_GREEDY_MARGIN,
            "{} margin {margin} is below {MIN_GREEDY_MARGIN}",
            fixture.name
        );
        let mut again = SingleBlockKv::new(micro_kv_layout()).unwrap();
        let second = greedy_generate(
            model.as_mut(),
            &mut again,
            1,
            fixture.prompt,
            fixture.new_tokens,
        )
        .unwrap();
        assert_eq!(second.tokens, output.tokens);
        assert_eq!(
            second
                .prefill_logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            output
                .prefill_logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        cases.push(ConformanceCase {
            name: fixture.name.to_string(),
            prompt: fixture.prompt.to_vec(),
            new_tokens: fixture.new_tokens,
            prefill_logits: output.prefill_logits,
            greedy_tokens: output.tokens,
            margin,
        });
    }
    let document = render_conformance(&source, &cases).unwrap();
    let path = conformance.join("expected.json");
    fs::write(&path, &document).unwrap();
    assert_eq!(fs::read_to_string(&path).unwrap(), document);
    assert!(document.contains(source.image_root.as_str()));
    assert!(document.contains(&cases[0].greedy_tokens[0].to_string()));
}

#[test]
fn prefill_matches_token_by_token_decode() {
    let (_dir, source) = loaded();
    let mut model = oracle(&source);
    let prompt = FIXTURE_CASES[0].prompt;
    let mut whole = SingleBlockKv::new(micro_kv_layout()).unwrap();
    whole.begin_sequence(1).unwrap();
    let full = model
        .prefill(
            &PrefillBatch {
                sequence_id: 1,
                token_ids: prompt.to_vec(),
            },
            &mut whole,
        )
        .unwrap();
    let mut stepped = SingleBlockKv::new(micro_kv_layout()).unwrap();
    stepped.begin_sequence(3).unwrap();
    let mut logits = model
        .prefill(
            &PrefillBatch {
                sequence_id: 3,
                token_ids: vec![prompt[0]],
            },
            &mut stepped,
        )
        .unwrap()
        .logits;
    for token in &prompt[1..] {
        let position = stepped.token_len(3).unwrap();
        logits = model
            .decode(
                &DecodeBatch {
                    sequence_id: 3,
                    token_ids: vec![*token],
                    positions: vec![position],
                },
                &mut stepped,
            )
            .unwrap()
            .logits;
    }
    assert_eq!(
        logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        full.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let left = whole.snapshot(1).unwrap();
    let right = stepped.snapshot(3).unwrap();
    assert_eq!(left.len, right.len);
    assert_eq!(left.block_table.len(), 1);
    assert_eq!(
        left.k.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        right.k.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        left.v.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        right.v.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn context_token_and_architecture_failures_are_closed() {
    let (dir, source) = loaded();
    let mut model = oracle(&source);
    let mut kv = SingleBlockKv::new(micro_kv_layout()).unwrap();
    kv.begin_sequence(1).unwrap();
    assert_eq!(
        code(
            model
                .prefill(
                    &PrefillBatch {
                        sequence_id: 1,
                        token_ids: vec![],
                    },
                    &mut kv,
                )
                .unwrap_err()
        ),
        ErrorCode::PromptCompilationFailed
    );
    assert_eq!(kv.token_len(1).unwrap(), 0);
    assert_eq!(
        code(
            model
                .prefill(
                    &PrefillBatch {
                        sequence_id: 1,
                        token_ids: vec![VOCAB as u32],
                    },
                    &mut kv,
                )
                .unwrap_err()
        ),
        ErrorCode::PromptCompilationFailed
    );
    assert_eq!(kv.token_len(1).unwrap(), 0);
    let too_long = vec![1u32; 17];
    assert_eq!(
        code(
            model
                .prefill(
                    &PrefillBatch {
                        sequence_id: 1,
                        token_ids: too_long,
                    },
                    &mut kv,
                )
                .unwrap_err()
        ),
        ErrorCode::ContextLimitExceeded
    );
    assert_eq!(kv.token_len(1).unwrap(), 0);
    assert_eq!(kv.block_table(1).unwrap().len(), 1);

    let mut limited = cpu_placement(&source).unwrap();
    limited.context_reservation_tokens = 8;
    accept_cpu_placement(&source, &limited).unwrap();
    let mut short = MicroAdapter
        .build(&source, &limited, &ReferenceF32Backend)
        .unwrap();
    let mut short_kv = SingleBlockKv::new(micro_kv_layout()).unwrap();
    short_kv.begin_sequence(1).unwrap();
    assert_eq!(
        code(
            short
                .prefill(
                    &PrefillBatch {
                        sequence_id: 1,
                        token_ids: vec![1; 9],
                    },
                    &mut short_kv,
                )
                .unwrap_err()
        ),
        ErrorCode::ContextLimitExceeded
    );

    assert_eq!(
        code(adapter_by_id("knolo.llama.v1").unwrap_err()),
        ErrorCode::UnsupportedArchitecture
    );
    let mut image = source.image.clone();
    image.architecture.family = "llama".into();
    assert_eq!(
        code(validate_micro_image(&image).unwrap_err()),
        ErrorCode::ModelImageInvalid
    );
    match MicroAdapter.build(&source, &cpu_placement(&source).unwrap(), &ForeignBackend) {
        Ok(_) => panic!("reference adapter accepted a foreign backend"),
        Err(err) => assert_eq!(err.code, ErrorCode::UnsupportedKernel),
    }

    let mut placement = cpu_placement(&source).unwrap();
    placement.devices = vec!["slot-0".into()];
    placement.tensor_groups[0].device = "slot-0".into();
    assert_eq!(
        code(accept_cpu_placement(&source, &placement).unwrap_err()),
        ErrorCode::PlacementUnsatisfiable
    );
    let cuda = cuda_placement(&source).unwrap();
    assert_eq!(cuda.devices, ["slot-0".to_string()]);
    assert_eq!(cuda.tensor_groups[0].device, "slot-0");
    assert_eq!(
        cuda.expected_total_bytes,
        cpu_placement(&source).unwrap().expected_total_bytes
    );
    accept_cuda_placement(&source, &cuda).unwrap();
    assert_eq!(
        code(accept_cpu_placement(&source, &cuda).unwrap_err()),
        ErrorCode::PlacementUnsatisfiable
    );
    assert_eq!(
        code(accept_cuda_placement(&source, &cpu_placement(&source).unwrap()).unwrap_err()),
        ErrorCode::PlacementUnsatisfiable
    );

    let mut weights = fs::read(dir.join("weights.safetensors")).unwrap();
    weights[8] ^= 0xff;
    fs::write(dir.join("weights.safetensors"), weights).unwrap();
    assert_eq!(
        code(load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap_err()),
        ErrorCode::ModelDigestMismatch
    );
}

#[test]
fn single_block_store_rejects_a_ninth_sequence() {
    let mut kv = SingleBlockKv::new(micro_kv_layout()).unwrap();
    for id in 0..8 {
        kv.begin_sequence(id).unwrap();
        assert_eq!(kv.block_table(id).unwrap().len(), 1);
    }
    assert_eq!(
        code(kv.begin_sequence(8).unwrap_err()),
        ErrorCode::InsufficientMemory
    );
    kv.release_sequence(0).unwrap();
    kv.begin_sequence(8).unwrap();
    assert_eq!(
        code(kv.token_len(0).unwrap_err()),
        ErrorCode::ContractInvalid
    );
}

#[test]
fn greedy_tie_keeps_the_lowest_index() {
    assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]).unwrap(), 1);
}

#[test]
fn paged_kv_matches_single_block_on_the_fixture() {
    let (_dir, source) = loaded();
    let mut model = oracle(&source);
    for fixture in FIXTURE_CASES {
        let mut single = SingleBlockKv::new(micro_kv_layout()).unwrap();
        let mut paged = PagedKv::new(micro_kv_layout(), CPU_KV_PAGE_POOL).unwrap();
        let left = greedy_generate(
            model.as_mut(),
            &mut single,
            1,
            fixture.prompt,
            fixture.new_tokens,
        )
        .unwrap();
        let right = greedy_generate(
            model.as_mut(),
            &mut paged,
            1,
            fixture.prompt,
            fixture.new_tokens,
        )
        .unwrap();
        assert_eq!(left.tokens, right.tokens, "{}", fixture.name);
        assert_eq!(
            left.prefill_logits, right.prefill_logits,
            "{}",
            fixture.name
        );
        let single_kv = single.snapshot(1).unwrap();
        let paged_kv = paged.snapshot(1).unwrap();
        assert_eq!(paged_kv.len, single_kv.len, "{}", fixture.name);
        assert_eq!(
            paged_kv.block_table, single_kv.block_table,
            "{}",
            fixture.name
        );
        assert_eq!(paged_kv.k, single_kv.k, "{}", fixture.name);
        assert_eq!(paged_kv.v, single_kv.v, "{}", fixture.name);
        assert_eq!(paged_kv.block_table.len(), 1, "{}", fixture.name);
    }
}
