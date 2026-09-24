use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use infer_contracts::ErrorCode;
#[cfg(feature = "cuda")]
use infer_engine::cuda_placement;
use infer_engine::{
    argmax, cpu_placement, greedy_generate, load_verified_micro, logit_margin, micro_kv_layout,
    write_synthetic_model, ArchitectureAdapter, ExecutableModel, KvSnapshot, KvStore,
    ReferenceF32Backend, SingleBlockKv, TensorBackend, ADAPTER_ID, FIXTURE_CASES,
    LOGIT_ABS_TOLERANCE, MIN_GREEDY_MARGIN, VOCAB,
};
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
    assert!(cpu_adapter_by_id("knolo.llama.v1").is_err());
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
