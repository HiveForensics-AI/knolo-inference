use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_cancellation_latency, reference_engine_build,
    reference_kernel_bundle, verify_cancellation_latency, write_cancellation_report,
    write_synthetic_model, CancellationObservation, MAX_CONTEXT, VOCAB,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-cancellation"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-cancellation-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    source: &infer_engine::VerifiedWeightSource,
    stage: &str,
    prompt: &[u32],
    output: &[u32],
    requested_nanos: u64,
    terminal_nanos: u64,
) -> CancellationObservation {
    CancellationObservation {
        model_image_root: source.image_root.clone(),
        artifact_root: source.artifact_root.clone(),
        engine_build_root: engine_root(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        cancel_stage: stage.into(),
        prefill_tokens: u32::try_from(prompt.len()).unwrap(),
        decode_tokens: u32::try_from(output.len()).unwrap(),
        prompt_tokens: prompt.to_vec(),
        output_tokens: output.to_vec(),
        requested_nanos,
        terminal_nanos,
    }
}

#[test]
fn prefill_and_decode_cancels_record_the_terminal_gap() {
    let dir = scratch("fixture");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    assert_eq!(plan.graph_capture_mode, "off");

    let prefill = observe(&source, "prefill", &[1, 4, 7], &[], 1_000, 1_400);
    let prefill = measure_cancellation_latency(&plan, &prefill).unwrap();
    assert_eq!(prefill.report.cancel_stage, "prefill");
    assert_eq!(prefill.report.decode_tokens, 0);
    assert_eq!(prefill.report.cancellation_latency_nanos, 400);
    assert_eq!(prefill.report.validation_result, "recorded");
    verify_cancellation_latency(&prefill).unwrap();

    let decode = observe(&source, "decode", &[1, 4, 7], &[13, 6], 2_000, 2_750);
    let decode = measure_cancellation_latency(&plan, &decode).unwrap();
    assert_eq!(decode.report.cancel_stage, "decode");
    assert_eq!(decode.report.decode_tokens, 2);
    assert_eq!(decode.report.cancellation_latency_nanos, 750);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = observe(&source, "prefill", &[1], &[], 3, 8);
    let on_slot = measure_cancellation_latency(&slot, &on_slot).unwrap();
    assert_eq!(on_slot.report.cancellation_latency_nanos, 5);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_failed_measurement_issues_no_report() {
    let dir = scratch("reject");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let root = observe(&source, "decode", &[1, 4, 7], &[13], 10, 25);

    let mut mode = root.clone();
    mode.execution_mode = "throughput".into();
    let err = measure_cancellation_latency(&plan, &mode).unwrap_err();
    assert_eq!(err.code, ErrorCode::BackendNotAllowed, "{err}");

    let mut stage = root.clone();
    stage.cancel_stage = "queue".into();
    let err = measure_cancellation_latency(&plan, &stage).unwrap_err();
    assert!(err.message.contains("cancelStage"), "{err}");

    let mut early = root.clone();
    early.decode_tokens = 0;
    early.output_tokens.clear();
    let err = measure_cancellation_latency(&plan, &early).unwrap_err();
    assert!(
        err.message.contains("decode cancel has no output tokens"),
        "{err}"
    );

    let mut late = observe(&source, "prefill", &[1], &[], 10, 25);
    late.decode_tokens = 1;
    late.output_tokens = vec![1];
    let err = measure_cancellation_latency(&plan, &late).unwrap_err();
    assert!(
        err.message.contains("prefill cancel has output tokens"),
        "{err}"
    );

    let mut zero = root.clone();
    zero.requested_nanos = 0;
    let err = measure_cancellation_latency(&plan, &zero).unwrap_err();
    assert!(err.message.contains("cancel request time is zero"), "{err}");

    let mut same = root.clone();
    same.terminal_nanos = same.requested_nanos;
    let err = measure_cancellation_latency(&plan, &same).unwrap_err();
    assert!(
        err.message
            .contains("cancel terminal is not after the request"),
        "{err}"
    );

    let mut over = root.clone();
    over.prompt_tokens = vec![1; MAX_CONTEXT as usize];
    over.prefill_tokens = MAX_CONTEXT;
    over.output_tokens = vec![1];
    over.decode_tokens = 1;
    let err = measure_cancellation_latency(&plan, &over).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContextLimitExceeded, "{err}");

    let mut vocab = root.clone();
    vocab.prompt_tokens = vec![VOCAB as u32];
    vocab.prefill_tokens = 1;
    let err = measure_cancellation_latency(&plan, &vocab).unwrap_err();
    assert!(
        err.message.contains("outside the micro vocabulary"),
        "{err}"
    );

    let mut graphs = plan.clone();
    graphs.graph_capture_mode = "decode-buckets".into();
    let err = measure_cancellation_latency(&graphs, &root).unwrap_err();
    assert!(err.message.contains("graph capture is off"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let observed = observe(&source, "prefill", &[1, 4, 7], &[], 4, 9);
    let measured = measure_cancellation_latency(&plan, &observed).unwrap();
    write_cancellation_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.cancellation-report"
    );
    let again = write_cancellation_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-cancellation-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_cancellation_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
