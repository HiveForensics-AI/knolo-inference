use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_receipt_finalization, reference_engine_build,
    reference_kernel_bundle, verify_receipt_finalization, write_finalization_report,
    write_synthetic_model, FinalizationObservation, MAX_CONTEXT, VOCAB,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-finalization"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-finalization-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    source: &infer_engine::VerifiedWeightSource,
    finish: &str,
    prompt: &[u32],
    output: &[u32],
    terminal_nanos: u64,
    finalized_nanos: u64,
) -> FinalizationObservation {
    FinalizationObservation {
        model_image_root: source.image_root.clone(),
        artifact_root: source.artifact_root.clone(),
        engine_build_root: engine_root(),
        receipt_root: sha256_prefixed(b"stored-receipt"),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        finish_reason: finish.into(),
        prefill_tokens: u32::try_from(prompt.len()).unwrap(),
        decode_tokens: u32::try_from(output.len()).unwrap(),
        prompt_tokens: prompt.to_vec(),
        output_tokens: output.to_vec(),
        terminal_nanos,
        finalized_nanos,
    }
}

#[test]
fn stop_and_length_record_the_gap_until_the_receipt_is_durable() {
    let dir = scratch("fixture");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    assert_eq!(plan.graph_capture_mode, "off");

    let stop = observe(&source, "stop", &[1, 4, 7], &[13], 5_000, 5_180);
    let stop = measure_receipt_finalization(&plan, &stop).unwrap();
    assert_eq!(stop.report.finish_reason, "stop");
    assert_eq!(stop.report.decode_tokens, 1);
    assert_eq!(stop.report.finalization_latency_nanos, 180);
    assert_eq!(stop.report.validation_result, "recorded");
    verify_receipt_finalization(&stop).unwrap();

    let length = observe(&source, "length", &[1], &[], 9_000, 9_010);
    let length = measure_receipt_finalization(&plan, &length).unwrap();
    assert_eq!(length.report.finish_reason, "length");
    assert_eq!(length.report.decode_tokens, 0);
    assert_eq!(length.report.finalization_latency_nanos, 10);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = observe(&source, "stop", &[1], &[2], 3, 8);
    let on_slot = measure_receipt_finalization(&slot, &on_slot).unwrap();
    assert_eq!(on_slot.report.finalization_latency_nanos, 5);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_failed_measurement_issues_no_report() {
    let dir = scratch("reject");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let root = observe(&source, "stop", &[1, 4, 7], &[13], 10, 25);

    let mut mode = root.clone();
    mode.execution_mode = "throughput".into();
    let err = measure_receipt_finalization(&plan, &mode).unwrap_err();
    assert_eq!(err.code, ErrorCode::BackendNotAllowed, "{err}");

    let mut cancelled = root.clone();
    cancelled.finish_reason = "cancelled".into();
    let err = measure_receipt_finalization(&plan, &cancelled).unwrap_err();
    assert!(
        err.message
            .contains("a cancelled request stores no receipt"),
        "{err}"
    );

    let mut failed = root.clone();
    failed.finish_reason = "timeout".into();
    let err = measure_receipt_finalization(&plan, &failed).unwrap_err();
    assert!(
        err.message.contains("a failed request stores no receipt"),
        "{err}"
    );

    let mut other = root.clone();
    other.finish_reason = "grammar".into();
    let err = measure_receipt_finalization(&plan, &other).unwrap_err();
    assert!(err.message.contains("finishReason"), "{err}");

    let mut empty = root.clone();
    empty.decode_tokens = 0;
    empty.output_tokens.clear();
    let err = measure_receipt_finalization(&plan, &empty).unwrap_err();
    assert!(
        err.message.contains("a stop receipt has no output tokens"),
        "{err}"
    );

    let mut zero = root.clone();
    zero.terminal_nanos = 0;
    let err = measure_receipt_finalization(&plan, &zero).unwrap_err();
    assert!(
        err.message.contains("receipt terminal time is zero"),
        "{err}"
    );

    let mut same = root.clone();
    same.finalized_nanos = same.terminal_nanos;
    let err = measure_receipt_finalization(&plan, &same).unwrap_err();
    assert!(
        err.message
            .contains("receipt is not finalized after the terminal"),
        "{err}"
    );

    let mut over = root.clone();
    over.prompt_tokens = vec![1; MAX_CONTEXT as usize];
    over.prefill_tokens = MAX_CONTEXT;
    over.output_tokens = vec![1];
    over.decode_tokens = 1;
    let err = measure_receipt_finalization(&plan, &over).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContextLimitExceeded, "{err}");

    let mut vocab = root.clone();
    vocab.prompt_tokens = vec![VOCAB as u32];
    vocab.prefill_tokens = 1;
    let err = measure_receipt_finalization(&plan, &vocab).unwrap_err();
    assert!(
        err.message.contains("outside the micro vocabulary"),
        "{err}"
    );

    let mut graphs = plan.clone();
    graphs.graph_capture_mode = "decode-buckets".into();
    let err = measure_receipt_finalization(&graphs, &root).unwrap_err();
    assert!(err.message.contains("graph capture is off"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let observed = observe(&source, "length", &[1, 4, 7], &[], 4, 9);
    let measured = measure_receipt_finalization(&plan, &observed).unwrap();
    write_finalization_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.finalization-report"
    );
    let again = write_finalization_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-finalization-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_finalization_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
