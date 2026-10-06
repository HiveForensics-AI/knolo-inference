use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_attention, reference_engine_build,
    reference_kernel_bundle, verify_attention, write_attention_report, write_synthetic_model,
    AttentionObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-attention"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-attention-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> AttentionObservation {
    AttentionObservation {
        engine_build_root: engine_root(),
        reason: "window".into(),
        window_tokens: 4,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        attention_exact: true,
        approximated: false,
        window_applied: false,
        sparsity_applied: false,
        kv_quantized: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn a_attention_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_attention(&plan, &observe()).unwrap();
    assert_eq!(measured.report.window_tokens, 4);
    assert!(measured.report.attention_exact);
    verify_attention(&measured).unwrap();

    let mut approx = observe();
    approx.reason = "approximation".into();
    approx.window_tokens = 0;
    let approx = measure_attention(&plan, &approx).unwrap();
    assert!(approx.report.attention_exact);
    assert!(!approx.report.approximated);
    let mut sparse = observe();
    sparse.reason = "sparsity".into();
    sparse.window_tokens = 0;
    measure_attention(&plan, &sparse).unwrap();
    let mut quant = observe();
    quant.reason = "quantized-kv".into();
    quant.window_tokens = 0;
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_attention(&slot, &quant).unwrap();
    assert!(!on_slot.report.kv_quantized);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_attention_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.approximated = true;
    let err = measure_attention(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("an approximation is not applied"),
        "{err}"
    );

    let mut bad = observe();
    bad.window_tokens = 17;
    let err = measure_attention(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("a window span exceeds the context"),
        "{err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_attention_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_attention(&plan, &observe()).unwrap();
    write_attention_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.attention-report"
    );
    let again = write_attention_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-attention-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_attention_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
