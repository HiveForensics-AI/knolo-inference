use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_hybrid, reference_engine_build,
    reference_kernel_bundle, verify_hybrid, write_hybrid_report, write_synthetic_model,
    HybridObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-hybrid"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-hybrid-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> HybridObservation {
    HybridObservation {
        engine_build_root: engine_root(),
        reason: "state".into(),
        state_tokens: 4,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        attention_exact: true,
        hybrid_allocated: false,
        sliding_applied: false,
        full_mixed: false,
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
fn a_hybrid_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_hybrid(&plan, &observe()).unwrap();
    assert_eq!(measured.report.state_tokens, 4);
    assert!(!measured.report.hybrid_allocated);
    verify_hybrid(&measured).unwrap();

    let mut varied = observe();
    varied.reason = "sliding".into();
    varied.state_tokens = 0;
    let varied = measure_hybrid(&plan, &varied).unwrap();
    assert_eq!(varied.report.state_tokens, 0);
    let mut varied = observe();
    varied.reason = "full".into();
    varied.state_tokens = 0;
    let varied = measure_hybrid(&plan, &varied).unwrap();
    assert!(!varied.report.full_mixed);

    let mut slot_obs = observe();
    slot_obs.state_tokens = 16;
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_hybrid(&slot, &slot_obs).unwrap();
    assert_eq!(on_slot.report.state_tokens, 16);
    assert!(on_slot.report.attention_exact);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_hybrid_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.hybrid_allocated = true;
    let err = measure_hybrid(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("hybrid state is not allocated"),
        "{err}"
    );

    let mut bad = observe();
    bad.state_tokens = 17;
    let err = measure_hybrid(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("a hybrid state exceeds the context"),
        "{err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_hybrid_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_hybrid(&plan, &observe()).unwrap();
    write_hybrid_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.hybrid-report"
    );
    let again = write_hybrid_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-hybrid-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_hybrid_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
