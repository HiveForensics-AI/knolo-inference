use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_capacity, reference_engine_build,
    reference_kernel_bundle, verify_capacity, write_capacity_report, write_synthetic_model,
    CapacityObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-capacity"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-capacity-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> CapacityObservation {
    CapacityObservation {
        engine_build_root: engine_root(),
        reason: "overflow".into(),
        requested_tokens: 4,
        capacity_tokens: 0,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        overflowed: false,
        dropped: false,
        balanced: false,
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
fn a_capacity_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_capacity(&plan, &observe()).unwrap();
    assert_eq!(measured.report.requested_tokens, 4);
    assert!(!measured.report.overflowed);
    verify_capacity(&measured).unwrap();

    let mut dropped_case = observe();
    dropped_case.reason = "drop".into();
    measure_capacity(&plan, &dropped_case).unwrap();
    let mut balance = observe();
    balance.reason = "balance".into();
    balance.requested_tokens = 16;
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_capacity(&slot, &balance).unwrap();
    assert_eq!(on_slot.report.capacity_tokens, 0);
    assert!(!on_slot.report.balanced);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_capacity_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.dropped = true;
    let err = measure_capacity(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("expert tokens are not dropped"),
        "{err}"
    );

    let mut bad = observe();
    bad.requested_tokens = 17;
    let err = measure_capacity(&plan, &bad).unwrap_err();
    assert!(
        err.message
            .contains("an expert capacity request exceeds the context"),
        "{err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_capacity_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_capacity(&plan, &observe()).unwrap();
    write_capacity_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.capacity-report"
    );
    let again = write_capacity_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-capacity-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_capacity_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
