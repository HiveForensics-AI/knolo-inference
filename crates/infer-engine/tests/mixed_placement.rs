use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_mixed_placement, reference_engine_build,
    reference_kernel_bundle, verify_mixed_placement, write_mixed_placement_report,
    write_synthetic_model, MixedPlacementObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-mixed_placement"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-mixed_placement-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> MixedPlacementObservation {
    MixedPlacementObservation {
        engine_build_root: engine_root(),
        reason: "mixed".into(),
        device_count: 2,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        single_recorded: false,
        mixed_selected: false,
        automatic_fallback: false,
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
fn a_mixed_placement_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_mixed_placement(&plan, &observe()).unwrap();
    assert_eq!(measured.report.device_count, 2);
    assert!(!measured.report.mixed_selected);
    verify_mixed_placement(&measured).unwrap();

    let mut single = observe();
    single.reason = "single".into();
    single.device_count = 1;
    single.code = "none".into();
    single.single_recorded = true;
    let single = measure_mixed_placement(&plan, &single).unwrap();
    assert!(single.report.single_recorded);
    assert!(!single.report.mixed_selected);
    let mut fallback = observe();
    fallback.reason = "fallback".into();
    fallback.device_count = 1;
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_mixed_placement(&slot, &fallback).unwrap();
    assert!(!on_slot.report.automatic_fallback);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_mixed_placement_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.mixed_selected = true;
    let err = measure_mixed_placement(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("mixed placement is not selected"),
        "{err}"
    );

    let mut bad = observe();
    bad.device_count = 3;
    let err = measure_mixed_placement(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("device count exceeds the record cap"),
        "{err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_mixed_placement_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_mixed_placement(&plan, &observe()).unwrap();
    write_mixed_placement_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.mixed-placement-report"
    );
    let again = write_mixed_placement_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-mixed_placement-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_mixed_placement_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
