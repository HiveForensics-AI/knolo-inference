use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_workstation, reference_engine_build,
    reference_kernel_bundle, verify_workstation, write_synthetic_model, write_workstation_report,
    WorkstationObservation,
};

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-workstation"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-workstation-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> WorkstationObservation {
    WorkstationObservation {
        engine_build_root: engine_root(),
        profile_root: pin(b"workstation-profile"),
        benchmark_root: pin(b"workstation-benchmark"),
        reason: "benchmark".into(),
        benchmark_count: 4,
        blessed: false,
        benchmark_recorded: false,
        fallback_selected: false,
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
fn a_workstation_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_workstation(&plan, &observe()).unwrap();
    assert_eq!(measured.report.benchmark_count, 4);
    assert!(!measured.report.blessed);
    verify_workstation(&measured).unwrap();

    let mut profile = observe();
    profile.reason = "profile".into();
    profile.benchmark_count = 0;
    measure_workstation(&plan, &profile).unwrap();
    let mut fallback = observe();
    fallback.reason = "fallback".into();
    fallback.benchmark_count = 0;
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_workstation(&slot, &fallback).unwrap();
    assert!(!on_slot.report.blessed);
    assert!(!on_slot.report.fallback_selected);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_workstation_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.blessed = true;
    let err = measure_workstation(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("a workstation recipe is not blessed"),
        "{err}"
    );

    let mut bad = observe();
    bad.benchmark_count = 33;
    let err = measure_workstation(&plan, &bad).unwrap_err();
    assert!(
        err.message
            .contains("benchmark count exceeds the record cap"),
        "{err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_workstation_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_workstation(&plan, &observe()).unwrap();
    write_workstation_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.workstation-report"
    );
    let again = write_workstation_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-workstation-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_workstation_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
