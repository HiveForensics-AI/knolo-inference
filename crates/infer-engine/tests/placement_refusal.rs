use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_placement_refusal, reference_engine_build,
    reference_kernel_bundle, verify_placement_refusal, write_placement_refusal_report,
    write_synthetic_model, PlacementRefusalObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-placement-refusal"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-placement-refusal-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(reason: &str, probe_reached: bool, slot_visible: bool) -> PlacementRefusalObservation {
    PlacementRefusalObservation {
        engine_build_root: engine_root(),
        reason: reason.into(),
        code: "PLACEMENT_UNSATISFIABLE".into(),
        retryable: false,
        probe_reached,
        slot_visible,
        device_opened: false,
        cpu_fallback: false,
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
fn the_placement_refusal_records_toolkit_device_capability_and_bytes() {
    let dir = scratch("placement");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let toolkit = measure_placement_refusal(&plan, &observe("toolkit", false, false)).unwrap();
    assert!(!toolkit.report.probe_reached);
    assert!(!toolkit.report.device_opened);
    assert!(!toolkit.report.cpu_fallback);
    verify_placement_refusal(&toolkit).unwrap();

    let device = measure_placement_refusal(&plan, &observe("device", true, false)).unwrap();
    assert!(!device.report.slot_visible);
    let capability = measure_placement_refusal(&plan, &observe("capability", true, true)).unwrap();
    assert!(capability.report.slot_visible);
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let bytes = measure_placement_refusal(&slot, &observe("bytes", true, true)).unwrap();
    assert_eq!(bytes.report.code, "PLACEMENT_UNSATISFIABLE");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_placement_refusal_rejects_a_visible_missing_device_and_a_fallback() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let err = measure_placement_refusal(&plan, &observe("device", true, true)).unwrap_err();
    assert!(
        err.message.contains("a missing device is not visible"),
        "{err}"
    );
    let err = measure_placement_refusal(&plan, &observe("toolkit", true, false)).unwrap_err();
    assert!(
        err.message
            .contains("a toolkit refusal does not probe the device"),
        "{err}"
    );
    let mut fallback = observe("bytes", true, true);
    fallback.cpu_fallback = true;
    let err = measure_placement_refusal(&plan, &fallback).unwrap_err();
    assert!(
        err.message
            .contains("an unsatisfiable placement does not fall back"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_placement_refusal_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_placement_refusal(&plan, &observe("device", true, false)).unwrap();
    write_placement_refusal_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.placement-refusal-report"
    );
    let again = write_placement_refusal_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-placement-refusal-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_placement_refusal_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
