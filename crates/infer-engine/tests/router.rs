use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_router, reference_engine_build,
    reference_kernel_bundle, verify_router, write_router_report, write_synthetic_model,
    RouterObservation,
};

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-router"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-router-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> RouterObservation {
    RouterObservation {
        engine_build_root: engine_root(),
        reference_root: pin(b"router-reference"),
        candidate_root: pin(b"router-candidate"),
        reason: "logits".into(),
        sample_count: 4,
        compared: false,
        parity: false,
        load_recorded: false,
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
fn a_router_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_router(&plan, &observe()).unwrap();
    assert_eq!(measured.report.sample_count, 4);
    assert!(!measured.report.compared);
    verify_router(&measured).unwrap();

    let mut selection = observe();
    selection.reason = "selection".into();
    measure_router(&plan, &selection).unwrap();
    let mut load = observe();
    load.reason = "load".into();
    load.sample_count = 16;
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_router(&slot, &load).unwrap();
    assert!(!on_slot.report.compared);
    assert!(!on_slot.report.load_recorded);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_router_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.compared = true;
    let err = measure_router(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("router outputs are not compared"),
        "{err}"
    );

    let mut bad = observe();
    bad.sample_count = 17;
    let err = measure_router(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("router sample exceeds the record cap"),
        "{err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_router_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_router(&plan, &observe()).unwrap();
    write_router_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.router-report"
    );
    let again = write_router_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-router-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_router_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
