use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_backend, reference_engine_build,
    reference_kernel_bundle, verify_backend, write_backend_report, write_synthetic_model,
    BackendObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-backend"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-backend-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(surface: &str) -> BackendObservation {
    BackendObservation {
        engine_build_root: engine_root(),
        surface: surface.into(),
        requested_mode: "throughput".into(),
        code: "BACKEND_NOT_ALLOWED".into(),
        retryable: false,
        backend_selected: false,
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
fn a_refused_backend_records_a_run_and_a_measurement() {
    let dir = scratch("backend");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let run = measure_backend(&plan, &observe("run")).unwrap();
    assert_eq!(run.report.requested_mode, "throughput");
    assert!(!run.report.backend_selected);
    assert!(!run.report.retryable);
    verify_backend(&run).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let measure = measure_backend(&slot, &observe("measure")).unwrap();
    assert_eq!(measure.report.surface, "measure");
    assert_eq!(measure.report.code, "BACKEND_NOT_ALLOWED");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_refused_backend_rejects_a_selected_backend_and_another_mode() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut selected = observe("run");
    selected.backend_selected = true;
    let err = measure_backend(&plan, &selected).unwrap_err();
    assert!(err.message.contains("the backend is not selected"), "{err}");

    let mut pinned = observe("measure");
    pinned.requested_mode = "pinned".into();
    let err = measure_backend(&plan, &pinned).unwrap_err();
    assert!(
        err.message.contains("the refused mode is throughput"),
        "{err}"
    );

    let mut throughput = observe("run");
    throughput.execution_mode = "throughput".into();
    let err = measure_backend(&plan, &throughput).unwrap_err();
    assert_eq!(err.code, infer_contracts::ErrorCode::BackendNotAllowed);
    assert!(
        err.message
            .contains("throughput execution mode is not enabled"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_backend_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_backend(&plan, &observe("run")).unwrap();
    write_backend_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.backend-report"
    );
    let again = write_backend_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-backend-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_backend_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
