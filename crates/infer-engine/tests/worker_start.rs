use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_worker_start, reference_engine_build,
    reference_kernel_bundle, verify_worker_start, write_synthetic_model, write_worker_start_report,
    WorkerStartObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-worker-start"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-worker-start-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(failure: &str, listener_up: bool, spawned: bool) -> WorkerStartObservation {
    WorkerStartObservation {
        engine_build_root: engine_root(),
        start_failure: failure.into(),
        code: "WORKER_START_FAILED".into(),
        retryable: true,
        receipt_stored: false,
        listener_up,
        process_spawned: spawned,
        worker_ready: false,
        restart_count: 0,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_worker_start_report_records_a_missing_binary_a_late_worker_and_a_socket() {
    let dir = scratch("start");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let missing = measure_worker_start(&plan, &observe("missing-binary", false, false)).unwrap();
    assert!(!missing.report.listener_up);
    assert!(!missing.report.process_spawned);
    assert_eq!(missing.report.restart_count, 0);
    verify_worker_start(&missing).unwrap();

    let late = measure_worker_start(&plan, &observe("not-ready", true, true)).unwrap();
    assert!(late.report.listener_up);
    assert!(late.report.process_spawned);
    assert!(late.report.retryable);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let socket = measure_worker_start(&slot, &observe("socket", true, false)).unwrap();
    assert!(!socket.report.process_spawned);
    assert_eq!(socket.report.code, "WORKER_START_FAILED");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_worker_start_record_refuses_a_bound_missing_binary_and_a_restart() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let err = measure_worker_start(&plan, &observe("missing-binary", true, false)).unwrap_err();
    assert!(
        err.message
            .contains("a missing worker binary does not bind"),
        "{err}"
    );

    let err = measure_worker_start(&plan, &observe("not-ready", true, false)).unwrap_err();
    assert!(
        err.message
            .contains("a worker that is not ready was spawned"),
        "{err}"
    );

    let err = measure_worker_start(&plan, &observe("socket", true, true)).unwrap_err();
    assert!(
        err.message
            .contains("a socket failure does not spawn the worker"),
        "{err}"
    );

    let mut restart = observe("not-ready", true, true);
    restart.restart_count = 1;
    let err = measure_worker_start(&plan, &restart).unwrap_err();
    assert!(
        err.message
            .contains("a start failure does not count a restart"),
        "{err}"
    );

    let mut ready = observe("socket", true, false);
    ready.worker_ready = true;
    let err = measure_worker_start(&plan, &ready).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message
            .contains("a start failure leaves the worker down"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_worker_start_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_worker_start(&plan, &observe("socket", true, false)).unwrap();
    write_worker_start_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.worker-start-report"
    );
    let again = write_worker_start_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-worker-start-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_worker_start_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
