use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, HTTP_SERVICE_UNAVAILABLE};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_worker_lost, reference_engine_build,
    reference_kernel_bundle, verify_worker_lost, write_synthetic_model, write_worker_lost_report,
    WorkerLostObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-worker-lost"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-worker-lost-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> WorkerLostObservation {
    WorkerLostObservation {
        engine_build_root: engine_root(),
        request_id: "req-1".into(),
        code: "WORKER_LOST".into(),
        retryable: true,
        receipt_stored: false,
        listener_up: true,
        supervisor_exited: false,
        http_status: HTTP_SERVICE_UNAVAILABLE,
        journal_sealed: true,
        restart_counted: true,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_worker_lost_report_records_a_sealed_journal_and_a_restart() {
    let dir = scratch("lost");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_worker_lost(&plan, &observe()).unwrap();
    assert!(measured.report.listener_up);
    assert!(measured.report.journal_sealed);
    assert!(measured.report.restart_counted);
    assert!(!measured.report.receipt_stored);
    assert_eq!(measured.report.http_status, 503);
    verify_worker_lost(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let slot_report = measure_worker_lost(&slot, &observe()).unwrap();
    assert_eq!(slot_report.report.code, "WORKER_LOST");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_worker_lost_record_refuses_an_open_journal_and_a_missed_restart() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut open = observe();
    open.journal_sealed = false;
    let err = measure_worker_lost(&plan, &open).unwrap_err();
    assert!(
        err.message.contains("a lost worker seals the open journal"),
        "{err}"
    );

    let mut restart = observe();
    restart.restart_counted = false;
    let err = measure_worker_lost(&plan, &restart).unwrap_err();
    assert!(
        err.message.contains("a lost worker counts a restart"),
        "{err}"
    );

    let mut stored = observe();
    stored.receipt_stored = true;
    let err = measure_worker_lost(&plan, &stored).unwrap_err();
    assert!(
        err.message.contains("a lost worker stores no receipt"),
        "{err}"
    );

    let mut down = observe();
    down.listener_up = false;
    let err = measure_worker_lost(&plan, &down).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(err.message.contains("the listener stays up"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_worker_lost_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_worker_lost(&plan, &observe()).unwrap();
    write_worker_lost_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.worker-lost-report"
    );
    let again = write_worker_lost_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-worker-lost-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_worker_lost_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
