use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, HTTP_TIMEOUT_STATUS};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_timeout, reference_engine_build,
    reference_kernel_bundle, verify_timeout, write_synthetic_model, write_timeout_report,
    TimeoutObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-timeout"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-timeout-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(stage: &str) -> TimeoutObservation {
    TimeoutObservation {
        engine_build_root: engine_root(),
        request_id: "req-1".into(),
        timeout_stage: stage.into(),
        waited_nanos: 1_000,
        code: "REQUEST_TIMEOUT".into(),
        retryable: true,
        receipt_stored: false,
        listener_up: true,
        worker_lost: false,
        http_status: HTTP_TIMEOUT_STATUS,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_timeout_report_records_admission_completion_and_cancellation() {
    let dir = scratch("timeout");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    for stage in ["admission", "completion", "cancellation"] {
        let measured = measure_timeout(&plan, &observe(stage)).unwrap();
        assert_eq!(measured.report.timeout_stage, stage);
        assert_eq!(measured.report.http_status, HTTP_TIMEOUT_STATUS);
        assert!(measured.report.retryable);
        assert!(!measured.report.worker_lost);
        verify_timeout(&measured).unwrap();
    }
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let device = measure_timeout(&slot, &observe("completion")).unwrap();
    assert_eq!(device.report.code, "REQUEST_TIMEOUT");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_timeout_record_refuses_a_zero_wait_and_a_lost_worker() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut zero = observe("admission");
    zero.waited_nanos = 0;
    let err = measure_timeout(&plan, &zero).unwrap_err();
    assert!(err.message.contains("a timeout record waits"), "{err}");

    let mut lost = observe("completion");
    lost.worker_lost = true;
    let err = measure_timeout(&plan, &lost).unwrap_err();
    assert!(
        err.message.contains("a timeout is not a lost worker"),
        "{err}"
    );

    let mut status = observe("cancellation");
    status.http_status = 503;
    let err = measure_timeout(&plan, &status).unwrap_err();
    assert!(
        err.message.contains("a timeout record is HTTP 504"),
        "{err}"
    );

    let other = observe("queue");
    let err = measure_timeout(&plan, &other).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message
            .contains("field timeoutStage has an unsupported value"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_timeout_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_timeout(&plan, &observe("completion")).unwrap();
    write_timeout_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.timeout-report"
    );
    let again = write_timeout_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-timeout-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_timeout_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
