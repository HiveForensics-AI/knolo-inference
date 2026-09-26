use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, HTTP_SERVICE_UNAVAILABLE};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_draining, reference_engine_build,
    reference_kernel_bundle, verify_draining, write_draining_report, write_synthetic_model,
    DrainingObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-draining"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-draining-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(lifecycle: &str) -> DrainingObservation {
    DrainingObservation {
        engine_build_root: engine_root(),
        lifecycle: lifecycle.into(),
        code: "SERVICE_DRAINING".into(),
        retryable: true,
        receipt_stored: false,
        body_parsed: false,
        listener_up: true,
        worker_loaded: true,
        restart_counted: false,
        http_status: HTTP_SERVICE_UNAVAILABLE,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_draining_report_records_a_draining_and_a_drained_refusal() {
    let dir = scratch("drain");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let draining = measure_draining(&plan, &observe("draining")).unwrap();
    assert!(draining.report.listener_up);
    assert!(draining.report.worker_loaded);
    assert!(!draining.report.body_parsed);
    assert!(!draining.report.restart_counted);
    assert_eq!(draining.report.http_status, 503);
    verify_draining(&draining).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let drained = measure_draining(&slot, &observe("drained")).unwrap();
    assert_eq!(drained.report.code, "SERVICE_DRAINING");
    assert!(drained.report.retryable);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_draining_record_refuses_a_parsed_body_and_a_counted_restart() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut parsed = observe("draining");
    parsed.body_parsed = true;
    let err = measure_draining(&plan, &parsed).unwrap_err();
    assert!(
        err.message
            .contains("a draining refusal does not parse the body"),
        "{err}"
    );

    let mut restart = observe("drained");
    restart.restart_counted = true;
    let err = measure_draining(&plan, &restart).unwrap_err();
    assert!(
        err.message
            .contains("a draining refusal does not count a restart"),
        "{err}"
    );

    let mut unloaded = observe("draining");
    unloaded.worker_loaded = false;
    let err = measure_draining(&plan, &unloaded).unwrap_err();
    assert!(
        err.message
            .contains("a draining refusal leaves the worker loaded"),
        "{err}"
    );

    let mut stored = observe("drained");
    stored.receipt_stored = true;
    let err = measure_draining(&plan, &stored).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message.contains("a draining refusal stores no receipt"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_draining_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_draining(&plan, &observe("draining")).unwrap();
    write_draining_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.draining-report"
    );
    let again = write_draining_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-draining-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_draining_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
