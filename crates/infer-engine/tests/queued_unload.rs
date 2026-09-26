use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_queued_unload, reference_engine_build,
    reference_kernel_bundle, verify_queued_unload, write_synthetic_model, write_unload_report,
    UnloadObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-unload"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-unload-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(queued: u32, admitted: u32) -> UnloadObservation {
    let idle = admitted == 0;
    UnloadObservation {
        engine_build_root: engine_root(),
        request_id: "req-1".into(),
        queued_requests: queued,
        admitted_requests: admitted,
        queued_started: false,
        admitted_finished: true,
        lifecycle: if idle { "unloaded" } else { "unloading" }.into(),
        worker_exited: idle,
        code: "SERVICE_UNLOADED".into(),
        retryable: false,
        listener_up: true,
        model_reloaded: false,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_unload_report_keeps_admitted_work_and_exits_when_the_queue_is_alone() {
    let dir = scratch("unload");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let busy = measure_queued_unload(&plan, &observe(2, 1)).unwrap();
    assert_eq!(busy.report.lifecycle, "unloading");
    assert!(!busy.report.worker_exited);
    assert!(!busy.report.queued_started);
    assert!(!busy.report.retryable);
    verify_queued_unload(&busy).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let idle = measure_queued_unload(&slot, &observe(1, 0)).unwrap();
    assert_eq!(idle.report.lifecycle, "unloaded");
    assert!(idle.report.worker_exited);
    assert!(idle.report.listener_up);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_queued_unload_refuses_an_empty_queue_and_a_retry() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let err = measure_queued_unload(&plan, &observe(0, 0)).unwrap_err();
    assert!(
        err.message
            .contains("an unload under queue has a queued request"),
        "{err}"
    );

    let mut started = observe(2, 1);
    started.queued_started = true;
    let err = measure_queued_unload(&plan, &started).unwrap_err();
    assert!(err.message.contains("queued work does not start"), "{err}");

    let mut retry = observe(1, 0);
    retry.retryable = true;
    let err = measure_queued_unload(&plan, &retry).unwrap_err();
    assert!(
        err.message.contains("a queued unload is not retryable"),
        "{err}"
    );

    let mut early = observe(2, 1);
    early.worker_exited = true;
    let err = measure_queued_unload(&plan, &early).unwrap_err();
    assert!(
        err.message
            .contains("the worker stays until admitted work finishes"),
        "{err}"
    );

    let measured = measure_queued_unload(&plan, &observe(1, 0)).unwrap();
    let mut stored = measured.report.clone();
    stored.lifecycle = "unloading".into();
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message.contains("an empty admission is unloaded"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_unload_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_queued_unload(&plan, &observe(1, 0)).unwrap();
    write_unload_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.unload-report"
    );
    let again = write_unload_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-unload-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_unload_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
