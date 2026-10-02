use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_concurrent_load, reference_engine_build,
    reference_kernel_bundle, verify_concurrent_load, write_concurrent_load_report,
    write_synthetic_model, ConcurrentLoadObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-concurrent-load"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-concurrent-load-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    phase: &str,
    lifecycle: &str,
    inflight: u32,
    started: bool,
) -> ConcurrentLoadObservation {
    ConcurrentLoadObservation {
        engine_build_root: engine_root(),
        phase: phase.into(),
        lifecycle: lifecycle.into(),
        inflight,
        worker_started: started,
        second_worker: false,
        worker_replaced: false,
        restart_count: 0,
        listener_up: true,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_concurrent_load_report_records_ready_unloading_and_down() {
    let dir = scratch("load");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let ready = measure_concurrent_load(&plan, &observe("ready", "serving", 2, false)).unwrap();
    assert!(!ready.report.worker_started);
    assert!(!ready.report.second_worker);
    assert!(!ready.report.worker_replaced);
    assert_eq!(ready.report.restart_count, 0);
    verify_concurrent_load(&ready).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let unloading =
        measure_concurrent_load(&slot, &observe("unloading", "unloading", 1, false)).unwrap();
    assert_eq!(unloading.report.lifecycle, "unloading");
    assert_eq!(unloading.report.inflight, 1);

    let down = measure_concurrent_load(&plan, &observe("down", "serving", 0, true)).unwrap();
    assert!(down.report.worker_started);
    assert_eq!(down.report.inflight, 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_concurrent_load_report_refuses_a_second_worker_and_an_empty_unload() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut second = observe("ready", "serving", 0, false);
    second.second_worker = true;
    let err = measure_concurrent_load(&plan, &second).unwrap_err();
    assert!(
        err.message.contains("a concurrent load starts one worker"),
        "{err}"
    );

    let mut replaced = observe("down", "serving", 0, true);
    replaced.worker_replaced = true;
    let err = measure_concurrent_load(&plan, &replaced).unwrap_err();
    assert!(
        err.message
            .contains("a concurrent load does not replace a live worker"),
        "{err}"
    );

    let err = measure_concurrent_load(&plan, &observe("ready", "serving", 0, true)).unwrap_err();
    assert!(
        err.message.contains("a ready worker is left running"),
        "{err}"
    );

    let err =
        measure_concurrent_load(&plan, &observe("unloading", "unloading", 0, false)).unwrap_err();
    assert!(
        err.message
            .contains("an unloading load still has admitted work"),
        "{err}"
    );

    let err = measure_concurrent_load(&plan, &observe("down", "serving", 1, true)).unwrap_err();
    assert!(
        err.message.contains("a down load has no admitted work"),
        "{err}"
    );

    let err = measure_concurrent_load(&plan, &observe("ready", "serving", 17, false)).unwrap_err();
    assert!(
        err.message
            .contains("a concurrent load admits at most 16 requests"),
        "{err}"
    );

    let measured = measure_concurrent_load(&plan, &observe("ready", "serving", 0, false)).unwrap();
    let mut stored = measured.report.clone();
    stored.restart_count = 1;
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message
            .contains("a concurrent load does not count a restart"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_concurrent_load_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_concurrent_load(&plan, &observe("down", "serving", 0, true)).unwrap();
    write_concurrent_load_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.concurrent-load-report"
    );
    let again = write_concurrent_load_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-concurrent-load-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_concurrent_load_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
