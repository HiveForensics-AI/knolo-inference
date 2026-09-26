use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, MAX_PEAK_BYTES};
use infer_engine::{
    cpu_placement, cuda_placement, load_verified_micro, measure_cuda_oom, reference_engine_build,
    reference_kernel_bundle, verify_cuda_oom, write_oom_report, write_synthetic_model,
    OomObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(sha256_prefixed(b"knolo-infer-oom"), bundle.root().unwrap())
        .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-oom-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(needed: u64) -> OomObservation {
    OomObservation {
        engine_build_root: engine_root(),
        free_bytes: 0,
        needed_bytes: needed,
        code: "CUDA_OOM".into(),
        retryable: true,
        receipt_stored: false,
        listener_up: true,
        supervisor_exited: false,
        cpu_fallback: false,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_oom_report_records_slot_0_and_refuses_cpu() {
    let dir = scratch("oom");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let slot = cuda_placement(&source).unwrap();
    let measured = measure_cuda_oom(&slot, &observe(MAX_PEAK_BYTES)).unwrap();
    assert_eq!(measured.report.device, "slot-0");
    assert_eq!(measured.report.needed_bytes, MAX_PEAK_BYTES);
    assert!(measured.report.retryable);
    assert!(!measured.report.cpu_fallback);
    assert!(measured.report.listener_up);
    verify_cuda_oom(&measured).unwrap();

    let cpu = cpu_placement(&source).unwrap();
    let err = measure_cuda_oom(&cpu, &observe(4096)).unwrap_err();
    assert!(
        err.message.contains("a cuda oom record names slot-0"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_cuda_oom_record_refuses_a_fallback_and_an_oversized_request() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let slot = cuda_placement(&source).unwrap();

    let mut fallback = observe(4096);
    fallback.cpu_fallback = true;
    let err = measure_cuda_oom(&slot, &fallback).unwrap_err();
    assert!(
        err.message.contains("a cuda oom does not fall back to cpu"),
        "{err}"
    );

    let mut exited = observe(4096);
    exited.supervisor_exited = true;
    let err = measure_cuda_oom(&slot, &exited).unwrap_err();
    assert!(
        err.message
            .contains("a cuda oom does not exit the supervisor"),
        "{err}"
    );

    let err = measure_cuda_oom(&slot, &observe(0)).unwrap_err();
    assert!(
        err.message.contains("a cuda oom record needs device bytes"),
        "{err}"
    );

    let err = measure_cuda_oom(&slot, &observe(MAX_PEAK_BYTES + 1)).unwrap_err();
    assert_eq!(err.code, ErrorCode::InsufficientMemory, "{err}");
    assert!(err.message.contains("needed bytes exceed 64 MiB"), "{err}");

    let measured = measure_cuda_oom(&slot, &observe(4096)).unwrap();
    let mut stored = measured.report.clone();
    stored.needed_bytes = MAX_PEAK_BYTES + 1;
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_oom_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let slot = cuda_placement(&source).unwrap();
    let measured = measure_cuda_oom(&slot, &observe(4096)).unwrap();
    write_oom_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.oom-report"
    );
    let again = write_oom_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-oom-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_oom_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
