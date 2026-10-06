use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, cuda_placement, load_verified_micro, measure_cuda_fault, reference_engine_build,
    reference_kernel_bundle, verify_cuda_fault, write_fault_report, write_synthetic_model,
    FaultObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-fault"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-fault-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(class: &str) -> FaultObservation {
    FaultObservation {
        engine_build_root: engine_root(),
        fault_class: class.into(),
        code: "CUDA_FAULT".into(),
        retryable: false,
        receipt_stored: false,
        listener_up: true,
        supervisor_exited: false,
        cpu_fallback: false,
        graph_captured: false,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_fault_report_records_a_kernel_fault_and_a_device_fault() {
    let dir = scratch("fault");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let slot = cuda_placement(&source).unwrap();
    let kernel = measure_cuda_fault(&slot, &observe("kernel")).unwrap();
    assert_eq!(kernel.report.code, "CUDA_FAULT");
    assert!(!kernel.report.retryable);
    assert!(!kernel.report.graph_captured);
    verify_cuda_fault(&kernel).unwrap();
    let device = measure_cuda_fault(&slot, &observe("device")).unwrap();
    assert_eq!(device.report.fault_class, "device");
    assert!(device.report.listener_up);

    let cpu = cpu_placement(&source).unwrap();
    let err = measure_cuda_fault(&cpu, &observe("kernel")).unwrap_err();
    assert!(
        err.message.contains("a cuda fault record names slot-0"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_cuda_fault_record_refuses_a_retry_and_a_graph() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let slot = cuda_placement(&source).unwrap();

    let mut retry = observe("kernel");
    retry.retryable = true;
    let err = measure_cuda_fault(&slot, &retry).unwrap_err();
    assert!(
        err.message.contains("a cuda fault record is not retryable"),
        "{err}"
    );

    let mut graph = observe("device");
    graph.graph_captured = true;
    let err = measure_cuda_fault(&slot, &graph).unwrap_err();
    assert!(err.message.contains("cuda graphs stay off"), "{err}");

    let mut fallback = observe("kernel");
    fallback.cpu_fallback = true;
    let err = measure_cuda_fault(&slot, &fallback).unwrap_err();
    assert!(
        err.message
            .contains("a cuda fault does not fall back to cpu"),
        "{err}"
    );

    let mut other = observe("kernel");
    other.fault_class = "host".into();
    let err = measure_cuda_fault(&slot, &other).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message
            .contains("field faultClass has an unsupported value"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_fault_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let slot = cuda_placement(&source).unwrap();
    let measured = measure_cuda_fault(&slot, &observe("kernel")).unwrap();
    write_fault_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.fault-report"
    );
    let again = write_fault_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-fault-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_fault_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
