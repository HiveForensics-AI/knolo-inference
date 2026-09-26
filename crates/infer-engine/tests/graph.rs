use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_cuda_graph, reference_engine_build,
    reference_kernel_bundle, verify_cuda_graph, write_cuda_graph_report, write_synthetic_model,
    GraphObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-graph"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-graph-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(reason: &str) -> GraphObservation {
    GraphObservation {
        engine_build_root: engine_root(),
        reason: reason.into(),
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        graph_captured: false,
        captured_nodes: 0,
        workspace_bytes: 0,
        shape_buckets: 0,
        kernel_repeated: false,
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
fn a_cuda_graph_record_names_slot_0_and_keeps_capture_at_zero() {
    let dir = scratch("graph");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let shape = measure_cuda_graph(&slot, &observe("shape")).unwrap();
    assert_eq!(shape.report.device, "slot-0");
    assert!(!shape.report.graph_captured);
    assert_eq!(shape.report.captured_nodes, 0);
    verify_cuda_graph(&shape).unwrap();

    let identity = measure_cuda_graph(&slot, &observe("identity")).unwrap();
    assert_eq!(identity.report.workspace_bytes, 0);
    let kernel = measure_cuda_graph(&slot, &observe("kernel")).unwrap();
    assert!(!kernel.report.kernel_repeated);
    let workspace = measure_cuda_graph(&slot, &observe("workspace")).unwrap();
    assert_eq!(workspace.report.shape_buckets, 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_cuda_graph_record_rejects_cpu_and_a_captured_graph() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let err = measure_cuda_graph(&plan, &observe("shape")).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid);
    assert!(
        err.message.contains("a cuda graph record names slot-0"),
        "{err}"
    );

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let mut captured = observe("identity");
    captured.graph_captured = true;
    let err = measure_cuda_graph(&slot, &captured).unwrap_err();
    assert!(err.message.contains("cuda graphs stay off"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_cuda_graph_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let measured = measure_cuda_graph(&slot, &observe("workspace")).unwrap();
    write_cuda_graph_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.graph-report"
    );
    let again = write_cuda_graph_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-graph-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_cuda_graph_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
