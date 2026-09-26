use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_model_load, reference_engine_build,
    reference_kernel_bundle, verify_model_load, write_load_report, write_synthetic_model,
    LoadObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build =
        reference_engine_build(sha256_prefixed(b"knolo-infer-load"), bundle.root().unwrap())
            .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-load-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(source: &infer_engine::VerifiedWeightSource, nanos: u64) -> LoadObservation {
    LoadObservation {
        model_image_root: source.image_root.clone(),
        artifact_root: source.artifact_root.clone(),
        engine_build_root: engine_root(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        model_load_nanos: nanos,
    }
}

#[test]
fn a_cold_load_records_the_duration_after_verification() {
    let dir = scratch("fixture");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_model_load(&plan, &observe(&source, 80)).unwrap();
    assert_eq!(measured.report.model_load_nanos, 80);
    assert_eq!(measured.report.validation_result, "recorded");
    verify_model_load(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_model_load(&slot, &observe(&source, 3)).unwrap();
    assert_eq!(on_slot.report.model_load_nanos, 3);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_failed_load_issues_no_report() {
    let dir = scratch("reject");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let root = observe(&source, 80);

    let mut zero = root.clone();
    zero.model_load_nanos = 0;
    let err = measure_model_load(&plan, &zero).unwrap_err();
    assert!(err.message.contains("model load time is zero"), "{err}");

    let mut mode = root.clone();
    mode.execution_mode = "throughput".into();
    let err = measure_model_load(&plan, &mode).unwrap_err();
    assert_eq!(err.code, ErrorCode::BackendNotAllowed, "{err}");

    let mut graphs = plan.clone();
    graphs.graph_capture_mode = "decode-buckets".into();
    let err = measure_model_load(&graphs, &root).unwrap_err();
    assert!(err.message.contains("graph capture is off"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_load_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_model_load(&plan, &observe(&source, 9)).unwrap();
    write_load_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.load-report"
    );
    let again = write_load_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-load-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_load_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
