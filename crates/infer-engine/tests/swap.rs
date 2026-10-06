use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_model_swap, reference_engine_build,
    reference_kernel_bundle, verify_model_swap, write_swap_report, write_synthetic_model,
    SwapObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build =
        reference_engine_build(sha256_prefixed(b"knolo-infer-swap"), bundle.root().unwrap())
            .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-swap-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    source: &infer_engine::VerifiedWeightSource,
    unload: u64,
    verify: u64,
    load: u64,
) -> SwapObservation {
    SwapObservation {
        model_image_root: source.image_root.clone(),
        artifact_root: source.artifact_root.clone(),
        resident_model_image_root: sha256_prefixed(b"resident-image"),
        resident_artifact_root: sha256_prefixed(b"resident-artifact"),
        engine_build_root: engine_root(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        unload_nanos: unload,
        incoming_verification_nanos: verify,
        incoming_load_nanos: load,
    }
}

#[test]
fn a_cold_swap_records_the_sum_of_the_three_durations() {
    let dir = scratch("fixture");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    assert_eq!(plan.graph_capture_mode, "off");

    let swap = measure_model_swap(&plan, &observe(&source, 50, 30, 80)).unwrap();
    assert_eq!(swap.report.model_swap_nanos, 160);
    assert_eq!(swap.report.validation_result, "recorded");
    assert_ne!(
        swap.report.resident_model_image_root,
        swap.report.model_image_root
    );
    verify_model_swap(&swap).unwrap();

    let short = measure_model_swap(&plan, &observe(&source, 10, 20, 5)).unwrap();
    assert_eq!(short.report.model_swap_nanos, 35);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_model_swap(&slot, &observe(&source, 1, 2, 3)).unwrap();
    assert_eq!(on_slot.report.model_swap_nanos, 6);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_failed_swap_issues_no_report() {
    let dir = scratch("reject");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let root = observe(&source, 50, 30, 80);

    let mut mode = root.clone();
    mode.execution_mode = "throughput".into();
    let err = measure_model_swap(&plan, &mode).unwrap_err();
    assert_eq!(err.code, ErrorCode::BackendNotAllowed, "{err}");

    let mut same_image = root.clone();
    same_image.resident_model_image_root = same_image.model_image_root.clone();
    let err = measure_model_swap(&plan, &same_image).unwrap_err();
    assert!(
        err.message
            .contains("a swap replaces a different model image"),
        "{err}"
    );

    let mut same_artifact = root.clone();
    same_artifact.resident_artifact_root = same_artifact.artifact_root.clone();
    let err = measure_model_swap(&plan, &same_artifact).unwrap_err();
    assert!(
        err.message.contains("a swap replaces a different artifact"),
        "{err}"
    );

    let mut zero = root.clone();
    zero.unload_nanos = 0;
    let err = measure_model_swap(&plan, &zero).unwrap_err();
    assert!(err.message.contains("unload time is zero"), "{err}");

    let mut overflow = root.clone();
    overflow.unload_nanos = u64::MAX;
    overflow.incoming_verification_nanos = 1;
    overflow.incoming_load_nanos = 1;
    let err = measure_model_swap(&plan, &overflow).unwrap_err();
    assert!(err.message.contains("model swap time overflows"), "{err}");

    let mut graphs = plan.clone();
    graphs.graph_capture_mode = "decode-buckets".into();
    let err = measure_model_swap(&graphs, &root).unwrap_err();
    assert!(err.message.contains("graph capture is off"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_swap_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_model_swap(&plan, &observe(&source, 4, 5, 6)).unwrap();
    write_swap_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.swap-report"
    );
    let again = write_swap_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-swap-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_swap_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
