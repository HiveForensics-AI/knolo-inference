use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_kv_utilization, reference_engine_build,
    reference_kernel_bundle, verify_kv_utilization, write_kv_report, write_synthetic_model,
    KvObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build =
        reference_engine_build(sha256_prefixed(b"knolo-infer-kv"), bundle.root().unwrap()).unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-kv-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(source: &infer_engine::VerifiedWeightSource, pages: u32, tokens: u32) -> KvObservation {
    KvObservation {
        model_image_root: source.image_root.clone(),
        artifact_root: source.artifact_root.clone(),
        engine_build_root: engine_root(),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        peak_pages: pages,
        peak_tokens: tokens,
    }
}

#[test]
fn a_cold_run_records_one_page_of_the_eight_page_pool() {
    let dir = scratch("fixture");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_kv_utilization(&plan, &observe(&source, 1, 16)).unwrap();
    assert_eq!(measured.report.page_total, 8);
    assert_eq!(measured.report.page_size_tokens, 16);
    assert_eq!(measured.report.peak_pages, 1);
    assert_eq!(measured.report.peak_tokens, 16);
    assert_eq!(measured.report.utilization_millionths, 125_000);
    assert_eq!(measured.report.validation_result, "recorded");
    verify_kv_utilization(&measured).unwrap();

    let one = measure_kv_utilization(&plan, &observe(&source, 1, 1)).unwrap();
    assert_eq!(one.report.utilization_millionths, 7_812);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_kv_utilization(&slot, &observe(&source, 1, 16)).unwrap();
    assert_eq!(on_slot.report.utilization_millionths, 125_000);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_failed_kv_measurement_issues_no_report() {
    let dir = scratch("reject");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let root = observe(&source, 1, 16);

    let mut zero = root.clone();
    zero.peak_tokens = 0;
    let err = measure_kv_utilization(&plan, &zero).unwrap_err();
    assert!(err.message.contains("kv peak tokens are zero"), "{err}");

    let mut over = root.clone();
    over.peak_tokens = 17;
    let err = measure_kv_utilization(&plan, &over).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContextLimitExceeded, "{err}");
    assert!(
        err.message
            .contains("kv peak tokens exceed the micro context"),
        "{err}"
    );

    let mut pages = root.clone();
    pages.peak_pages = 2;
    let err = measure_kv_utilization(&plan, &pages).unwrap_err();
    assert!(
        err.message.contains("the micro fixture occupies one page"),
        "{err}"
    );

    let mut mode = root.clone();
    mode.execution_mode = "throughput".into();
    let err = measure_kv_utilization(&plan, &mode).unwrap_err();
    assert_eq!(err.code, ErrorCode::BackendNotAllowed, "{err}");

    let mut graphs = plan.clone();
    graphs.graph_capture_mode = "decode-buckets".into();
    let err = measure_kv_utilization(&graphs, &root).unwrap_err();
    assert!(err.message.contains("graph capture is off"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_kv_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_kv_utilization(&plan, &observe(&source, 1, 4)).unwrap();
    assert_eq!(measured.report.utilization_millionths, 31_250);
    write_kv_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.kv-report"
    );
    let again = write_kv_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-kv-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_kv_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
