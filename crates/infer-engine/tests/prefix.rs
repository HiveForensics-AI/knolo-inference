use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_prefix_reuse, reference_engine_build,
    reference_kernel_bundle, verify_prefix_reuse, write_prefix_report, write_synthetic_model,
    PrefixObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-prefix"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-prefix-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(source: &infer_engine::VerifiedWeightSource) -> PrefixObservation {
    PrefixObservation {
        model_image_root: source.image_root.clone(),
        artifact_root: source.artifact_root.clone(),
        engine_build_root: engine_root(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        lookup_count: 0,
        hit_count: 0,
        miss_count: 0,
        reused_tokens: 0,
    }
}

#[test]
fn a_cold_run_records_zero_reuse_while_the_cache_is_off() {
    let dir = scratch("fixture");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_prefix_reuse(&plan, &observe(&source)).unwrap();
    assert_eq!(measured.report.cache_policy, "off");
    assert_eq!(measured.report.lookup_count, 0);
    assert_eq!(measured.report.hit_count, 0);
    assert_eq!(measured.report.miss_count, 0);
    assert_eq!(measured.report.reused_tokens, 0);
    assert_eq!(measured.report.validation_result, "recorded");
    verify_prefix_reuse(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_prefix_reuse(&slot, &observe(&source)).unwrap();
    assert_eq!(on_slot.report.reused_tokens, 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_failed_prefix_measurement_issues_no_report() {
    let dir = scratch("reject");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let root = observe(&source);

    let mut lookups = root.clone();
    lookups.lookup_count = 1;
    let err = measure_prefix_reuse(&plan, &lookups).unwrap_err();
    assert!(
        err.message
            .contains("prefix lookups are zero while the cache is off"),
        "{err}"
    );

    let mut hits = root.clone();
    hits.hit_count = 1;
    let err = measure_prefix_reuse(&plan, &hits).unwrap_err();
    assert!(
        err.message
            .contains("prefix hits are zero while the cache is off"),
        "{err}"
    );

    let mut misses = root.clone();
    misses.miss_count = 1;
    let err = measure_prefix_reuse(&plan, &misses).unwrap_err();
    assert!(
        err.message
            .contains("prefix misses are zero while the cache is off"),
        "{err}"
    );

    let mut reused = root.clone();
    reused.reused_tokens = 1;
    let err = measure_prefix_reuse(&plan, &reused).unwrap_err();
    assert!(
        err.message
            .contains("reused tokens are zero while the cache is off"),
        "{err}"
    );

    let mut cache = root.clone();
    cache.cache_policy = "on".into();
    let err = measure_prefix_reuse(&plan, &cache).unwrap_err();
    assert!(err.message.contains("prefix cache is off"), "{err}");

    let mut mode = root.clone();
    mode.execution_mode = "throughput".into();
    let err = measure_prefix_reuse(&plan, &mode).unwrap_err();
    assert_eq!(err.code, ErrorCode::BackendNotAllowed, "{err}");

    let mut graphs = plan.clone();
    graphs.graph_capture_mode = "decode-buckets".into();
    let err = measure_prefix_reuse(&graphs, &root).unwrap_err();
    assert!(err.message.contains("graph capture is off"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_prefix_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_prefix_reuse(&plan, &observe(&source)).unwrap();
    write_prefix_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.prefix-report"
    );
    let again = write_prefix_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-prefix-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_prefix_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
