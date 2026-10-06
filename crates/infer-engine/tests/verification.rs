use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, MAX_VERIFIED_BYTES};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_model_verification, reference_engine_build,
    reference_kernel_bundle, verify_model_verification, write_synthetic_model,
    write_verification_report, VerificationObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-verification"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-verification-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    source: &infer_engine::VerifiedWeightSource,
    bytes: u64,
    nanos: u64,
) -> VerificationObservation {
    VerificationObservation {
        model_image_root: source.image_root.clone(),
        artifact_root: source.artifact_root.clone(),
        engine_build_root: engine_root(),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        verified_bytes: bytes,
        model_verification_nanos: nanos,
    }
}

#[test]
fn a_cold_verification_records_the_byte_count_and_the_duration() {
    let dir = scratch("fixture");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_model_verification(&plan, &observe(&source, 4096, 40)).unwrap();
    assert_eq!(measured.report.verified_bytes, 4096);
    assert_eq!(measured.report.model_verification_nanos, 40);
    assert_eq!(measured.report.validation_result, "recorded");
    verify_model_verification(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_model_verification(&slot, &observe(&source, 8, 1)).unwrap();
    assert_eq!(on_slot.report.model_verification_nanos, 1);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_failed_verification_issues_no_report() {
    let dir = scratch("reject");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let root = observe(&source, 4096, 40);

    let mut zero_bytes = root.clone();
    zero_bytes.verified_bytes = 0;
    let err = measure_model_verification(&plan, &zero_bytes).unwrap_err();
    assert!(err.message.contains("verified byte count is zero"), "{err}");

    let mut over = root.clone();
    over.verified_bytes = MAX_VERIFIED_BYTES + 1;
    let err = measure_model_verification(&plan, &over).unwrap_err();
    assert_eq!(err.code, ErrorCode::InsufficientMemory, "{err}");
    assert!(
        err.message.contains("verified bytes exceed the parser cap"),
        "{err}"
    );

    let cap = measure_model_verification(&plan, &observe(&source, MAX_VERIFIED_BYTES, 2)).unwrap();
    assert_eq!(cap.report.verified_bytes, MAX_VERIFIED_BYTES);

    let mut zero_time = root.clone();
    zero_time.model_verification_nanos = 0;
    let err = measure_model_verification(&plan, &zero_time).unwrap_err();
    assert!(
        err.message.contains("model verification time is zero"),
        "{err}"
    );

    let mut mode = root.clone();
    mode.execution_mode = "throughput".into();
    let err = measure_model_verification(&plan, &mode).unwrap_err();
    assert_eq!(err.code, ErrorCode::BackendNotAllowed, "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_verification_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_model_verification(&plan, &observe(&source, 12, 7)).unwrap();
    write_verification_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.verification-report"
    );
    let again = write_verification_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-verification-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_verification_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
