use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use infer_contracts::{
    decode_contract, sha256_prefixed, ErrorCode, ModelConformanceReceiptV1, MICRO_ADAPTER,
};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_hub_record, reference_engine_build,
    reference_kernel_bundle, verify_hub_record, write_hub_report, write_synthetic_model,
    HubObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(sha256_prefixed(b"knolo-infer-hub"), bundle.root().unwrap())
        .unwrap();
    build.root().unwrap()
}

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-hub-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn conformance(support: &str, parity: bool) -> ModelConformanceReceiptV1 {
    ModelConformanceReceiptV1 {
        model_runtime_root: pin(b"runtime"),
        engine_build_root: engine_root(),
        adapter_id: MICRO_ADAPTER.into(),
        logit_abs_tolerance_millionths: 1_000,
        greedy_token_parity: parity,
        support_level: support.into(),
        notes_root: pin(b"notes"),
        extensions: BTreeMap::new(),
    }
}

fn observe(source: &infer_engine::VerifiedWeightSource, distribution: &str) -> HubObservation {
    HubObservation {
        publisher: "knolo".into(),
        model_image_root: source.image_root.clone(),
        artifact_root: source.artifact_root.clone(),
        engine_build_root: engine_root(),
        adapter_id: MICRO_ADAPTER.into(),
        quantization: "f32".into(),
        license_id: "apache-2.0".into(),
        source_provider: "local".into(),
        benchmark_root: pin(b"benchmark"),
        distribution: distribution.into(),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn an_experimental_micro_record_is_not_native_supported() {
    let dir = scratch("experimental");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_hub_record(
        &plan,
        &observe(&source, "active"),
        &conformance("experimental", true),
    )
    .unwrap();
    assert!(!measured.report.native_supported);
    assert_eq!(measured.report.source_provider, "local");
    assert_eq!(measured.report.validation_result, "recorded");
    verify_hub_record(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_hub_record(
        &slot,
        &observe(&source, "active"),
        &conformance("experimental", false),
    )
    .unwrap();
    assert!(!on_slot.report.native_supported);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_support_requires_an_active_conformant_receipt() {
    let dir = scratch("native");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let blessed = measure_hub_record(
        &plan,
        &observe(&source, "active"),
        &conformance("blessed", true),
    )
    .unwrap();
    assert!(blessed.report.native_supported);
    let conformant = measure_hub_record(
        &plan,
        &observe(&source, "active"),
        &conformance("conformant", true),
    )
    .unwrap();
    assert!(conformant.report.native_supported);

    let yanked = measure_hub_record(
        &plan,
        &observe(&source, "yanked"),
        &conformance("blessed", true),
    )
    .unwrap();
    assert!(!yanked.report.native_supported);

    let err = measure_hub_record(
        &plan,
        &observe(&source, "active"),
        &conformance("blessed", false),
    )
    .unwrap_err();
    assert!(
        err.message
            .contains("blessed support requires greedy token parity"),
        "{err}"
    );

    let err = measure_hub_record(
        &plan,
        &observe(&source, "active"),
        &conformance("conformant", false),
    )
    .unwrap_err();
    assert!(
        err.message
            .contains("conformant support requires greedy token parity"),
        "{err}"
    );

    let mut remote = observe(&source, "active");
    remote.source_provider = "huggingface".into();
    let err = measure_hub_record(&plan, &remote, &conformance("experimental", true)).unwrap_err();
    assert!(
        err.message
            .contains("the hub record does not download weights"),
        "{err}"
    );

    let mut adapter = observe(&source, "active");
    adapter.adapter_id = "knolo.llama.v1".into();
    let err = measure_hub_record(&plan, &adapter, &conformance("experimental", true)).unwrap_err();
    assert_eq!(err.code, ErrorCode::UnsupportedArchitecture, "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_hub_record_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_hub_record(
        &plan,
        &observe(&source, "active"),
        &conformance("experimental", true),
    )
    .unwrap();
    write_hub_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.hub-report"
    );
    assert!(!stored
        .windows(b"huggingface".len())
        .any(|window| window == b"huggingface"));
    let again = write_hub_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-hub-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_hub_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
