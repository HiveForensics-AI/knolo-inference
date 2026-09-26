use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use infer_contracts::{
    decode_contract, sha256_prefixed, ErrorCode, ModelConformanceReceiptV1, RECIPE_ADAPTER,
};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_recipe_status, reference_engine_build,
    reference_kernel_bundle, verify_recipe_status, write_recipe_report, write_synthetic_model,
    RecipeObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-recipe"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-recipe-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn conformance(support: &str, parity: bool) -> ModelConformanceReceiptV1 {
    ModelConformanceReceiptV1 {
        model_runtime_root: pin(b"runtime"),
        engine_build_root: engine_root(),
        adapter_id: RECIPE_ADAPTER.into(),
        logit_abs_tolerance_millionths: 1_000,
        greedy_token_parity: parity,
        support_level: support.into(),
        notes_root: pin(b"notes"),
        extensions: BTreeMap::new(),
    }
}

fn observe(source: &infer_engine::VerifiedWeightSource, support: &str) -> RecipeObservation {
    RecipeObservation {
        model_image_root: source.image_root.clone(),
        artifact_root: source.artifact_root.clone(),
        engine_build_root: engine_root(),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        adapter_id: RECIPE_ADAPTER.into(),
        security_root: pin(b"security"),
        stability_root: pin(b"stability"),
        benchmark_root: pin(b"benchmark"),
        support_level: support.into(),
    }
}

#[test]
fn the_micro_fixture_records_an_experimental_recipe() {
    let dir = scratch("experimental");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_recipe_status(
        &plan,
        &observe(&source, "experimental"),
        &conformance("experimental", true),
    )
    .unwrap();
    assert_eq!(measured.report.support_level, "experimental");
    assert_eq!(measured.report.adapter_id, "knolo.micro.v1");
    assert_eq!(measured.report.validation_result, "recorded");
    verify_recipe_status(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_recipe_status(
        &slot,
        &observe(&source, "experimental"),
        &conformance("experimental", true),
    )
    .unwrap();
    assert_eq!(on_slot.report.support_level, "experimental");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_blessed_recipe_requires_parity_and_four_distinct_receipts() {
    let dir = scratch("blessed");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_recipe_status(
        &plan,
        &observe(&source, "blessed"),
        &conformance("blessed", true),
    )
    .unwrap();
    assert_eq!(measured.report.support_level, "blessed");
    verify_recipe_status(&measured).unwrap();

    let conformant = measure_recipe_status(
        &plan,
        &observe(&source, "conformant"),
        &conformance("conformant", true),
    )
    .unwrap();
    assert_eq!(conformant.report.support_level, "conformant");

    let err = measure_recipe_status(
        &plan,
        &observe(&source, "blessed"),
        &conformance("blessed", false),
    )
    .unwrap_err();
    assert!(
        err.message.contains("fast execution is not blessed"),
        "{err}"
    );

    let err = measure_recipe_status(
        &plan,
        &observe(&source, "conformant"),
        &conformance("conformant", false),
    )
    .unwrap_err();
    assert!(
        err.message
            .contains("conformant support requires greedy token parity"),
        "{err}"
    );

    let mut repeated = observe(&source, "blessed");
    repeated.benchmark_root = repeated.security_root.clone();
    let err = measure_recipe_status(&plan, &repeated, &conformance("blessed", true)).unwrap_err();
    assert!(
        err.message.contains("blessed recipe repeats a receipt"),
        "{err}"
    );

    let mut adapter = observe(&source, "experimental");
    adapter.adapter_id = "knolo.llama.v1".into();
    let err =
        measure_recipe_status(&plan, &adapter, &conformance("experimental", true)).unwrap_err();
    assert_eq!(err.code, ErrorCode::UnsupportedArchitecture, "{err}");
    assert!(
        err.message
            .contains("architecture adapter knolo.llama.v1 is not compiled in"),
        "{err}"
    );

    let err = measure_recipe_status(
        &plan,
        &observe(&source, "blessed"),
        &conformance("experimental", true),
    )
    .unwrap_err();
    assert!(
        err.message.contains("support level does not match"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_recipe_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_recipe_status(
        &plan,
        &observe(&source, "experimental"),
        &conformance("experimental", true),
    )
    .unwrap();
    write_recipe_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.recipe-report"
    );
    let again = write_recipe_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-recipe-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_recipe_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
