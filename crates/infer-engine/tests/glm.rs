use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_glm, reference_engine_build,
    reference_kernel_bundle, verify_glm, write_glm_report, write_synthetic_model, GlmObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(sha256_prefixed(b"knolo-infer-glm"), bundle.root().unwrap())
        .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-glm-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> GlmObservation {
    GlmObservation {
        engine_build_root: engine_root(),
        reason: "inventory".into(),
        rejected_adapter: "knolo.glm.v1".into(),
        code: "UNSUPPORTED_ARCHITECTURE".into(),
        retryable: false,
        weights_opened: true,
        inventory_read: false,
        recipe_blessed: false,
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
fn a_glm_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_glm(&plan, &observe()).unwrap();
    assert!(measured.report.weights_opened);
    assert!(!measured.report.inventory_read);
    verify_glm(&measured).unwrap();

    let mut adapter = observe();
    adapter.reason = "adapter".into();
    adapter.weights_opened = false;
    adapter.inventory_read = false;
    measure_glm(&plan, &adapter).unwrap();
    let mut recipe = observe();
    recipe.reason = "recipe".into();
    recipe.weights_opened = true;
    recipe.inventory_read = true;
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_glm(&slot, &recipe).unwrap();
    assert!(!on_slot.report.recipe_blessed);
    assert_eq!(on_slot.report.rejected_adapter, "knolo.glm.v1");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_glm_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.rejected_adapter = "knolo.micro.v1".into();
    let err = measure_glm(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("the micro adapter is compiled in"),
        "{err}"
    );

    let mut bad = observe();
    bad.recipe_blessed = true;
    let err = measure_glm(&plan, &bad).unwrap_err();
    assert!(err.message.contains("a glm recipe is not blessed"), "{err}");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_glm_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_glm(&plan, &observe()).unwrap();
    write_glm_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.glm-report"
    );
    let again = write_glm_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-glm-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_glm_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
