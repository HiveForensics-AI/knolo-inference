use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_positional, reference_engine_build,
    reference_kernel_bundle, verify_positional, write_positional_report, write_synthetic_model,
    PositionalObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-positional"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-positional-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> PositionalObservation {
    PositionalObservation {
        engine_build_root: engine_root(),
        reason: "yarn".into(),
        requested_scale: 4,
        applied_scale: 0,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        yarn_applied: false,
        ntk_applied: false,
        rope_scaled: false,
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
fn a_positional_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_positional(&plan, &observe()).unwrap();
    assert_eq!(measured.report.requested_scale, 4);
    assert!(!measured.report.yarn_applied);
    verify_positional(&measured).unwrap();

    let mut varied = observe();
    varied.reason = "ntk".into();
    let varied = measure_positional(&plan, &varied).unwrap();
    assert!(!varied.report.ntk_applied);
    let mut varied = observe();
    varied.reason = "rope".into();
    let varied = measure_positional(&plan, &varied).unwrap();
    assert_eq!(varied.report.applied_scale, 0);

    let mut slot_obs = observe();
    slot_obs.requested_scale = 16;
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_positional(&slot, &slot_obs).unwrap();
    assert_eq!(on_slot.report.requested_scale, 16);
    assert!(!on_slot.report.rope_scaled);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_positional_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.yarn_applied = true;
    let err = measure_positional(&plan, &bad).unwrap_err();
    assert!(err.message.contains("a YaRN scale is not applied"), "{err}");

    let mut bad = observe();
    bad.requested_scale = 17;
    let err = measure_positional(&plan, &bad).unwrap_err();
    assert!(
        err.message
            .contains("positional scale exceeds the record cap"),
        "{err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_positional_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_positional(&plan, &observe()).unwrap();
    write_positional_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.positional-report"
    );
    let again = write_positional_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-positional-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_positional_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
