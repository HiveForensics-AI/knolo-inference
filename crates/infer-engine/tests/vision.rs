use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_vision, reference_engine_build,
    reference_kernel_bundle, verify_vision, write_synthetic_model, write_vision_report,
    VisionObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-vision"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-vision-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> VisionObservation {
    VisionObservation {
        engine_build_root: engine_root(),
        reason: "projector".into(),
        requested_patches: 4,
        projected_tokens: 0,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        projector_applied: false,
        patches_opened: false,
        embedded: false,
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
fn a_vision_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_vision(&plan, &observe()).unwrap();
    assert_eq!(measured.report.requested_patches, 4);
    assert!(!measured.report.projector_applied);
    verify_vision(&measured).unwrap();

    let mut varied = observe();
    varied.reason = "patch".into();
    let varied = measure_vision(&plan, &varied).unwrap();
    assert!(!varied.report.patches_opened);
    let mut varied = observe();
    varied.reason = "embed".into();
    let varied = measure_vision(&plan, &varied).unwrap();
    assert_eq!(varied.report.projected_tokens, 0);

    let mut slot_obs = observe();
    slot_obs.requested_patches = 16;
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_vision(&slot, &slot_obs).unwrap();
    assert_eq!(on_slot.report.requested_patches, 16);
    assert!(!on_slot.report.embedded);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_vision_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.projector_applied = true;
    let err = measure_vision(&plan, &bad).unwrap_err();
    assert!(err.message.contains("a projector is not applied"), "{err}");

    let mut bad = observe();
    bad.requested_patches = 17;
    let err = measure_vision(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("a vision patch exceeds the context"),
        "{err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_vision_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_vision(&plan, &observe()).unwrap();
    write_vision_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.vision-report"
    );
    let again = write_vision_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-vision-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_vision_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
