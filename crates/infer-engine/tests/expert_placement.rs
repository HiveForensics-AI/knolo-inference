use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_expert_placement, reference_engine_build,
    reference_kernel_bundle, verify_expert_placement, write_expert_placement_report,
    write_synthetic_model, ExpertPlacementObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-expert_placement"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-expert_placement-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> ExpertPlacementObservation {
    ExpertPlacementObservation {
        engine_build_root: engine_root(),
        reason: "asymmetric".into(),
        device_count: 2,
        experts_placed: 0,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        cpu_offload: false,
        resident_moved: false,
        tensor_parallel: false,
        automatic_fallback: false,
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
fn a_expert_placement_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_expert_placement(&plan, &observe()).unwrap();
    assert_eq!(measured.report.device_count, 2);
    assert!(!measured.report.tensor_parallel);
    verify_expert_placement(&measured).unwrap();

    let mut residency = observe();
    residency.reason = "residency".into();
    residency.device_count = 1;
    measure_expert_placement(&plan, &residency).unwrap();
    let mut offload = observe();
    offload.reason = "offload".into();
    offload.device_count = 1;
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_expert_placement(&slot, &offload).unwrap();
    assert!(!on_slot.report.cpu_offload);
    assert_eq!(on_slot.report.experts_placed, 0);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_expert_placement_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.tensor_parallel = true;
    let err = measure_expert_placement(&plan, &bad).unwrap_err();
    assert!(err.message.contains("tensor parallel stays off"), "{err}");

    let mut bad = observe();
    bad.device_count = 3;
    let err = measure_expert_placement(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("device count exceeds the record cap"),
        "{err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_expert_placement_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_expert_placement(&plan, &observe()).unwrap();
    write_expert_placement_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.expert-placement-report"
    );
    let again = write_expert_placement_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-expert_placement-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_expert_placement_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
