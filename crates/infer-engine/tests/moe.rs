use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_moe, reference_engine_build,
    reference_kernel_bundle, verify_moe, write_moe_report, write_synthetic_model, MoeObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(sha256_prefixed(b"knolo-infer-moe"), bundle.root().unwrap())
        .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-moe-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> MoeObservation {
    MoeObservation {
        engine_build_root: engine_root(),
        reason: "router".into(),
        requested_experts: 4,
        selected_experts: 0,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        routed: false,
        shared_used: false,
        tie_broken: false,
        grouped: false,
        expert_placed: false,
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
fn a_moe_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_moe(&plan, &observe()).unwrap();
    assert_eq!(measured.report.requested_experts, 4);
    assert!(!measured.report.routed);
    verify_moe(&measured).unwrap();

    let mut expert = observe();
    expert.reason = "expert".into();
    measure_moe(&plan, &expert).unwrap();
    let mut shared = observe();
    shared.reason = "shared".into();
    shared.requested_experts = 16;
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_moe(&slot, &shared).unwrap();
    assert_eq!(on_slot.report.selected_experts, 0);
    assert!(!on_slot.report.expert_placed);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_moe_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.routed = true;
    let err = measure_moe(&plan, &bad).unwrap_err();
    assert!(err.message.contains("a router does not run"), "{err}");

    let mut bad = observe();
    bad.requested_experts = 17;
    let err = measure_moe(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("expert count exceeds the record cap"),
        "{err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_moe_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_moe(&plan, &observe()).unwrap();
    write_moe_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.moe-report"
    );
    let again = write_moe_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-moe-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_moe_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
