use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_mtp, reference_engine_build,
    reference_kernel_bundle, verify_mtp, write_mtp_report, write_synthetic_model, MtpObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(sha256_prefixed(b"knolo-infer-mtp"), bundle.root().unwrap())
        .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-mtp-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> MtpObservation {
    MtpObservation {
        engine_build_root: engine_root(),
        reason: "head".into(),
        proposal_tokens: 4,
        accepted_tokens: 0,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        head_ran: false,
        drafted: false,
        accepted: false,
        speculated: false,
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
fn a_mtp_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_mtp(&plan, &observe()).unwrap();
    assert_eq!(measured.report.proposal_tokens, 4);
    assert!(!measured.report.head_ran);
    verify_mtp(&measured).unwrap();

    let mut varied = observe();
    varied.reason = "draft".into();
    let varied = measure_mtp(&plan, &varied).unwrap();
    assert!(!varied.report.drafted);
    let mut varied = observe();
    varied.reason = "accept".into();
    let varied = measure_mtp(&plan, &varied).unwrap();
    assert_eq!(varied.report.accepted_tokens, 0);

    let mut slot_obs = observe();
    slot_obs.proposal_tokens = 16;
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_mtp(&slot, &slot_obs).unwrap();
    assert_eq!(on_slot.report.proposal_tokens, 16);
    assert!(!on_slot.report.speculated);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_mtp_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.speculated = true;
    let err = measure_mtp(&plan, &bad).unwrap_err();
    assert!(err.message.contains("speculation stays off"), "{err}");

    let mut bad = observe();
    bad.proposal_tokens = 17;
    let err = measure_mtp(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("an MTP proposal exceeds the context"),
        "{err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_mtp_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_mtp(&plan, &observe()).unwrap();
    write_mtp_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.mtp-report"
    );
    let again = write_mtp_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-mtp-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_mtp_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
