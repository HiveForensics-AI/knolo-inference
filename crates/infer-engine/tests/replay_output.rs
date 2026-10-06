use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_replay_output, reference_engine_build,
    reference_kernel_bundle, verify_replay_output, write_replay_output_report,
    write_synthetic_model, ReplayOutputObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-replay-output"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-replay-output-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> ReplayOutputObservation {
    ReplayOutputObservation {
        engine_build_root: engine_root(),
        receipt_root: sha256_prefixed(b"receipt"),
        candidate_output_root: sha256_prefixed(b"candidate"),
        code: "REPLAY_OUTPUT_MISMATCH".into(),
        retryable: false,
        check_stored: false,
        forward_ran: true,
        environment_matched: true,
        assurance: "incomplete".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_replay_output_report_records_a_candidate_that_differs() {
    let dir = scratch("output");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_replay_output(&plan, &observe()).unwrap();
    assert!(measured.report.forward_ran);
    assert!(measured.report.environment_matched);
    assert!(!measured.report.check_stored);
    assert_eq!(measured.report.assurance, "incomplete");
    verify_replay_output(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let slot_report = measure_replay_output(&slot, &observe()).unwrap();
    assert_eq!(slot_report.report.code, "REPLAY_OUTPUT_MISMATCH");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_replay_output_record_refuses_a_skipped_forward_and_a_repeated_receipt() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut skipped = observe();
    skipped.forward_ran = false;
    let err = measure_replay_output(&plan, &skipped).unwrap_err();
    assert!(
        err.message.contains("an output mismatch ran the forward"),
        "{err}"
    );

    let mut environment = observe();
    environment.environment_matched = false;
    let err = measure_replay_output(&plan, &environment).unwrap_err();
    assert!(
        err.message
            .contains("an output mismatch follows a matching environment"),
        "{err}"
    );

    let mut repeated = observe();
    repeated.candidate_output_root = repeated.receipt_root.clone();
    let err = measure_replay_output(&plan, &repeated).unwrap_err();
    assert!(
        err.message
            .contains("the candidate output repeats the receipt"),
        "{err}"
    );

    let mut retry = observe();
    retry.retryable = true;
    let err = measure_replay_output(&plan, &retry).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message.contains("an output mismatch is not retryable"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_replay_output_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_replay_output(&plan, &observe()).unwrap();
    write_replay_output_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.replay-output-report"
    );
    let again = write_replay_output_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-replay-output-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_replay_output_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
