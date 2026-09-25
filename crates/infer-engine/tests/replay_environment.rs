use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_replay_environment, reference_engine_build,
    reference_kernel_bundle, verify_replay_environment, write_replay_environment_report,
    write_synthetic_model, ReplayEnvironmentObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-replay-environment"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-replay-environment-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(field: &str) -> ReplayEnvironmentObservation {
    ReplayEnvironmentObservation {
        engine_build_root: engine_root(),
        receipt_root: sha256_prefixed(b"receipt"),
        mismatched_field: field.into(),
        code: "REPLAY_ENVIRONMENT_MISMATCH".into(),
        retryable: false,
        check_stored: false,
        forward_ran: false,
        output_compared: false,
        assurance: "incomplete".into(),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_replay_environment_report_records_each_mismatched_field() {
    let dir = scratch("environment");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    for field in ["prompt", "sampler", "model", "engine", "placement"] {
        let measured = measure_replay_environment(&plan, &observe(field)).unwrap();
        assert!(!measured.report.forward_ran);
        assert!(!measured.report.output_compared);
        assert!(!measured.report.retryable);
        assert_eq!(measured.report.assurance, "incomplete");
        verify_replay_environment(&measured).unwrap();
    }
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let slot_report = measure_replay_environment(&slot, &observe("engine")).unwrap();
    assert_eq!(slot_report.report.code, "REPLAY_ENVIRONMENT_MISMATCH");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_replay_environment_record_refuses_a_forward_and_a_stored_check() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut forward = observe("prompt");
    forward.forward_ran = true;
    let err = measure_replay_environment(&plan, &forward).unwrap_err();
    assert!(
        err.message
            .contains("an environment mismatch does not run the forward"),
        "{err}"
    );

    let mut stored = observe("sampler");
    stored.check_stored = true;
    let err = measure_replay_environment(&plan, &stored).unwrap_err();
    assert!(
        err.message
            .contains("an environment mismatch stores no replay check"),
        "{err}"
    );

    let mut compared = observe("model");
    compared.output_compared = true;
    let err = measure_replay_environment(&plan, &compared).unwrap_err();
    assert!(
        err.message
            .contains("an environment mismatch does not compare output tokens"),
        "{err}"
    );

    let mut retry = observe("placement");
    retry.retryable = true;
    let err = measure_replay_environment(&plan, &retry).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message
            .contains("an environment mismatch is not retryable"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_replay_environment_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_replay_environment(&plan, &observe("engine")).unwrap();
    write_replay_environment_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.replay-environment-report"
    );
    let again = write_replay_environment_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!(
        "knolo-replay-environment-not-created-{}",
        std::process::id()
    );
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_replay_environment_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
