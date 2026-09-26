use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_rollback, reference_engine_build,
    reference_kernel_bundle, verify_rollback, write_rollback_report, write_synthetic_model,
    RollbackObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-rollback"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-rollback-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(reason: &str) -> RollbackObservation {
    RollbackObservation {
        engine_build_root: engine_root(),
        previous_image_root: pin(b"previous-image"),
        incoming_image_root: pin(b"incoming-image"),
        previous_artifact_root: pin(b"previous-artifact"),
        incoming_artifact_root: pin(b"incoming-artifact"),
        reason: reason.into(),
        lockfile_mutated: false,
        core_lock_touched: false,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_rollback_report_records_a_distinct_incoming_pin() {
    let dir = scratch("rollback");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let recorded = measure_rollback(&plan, &observe("pin-mismatch")).unwrap();
    assert!(!recorded.report.lockfile_mutated);
    assert!(!recorded.report.core_lock_touched);
    assert_eq!(recorded.report.validation_result, "recorded");
    verify_rollback(&recorded).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let operator = measure_rollback(&slot, &observe("operator")).unwrap();
    assert_eq!(operator.report.reason, "operator");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_rollback_report_refuses_a_mutated_lockfile_and_a_repeated_pin() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut mutated = observe("failed-load");
    mutated.lockfile_mutated = true;
    let err = measure_rollback(&plan, &mutated).unwrap_err();
    assert!(
        err.message
            .contains("a rollback record leaves the lockfile unchanged"),
        "{err}"
    );

    let mut core = observe("operator");
    core.core_lock_touched = true;
    let err = measure_rollback(&plan, &core).unwrap_err();
    assert!(
        err.message
            .contains("a rollback record leaves the core lockfile unchanged"),
        "{err}"
    );

    let mut same = observe("pin-mismatch");
    same.incoming_image_root = same.previous_image_root.clone();
    let err = measure_rollback(&plan, &same).unwrap_err();
    assert!(
        err.message
            .contains("the incoming image repeats the previous image"),
        "{err}"
    );

    let err = measure_rollback(&plan, &observe("download")).unwrap_err();
    assert!(
        err.message
            .contains("field reason has an unsupported value"),
        "{err}"
    );

    let measured = measure_rollback(&plan, &observe("pin-mismatch")).unwrap();
    let mut stored = measured.report.clone();
    stored.validation_result = "verified".into();
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message
            .contains("field validationResult has an unsupported value"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_rollback_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_rollback(&plan, &observe("operator")).unwrap();
    write_rollback_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.rollback-report"
    );
    let again = write_rollback_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-rollback-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_rollback_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
