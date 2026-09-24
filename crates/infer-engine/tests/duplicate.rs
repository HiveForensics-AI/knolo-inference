use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_duplicate, reference_engine_build,
    reference_kernel_bundle, verify_duplicate, write_duplicate_report, write_synthetic_model,
    DuplicateObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-duplicate"),
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
        "knolo-infer-duplicate-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> DuplicateObservation {
    DuplicateObservation {
        engine_build_root: engine_root(),
        request_id: "held".into(),
        occupant_root: pin(b"occupant"),
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        duplicate_started: false,
        occupant_kept: true,
        second_journal: false,
        listener_up: true,
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
fn the_duplicate_report_keeps_the_admitted_request() {
    let dir = scratch("duplicate");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_duplicate(&plan, &observe()).unwrap();
    assert_eq!(measured.report.code, "CONTRACT_INVALID");
    assert!(!measured.report.duplicate_started);
    assert!(measured.report.occupant_kept);
    assert!(!measured.report.second_journal);
    assert!(!measured.report.receipt_stored);
    assert!(measured.report.listener_up);
    verify_duplicate(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_duplicate(&slot, &observe()).unwrap();
    assert_eq!(on_slot.report.request_id, "held");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_duplicate_report_refuses_a_started_copy_and_a_repeated_occupant() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut started = observe();
    started.duplicate_started = true;
    let err = measure_duplicate(&plan, &started).unwrap_err();
    assert!(
        err.message.contains("a duplicate request does not start"),
        "{err}"
    );

    let mut journal = observe();
    journal.second_journal = true;
    let err = measure_duplicate(&plan, &journal).unwrap_err();
    assert!(
        err.message.contains("a duplicate request opens no journal"),
        "{err}"
    );

    let mut retryable = observe();
    retryable.retryable = true;
    let err = measure_duplicate(&plan, &retryable).unwrap_err();
    assert!(
        err.message
            .contains("a duplicate request id is not retryable"),
        "{err}"
    );

    let mut repeated = observe();
    repeated.occupant_root = engine_root();
    let err = measure_duplicate(&plan, &repeated).unwrap_err();
    assert!(
        err.message
            .contains("the occupant repeats the engine build"),
        "{err}"
    );

    let measured = measure_duplicate(&plan, &observe()).unwrap();
    let mut stored = measured.report.clone();
    stored.receipt_stored = true;
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message
            .contains("a duplicate request stores no receipt"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_duplicate_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_duplicate(&plan, &observe()).unwrap();
    write_duplicate_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.duplicate-report"
    );
    let again = write_duplicate_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-duplicate-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_duplicate_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
