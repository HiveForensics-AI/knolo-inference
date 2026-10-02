use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_receipt_store, reference_engine_build,
    reference_kernel_bundle, verify_receipt_store, write_receipt_store_report,
    write_synthetic_model, ReceiptStoreObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-receipt-store"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-receipt-store-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> ReceiptStoreObservation {
    ReceiptStoreObservation {
        engine_build_root: engine_root(),
        request_id: "req-1".into(),
        code: "RECEIPT_PERSIST_FAILED".into(),
        retryable: true,
        receipt_stored: false,
        journal_event: "failed".into(),
        partial_receipt: "absent".into(),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_receipt_store_report_records_a_retryable_failure_without_a_receipt() {
    let dir = scratch("store");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let recorded = measure_receipt_store(&plan, &observe()).unwrap();
    assert!(!recorded.report.receipt_stored);
    assert!(recorded.report.retryable);
    assert_eq!(recorded.report.journal_event, "failed");
    assert_eq!(recorded.report.partial_receipt, "absent");
    assert_eq!(recorded.report.validation_result, "recorded");
    verify_receipt_store(&recorded).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let cuda = measure_receipt_store(&slot, &observe()).unwrap();
    assert_eq!(cuda.report.code, "RECEIPT_PERSIST_FAILED");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_receipt_store_report_refuses_a_stored_receipt_and_a_quiet_journal() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut stored_receipt = observe();
    stored_receipt.receipt_stored = true;
    let err = measure_receipt_store(&plan, &stored_receipt).unwrap_err();
    assert!(
        err.message
            .contains("a receipt-store failure stores no receipt"),
        "{err}"
    );

    let mut quiet = observe();
    quiet.journal_event = "accepted".into();
    let err = measure_receipt_store(&plan, &quiet).unwrap_err();
    assert!(err.message.contains("the journal seals failed"), "{err}");

    let mut held = observe();
    held.retryable = false;
    let err = measure_receipt_store(&plan, &held).unwrap_err();
    assert!(
        err.message.contains("a receipt-store failure is retryable"),
        "{err}"
    );

    let mut other = observe();
    other.code = "WORKER_LOST".into();
    let err = measure_receipt_store(&plan, &other).unwrap_err();
    assert!(
        err.message
            .contains("a receipt-store failure is RECEIPT_PERSIST_FAILED"),
        "{err}"
    );

    let mut repeated = observe();
    repeated.partial_receipt = repeated.engine_build_root.as_str().into();
    let err = measure_receipt_store(&plan, &repeated).unwrap_err();
    assert!(
        err.message
            .contains("the partial receipt repeats the engine build"),
        "{err}"
    );

    let measured = measure_receipt_store(&plan, &observe()).unwrap();
    let mut stored = measured.report.clone();
    stored.validation_result = "verified".into();
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_receipt_store_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_receipt_store(&plan, &observe()).unwrap();
    write_receipt_store_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.receipt-store-report"
    );
    let again = write_receipt_store_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-receipt-store-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_receipt_store_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
