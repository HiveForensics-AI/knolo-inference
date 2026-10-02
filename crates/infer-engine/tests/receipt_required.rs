use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, RECEIPT_HTTP_STATUS};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_receipt_required, reference_engine_build,
    reference_kernel_bundle, verify_receipt_required, write_receipt_required_report,
    write_synthetic_model, ReceiptRequiredObservation,
};

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-receipt-required"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-receipt-required-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    reason: &str,
    status: u32,
    read: bool,
    bound: bool,
    journal: bool,
) -> ReceiptRequiredObservation {
    ReceiptRequiredObservation {
        engine_build_root: engine_root(),
        receipt_root: pin(b"receipt"),
        reason: reason.into(),
        code: "RECEIPT_REQUIRED".into(),
        retryable: false,
        http_status: status,
        receipt_read: read,
        request_bound: bound,
        journal_opened: journal,
        event_count: 0,
        listener_up: true,
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
fn a_required_receipt_records_a_missing_file_a_missing_request_and_an_empty_journal() {
    let dir = scratch("required");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let file = measure_receipt_required(
        &plan,
        &observe("file", RECEIPT_HTTP_STATUS, false, false, false),
    )
    .unwrap();
    assert_eq!(file.report.http_status, 404);
    assert!(!file.report.receipt_read);
    assert!(file.report.listener_up);
    verify_receipt_required(&file).unwrap();

    let request =
        measure_receipt_required(&plan, &observe("request", 0, true, false, false)).unwrap();
    assert!(request.report.receipt_read);
    assert!(!request.report.request_bound);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let journal =
        measure_receipt_required(&slot, &observe("journal", 0, true, true, true)).unwrap();
    assert!(journal.report.journal_opened);
    assert_eq!(journal.report.event_count, 0);
    assert_eq!(journal.report.code, "RECEIPT_REQUIRED");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_required_receipt_refuses_a_stored_receipt_and_a_journal_that_has_events() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut stored = observe("file", RECEIPT_HTTP_STATUS, false, false, false);
    stored.receipt_stored = true;
    let err = measure_receipt_required(&plan, &stored).unwrap_err();
    assert!(
        err.message.contains("a missing receipt stores no receipt"),
        "{err}"
    );

    let mut events = observe("journal", 0, true, true, true);
    events.event_count = 1;
    let err = measure_receipt_required(&plan, &events).unwrap_err();
    assert!(
        err.message
            .contains("a required receipt has no journal events"),
        "{err}"
    );

    let err =
        measure_receipt_required(&plan, &observe("request", 404, true, false, false)).unwrap_err();
    assert!(
        err.message
            .contains("a verify-path refusal has no HTTP status"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_receipt_required_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_receipt_required(
        &plan,
        &observe("file", RECEIPT_HTTP_STATUS, false, false, false),
    )
    .unwrap();
    write_receipt_required_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.receipt-required-report"
    );
    let again = write_receipt_required_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-receipt-required-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_receipt_required_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
