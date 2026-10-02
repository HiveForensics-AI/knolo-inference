use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, MAX_FIELD_RECORD_BYTES};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_contract_invalid, reference_engine_build,
    reference_kernel_bundle, verify_contract_invalid, write_contract_invalid_report,
    write_synthetic_model, ContractInvalidObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-contract-invalid"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-contract-invalid-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    reason: &str,
    field_bytes: u32,
    present: bool,
    type_accepted: bool,
) -> ContractInvalidObservation {
    ContractInvalidObservation {
        engine_build_root: engine_root(),
        reason: reason.into(),
        field_bytes,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        field_present: present,
        type_accepted,
        value_accepted: false,
        decoded: false,
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
fn a_contract_refusal_records_a_missing_field_a_type_an_unknown_field_and_a_value() {
    let dir = scratch("contract");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let missing = measure_contract_invalid(&plan, &observe("missing", 0, false, false)).unwrap();
    assert!(!missing.report.field_present);
    assert!(!missing.report.decoded);
    verify_contract_invalid(&missing).unwrap();

    let typed = measure_contract_invalid(&plan, &observe("type", 4, true, false)).unwrap();
    assert!(typed.report.field_present);
    assert!(!typed.report.type_accepted);

    let unknown = measure_contract_invalid(&plan, &observe("unknown", 80, true, false)).unwrap();
    assert_eq!(unknown.report.field_bytes, MAX_FIELD_RECORD_BYTES);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let value = measure_contract_invalid(&slot, &observe("value", 6, true, true)).unwrap();
    assert!(value.report.type_accepted);
    assert!(!value.report.value_accepted);
    assert!(!value.report.retryable);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_contract_refusal_rejects_a_decoded_document_and_a_field_past_the_cap() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut decoded = observe("unknown", 12, true, false);
    decoded.decoded = true;
    let err = measure_contract_invalid(&plan, &decoded).unwrap_err();
    assert!(
        err.message
            .contains("a contract refusal does not decode the document"),
        "{err}"
    );

    let err = measure_contract_invalid(&plan, &observe("unknown", 81, true, false)).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid);
    assert!(
        err.message.contains("field exceeds the record cap"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_contract_invalid_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_contract_invalid(&plan, &observe("value", 3, true, true)).unwrap();
    write_contract_invalid_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.contract-invalid-report"
    );
    let again = write_contract_invalid_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-contract-invalid-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_contract_invalid_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
