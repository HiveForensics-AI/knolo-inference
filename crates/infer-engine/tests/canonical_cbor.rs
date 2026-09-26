use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_canonical_cbor, reference_engine_build,
    reference_kernel_bundle, verify_canonical_cbor, write_canonical_cbor_report,
    write_synthetic_model, CanonicalCborObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-canonical-cbor"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-canonical-cbor-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    reason: &str,
    major: u32,
    additional: u32,
    argument_read: bool,
) -> CanonicalCborObservation {
    CanonicalCborObservation {
        engine_build_root: engine_root(),
        reason: reason.into(),
        major,
        additional_info: additional,
        code: "CANONICAL_CBOR_INVALID".into(),
        retryable: false,
        argument_read,
        value_accepted: false,
        keys_ordered: false,
        reencoded: false,
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
fn a_canonical_refusal_records_indefinite_tag_order_and_shortest() {
    let dir = scratch("cbor");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let indefinite = measure_canonical_cbor(&plan, &observe("indefinite", 4, 31, false)).unwrap();
    assert!(!indefinite.report.argument_read);
    assert!(!indefinite.report.value_accepted);
    verify_canonical_cbor(&indefinite).unwrap();

    let tag = measure_canonical_cbor(&plan, &observe("tag", 7, 25, false)).unwrap();
    assert!(!tag.report.reencoded);

    let order = measure_canonical_cbor(&plan, &observe("order", 5, 2, true)).unwrap();
    assert!(order.report.argument_read);
    assert!(!order.report.keys_ordered);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let shortest = measure_canonical_cbor(&slot, &observe("shortest", 0, 24, true)).unwrap();
    assert_eq!(shortest.report.code, "CANONICAL_CBOR_INVALID");
    assert!(!shortest.report.receipt_stored);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_canonical_refusal_rejects_an_accepted_value_and_a_major_past_the_cap() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut accepted = observe("order", 5, 2, true);
    accepted.value_accepted = true;
    let err = measure_canonical_cbor(&plan, &accepted).unwrap_err();
    assert!(
        err.message
            .contains("a canonical refusal does not accept the value"),
        "{err}"
    );

    let err = measure_canonical_cbor(&plan, &observe("order", 8, 2, true)).unwrap_err();
    assert_eq!(err.code, ErrorCode::CanonicalCborInvalid);
    assert!(
        err.message.contains("CBOR major exceeds the record cap"),
        "{err}"
    );

    let err = measure_canonical_cbor(&plan, &observe("order", 5, 24, true)).unwrap_err();
    assert_eq!(err.code, ErrorCode::CanonicalCborInvalid);
    assert!(
        err.message.contains("map length exceeds the record cap"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_canonical_cbor_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_canonical_cbor(&plan, &observe("shortest", 1, 26, true)).unwrap();
    write_canonical_cbor_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.canonical-cbor-report"
    );
    let again = write_canonical_cbor_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-canonical-cbor-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_canonical_cbor_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
