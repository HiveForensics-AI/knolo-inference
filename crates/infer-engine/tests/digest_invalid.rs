use std::fs;
use std::path::PathBuf;

use infer_contracts::{
    decode_contract, sha256_prefixed, MAX_DIGEST_DOMAIN_BYTES, MAX_DIGEST_RECORD_HEX,
};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_digest_invalid, reference_engine_build,
    reference_kernel_bundle, verify_digest_invalid, write_digest_invalid_report,
    write_synthetic_model, DigestInvalidObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-digest-invalid"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-digest-invalid-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    reason: &str,
    hex_length: u32,
    domain: &str,
    prefix: bool,
    length: bool,
    alphabet: bool,
) -> DigestInvalidObservation {
    DigestInvalidObservation {
        engine_build_root: engine_root(),
        reason: reason.into(),
        hex_length,
        domain: domain.into(),
        code: "DIGEST_INVALID".into(),
        retryable: false,
        prefix_accepted: prefix,
        length_accepted: length,
        alphabet_accepted: alphabet,
        domain_accepted: false,
        hashed: false,
        file_opened: false,
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
fn an_invalid_digest_records_the_prefix_the_length_the_alphabet_and_the_domain() {
    let dir = scratch("digest");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let prefix =
        measure_digest_invalid(&plan, &observe("prefix", 0, "none", false, false, false)).unwrap();
    assert!(!prefix.report.prefix_accepted);
    assert!(!prefix.report.hashed);
    verify_digest_invalid(&prefix).unwrap();

    let length =
        measure_digest_invalid(&plan, &observe("length", 32, "none", true, false, false)).unwrap();
    assert!(length.report.prefix_accepted);
    assert!(!length.report.length_accepted);

    let alphabet =
        measure_digest_invalid(&plan, &observe("alphabet", 64, "none", true, true, false)).unwrap();
    assert!(alphabet.report.length_accepted);
    assert!(!alphabet.report.alphabet_accepted);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let domain = measure_digest_invalid(
        &slot,
        &observe("domain", 64, "not-a-domain", true, true, true),
    )
    .unwrap();
    assert_eq!(domain.report.domain, "not-a-domain");
    assert!(!domain.report.domain_accepted);
    assert!(!domain.report.file_opened);
    assert_eq!(domain.report.code, "DIGEST_INVALID");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_invalid_digest_refuses_a_hash_a_known_domain_and_a_hex_past_the_cap() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut hashed = observe("prefix", 0, "none", false, false, false);
    hashed.hashed = true;
    let err = measure_digest_invalid(&plan, &hashed).unwrap_err();
    assert!(
        err.message
            .contains("a digest refusal does not hash the payload"),
        "{err}"
    );

    let err = measure_digest_invalid(
        &plan,
        &observe("domain", 64, "infer-receipt", true, true, true),
    )
    .unwrap_err();
    assert!(
        err.message
            .contains("a domain refusal names a domain outside the allowlist"),
        "{err}"
    );

    let err = measure_digest_invalid(
        &plan,
        &observe(
            "length",
            MAX_DIGEST_RECORD_HEX + 1,
            "none",
            true,
            false,
            false,
        ),
    )
    .unwrap_err();
    assert_eq!(err.code, infer_contracts::ErrorCode::DigestInvalid);
    assert!(
        err.message.contains("digest hex exceeds the record cap"),
        "{err}"
    );

    let domain = "d".repeat(MAX_DIGEST_DOMAIN_BYTES as usize + 1);
    let err = measure_digest_invalid(&plan, &observe("domain", 64, &domain, true, true, true))
        .unwrap_err();
    assert_eq!(err.code, infer_contracts::ErrorCode::DigestInvalid);
    assert!(
        err.message.contains("domain exceeds the record cap"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_digest_invalid_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured =
        measure_digest_invalid(&plan, &observe("length", 63, "none", true, false, false)).unwrap();
    write_digest_invalid_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.digest-invalid-report"
    );
    let again = write_digest_invalid_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-digest-invalid-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_digest_invalid_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
