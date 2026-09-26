use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_digest_mismatch, reference_engine_build,
    reference_kernel_bundle, verify_digest_mismatch, write_digest_mismatch_report,
    write_synthetic_model, DigestMismatchObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-digest-mismatch"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-digest-mismatch-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(mismatch: &str, body_read: bool) -> DigestMismatchObservation {
    DigestMismatchObservation {
        engine_build_root: engine_root(),
        artifact_root: sha256_prefixed(b"artifact"),
        mismatch: mismatch.into(),
        code: "MODEL_DIGEST_MISMATCH".into(),
        retryable: false,
        header_parsed: false,
        body_read,
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
fn the_digest_mismatch_report_records_a_size_and_a_digest() {
    let dir = scratch("mismatch");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let size = measure_digest_mismatch(&plan, &observe("size", false)).unwrap();
    assert!(!size.report.body_read);
    assert!(!size.report.header_parsed);
    assert!(!size.report.forward_ran);
    verify_digest_mismatch(&size).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let digest = measure_digest_mismatch(&slot, &observe("digest", true)).unwrap();
    assert!(digest.report.body_read);
    assert_eq!(digest.report.code, "MODEL_DIGEST_MISMATCH");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_digest_mismatch_refuses_a_parsed_header_and_a_read_size() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut parsed = observe("digest", true);
    parsed.header_parsed = true;
    let err = measure_digest_mismatch(&plan, &parsed).unwrap_err();
    assert!(
        err.message
            .contains("a digest mismatch does not parse the header"),
        "{err}"
    );

    let err = measure_digest_mismatch(&plan, &observe("size", true)).unwrap_err();
    assert!(
        err.message
            .contains("a size mismatch does not read the body"),
        "{err}"
    );

    let err = measure_digest_mismatch(&plan, &observe("digest", false)).unwrap_err();
    assert!(
        err.message.contains("a digest mismatch reads the body"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_digest_mismatch_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_digest_mismatch(&plan, &observe("digest", true)).unwrap();
    write_digest_mismatch_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.digest-mismatch-report"
    );
    let again = write_digest_mismatch_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-digest-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_digest_mismatch_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
