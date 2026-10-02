use std::fs;
use std::path::PathBuf;

use infer_contracts::{
    decode_contract, sha256_prefixed, ED25519_PUBLIC_KEY_BYTES, ED25519_SCALAR_BYTES,
    ED25519_SIGNATURE_BYTES,
};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_signature_check, reference_engine_build,
    reference_kernel_bundle, verify_signature_check, write_signature_check_report,
    write_synthetic_model, SignatureCheckObservation,
};

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-signature-check"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-signature-check-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(status: &str, checked: bool, bytes: u32) -> SignatureCheckObservation {
    let (public_key, signature, scalar) = if bytes == 0 {
        (0, 0, 0)
    } else {
        (
            ED25519_PUBLIC_KEY_BYTES,
            ED25519_SIGNATURE_BYTES,
            ED25519_SCALAR_BYTES,
        )
    };
    SignatureCheckObservation {
        engine_build_root: engine_root(),
        release_root: pin(b"release"),
        message_root: pin(b"message"),
        check_status: status.into(),
        signature_checked: checked,
        cofactor_cleared: false,
        public_key_bytes: public_key,
        signature_bytes: signature,
        scalar_bytes: scalar,
        key_material_present: false,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_signature_check_records_a_check_a_rejection_and_an_unsigned_release() {
    let dir = scratch("check");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let checked = measure_signature_check(&plan, &observe("checked", true, 32)).unwrap();
    assert!(checked.report.signature_checked);
    assert!(!checked.report.cofactor_cleared);
    assert_eq!(checked.report.validation_result, "verified");
    verify_signature_check(&checked).unwrap();

    let rejected = measure_signature_check(&plan, &observe("rejected", true, 32)).unwrap();
    assert!(rejected.report.signature_checked);
    assert_eq!(rejected.report.validation_result, "recorded");

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let unsigned = measure_signature_check(&slot, &observe("unsigned-local", false, 0)).unwrap();
    assert!(!unsigned.report.signature_checked);
    assert_eq!(unsigned.report.public_key_bytes, 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_signature_check_refuses_a_cleared_cofactor_and_an_unrecorded_check() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut cleared = observe("checked", true, 32);
    cleared.cofactor_cleared = true;
    let err = measure_signature_check(&plan, &cleared).unwrap_err();
    assert!(
        err.message.contains("the cofactor stays uncleared"),
        "{err}"
    );

    let err = measure_signature_check(&plan, &observe("checked", false, 32)).unwrap_err();
    assert!(
        err.message
            .contains("a checked signature records the check"),
        "{err}"
    );

    let err = measure_signature_check(&plan, &observe("unsigned-local", true, 0)).unwrap_err();
    assert!(
        err.message
            .contains("an unsigned release checks a signature"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_signature_check_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_signature_check(&plan, &observe("checked", true, 32)).unwrap();
    write_signature_check_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.signature-check-report"
    );
    let again = write_signature_check_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-signature-check-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_signature_check_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
