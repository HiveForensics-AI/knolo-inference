use std::fs;
use std::path::PathBuf;

use infer_contracts::{
    decode_contract, sha256_prefixed, ED25519_PUBLIC_KEY_BYTES, ED25519_SCALAR_BYTES,
    ED25519_SIGNATURE_BYTES,
};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_cofactor, reference_engine_build,
    reference_kernel_bundle, verify_cofactor, write_cofactor_report, write_synthetic_model,
    CofactorObservation,
};

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-cofactor"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-cofactor-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(status: &str, cleared: bool, bytes: u32) -> CofactorObservation {
    let (public_key, signature, scalar) = if bytes == 0 {
        (0, 0, 0)
    } else {
        (
            ED25519_PUBLIC_KEY_BYTES,
            ED25519_SIGNATURE_BYTES,
            ED25519_SCALAR_BYTES,
        )
    };
    CofactorObservation {
        engine_build_root: engine_root(),
        release_root: pin(b"release"),
        message_root: pin(b"message"),
        clear_status: status.into(),
        cofactor_cleared: cleared,
        receipt_signed: false,
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
fn the_cofactor_clear_records_a_clear_a_rejection_and_an_unsigned_release() {
    let dir = scratch("clear");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let cleared = measure_cofactor(&plan, &observe("cleared", true, 32)).unwrap();
    assert!(cleared.report.cofactor_cleared);
    assert!(!cleared.report.receipt_signed);
    assert_eq!(cleared.report.validation_result, "verified");
    verify_cofactor(&cleared).unwrap();

    let rejected = measure_cofactor(&plan, &observe("rejected", true, 32)).unwrap();
    assert!(rejected.report.cofactor_cleared);
    assert_eq!(rejected.report.validation_result, "recorded");

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let unsigned = measure_cofactor(&slot, &observe("unsigned-local", false, 0)).unwrap();
    assert!(!unsigned.report.cofactor_cleared);
    assert_eq!(unsigned.report.public_key_bytes, 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_cofactor_clear_refuses_a_signed_receipt_and_an_unrecorded_clear() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut signed = observe("cleared", true, 32);
    signed.receipt_signed = true;
    let err = measure_cofactor(&plan, &signed).unwrap_err();
    assert!(err.message.contains("the receipt stays unsigned"), "{err}");

    let err = measure_cofactor(&plan, &observe("cleared", false, 32)).unwrap_err();
    assert!(
        err.message.contains("a cleared cofactor records the clear"),
        "{err}"
    );

    let err = measure_cofactor(&plan, &observe("unsigned-local", true, 0)).unwrap_err();
    assert!(
        err.message
            .contains("an unsigned release clears the cofactor"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_cofactor_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_cofactor(&plan, &observe("cleared", true, 32)).unwrap();
    write_cofactor_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.cofactor-report"
    );
    let again = write_cofactor_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-cofactor-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_cofactor_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
