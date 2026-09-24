use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_signature_gate, reference_engine_build,
    reference_kernel_bundle, verify_signature_gate, write_signature_report, write_synthetic_model,
    SignatureObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-signature"),
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
        "knolo-infer-signature-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(status: &str, count: u32, key_id: &str, bytes: u32) -> SignatureObservation {
    SignatureObservation {
        engine_build_root: engine_root(),
        release_root: pin(b"release"),
        signature_status: status.into(),
        signature_count: count,
        key_id: key_id.into(),
        signature_bytes: bytes,
        key_verified: false,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_signature_gate_records_unsigned_and_shape_checked_blocks() {
    let dir = scratch("signature");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let unsigned = measure_signature_gate(&plan, &observe("unsigned-local", 0, "", 0)).unwrap();
    assert_eq!(unsigned.report.signature_status, "unsigned-local");
    assert!(!unsigned.report.key_verified);
    assert!(unsigned.report.key_id.is_empty());
    verify_signature_gate(&unsigned).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let checked =
        measure_signature_gate(&slot, &observe("shape-checked", 1, "local-dev", 64)).unwrap();
    assert_eq!(checked.report.key_id, "local-dev");
    assert_eq!(checked.report.signature_bytes, 64);
    assert!(!checked.report.key_verified);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_signature_gate_refuses_key_verification_and_the_wrong_shape() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut verified = observe("shape-checked", 1, "local-dev", 64);
    verified.key_verified = true;
    let err = measure_signature_gate(&plan, &verified).unwrap_err();
    assert!(
        err.message.contains("signature keys stay unverified"),
        "{err}"
    );

    let err =
        measure_signature_gate(&plan, &observe("shape-checked", 1, "local-dev", 63)).unwrap_err();
    assert!(
        err.message.contains("ed25519 signatures are 64 bytes"),
        "{err}"
    );

    let err =
        measure_signature_gate(&plan, &observe("unsigned-local", 0, "local-dev", 0)).unwrap_err();
    assert!(
        err.message.contains("an unsigned release names a key"),
        "{err}"
    );

    let err = measure_signature_gate(&plan, &observe("unsigned-local", 0, "", 64)).unwrap_err();
    assert!(
        err.message
            .contains("an unsigned release carries signature bytes"),
        "{err}"
    );

    let mut repeated = observe("unsigned-local", 0, "", 0);
    repeated.release_root = repeated.engine_build_root.clone();
    let err = measure_signature_gate(&plan, &repeated).unwrap_err();
    assert!(
        err.message.contains("the release repeats the engine build"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_signature_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured =
        measure_signature_gate(&plan, &observe("shape-checked", 1, "local-dev", 64)).unwrap();
    write_signature_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.signature-report"
    );
    let secret = b"ed25519-private-key-material";
    assert!(!stored.windows(secret.len()).any(|window| window == secret));
    let again = write_signature_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-signature-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_signature_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
