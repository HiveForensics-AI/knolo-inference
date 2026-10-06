use std::fs;
use std::path::PathBuf;

use infer_contracts::{
    decode_contract, sha256_prefixed, ErrorCode, ED25519_PUBLIC_KEY_BYTES, ED25519_SCALAR_BYTES,
    ED25519_SIGNATURE_BYTES,
};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_scalar, reference_engine_build,
    reference_kernel_bundle, verify_scalar, write_scalar_report, write_synthetic_model,
    ScalarObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-scalar"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-scalar-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(status: &str, reduced: bool, bytes: u32) -> ScalarObservation {
    let (public_key, signature, scalar) = if bytes == 0 {
        (0, 0, 0)
    } else {
        (
            ED25519_PUBLIC_KEY_BYTES,
            ED25519_SIGNATURE_BYTES,
            ED25519_SCALAR_BYTES,
        )
    };
    ScalarObservation {
        engine_build_root: engine_root(),
        release_root: pin(b"release"),
        message_root: pin(b"message"),
        scalar_status: status.into(),
        scalar_reduced: reduced,
        public_multiplied: false,
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
fn the_scalar_report_records_a_reduction_a_rejection_and_an_unsigned_release() {
    let dir = scratch("scalar");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let reduced = measure_scalar(&plan, &observe("reduced", true, 32)).unwrap();
    assert!(reduced.report.scalar_reduced);
    assert!(!reduced.report.public_multiplied);
    assert_eq!(reduced.report.validation_result, "verified");
    verify_scalar(&reduced).unwrap();

    let rejected = measure_scalar(&plan, &observe("rejected", true, 32)).unwrap();
    assert!(rejected.report.scalar_reduced);
    assert_eq!(rejected.report.validation_result, "recorded");

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let unsigned = measure_scalar(&slot, &observe("unsigned-local", false, 0)).unwrap();
    assert!(!unsigned.report.scalar_reduced);
    assert_eq!(unsigned.report.public_key_bytes, 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_scalar_record_refuses_a_multiplied_public_key_and_an_unreduced_scalar() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut multiplied = observe("reduced", true, 32);
    multiplied.public_multiplied = true;
    let err = measure_scalar(&plan, &multiplied).unwrap_err();
    assert!(
        err.message
            .contains("the public-key multiplication stays uncomputed"),
        "{err}"
    );

    let err = measure_scalar(&plan, &observe("reduced", false, 32)).unwrap_err();
    assert!(
        err.message
            .contains("a reduced scalar records the reduction"),
        "{err}"
    );

    let err = measure_scalar(&plan, &observe("unsigned-local", true, 0)).unwrap_err();
    assert!(
        err.message
            .contains("an unsigned release reduces the scalar"),
        "{err}"
    );

    let mut material = observe("rejected", true, 32);
    material.key_material_present = true;
    let err = measure_scalar(&plan, &material).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message.contains("key material stays in host storage"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_scalar_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_scalar(&plan, &observe("reduced", true, 32)).unwrap();
    write_scalar_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.scalar-report"
    );
    let again = write_scalar_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-scalar-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_scalar_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
