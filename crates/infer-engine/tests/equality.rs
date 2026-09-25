use std::fs;
use std::path::PathBuf;

use infer_contracts::{
    decode_contract, sha256_prefixed, ErrorCode, ED25519_PUBLIC_KEY_BYTES, ED25519_SCALAR_BYTES,
    ED25519_SIGNATURE_BYTES,
};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_point_equality, reference_engine_build,
    reference_kernel_bundle, verify_point_equality, write_equality_report, write_synthetic_model,
    EqualityObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-equality"),
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
        "knolo-infer-equality-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(status: &str, compared: bool, equal: bool, bytes: u32) -> EqualityObservation {
    let (public_key, signature, scalar) = if bytes == 0 {
        (0, 0, 0)
    } else {
        (
            ED25519_PUBLIC_KEY_BYTES,
            ED25519_SIGNATURE_BYTES,
            ED25519_SCALAR_BYTES,
        )
    };
    EqualityObservation {
        engine_build_root: engine_root(),
        release_root: pin(b"release"),
        message_root: pin(b"message"),
        equality_status: status.into(),
        points_compared: compared,
        points_equal: equal,
        challenge_hashed: false,
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
fn the_equality_report_records_an_unsigned_release_and_an_equal_comparison() {
    let dir = scratch("equality");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let unsigned =
        measure_point_equality(&plan, &observe("unsigned-local", false, false, 0)).unwrap();
    assert!(!unsigned.report.points_compared);
    assert!(!unsigned.report.points_equal);
    assert!(!unsigned.report.challenge_hashed);
    assert_eq!(unsigned.report.validation_result, "recorded");
    verify_point_equality(&unsigned).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let equal = measure_point_equality(&slot, &observe("equal", true, true, 32)).unwrap();
    assert!(equal.report.points_equal);
    assert_eq!(equal.report.validation_result, "verified");
    let unequal = measure_point_equality(&plan, &observe("unequal", true, false, 32)).unwrap();
    assert!(!unequal.report.points_equal);
    assert!(unequal.report.points_compared);
    assert_eq!(unequal.report.validation_result, "recorded");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_point_comparison_refuses_a_challenge_hash_and_a_false_equal_result() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut hashed = observe("equal", true, true, 32);
    hashed.challenge_hashed = true;
    let err = measure_point_equality(&plan, &hashed).unwrap_err();
    assert!(
        err.message.contains("the challenge hash stays uncomputed"),
        "{err}"
    );

    let err = measure_point_equality(&plan, &observe("equal", true, false, 32)).unwrap_err();
    assert!(
        err.message
            .contains("an equal comparison says the points are equal"),
        "{err}"
    );

    let err = measure_point_equality(&plan, &observe("unequal", true, true, 32)).unwrap_err();
    assert!(
        err.message
            .contains("an unequal comparison says the points differ"),
        "{err}"
    );

    let err =
        measure_point_equality(&plan, &observe("unsigned-local", true, false, 0)).unwrap_err();
    assert!(
        err.message
            .contains("an unsigned release compares the points"),
        "{err}"
    );

    let err = measure_point_equality(&plan, &observe("equal", true, true, 0)).unwrap_err();
    assert!(
        err.message.contains("ed25519 public keys are 32 bytes"),
        "{err}"
    );

    let measured =
        measure_point_equality(&plan, &observe("unsigned-local", false, false, 0)).unwrap();
    let mut stored = measured.report.clone();
    stored.validation_result = "verified".into();
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message
            .contains("only an equal point comparison is verified"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_equality_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured =
        measure_point_equality(&plan, &observe("unsigned-local", false, false, 0)).unwrap();
    write_equality_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.equality-report"
    );
    let again = write_equality_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-equality-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_equality_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
