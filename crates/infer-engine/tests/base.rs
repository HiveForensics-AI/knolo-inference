use std::fs;
use std::path::PathBuf;

use infer_contracts::{
    decode_contract, sha256_prefixed, ErrorCode, ED25519_PUBLIC_KEY_BYTES, ED25519_SCALAR_BYTES,
    ED25519_SIGNATURE_BYTES,
};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_base, reference_engine_build,
    reference_kernel_bundle, verify_base, write_base_report, write_synthetic_model,
    BaseObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build =
        reference_engine_build(sha256_prefixed(b"knolo-infer-base"), bundle.root().unwrap())
            .unwrap();
    build.root().unwrap()
}

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-base-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    status: &str,
    multiplied: bool,
    public_key: u32,
    signature: u32,
    scalar: u32,
) -> BaseObservation {
    BaseObservation {
        engine_build_root: engine_root(),
        release_root: pin(b"release"),
        message_root: pin(b"message"),
        base_status: status.into(),
        base_multiplied: multiplied,
        public_key_bytes: public_key,
        signature_bytes: signature,
        scalar_bytes: scalar,
        point_added: false,
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
fn the_base_report_records_an_unsigned_release_and_a_multiplied_point() {
    let dir = scratch("base");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let unsigned = measure_base(&plan, &observe("unsigned-local", false, 0, 0, 0)).unwrap();
    assert!(!unsigned.report.base_multiplied);
    assert!(!unsigned.report.point_added);
    assert_eq!(unsigned.report.validation_result, "recorded");
    verify_base(&unsigned).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let multiplied = measure_base(
        &slot,
        &observe(
            "multiplied",
            true,
            ED25519_PUBLIC_KEY_BYTES,
            ED25519_SIGNATURE_BYTES,
            ED25519_SCALAR_BYTES,
        ),
    )
    .unwrap();
    assert!(multiplied.report.base_multiplied);
    assert_eq!(multiplied.report.scalar_bytes, ED25519_SCALAR_BYTES);
    assert_eq!(multiplied.report.validation_result, "verified");
    let rejected = measure_base(
        &plan,
        &observe(
            "rejected",
            true,
            ED25519_PUBLIC_KEY_BYTES,
            ED25519_SIGNATURE_BYTES,
            ED25519_SCALAR_BYTES,
        ),
    )
    .unwrap();
    assert_eq!(rejected.report.validation_result, "recorded");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_base_report_refuses_point_addition_and_a_short_scalar() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut added = observe(
        "multiplied",
        true,
        ED25519_PUBLIC_KEY_BYTES,
        ED25519_SIGNATURE_BYTES,
        ED25519_SCALAR_BYTES,
    );
    added.point_added = true;
    let err = measure_base(&plan, &added).unwrap_err();
    assert!(
        err.message.contains("the public-key point stays unadded"),
        "{err}"
    );

    let err = measure_base(&plan, &observe("unsigned-local", true, 0, 0, 0)).unwrap_err();
    assert!(
        err.message
            .contains("an unsigned release multiplies the base point"),
        "{err}"
    );

    let err = measure_base(
        &plan,
        &observe(
            "multiplied",
            true,
            ED25519_PUBLIC_KEY_BYTES,
            ED25519_SIGNATURE_BYTES,
            31,
        ),
    )
    .unwrap_err();
    assert!(
        err.message.contains("ed25519 scalars are 32 bytes"),
        "{err}"
    );

    let err = measure_base(
        &plan,
        &observe(
            "rejected",
            false,
            ED25519_PUBLIC_KEY_BYTES,
            ED25519_SIGNATURE_BYTES,
            ED25519_SCALAR_BYTES,
        ),
    )
    .unwrap_err();
    assert!(
        err.message
            .contains("a multiplied release multiplies the base point"),
        "{err}"
    );

    let measured = measure_base(&plan, &observe("unsigned-local", false, 0, 0, 0)).unwrap();
    let mut stored = measured.report.clone();
    stored.validation_result = "verified".into();
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message
            .contains("only a multiplied base point is verified"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_base_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_base(&plan, &observe("unsigned-local", false, 0, 0, 0)).unwrap();
    write_base_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.base-report"
    );
    let again = write_base_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-base-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_base_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
