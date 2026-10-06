use std::fs;
use std::path::PathBuf;

use infer_contracts::{
    decode_contract, sha256_prefixed, ErrorCode, ED25519_PUBLIC_KEY_BYTES, ED25519_SCALAR_BYTES,
    ED25519_SIGNATURE_BYTES,
};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_point, reference_engine_build,
    reference_kernel_bundle, verify_point, write_point_report, write_synthetic_model,
    PointObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-point"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-point-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    status: &str,
    added: bool,
    public_key: u32,
    signature: u32,
    scalar: u32,
) -> PointObservation {
    PointObservation {
        engine_build_root: engine_root(),
        release_root: pin(b"release"),
        message_root: pin(b"message"),
        point_status: status.into(),
        point_added: added,
        public_key_bytes: public_key,
        signature_bytes: signature,
        scalar_bytes: scalar,
        points_equal: false,
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
fn the_point_report_records_an_unsigned_release_and_an_added_point() {
    let dir = scratch("point");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let unsigned = measure_point(&plan, &observe("unsigned-local", false, 0, 0, 0)).unwrap();
    assert!(!unsigned.report.point_added);
    assert!(!unsigned.report.points_equal);
    assert_eq!(unsigned.report.validation_result, "recorded");
    verify_point(&unsigned).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let added = measure_point(
        &slot,
        &observe(
            "added",
            true,
            ED25519_PUBLIC_KEY_BYTES,
            ED25519_SIGNATURE_BYTES,
            ED25519_SCALAR_BYTES,
        ),
    )
    .unwrap();
    assert!(added.report.point_added);
    assert_eq!(added.report.scalar_bytes, ED25519_SCALAR_BYTES);
    assert_eq!(added.report.validation_result, "verified");
    let rejected = measure_point(
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
fn a_point_report_refuses_a_comparison_and_a_short_scalar() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut compared = observe(
        "added",
        true,
        ED25519_PUBLIC_KEY_BYTES,
        ED25519_SIGNATURE_BYTES,
        ED25519_SCALAR_BYTES,
    );
    compared.points_equal = true;
    let err = measure_point(&plan, &compared).unwrap_err();
    assert!(err.message.contains("the points stay uncompared"), "{err}");

    let err = measure_point(&plan, &observe("unsigned-local", true, 0, 0, 0)).unwrap_err();
    assert!(
        err.message
            .contains("an unsigned release adds the public-key point"),
        "{err}"
    );

    let err = measure_point(
        &plan,
        &observe(
            "added",
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

    let err = measure_point(
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
            .contains("an added release adds the public-key point"),
        "{err}"
    );

    let measured = measure_point(&plan, &observe("unsigned-local", false, 0, 0, 0)).unwrap();
    let mut stored = measured.report.clone();
    stored.validation_result = "verified".into();
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message
            .contains("only an added public-key point is verified"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_point_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_point(&plan, &observe("unsigned-local", false, 0, 0, 0)).unwrap();
    write_point_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.point-report"
    );
    let again = write_point_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-point-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_point_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
