use std::fs;
use std::path::PathBuf;

use infer_contracts::{
    decode_contract, sha256_prefixed, ErrorCode, ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES,
};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_curve, reference_engine_build,
    reference_kernel_bundle, verify_curve, write_curve_report, write_synthetic_model,
    CurveObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-curve"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-curve-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    status: &str,
    computed: bool,
    public_key: u32,
    signature: u32,
    checked: bool,
) -> CurveObservation {
    CurveObservation {
        engine_build_root: engine_root(),
        release_root: pin(b"release"),
        message_root: pin(b"message"),
        curve_status: status.into(),
        curve_computed: computed,
        public_key_bytes: public_key,
        signature_bytes: signature,
        point_checked: checked,
        base_multiplied: false,
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
fn the_curve_report_records_an_unsigned_release_and_an_on_curve_point() {
    let dir = scratch("curve");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let unsigned = measure_curve(&plan, &observe("unsigned-local", false, 0, 0, false)).unwrap();
    assert!(!unsigned.report.curve_computed);
    assert!(!unsigned.report.base_multiplied);
    assert_eq!(unsigned.report.validation_result, "recorded");
    verify_curve(&unsigned).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_curve = measure_curve(
        &slot,
        &observe(
            "on-curve",
            true,
            ED25519_PUBLIC_KEY_BYTES,
            ED25519_SIGNATURE_BYTES,
            true,
        ),
    )
    .unwrap();
    assert!(on_curve.report.curve_computed);
    assert!(on_curve.report.point_checked);
    assert_eq!(on_curve.report.validation_result, "verified");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_curve_report_refuses_base_multiplication_and_an_unverified_point() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut multiplied = observe(
        "on-curve",
        true,
        ED25519_PUBLIC_KEY_BYTES,
        ED25519_SIGNATURE_BYTES,
        true,
    );
    multiplied.base_multiplied = true;
    let err = measure_curve(&plan, &multiplied).unwrap_err();
    assert!(
        err.message.contains("the base point stays unmultiplied"),
        "{err}"
    );

    let err = measure_curve(&plan, &observe("unsigned-local", true, 0, 0, false)).unwrap_err();
    assert!(
        err.message
            .contains("an unsigned release computes the curve"),
        "{err}"
    );

    let err = measure_curve(
        &plan,
        &observe(
            "off-curve",
            false,
            ED25519_PUBLIC_KEY_BYTES,
            ED25519_SIGNATURE_BYTES,
            true,
        ),
    )
    .unwrap_err();
    assert!(
        err.message.contains("a checked release computes the curve"),
        "{err}"
    );

    let err = measure_curve(
        &plan,
        &observe("on-curve", true, ED25519_PUBLIC_KEY_BYTES, 63, true),
    )
    .unwrap_err();
    assert!(
        err.message.contains("ed25519 signatures are 64 bytes"),
        "{err}"
    );

    let measured = measure_curve(&plan, &observe("unsigned-local", false, 0, 0, false)).unwrap();
    let mut stored = measured.report.clone();
    stored.validation_result = "verified".into();
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message.contains("only an on-curve point is verified"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_curve_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_curve(&plan, &observe("unsigned-local", false, 0, 0, false)).unwrap();
    write_curve_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.curve-report"
    );
    let again = write_curve_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-curve-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_curve_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
