use std::fs;
use std::path::PathBuf;

use infer_contracts::{
    decode_contract, sha256_prefixed, ErrorCode, ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES,
};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_signature_equation, reference_engine_build,
    reference_kernel_bundle, verify_signature_equation, write_equation_report,
    write_synthetic_model, EquationObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-equation"),
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
        "knolo-infer-equation-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(status: &str, evaluated: bool, public_key: u32, signature: u32) -> EquationObservation {
    EquationObservation {
        engine_build_root: engine_root(),
        release_root: pin(b"release"),
        message_root: pin(b"message"),
        equation_status: status.into(),
        equation_evaluated: evaluated,
        public_key_bytes: public_key,
        signature_bytes: signature,
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
fn the_equation_report_records_an_unsigned_release_and_an_accepted_check() {
    let dir = scratch("equation");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let unsigned =
        measure_signature_equation(&plan, &observe("unsigned-local", false, 0, 0)).unwrap();
    assert!(!unsigned.report.equation_evaluated);
    assert_eq!(unsigned.report.validation_result, "recorded");
    assert_eq!(unsigned.report.public_key_bytes, 0);
    verify_signature_equation(&unsigned).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let accepted = measure_signature_equation(
        &slot,
        &observe(
            "accepted",
            true,
            ED25519_PUBLIC_KEY_BYTES,
            ED25519_SIGNATURE_BYTES,
        ),
    )
    .unwrap();
    assert!(accepted.report.equation_evaluated);
    assert_eq!(accepted.report.validation_result, "verified");
    assert_eq!(accepted.report.signature_bytes, ED25519_SIGNATURE_BYTES);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_equation_report_refuses_key_bytes_and_an_unverified_acceptance() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut material = observe(
        "accepted",
        true,
        ED25519_PUBLIC_KEY_BYTES,
        ED25519_SIGNATURE_BYTES,
    );
    material.key_material_present = true;
    let err = measure_signature_equation(&plan, &material).unwrap_err();
    assert!(
        err.message.contains("key material stays in host storage"),
        "{err}"
    );

    let err =
        measure_signature_equation(&plan, &observe("unsigned-local", true, 0, 0)).unwrap_err();
    assert!(
        err.message
            .contains("an unsigned release evaluates an equation"),
        "{err}"
    );

    let err =
        measure_signature_equation(&plan, &observe("unsigned-local", false, 32, 0)).unwrap_err();
    assert!(
        err.message
            .contains("an unsigned release carries a public key"),
        "{err}"
    );

    let err = measure_signature_equation(
        &plan,
        &observe(
            "accepted",
            false,
            ED25519_PUBLIC_KEY_BYTES,
            ED25519_SIGNATURE_BYTES,
        ),
    )
    .unwrap_err();
    assert!(
        err.message
            .contains("a checked release evaluates the equation"),
        "{err}"
    );

    let err = measure_signature_equation(
        &plan,
        &observe("rejected", true, ED25519_PUBLIC_KEY_BYTES, 63),
    )
    .unwrap_err();
    assert!(
        err.message.contains("ed25519 signatures are 64 bytes"),
        "{err}"
    );

    let err = measure_signature_equation(
        &plan,
        &observe("accepted", true, 31, ED25519_SIGNATURE_BYTES),
    )
    .unwrap_err();
    assert!(
        err.message.contains("ed25519 public keys are 32 bytes"),
        "{err}"
    );

    let mut message = observe(
        "accepted",
        true,
        ED25519_PUBLIC_KEY_BYTES,
        ED25519_SIGNATURE_BYTES,
    );
    message.message_root = message.release_root.clone();
    let err = measure_signature_equation(&plan, &message).unwrap_err();
    assert!(
        err.message
            .contains("the signed message repeats the release"),
        "{err}"
    );

    let measured =
        measure_signature_equation(&plan, &observe("unsigned-local", false, 0, 0)).unwrap();
    let mut stored = measured.report.clone();
    stored.validation_result = "verified".into();
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message
            .contains("only an accepted equation is verified"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_equation_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured =
        measure_signature_equation(&plan, &observe("unsigned-local", false, 0, 0)).unwrap();
    write_equation_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.equation-report"
    );
    let again = write_equation_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-equation-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_equation_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
