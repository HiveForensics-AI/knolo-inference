use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_image_signature, reference_engine_build,
    reference_kernel_bundle, verify_image_signature, write_image_signature_report,
    write_synthetic_model, ImageSignatureObservation,
};

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-image-signature"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-image-signature-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(reason: &str, count: u32, bytes: u32, accepted: bool) -> ImageSignatureObservation {
    ImageSignatureObservation {
        engine_build_root: engine_root(),
        image_root: pin(b"image"),
        reason: reason.into(),
        signature_count: count,
        signature_bytes: bytes,
        code: "MODEL_IMAGE_SIGNATURE_INVALID".into(),
        retryable: false,
        algorithm_accepted: accepted,
        weights_opened: false,
        forward_ran: false,
        receipt_stored: false,
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
fn the_image_signature_records_an_algorithm_a_length_and_a_count() {
    let dir = scratch("signature");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let algorithm = measure_image_signature(&plan, &observe("algorithm", 1, 0, false)).unwrap();
    assert!(!algorithm.report.algorithm_accepted);
    assert!(!algorithm.report.weights_opened);
    verify_image_signature(&algorithm).unwrap();

    let length = measure_image_signature(&plan, &observe("length", 1, 32, true)).unwrap();
    assert_eq!(length.report.signature_bytes, 32);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let count = measure_image_signature(&slot, &observe("count", 9, 0, false)).unwrap();
    assert_eq!(count.report.signature_count, 9);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_image_signature_refuses_an_accepted_algorithm_and_a_count_past_the_cap() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let err = measure_image_signature(&plan, &observe("algorithm", 1, 0, true)).unwrap_err();
    assert!(
        err.message
            .contains("an algorithm refusal does not accept the algorithm"),
        "{err}"
    );

    let err = measure_image_signature(&plan, &observe("length", 1, 64, true)).unwrap_err();
    assert!(
        err.message
            .contains("a length refusal read a signature that is not 64 bytes"),
        "{err}"
    );

    let err = measure_image_signature(&plan, &observe("count", 17, 0, false)).unwrap_err();
    assert_eq!(err.code, ErrorCode::ModelImageSignatureInvalid, "{err}");
    assert!(
        err.message
            .contains("signature count exceeds the record cap"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_image_signature_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_image_signature(&plan, &observe("length", 1, 32, true)).unwrap();
    write_image_signature_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.image-signature-report"
    );
    let again = write_image_signature_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-image-signature-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_image_signature_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
