use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_image_invalid, reference_engine_build,
    reference_kernel_bundle, verify_image_invalid, write_image_invalid_report,
    write_synthetic_model, ImageInvalidObservation,
};

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-image-invalid"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-image-invalid-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(reason: &str, parsed: bool, opened: bool) -> ImageInvalidObservation {
    ImageInvalidObservation {
        engine_build_root: engine_root(),
        image_root: pin(b"image"),
        reason: reason.into(),
        code: "MODEL_IMAGE_INVALID".into(),
        retryable: false,
        image_parsed: parsed,
        weights_opened: opened,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_image_invalid_report_records_an_empty_image_a_format_and_an_inventory() {
    let dir = scratch("image");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let empty = measure_image_invalid(&plan, &observe("empty", false, false)).unwrap();
    assert!(!empty.report.image_parsed);
    assert!(!empty.report.weights_opened);
    verify_image_invalid(&empty).unwrap();

    let format = measure_image_invalid(&plan, &observe("format", true, false)).unwrap();
    assert!(format.report.image_parsed);
    assert!(!format.report.weights_opened);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let inventory = measure_image_invalid(&slot, &observe("inventory", true, true)).unwrap();
    assert!(inventory.report.weights_opened);
    let canonical = measure_image_invalid(&plan, &observe("canonical", false, false)).unwrap();
    assert!(!canonical.report.image_parsed);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_image_invalid_record_refuses_a_parsed_empty_image() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let err = measure_image_invalid(&plan, &observe("empty", true, false)).unwrap_err();
    assert!(
        err.message.contains("an empty image is not parsed"),
        "{err}"
    );

    let err = measure_image_invalid(&plan, &observe("format", true, true)).unwrap_err();
    assert!(
        err.message
            .contains("a format refusal does not open weights"),
        "{err}"
    );

    let err = measure_image_invalid(&plan, &observe("inventory", true, false)).unwrap_err();
    assert!(
        err.message
            .contains("an inventory refusal opens the weights"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_image_invalid_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_image_invalid(&plan, &observe("format", true, false)).unwrap();
    write_image_invalid_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.image-invalid-report"
    );
    let again = write_image_invalid_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-image-invalid-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_image_invalid_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
