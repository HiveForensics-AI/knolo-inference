use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_quantization, reference_engine_build,
    reference_kernel_bundle, verify_quantization, write_quantization_report, write_synthetic_model,
    QuantizationObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-quantization"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-quantization-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(reason: &str, weights_opened: bool, payload_read: bool) -> QuantizationObservation {
    QuantizationObservation {
        engine_build_root: engine_root(),
        artifact_root: sha256_prefixed(b"artifact"),
        reason: reason.into(),
        code: "UNSUPPORTED_QUANTIZATION".into(),
        retryable: false,
        weights_opened,
        payload_read,
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
fn the_quantization_report_records_precision_dtype_ggml_and_version() {
    let dir = scratch("quantization");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let precision = measure_quantization(&plan, &observe("precision", false, false)).unwrap();
    assert!(!precision.report.weights_opened);
    assert!(!precision.report.payload_read);
    verify_quantization(&precision).unwrap();

    let dtype = measure_quantization(&plan, &observe("dtype", true, true)).unwrap();
    assert!(dtype.report.payload_read);
    let ggml = measure_quantization(&plan, &observe("ggml", true, false)).unwrap();
    assert!(!ggml.report.payload_read);
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let version = measure_quantization(&slot, &observe("version", true, false)).unwrap();
    assert_eq!(version.report.code, "UNSUPPORTED_QUANTIZATION");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_quantization_record_refuses_opened_weights_on_a_precision_check() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let err = measure_quantization(&plan, &observe("precision", true, false)).unwrap_err();
    assert!(
        err.message
            .contains("a precision refusal does not open weights"),
        "{err}"
    );
    let err = measure_quantization(&plan, &observe("dtype", true, false)).unwrap_err();
    assert!(
        err.message.contains("a dtype refusal reads the payload"),
        "{err}"
    );
    let err = measure_quantization(&plan, &observe("ggml", true, true)).unwrap_err();
    assert!(
        err.message
            .contains("a ggml type does not read the payload"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_quantization_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_quantization(&plan, &observe("dtype", true, true)).unwrap();
    write_quantization_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.quantization-report"
    );
    let again = write_quantization_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-quantization-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_quantization_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
