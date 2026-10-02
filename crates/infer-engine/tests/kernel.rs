use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_kernel, reference_engine_build,
    reference_kernel_bundle, verify_kernel, write_kernel_report, write_synthetic_model,
    KernelObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-kernel"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-kernel-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(reason: &str, cuda_requested: bool) -> KernelObservation {
    KernelObservation {
        engine_build_root: engine_root(),
        reason: reason.into(),
        code: "UNSUPPORTED_KERNEL".into(),
        retryable: false,
        cuda_requested,
        kernel_selected: false,
        device_opened: false,
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
fn the_kernel_report_records_a_foreign_backend_and_a_missing_feature() {
    let dir = scratch("kernel");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let backend = measure_kernel(&plan, &observe("backend", false)).unwrap();
    assert!(!backend.report.cuda_requested);
    assert!(!backend.report.kernel_selected);
    assert!(!backend.report.device_opened);
    verify_kernel(&backend).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let feature = measure_kernel(&slot, &observe("feature", true)).unwrap();
    assert!(feature.report.cuda_requested);
    assert_eq!(feature.report.code, "UNSUPPORTED_KERNEL");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_kernel_record_refuses_a_selected_kernel_and_a_cuda_request_on_a_foreign_backend() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut selected = observe("backend", false);
    selected.kernel_selected = true;
    let err = measure_kernel(&plan, &selected).unwrap_err();
    assert!(
        err.message
            .contains("an unsupported kernel is not selected"),
        "{err}"
    );

    let err = measure_kernel(&plan, &observe("backend", true)).unwrap_err();
    assert!(
        err.message
            .contains("a foreign backend does not request cuda"),
        "{err}"
    );
    let err = measure_kernel(&plan, &observe("feature", false)).unwrap_err();
    assert!(
        err.message
            .contains("a missing feature requests candle-cuda"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_kernel_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_kernel(&plan, &observe("feature", true)).unwrap();
    write_kernel_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.kernel-report"
    );
    let again = write_kernel_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-kernel-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_kernel_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
