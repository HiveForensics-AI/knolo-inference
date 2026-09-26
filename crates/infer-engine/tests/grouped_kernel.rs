use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_grouped_kernel, reference_engine_build,
    reference_kernel_bundle, verify_grouped_kernel, write_grouped_kernel_report,
    write_synthetic_model, GroupedKernelObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-grouped_kernel"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-grouped_kernel-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> GroupedKernelObservation {
    GroupedKernelObservation {
        engine_build_root: engine_root(),
        reason: "gemm".into(),
        group_count: 4,
        code: "UNSUPPORTED_KERNEL".into(),
        retryable: false,
        kernel_selected: false,
        grouped: false,
        routing_ran: false,
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
fn a_grouped_kernel_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_grouped_kernel(&plan, &observe()).unwrap();
    assert_eq!(measured.report.group_count, 4);
    assert!(!measured.report.grouped);
    verify_grouped_kernel(&measured).unwrap();

    let mut routing = observe();
    routing.reason = "routing".into();
    routing.group_count = 0;
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_grouped_kernel(&slot, &routing).unwrap();
    assert!(!on_slot.report.routing_ran);
    assert!(!on_slot.report.kernel_selected);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_grouped_kernel_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.kernel_selected = true;
    let err = measure_grouped_kernel(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("a grouped kernel is not selected"),
        "{err}"
    );

    let mut bad = observe();
    bad.group_count = 17;
    let err = measure_grouped_kernel(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("group count exceeds the record cap"),
        "{err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_grouped_kernel_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_grouped_kernel(&plan, &observe()).unwrap();
    write_grouped_kernel_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.grouped-kernel-report"
    );
    let again = write_grouped_kernel_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-grouped_kernel-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_grouped_kernel_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
