use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, TensorGroupV1, MAX_PEAK_BYTES};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_peak_memory, reference_engine_build,
    reference_kernel_bundle, verify_peak_memory, write_peak_report, write_synthetic_model,
    PeakObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build =
        reference_engine_build(sha256_prefixed(b"knolo-infer-peak"), bundle.root().unwrap())
            .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-peak-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(source: &infer_engine::VerifiedWeightSource, ram: u64, vram: u64) -> PeakObservation {
    PeakObservation {
        model_image_root: source.image_root.clone(),
        artifact_root: source.artifact_root.clone(),
        engine_build_root: engine_root(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        peak_ram_bytes: ram,
        peak_vram_bytes: vram,
    }
}

#[test]
fn a_cold_run_records_peak_ram_and_vram() {
    let dir = scratch("fixture");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_peak_memory(&plan, &observe(&source, 4096, 0)).unwrap();
    assert_eq!(measured.report.device, "cpu");
    assert_eq!(measured.report.peak_ram_bytes, 4096);
    assert_eq!(measured.report.peak_vram_bytes, 0);
    assert_eq!(measured.report.validation_result, "recorded");
    verify_peak_memory(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_peak_memory(&slot, &observe(&source, 4096, 2048)).unwrap();
    assert_eq!(on_slot.report.device, "slot-0");
    assert_eq!(on_slot.report.peak_vram_bytes, 2048);
    let at_cap = measure_peak_memory(&slot, &observe(&source, MAX_PEAK_BYTES, MAX_PEAK_BYTES));
    assert!(at_cap.is_ok());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_failed_peak_issues_no_report() {
    let dir = scratch("reject");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let root = observe(&source, 4096, 0);

    let mut zero = root.clone();
    zero.peak_ram_bytes = 0;
    let err = measure_peak_memory(&plan, &zero).unwrap_err();
    assert!(err.message.contains("peak ram is zero"), "{err}");

    let mut host_vram = root.clone();
    host_vram.peak_vram_bytes = 16;
    let err = measure_peak_memory(&plan, &host_vram).unwrap_err();
    assert!(
        err.message.contains("cpu placement has no device memory"),
        "{err}"
    );

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let err = measure_peak_memory(&slot, &root).unwrap_err();
    assert!(
        err.message
            .contains("slot-0 placement records device memory"),
        "{err}"
    );

    let mut over = root.clone();
    over.peak_ram_bytes = MAX_PEAK_BYTES + 1;
    let err = measure_peak_memory(&plan, &over).unwrap_err();
    assert_eq!(err.code, ErrorCode::InsufficientMemory, "{err}");
    assert!(err.message.contains("peak ram exceeds 64 MiB"), "{err}");

    let mut device = observe(&source, 4096, MAX_PEAK_BYTES + 1);
    let err = measure_peak_memory(&slot, &device).unwrap_err();
    assert_eq!(err.code, ErrorCode::InsufficientMemory, "{err}");
    assert!(err.message.contains("peak vram exceeds 64 MiB"), "{err}");

    let mut other = slot.clone();
    other.devices = vec!["slot-1".into()];
    other.tensor_groups = vec![TensorGroupV1 {
        name: "weights".into(),
        device: "slot-1".into(),
        compute_precision: "f32".into(),
        storage_precision: "f32".into(),
    }];
    device.peak_vram_bytes = 2048;
    let err = measure_peak_memory(&other, &device).unwrap_err();
    assert!(
        err.message.contains("the peak report is the micro fixture"),
        "{err}"
    );

    let mut mode = root.clone();
    mode.execution_mode = "throughput".into();
    let err = measure_peak_memory(&plan, &mode).unwrap_err();
    assert_eq!(err.code, ErrorCode::BackendNotAllowed, "{err}");

    let mut graphs = plan.clone();
    graphs.graph_capture_mode = "decode-buckets".into();
    let err = measure_peak_memory(&graphs, &root).unwrap_err();
    assert!(err.message.contains("graph capture is off"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_peak_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_peak_memory(&plan, &observe(&source, 9, 0)).unwrap();
    write_peak_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.peak-report"
    );
    let again = write_peak_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-peak-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_peak_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
