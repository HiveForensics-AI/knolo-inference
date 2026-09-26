use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, MAX_BINARY_BYTES};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_binary_inventory, reference_engine_build,
    reference_kernel_bundle, verify_binary_inventory, write_binary_report, write_synthetic_model,
    BinaryObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-binary"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-binary-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> BinaryObservation {
    BinaryObservation {
        engine_build_root: engine_root(),
        supervisor_name: "knolo-infer".into(),
        supervisor_hash: pin(b"supervisor"),
        supervisor_bytes: 1024,
        worker_name: "knolo-infer-worker".into(),
        worker_hash: pin(b"worker"),
        worker_bytes: 2048,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_binary_inventory_records_both_hashes_on_cpu_and_cuda() {
    let dir = scratch("binary");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_binary_inventory(&plan, &observe()).unwrap();
    assert_eq!(measured.report.supervisor_name, "knolo-infer");
    assert_eq!(measured.report.worker_name, "knolo-infer-worker");
    assert_eq!(measured.report.validation_result, "recorded");
    assert_ne!(measured.report.supervisor_hash, measured.report.worker_hash);
    verify_binary_inventory(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    measure_binary_inventory(&slot, &observe()).unwrap();
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_binary_inventory_refuses_a_repeated_hash_or_an_oversized_file() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut same = observe();
    same.worker_hash = same.supervisor_hash.clone();
    let err = measure_binary_inventory(&plan, &same).unwrap_err();
    assert!(
        err.message
            .contains("the worker hash repeats the supervisor"),
        "{err}"
    );

    let mut engine = observe();
    engine.supervisor_hash = engine.engine_build_root.clone();
    let err = measure_binary_inventory(&plan, &engine).unwrap_err();
    assert!(
        err.message
            .contains("the binary hash repeats the engine build"),
        "{err}"
    );

    let mut name = observe();
    name.worker_name = "llama.cpp".into();
    let err = measure_binary_inventory(&plan, &name).unwrap_err();
    assert!(
        err.message
            .contains("the worker binary is knolo-infer-worker"),
        "{err}"
    );

    let mut empty = observe();
    empty.supervisor_bytes = 0;
    let err = measure_binary_inventory(&plan, &empty).unwrap_err();
    assert!(
        err.message.contains("the supervisor binary is empty"),
        "{err}"
    );

    let mut over = observe();
    over.worker_bytes = MAX_BINARY_BYTES + 1;
    let err = measure_binary_inventory(&plan, &over).unwrap_err();
    assert_eq!(err.code, ErrorCode::InsufficientMemory, "{err}");
    assert!(
        err.message.contains("the worker binary exceeds 64 MiB"),
        "{err}"
    );

    let mut exact = observe();
    exact.supervisor_bytes = MAX_BINARY_BYTES;
    let measured = measure_binary_inventory(&plan, &exact).unwrap();
    assert_eq!(measured.report.supervisor_bytes, MAX_BINARY_BYTES);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_binary_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_binary_inventory(&plan, &observe()).unwrap();
    write_binary_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.binary-report"
    );
    let path = b"/usr/local/bin/knolo-infer";
    assert!(!stored.windows(path.len()).any(|window| window == path));
    let again = write_binary_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-binary-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_binary_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
