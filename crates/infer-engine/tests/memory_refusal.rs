use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, MAX_PEAK_BYTES};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_memory_refusal, reference_engine_build,
    reference_kernel_bundle, verify_memory_refusal, write_memory_refusal_report,
    write_synthetic_model, MemoryRefusalObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-memory-refusal"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-memory-refusal-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    reason: &str,
    needed: u64,
    resident_full: bool,
    queue_held: bool,
) -> MemoryRefusalObservation {
    MemoryRefusalObservation {
        engine_build_root: engine_root(),
        reason: reason.into(),
        free_bytes: 0,
        needed_bytes: needed,
        code: "INSUFFICIENT_MEMORY".into(),
        retryable: true,
        resident_full,
        queue_held,
        allocated: false,
        forward_ran: false,
        receipt_stored: false,
        listener_up: true,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_memory_refusal_records_a_full_pool_an_output_cap_and_an_admission_cap() {
    let dir = scratch("memory");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let pool =
        measure_memory_refusal(&plan, &observe("pool", MAX_PEAK_BYTES, true, false)).unwrap();
    assert_eq!(pool.report.needed_bytes, MAX_PEAK_BYTES);
    assert!(pool.report.resident_full);
    assert!(pool.report.retryable);
    assert!(!pool.report.allocated);
    verify_memory_refusal(&pool).unwrap();

    let output = measure_memory_refusal(&plan, &observe("output", 4096, false, false)).unwrap();
    assert!(!output.report.resident_full);
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let admission = measure_memory_refusal(&slot, &observe("admission", 1, false, true)).unwrap();
    assert!(admission.report.queue_held);
    assert!(admission.report.listener_up);
    assert_eq!(admission.report.code, "INSUFFICIENT_MEMORY");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_memory_refusal_rejects_a_count_above_64_mib_and_a_held_queue_on_a_full_pool() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let err = measure_memory_refusal(&plan, &observe("pool", MAX_PEAK_BYTES + 1, true, false))
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::InsufficientMemory, "{err}");
    assert!(err.message.contains("needed bytes exceed 64 MiB"), "{err}");

    let err = measure_memory_refusal(&plan, &observe("pool", 4096, true, true)).unwrap_err();
    assert!(
        err.message.contains("a full pool does not hold the queue"),
        "{err}"
    );
    let err = measure_memory_refusal(&plan, &observe("admission", 4096, false, false)).unwrap_err();
    assert!(
        err.message.contains("an admission cap holds the queue"),
        "{err}"
    );
    let mut allocated = observe("output", 4096, false, false);
    allocated.allocated = true;
    let err = measure_memory_refusal(&plan, &allocated).unwrap_err();
    assert!(
        err.message
            .contains("an insufficient memory record does not allocate"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_memory_refusal_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_memory_refusal(&plan, &observe("pool", 4096, true, false)).unwrap();
    write_memory_refusal_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.memory-refusal-report"
    );
    let again = write_memory_refusal_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-memory-refusal-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_memory_refusal_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
