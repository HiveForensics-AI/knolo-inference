use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, MAX_DISK_NEEDED_BYTES};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_disk, reference_engine_build,
    reference_kernel_bundle, verify_disk, write_disk_report, write_synthetic_model,
    DiskObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build =
        reference_engine_build(sha256_prefixed(b"knolo-infer-disk"), bundle.root().unwrap())
            .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-disk-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(store: &str) -> DiskObservation {
    let durable = store != "trace";
    DiskObservation {
        engine_build_root: engine_root(),
        request_id: "req-1".into(),
        store: store.into(),
        free_bytes: 0,
        needed_bytes: if store == "lock" {
            MAX_DISK_NEEDED_BYTES
        } else {
            4096
        },
        file_written: false,
        space_reclaimed: false,
        listener_up: store != "lock",
        code: if durable {
            "RECEIPT_PERSIST_FAILED".into()
        } else {
            "none".into()
        },
        retryable: durable,
        receipt_stored: !durable,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_disk_report_records_a_full_journal_and_a_full_trace() {
    let dir = scratch("disk");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let journal = measure_disk(&plan, &observe("journal")).unwrap();
    assert_eq!(journal.report.code, "RECEIPT_PERSIST_FAILED");
    assert!(journal.report.retryable);
    assert!(!journal.report.receipt_stored);
    assert!(journal.report.listener_up);
    verify_disk(&journal).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let trace = measure_disk(&slot, &observe("trace")).unwrap();
    assert_eq!(trace.report.code, "none");
    assert!(trace.report.receipt_stored);
    assert!(!trace.report.retryable);

    let lock = measure_disk(&plan, &observe("lock")).unwrap();
    assert!(!lock.report.listener_up);
    assert_eq!(lock.report.needed_bytes, MAX_DISK_NEEDED_BYTES);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_disk_report_refuses_free_space_reclaimed_files_and_an_oversized_need() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut free = observe("journal");
    free.free_bytes = 1;
    let err = measure_disk(&plan, &free).unwrap_err();
    assert!(
        err.message.contains("a disk-full record has no free bytes"),
        "{err}"
    );

    let mut written = observe("receipt");
    written.file_written = true;
    let err = measure_disk(&plan, &written).unwrap_err();
    assert!(
        err.message.contains("a disk-full record writes no file"),
        "{err}"
    );

    let mut trace = observe("trace");
    trace.code = "RECEIPT_PERSIST_FAILED".into();
    let err = measure_disk(&plan, &trace).unwrap_err();
    assert!(
        err.message
            .contains("a trace write does not fail the completion"),
        "{err}"
    );

    let mut huge = observe("journal");
    huge.needed_bytes = MAX_DISK_NEEDED_BYTES + 1;
    let err = measure_disk(&plan, &huge).unwrap_err();
    assert_eq!(err.code, ErrorCode::InsufficientMemory, "{err}");
    assert!(err.message.contains("needed bytes exceed 1 MiB"), "{err}");

    let measured = measure_disk(&plan, &observe("journal")).unwrap();
    let mut stored = measured.report.clone();
    stored.needed_bytes = MAX_DISK_NEEDED_BYTES + 1;
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(err.message.contains("needed bytes exceed 1 MiB"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_disk_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_disk(&plan, &observe("journal")).unwrap();
    write_disk_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.disk-report"
    );
    let again = write_disk_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-disk-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_disk_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
