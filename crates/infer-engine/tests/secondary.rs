use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_secondary, reference_engine_build,
    reference_kernel_bundle, verify_secondary, write_secondary_report, write_synthetic_model,
    SecondaryObservation,
};

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-secondary"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-secondary-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(reason: &str, named: bool, embeddings: bool, overflow: bool) -> SecondaryObservation {
    SecondaryObservation {
        engine_build_root: engine_root(),
        primary_root: pin(b"primary-model"),
        secondary_root: pin(b"secondary-model"),
        reason: reason.into(),
        secondary_slot: "slot-1".into(),
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        model_named: named,
        embeddings_requested: embeddings,
        overflow_requested: overflow,
        service_started: false,
        embeddings_ran: false,
        overflow_placed: false,
        device_opened: false,
        tensor_parallel: false,
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
fn a_secondary_service_records_a_tiny_model_embeddings_and_overflow() {
    let dir = scratch("secondary");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let tiny = measure_secondary(&plan, &observe("tiny", true, false, false)).unwrap();
    assert!(tiny.report.model_named);
    assert!(!tiny.report.service_started);
    assert_eq!(tiny.report.secondary_slot, "slot-1");
    verify_secondary(&tiny).unwrap();

    let embeddings = measure_secondary(&plan, &observe("embeddings", false, true, false)).unwrap();
    assert!(embeddings.report.embeddings_requested);
    assert!(!embeddings.report.embeddings_ran);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let overflow = measure_secondary(&slot, &observe("overflow", true, false, true)).unwrap();
    assert!(overflow.report.overflow_requested);
    assert!(!overflow.report.overflow_placed);
    assert!(!overflow.report.device_opened);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_secondary_service_rejects_a_start_and_the_primary_slot() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut started = observe("tiny", true, false, false);
    started.service_started = true;
    let err = measure_secondary(&plan, &started).unwrap_err();
    assert!(
        err.message.contains("a secondary service does not start"),
        "{err}"
    );

    let mut primary = observe("overflow", true, false, true);
    primary.secondary_slot = "slot-0".into();
    let err = measure_secondary(&plan, &primary).unwrap_err();
    assert!(
        err.message.contains("a secondary service names slot-1"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_secondary_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_secondary(&plan, &observe("embeddings", false, true, false)).unwrap();
    write_secondary_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.secondary-report"
    );
    let again = write_secondary_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-secondary-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_secondary_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
