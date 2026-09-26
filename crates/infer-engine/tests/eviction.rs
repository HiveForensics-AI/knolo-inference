use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_prefix_eviction, reference_engine_build,
    reference_kernel_bundle, verify_prefix_eviction, write_eviction_report, write_synthetic_model,
    EvictionObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-eviction"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-eviction-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(load_requests: u32) -> EvictionObservation {
    EvictionObservation {
        engine_build_root: engine_root(),
        load_requests,
        evicted_pages: 0,
        evicted_tokens: 0,
        active_evicted: false,
        prefix_allocated: false,
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
fn the_eviction_report_records_zero_pages_under_load() {
    let dir = scratch("eviction");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_prefix_eviction(&plan, &observe(1)).unwrap();
    assert_eq!(measured.report.load_requests, 1);
    assert_eq!(measured.report.evicted_pages, 0);
    assert_eq!(measured.report.evicted_tokens, 0);
    assert!(!measured.report.active_evicted);
    assert!(!measured.report.prefix_allocated);
    verify_prefix_eviction(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let full = measure_prefix_eviction(&slot, &observe(16)).unwrap();
    assert_eq!(full.report.load_requests, 16);
    assert!(full.report.listener_up);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_eviction_report_refuses_an_allocated_index_and_an_evicted_page() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut allocated = observe(2);
    allocated.prefix_allocated = true;
    let err = measure_prefix_eviction(&plan, &allocated).unwrap_err();
    assert!(
        err.message.contains("prefix cache is not allocated"),
        "{err}"
    );

    let mut pages = observe(2);
    pages.evicted_pages = 1;
    let err = measure_prefix_eviction(&plan, &pages).unwrap_err();
    assert!(
        err.message
            .contains("prefix eviction is zero while the cache is off"),
        "{err}"
    );

    let mut active = observe(2);
    active.active_evicted = true;
    let err = measure_prefix_eviction(&plan, &active).unwrap_err();
    assert!(
        err.message.contains("an active sequence is not evicted"),
        "{err}"
    );

    let err = measure_prefix_eviction(&plan, &observe(0)).unwrap_err();
    assert!(
        err.message.contains("eviction under load names a request"),
        "{err}"
    );

    let err = measure_prefix_eviction(&plan, &observe(17)).unwrap_err();
    assert!(
        err.message
            .contains("eviction under load admits at most 16 requests"),
        "{err}"
    );

    let measured = measure_prefix_eviction(&plan, &observe(1)).unwrap();
    let mut stored = measured.report.clone();
    stored.evicted_tokens = 1;
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message
            .contains("evicted tokens are zero while the cache is off"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_eviction_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_prefix_eviction(&plan, &observe(4)).unwrap();
    write_eviction_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.eviction-report"
    );
    let again = write_eviction_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-eviction-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_eviction_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
