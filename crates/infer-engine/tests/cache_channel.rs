use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_cache_channel, reference_engine_build,
    reference_kernel_bundle, verify_cache_channel, write_cache_channel_report,
    write_synthetic_model, CacheChannelObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-cache-channel"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-cache-channel-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> CacheChannelObservation {
    CacheChannelObservation {
        engine_build_root: engine_root(),
        tenant_root: pin(b"tenant"),
        project_root: pin(b"project"),
        sharing_policy: "isolated".into(),
        cross_tenant: false,
        existence_disclosure: "hidden".into(),
        metrics_scope: "aggregate".into(),
        prefix_allocated: false,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_cache_channel_hides_another_tenant_and_stays_unallocated() {
    let dir = scratch("channel");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let local = measure_cache_channel(&plan, &observe()).unwrap();
    assert_eq!(local.report.sharing_policy, "isolated");
    assert!(!local.report.cross_tenant);
    assert_eq!(local.report.existence_disclosure, "hidden");
    assert_eq!(local.report.metrics_scope, "aggregate");
    assert!(!local.report.prefix_allocated);
    verify_cache_channel(&local).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let remote = measure_cache_channel(&slot, &observe()).unwrap();
    assert_eq!(remote.report.cache_policy, "off");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_cache_channel_refuses_sharing_disclosure_and_an_allocated_index() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut shared = observe();
    shared.sharing_policy = "shared".into();
    let err = measure_cache_channel(&plan, &shared).unwrap_err();
    assert!(err.message.contains("cache sharing is isolated"), "{err}");

    let mut cross = observe();
    cross.cross_tenant = true;
    let err = measure_cache_channel(&plan, &cross).unwrap_err();
    assert!(err.message.contains("cross-tenant sharing is off"), "{err}");

    let mut revealed = observe();
    revealed.existence_disclosure = "revealed".into();
    let err = measure_cache_channel(&plan, &revealed).unwrap_err();
    assert!(
        err.message.contains("another tenant prefix stays hidden"),
        "{err}"
    );

    let mut metrics = observe();
    metrics.metrics_scope = "tenant".into();
    let err = measure_cache_channel(&plan, &metrics).unwrap_err();
    assert!(
        err.message.contains("cache metrics stay aggregated"),
        "{err}"
    );

    let mut allocated = observe();
    allocated.prefix_allocated = true;
    let err = measure_cache_channel(&plan, &allocated).unwrap_err();
    assert!(
        err.message.contains("the prefix index stays unallocated"),
        "{err}"
    );

    let mut project = observe();
    project.project_root = project.tenant_root.clone();
    let err = measure_cache_channel(&plan, &project).unwrap_err();
    assert!(
        err.message.contains("the project repeats the tenant"),
        "{err}"
    );

    let measured = measure_cache_channel(&plan, &observe()).unwrap();
    let mut stored = measured.report.clone();
    stored.prefix_allocated = true;
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_cache_channel_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_cache_channel(&plan, &observe()).unwrap();
    write_cache_channel_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.cache-channel-report"
    );
    let again = write_cache_channel_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-cache-channel-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_cache_channel_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
