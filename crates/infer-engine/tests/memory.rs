use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_placement_memory, micro_kv_bytes, micro_kv_layout,
    planned_kv_bytes, reference_engine_build, reference_kernel_bundle, verify_memory_estimate,
    write_memory_estimate, write_synthetic_model, KvStore, MemoryObservation, PagedKv,
    CPU_KV_PAGE_POOL,
};

fn estimator_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-memory"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-memory-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn the_micro_pool_matches_the_placement_bound() {
    let dir = scratch("micro");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut kv = PagedKv::new(micro_kv_layout(), CPU_KV_PAGE_POOL).unwrap();
    assert_eq!(
        planned_kv_bytes(&micro_kv_layout(), 1).unwrap(),
        micro_kv_bytes()
    );
    assert_eq!(
        planned_kv_bytes(&micro_kv_layout(), CPU_KV_PAGE_POOL).unwrap(),
        kv.resident_bytes()
    );
    assert_eq!(plan.expected_kv_bytes, kv.resident_bytes());
    assert_eq!(plan.expected_weight_bytes, source.weight_bytes);
    assert_eq!(plan.expected_workspace_bytes, 4096);
    assert_eq!(plan.workspace_bytes, 4096);
    assert_eq!(plan.safety_margin_bytes, 1024);
    assert_eq!(plan.expected_staging_bytes, 0);
    assert_eq!(plan.expected_overhead_bytes, 0);
    assert_eq!(plan.graph_capture_mode, "off");
    assert!(plan.rejection_reason.is_none());

    kv.begin_sequence(1).unwrap();
    assert_eq!(kv.resident_bytes(), plan.expected_kv_bytes);
    kv.release_sequence(1).unwrap();
    assert_eq!(kv.resident_bytes(), plan.expected_kv_bytes);

    let observed = MemoryObservation {
        weight_bytes: source.weight_bytes,
        kv_bytes: kv.resident_bytes(),
        workspace_bytes: 0,
        staging_bytes: 0,
        overhead_bytes: 0,
    };
    let estimate = measure_placement_memory(&plan, &observed, &estimator_root()).unwrap();
    assert_eq!(estimate.report.validation_result, "within-bounds");
    assert_eq!(estimate.report.measured_kv_bytes, plan.expected_kv_bytes);
    assert_eq!(estimate.report.measured_weight_bytes, source.weight_bytes);
    assert_eq!(
        estimate.report.measured_total_bytes,
        source.weight_bytes + kv.resident_bytes()
    );
    assert_eq!(estimate.report.headroom_bytes, 4096 + 1024);
    assert_eq!(
        estimate.report.declared_total_bytes,
        source.weight_bytes + kv.resident_bytes() + 4096 + 1024
    );
    verify_memory_estimate(&estimate).unwrap();
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_count_above_the_bound_issues_no_report() {
    let dir = scratch("bound");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let root = estimator_root();
    let mut observed = MemoryObservation {
        weight_bytes: source.weight_bytes,
        kv_bytes: plan.expected_kv_bytes,
        workspace_bytes: 0,
        staging_bytes: 0,
        overhead_bytes: 0,
    };

    observed.kv_bytes = plan.expected_kv_bytes + 1;
    let kv = measure_placement_memory(&plan, &observed, &root).unwrap_err();
    assert_eq!(kv.code, ErrorCode::InsufficientMemory, "{kv}");
    assert!(kv.message.contains("measured kv bytes exceed"), "{kv}");

    observed.kv_bytes = plan.expected_kv_bytes;
    observed.weight_bytes = source.weight_bytes + 1;
    let weight = measure_placement_memory(&plan, &observed, &root).unwrap_err();
    assert!(
        weight.message.contains("measured weight bytes exceed"),
        "{weight}"
    );

    observed.weight_bytes = source.weight_bytes;
    observed.workspace_bytes = 4097;
    let workspace = measure_placement_memory(&plan, &observed, &root).unwrap_err();
    assert!(
        workspace
            .message
            .contains("measured workspace bytes exceed"),
        "{workspace}"
    );

    observed.workspace_bytes = 0;
    observed.staging_bytes = 1;
    let staging = measure_placement_memory(&plan, &observed, &root).unwrap_err();
    assert!(
        staging.message.contains("measured staging bytes exceed"),
        "{staging}"
    );

    observed.staging_bytes = 0;
    observed.overhead_bytes = 1;
    let overhead = measure_placement_memory(&plan, &observed, &root).unwrap_err();
    assert!(
        overhead.message.contains("measured overhead bytes exceed"),
        "{overhead}"
    );

    let mut graphs = plan.clone();
    graphs.graph_capture_mode = "decode-buckets".into();
    observed.overhead_bytes = 0;
    let graph = measure_placement_memory(&graphs, &observed, &root).unwrap_err();
    assert_eq!(graph.code, ErrorCode::ContractInvalid, "{graph}");
    assert!(graph.message.contains("graph capture is off"), "{graph}");

    let mut rejected = plan.clone();
    rejected.rejection_reason = Some("INSUFFICIENT_MEMORY".into());
    let reason = measure_placement_memory(&rejected, &observed, &root).unwrap_err();
    assert_eq!(reason.code, ErrorCode::PlacementUnsatisfiable, "{reason}");
    assert!(reason.message.contains("already rejected"), "{reason}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let observed = MemoryObservation {
        weight_bytes: source.weight_bytes,
        kv_bytes: plan.expected_kv_bytes,
        workspace_bytes: plan.expected_workspace_bytes,
        staging_bytes: 0,
        overhead_bytes: 0,
    };
    let estimate = measure_placement_memory(&plan, &observed, &estimator_root()).unwrap();
    assert_eq!(estimate.report.headroom_bytes, plan.safety_margin_bytes);
    write_memory_estimate(&dir, &estimate, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, estimate.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.memory-estimate"
    );
    let again = write_memory_estimate(&dir, &estimate, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");
    assert_eq!(fs::read(dir.join("report.cbor")).unwrap(), stored);

    let outside_name = format!("knolo-mem-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_memory_estimate(&dir, &estimate, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());

    std::os::unix::fs::symlink(&outside, dir.join("linked.cbor")).unwrap();
    let linked = write_memory_estimate(&dir, &estimate, "linked.cbor").unwrap_err();
    assert!(linked.message.contains("already exists"), "{linked}");
    assert!(!outside.exists());

    let missing = scratch("missing");
    fs::remove_dir_all(&missing).unwrap();
    let err = write_memory_estimate(&missing, &estimate, "report.cbor").unwrap_err();
    assert!(err.message.contains("does not exist"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_tampered_count_is_not_written() {
    let dir = scratch("tamper");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let observed = MemoryObservation {
        weight_bytes: source.weight_bytes,
        kv_bytes: plan.expected_kv_bytes,
        workspace_bytes: 0,
        staging_bytes: 0,
        overhead_bytes: 0,
    };
    let mut estimate = measure_placement_memory(&plan, &observed, &estimator_root()).unwrap();
    estimate.observed.kv_bytes -= 1;
    let err = write_memory_estimate(&dir, &estimate, "report.cbor").unwrap_err();
    assert!(
        err.message.contains("measured kv bytes do not match"),
        "{err}"
    );
    assert!(!dir.join("report.cbor").exists());

    estimate.observed.kv_bytes = plan.expected_kv_bytes;
    estimate.report.validation_result = "recorded".into();
    let rejected = write_memory_estimate(&dir, &estimate, "report.cbor").unwrap_err();
    assert!(rejected.message.contains("validationResult"), "{rejected}");
    assert!(!dir.join("report.cbor").exists());
    let _ = fs::remove_dir_all(&dir);
}
