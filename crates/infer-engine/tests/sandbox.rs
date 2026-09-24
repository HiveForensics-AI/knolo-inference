use std::fs;
use std::path::PathBuf;

use infer_contracts::{
    decode_contract, sha256_prefixed, ErrorCode, MAX_SANDBOX_MEMORY, MAX_SHARED_MEMORY,
};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_sandbox_profile, reference_engine_build,
    reference_kernel_bundle, verify_sandbox_profile, write_sandbox_report, write_synthetic_model,
    SandboxObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-sandbox"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-sandbox-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(memory: u64, shared: u64) -> SandboxObservation {
    SandboxObservation {
        engine_build_root: engine_root(),
        user_class: "unprivileged".into(),
        network: "none".into(),
        model_cas: "read-only".into(),
        scratch: "worker-only".into(),
        seccomp: "deferred".into(),
        memory_limit_bytes: memory,
        process_group: "isolated".into(),
        parent_death: "socket-eof".into(),
        shared_memory_bytes: shared,
        arguments: "direct-array".into(),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_sandbox_profile_records_the_production_bounds() {
    let dir = scratch("sandbox");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_sandbox_profile(&plan, &observe(MAX_SANDBOX_MEMORY, 0)).unwrap();
    assert_eq!(measured.report.user_class, "unprivileged");
    assert_eq!(measured.report.seccomp, "deferred");
    assert_eq!(measured.report.parent_death, "socket-eof");
    assert_eq!(measured.report.memory_limit_bytes, MAX_SANDBOX_MEMORY);
    verify_sandbox_profile(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let shared = measure_sandbox_profile(&slot, &observe(1024, MAX_SHARED_MEMORY)).unwrap();
    assert_eq!(shared.report.shared_memory_bytes, MAX_SHARED_MEMORY);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_sandbox_profile_refuses_privilege_network_seccomp_and_oversized_memory() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut root = observe(1024, 0);
    root.user_class = "root".into();
    let err = measure_sandbox_profile(&plan, &root).unwrap_err();
    assert!(
        err.message.contains("the worker user is unprivileged"),
        "{err}"
    );

    let mut network = observe(1024, 0);
    network.network = "external".into();
    let err = measure_sandbox_profile(&plan, &network).unwrap_err();
    assert!(
        err.message.contains("the worker has no external network"),
        "{err}"
    );

    let mut seccomp = observe(1024, 0);
    seccomp.seccomp = "enforced".into();
    let err = measure_sandbox_profile(&plan, &seccomp).unwrap_err();
    assert!(
        err.message
            .contains("seccomp stays deferred until the profile is stable"),
        "{err}"
    );

    let mut death = observe(1024, 0);
    death.parent_death = "pdeathsig".into();
    let err = measure_sandbox_profile(&plan, &death).unwrap_err();
    assert!(
        err.message
            .contains("the worker lifetime is the supervisor socket"),
        "{err}"
    );

    let mut shell = observe(1024, 0);
    shell.arguments = "shell".into();
    let err = measure_sandbox_profile(&plan, &shell).unwrap_err();
    assert!(
        err.message
            .contains("the worker arguments are a direct array"),
        "{err}"
    );

    let err = measure_sandbox_profile(&plan, &observe(0, 0)).unwrap_err();
    assert!(
        err.message.contains("the worker memory limit is zero"),
        "{err}"
    );

    let err = measure_sandbox_profile(&plan, &observe(MAX_SANDBOX_MEMORY + 1, 0)).unwrap_err();
    assert_eq!(err.code, ErrorCode::InsufficientMemory, "{err}");
    assert!(
        err.message.contains("worker memory exceeds 64 MiB"),
        "{err}"
    );

    let err = measure_sandbox_profile(&plan, &observe(1024, MAX_SHARED_MEMORY + 1)).unwrap_err();
    assert!(
        err.message
            .contains("shared memory exceeds the worker bound"),
        "{err}"
    );

    let measured = measure_sandbox_profile(&plan, &observe(1024, 0)).unwrap();
    let mut stored = measured.report.clone();
    stored.memory_limit_bytes = MAX_SANDBOX_MEMORY + 1;
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message.contains("worker memory exceeds 64 MiB"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_sandbox_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_sandbox_profile(&plan, &observe(1024, 0)).unwrap();
    write_sandbox_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.sandbox-report"
    );
    let again = write_sandbox_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-sandbox-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_sandbox_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
