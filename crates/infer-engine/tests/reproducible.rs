use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_reproducible_build, reference_engine_build,
    reference_kernel_bundle, verify_reproducible_build, write_reproducible_report,
    write_synthetic_model, ReproducibleObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-reproducible"),
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
        "knolo-infer-reproducible-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(feature: &str) -> ReproducibleObservation {
    ReproducibleObservation {
        engine_build_root: engine_root(),
        lock_root: pin(b"cargo-lock"),
        source_root: pin(b"instructions"),
        feature_set: feature.into(),
        build_profile: "debug".into(),
        instruction_count: 4,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_reproducible_record_pins_cpu_and_cuda_builds() {
    let dir = scratch("build");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_reproducible_build(&plan, &observe("cpu")).unwrap();
    assert_eq!(measured.report.feature_set, "cpu");
    assert_eq!(measured.report.build_profile, "debug");
    assert_eq!(measured.report.instruction_count, 4);
    assert_eq!(measured.report.validation_result, "recorded");
    verify_reproducible_build(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let cuda = measure_reproducible_build(&slot, &observe("cuda")).unwrap();
    assert_eq!(cuda.report.feature_set, "cuda");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_reproducible_record_refuses_a_repeated_root_or_an_empty_instruction_list() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut lock = observe("cpu");
    lock.lock_root = lock.engine_build_root.clone();
    let err = measure_reproducible_build(&plan, &lock).unwrap_err();
    assert!(
        err.message.contains("the lock repeats the engine build"),
        "{err}"
    );

    let mut source_root = observe("cpu");
    source_root.source_root = source_root.lock_root.clone();
    let err = measure_reproducible_build(&plan, &source_root).unwrap_err();
    assert!(err.message.contains("the source repeats the lock"), "{err}");

    let mut empty = observe("cpu");
    empty.instruction_count = 0;
    let err = measure_reproducible_build(&plan, &empty).unwrap_err();
    assert!(
        err.message.contains("the build names no instruction"),
        "{err}"
    );

    let mut wide = observe("cpu");
    wide.instruction_count = 33;
    let err = measure_reproducible_build(&plan, &wide).unwrap_err();
    assert!(
        err.message.contains("the build instructions are too large"),
        "{err}"
    );

    let mut profile = observe("cpu");
    profile.build_profile = "dev".into();
    let err = measure_reproducible_build(&plan, &profile).unwrap_err();
    assert!(
        err.message
            .contains("field buildProfile has an unsupported value"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_reproducible_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_reproducible_build(&plan, &observe("cpu")).unwrap();
    write_reproducible_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.reproducible-report"
    );
    let command = b"cargo build --release";
    assert!(!stored
        .windows(command.len())
        .any(|window| window == command));
    let again = write_reproducible_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-reproducible-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_reproducible_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
