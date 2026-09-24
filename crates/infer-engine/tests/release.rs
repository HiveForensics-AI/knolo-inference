use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_release_manifest, reference_engine_build,
    reference_kernel_bundle, verify_release_manifest, write_release_report, write_synthetic_model,
    ReleaseObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-release"),
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
        std::env::temp_dir().join(format!("knolo-infer-release-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(status: &str, count: u32) -> ReleaseObservation {
    ReleaseObservation {
        engine_build_root: engine_root(),
        notice_root: pin(b"notice"),
        sbom_root: pin(b"sbom"),
        binary_set_root: pin(b"binaries"),
        signature_status: status.into(),
        signature_count: count,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_release_records_unsigned_and_shape_checked_manifests() {
    let dir = scratch("release");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_release_manifest(&plan, &observe("unsigned-local", 0)).unwrap();
    assert_eq!(measured.report.signature_status, "unsigned-local");
    assert_eq!(measured.report.signature_count, 0);
    assert_eq!(measured.report.validation_result, "recorded");
    verify_release_manifest(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let signed = measure_release_manifest(&slot, &observe("shape-checked", 1)).unwrap();
    assert_eq!(signed.report.signature_count, 1);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_release_refuses_a_repeated_root_or_the_wrong_signature_count() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut repeated = observe("unsigned-local", 0);
    repeated.sbom_root = repeated.notice_root.clone();
    let err = measure_release_manifest(&plan, &repeated).unwrap_err();
    assert!(err.message.contains("the sbom repeats the notice"), "{err}");

    let mut binaries = observe("unsigned-local", 0);
    binaries.binary_set_root = binaries.engine_build_root.clone();
    let err = measure_release_manifest(&plan, &binaries).unwrap_err();
    assert!(
        err.message
            .contains("the binary set repeats the engine build"),
        "{err}"
    );

    let err = measure_release_manifest(&plan, &observe("unsigned-local", 1)).unwrap_err();
    assert!(
        err.message
            .contains("an unsigned release carries a signature"),
        "{err}"
    );

    let err = measure_release_manifest(&plan, &observe("shape-checked", 0)).unwrap_err();
    assert!(
        err.message
            .contains("a shape-checked release carries one signature"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_release_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_release_manifest(&plan, &observe("unsigned-local", 0)).unwrap();
    write_release_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.release-report"
    );
    let secret = b"ed25519-private-key";
    assert!(!stored.windows(secret.len()).any(|window| window == secret));
    let again = write_release_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-release-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_release_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
