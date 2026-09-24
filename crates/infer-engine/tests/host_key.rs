use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_host_key, reference_engine_build,
    reference_kernel_bundle, verify_host_key, write_host_key_report, write_synthetic_model,
    HostKeyObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-host-key"),
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
        "knolo-infer-host-key-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(status: &str, key_id: &str, bytes: u32, verified: bool) -> HostKeyObservation {
    HostKeyObservation {
        engine_build_root: engine_root(),
        release_root: pin(b"release"),
        host_key_root: pin(b"host-key"),
        signature_status: status.into(),
        key_id: key_id.into(),
        signature_bytes: bytes,
        key_verified: verified,
        key_material_present: false,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_host_key_records_a_match_a_rejection_and_an_unsigned_release() {
    let dir = scratch("host-key");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let matched = measure_host_key(&plan, &observe("matched", "local-dev", 64, true)).unwrap();
    assert!(matched.report.key_verified);
    assert_eq!(matched.report.validation_result, "verified");
    assert!(!matched.report.key_material_present);
    verify_host_key(&matched).unwrap();

    let rejected = measure_host_key(&plan, &observe("rejected", "local-dev", 64, false)).unwrap();
    assert!(!rejected.report.key_verified);
    assert_eq!(rejected.report.validation_result, "recorded");

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let unsigned = measure_host_key(&slot, &observe("unsigned-local", "", 0, false)).unwrap();
    assert!(unsigned.report.key_id.is_empty());
    assert_eq!(unsigned.report.signature_bytes, 0);
    assert_eq!(unsigned.report.validation_result, "recorded");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_host_key_refuses_key_material_and_a_mismatched_verdict() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut material = observe("matched", "local-dev", 64, true);
    material.key_material_present = true;
    let err = measure_host_key(&plan, &material).unwrap_err();
    assert!(
        err.message.contains("key material stays in host storage"),
        "{err}"
    );

    let err = measure_host_key(&plan, &observe("matched", "local-dev", 64, false)).unwrap_err();
    assert!(
        err.message
            .contains("a matched release verifies the host key"),
        "{err}"
    );

    let err = measure_host_key(&plan, &observe("rejected", "local-dev", 64, true)).unwrap_err();
    assert!(
        err.message.contains("a rejected release verifies a key"),
        "{err}"
    );

    let err =
        measure_host_key(&plan, &observe("unsigned-local", "local-dev", 0, false)).unwrap_err();
    assert!(
        err.message.contains("an unsigned release names a key"),
        "{err}"
    );

    let err = measure_host_key(&plan, &observe("matched", "local-dev", 63, true)).unwrap_err();
    assert!(
        err.message.contains("ed25519 signatures are 64 bytes"),
        "{err}"
    );

    let mut repeated = observe("matched", "local-dev", 64, true);
    repeated.host_key_root = repeated.release_root.clone();
    let err = measure_host_key(&plan, &repeated).unwrap_err();
    assert!(
        err.message.contains("the host key repeats the release"),
        "{err}"
    );

    let measured = measure_host_key(&plan, &observe("matched", "local-dev", 64, true)).unwrap();
    let mut stored = measured.report.clone();
    stored.validation_result = "recorded".into();
    let err = stored.validate().unwrap_err();
    assert!(
        err.message.contains("a matched release is verified"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_host_key_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_host_key(&plan, &observe("matched", "local-dev", 64, true)).unwrap();
    write_host_key_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.host-key-report"
    );
    let secret = b"ed25519-private-key-material";
    assert!(!stored.windows(secret.len()).any(|window| window == secret));
    let again = write_host_key_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-host-key-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_host_key_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
