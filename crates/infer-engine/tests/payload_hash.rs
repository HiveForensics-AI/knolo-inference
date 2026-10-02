use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_payload, reference_engine_build,
    reference_kernel_bundle, verify_payload, write_payload_report, write_synthetic_model,
    PayloadObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-payload"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-payload-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> PayloadObservation {
    PayloadObservation {
        engine_build_root: engine_root(),
        payload_root: sha256_prefixed(b"payload-body"),
        domain_root: sha256_prefixed(b"payload-domain"),
        hash_status: "hashed".into(),
        payload_hashed: true,
        digest_bound: false,
        public_key_bytes: 32,
        signature_bytes: 64,
        scalar_bytes: 32,
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
fn a_payload_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_payload(&plan, &observe()).unwrap();
    assert!(measured.report.payload_hashed);
    assert!(!measured.report.digest_bound);
    verify_payload(&measured).unwrap();

    let mut varied = observe();
    varied.hash_status = "rejected".into();
    let varied = measure_payload(&plan, &varied).unwrap();
    assert!(varied.report.payload_hashed);
    assert_eq!(varied.report.validation_result, "recorded");
    let mut varied = observe();
    varied.hash_status = "unsigned-local".into();
    varied.payload_hashed = false;
    varied.public_key_bytes = 0;
    varied.signature_bytes = 0;
    varied.scalar_bytes = 0;
    let varied = measure_payload(&plan, &varied).unwrap();
    assert!(!varied.report.payload_hashed);
    assert!(!varied.report.digest_bound);

    let slot_obs = observe();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_payload(&slot, &slot_obs).unwrap();
    assert_eq!(on_slot.report.validation_result, "verified");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_payload_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.digest_bound = true;
    let err = measure_payload(&plan, &bad).unwrap_err();
    assert!(err.message.contains("the digest stays unbound"), "{err}");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_payload_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_payload(&plan, &observe()).unwrap();
    write_payload_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.payload-report"
    );
    let again = write_payload_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-payload-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_payload_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
