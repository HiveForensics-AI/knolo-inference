use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_domain, reference_engine_build,
    reference_kernel_bundle, verify_domain, write_domain_report, write_synthetic_model,
    DomainObservation,
};

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-domain_sep"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-domain_sep-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> DomainObservation {
    DomainObservation {
        engine_build_root: engine_root(),
        domain_root: pin(b"domain-prefix"),
        message_root: pin(b"domain-message"),
        separate_status: "separated".into(),
        domain_separated: true,
        payload_hashed: false,
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
fn a_domain_sep_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_domain(&plan, &observe()).unwrap();
    assert!(measured.report.domain_separated);
    assert!(!measured.report.payload_hashed);
    verify_domain(&measured).unwrap();

    let mut rejected = observe();
    rejected.separate_status = "rejected".into();
    let rejected = measure_domain(&plan, &rejected).unwrap();
    assert!(rejected.report.domain_separated);
    assert_eq!(rejected.report.validation_result, "recorded");

    let mut unsigned = observe();
    unsigned.separate_status = "unsigned-local".into();
    unsigned.domain_separated = false;
    unsigned.public_key_bytes = 0;
    unsigned.signature_bytes = 0;
    unsigned.scalar_bytes = 0;
    let unsigned = measure_domain(&plan, &unsigned).unwrap();
    assert!(!unsigned.report.domain_separated);
    assert!(!unsigned.report.payload_hashed);
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_domain(&slot, &observe()).unwrap();
    assert_eq!(on_slot.report.validation_result, "verified");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_domain_sep_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.payload_hashed = true;
    let err = measure_domain(&plan, &bad).unwrap_err();
    assert!(err.message.contains("the payload stays unhashed"), "{err}");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_domain_sep_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_domain(&plan, &observe()).unwrap();
    write_domain_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.domain-report"
    );
    let again = write_domain_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-domain_sep-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_domain_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
