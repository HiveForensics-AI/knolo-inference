use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_binding, reference_engine_build,
    reference_kernel_bundle, verify_binding, write_binding_report, write_synthetic_model,
    BindingObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-binding"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-binding-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> BindingObservation {
    BindingObservation {
        engine_build_root: engine_root(),
        digest_root: sha256_prefixed(b"bound-digest"),
        evidence_root: sha256_prefixed(b"bound-evidence"),
        bind_status: "bound".into(),
        digest_bound: true,
        evidence_bound: false,
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
fn a_binding_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_binding(&plan, &observe()).unwrap();
    assert!(measured.report.digest_bound);
    assert!(!measured.report.evidence_bound);
    verify_binding(&measured).unwrap();

    let mut varied = observe();
    varied.bind_status = "rejected".into();
    let varied = measure_binding(&plan, &varied).unwrap();
    assert!(varied.report.digest_bound);
    assert_eq!(varied.report.validation_result, "recorded");
    let mut varied = observe();
    varied.bind_status = "unsigned-local".into();
    varied.digest_bound = false;
    varied.public_key_bytes = 0;
    varied.signature_bytes = 0;
    varied.scalar_bytes = 0;
    let varied = measure_binding(&plan, &varied).unwrap();
    assert!(!varied.report.digest_bound);
    assert!(!varied.report.evidence_bound);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_binding(&slot, &observe()).unwrap();
    assert_eq!(on_slot.report.validation_result, "verified");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_binding_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.evidence_bound = true;
    let err = measure_binding(&plan, &bad).unwrap_err();
    assert!(err.message.contains("evidence stays unbound"), "{err}");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_binding_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_binding(&plan, &observe()).unwrap();
    write_binding_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.binding-report"
    );
    let again = write_binding_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-binding-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_binding_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
