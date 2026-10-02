use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, EvidenceBindingV1};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_evidence_composition, reference_engine_build,
    reference_kernel_bundle, verify_evidence_composition, write_composition_report,
    write_synthetic_model, CompositionObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-composition"),
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
        "knolo-infer-composition-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn binding(image: &[u8], commit: &[u8]) -> EvidenceBindingV1 {
    EvidenceBindingV1 {
        knowledge_image_root: Some(pin(image)),
        knowledge_commit_root: Some(pin(commit)),
        query_receipt_ids: vec![pin(b"query")],
        reflex_receipt_ids: vec![pin(b"reflex")],
        context_root: pin(b"context"),
        ordered_evidence_ids: vec!["block-1".into()],
        extensions: BTreeMap::new(),
    }
}

fn observe(source: &infer_engine::VerifiedWeightSource) -> CompositionObservation {
    CompositionObservation {
        model_image_root: source.image_root.clone(),
        artifact_root: source.artifact_root.clone(),
        engine_build_root: engine_root(),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        knowledge_image_root: pin(b"knowledge-image"),
        knowledge_commit_root: pin(b"knowledge-commit"),
        context_root: pin(b"context"),
        query_receipt_count: 1,
        reflex_receipt_count: 1,
        ordered_evidence_count: 1,
    }
}

#[test]
fn the_micro_fixture_records_a_host_supplied_binding() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let host = binding(b"knowledge-image", b"knowledge-commit");
    let measured = measure_evidence_composition(&plan, &observe(&source), &host).unwrap();
    assert_eq!(measured.report.query_receipt_count, 1);
    assert_eq!(measured.report.reflex_receipt_count, 1);
    assert_eq!(measured.report.validation_result, "recorded");
    assert_ne!(
        measured.report.knowledge_image_root,
        measured.report.knowledge_commit_root
    );
    verify_evidence_composition(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_evidence_composition(&slot, &observe(&source), &host).unwrap();
    assert_eq!(on_slot.report.evidence_root, measured.report.evidence_root);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_composition_refuses_a_binding_the_host_did_not_supply() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut host = binding(b"knowledge-image", b"knowledge-commit");
    host.knowledge_image_root = None;
    let err = measure_evidence_composition(&plan, &observe(&source), &host).unwrap_err();
    assert!(
        err.message
            .contains("the composition has no knowledge image"),
        "{err}"
    );

    host = binding(b"knowledge-image", b"knowledge-image");
    let err = measure_evidence_composition(&plan, &observe(&source), &host).unwrap_err();
    assert!(
        err.message
            .contains("the knowledge commit repeats the image"),
        "{err}"
    );

    host = binding(b"knowledge-image", b"knowledge-commit");
    host.query_receipt_ids.clear();
    let err = measure_evidence_composition(&plan, &observe(&source), &host).unwrap_err();
    assert!(
        err.message.contains("the composition has no query receipt"),
        "{err}"
    );

    host = binding(b"knowledge-image", b"knowledge-commit");
    host.reflex_receipt_ids.clear();
    let err = measure_evidence_composition(&plan, &observe(&source), &host).unwrap_err();
    assert!(
        err.message
            .contains("the composition has no reflex receipt"),
        "{err}"
    );

    host = binding(b"knowledge-image", b"knowledge-commit");
    host.ordered_evidence_ids.clear();
    let err = measure_evidence_composition(&plan, &observe(&source), &host).unwrap_err();
    assert!(
        err.message.contains("the composition has no evidence id"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_composition_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_evidence_composition(
        &plan,
        &observe(&source),
        &binding(b"knowledge-image", b"knowledge-commit"),
    )
    .unwrap();
    write_composition_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.composition-report"
    );
    let again = write_composition_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-composition-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_composition_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
