use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_evidence_output, verify_evidence_output,
    write_evidence_output_report, write_synthetic_model, EvidenceObservation,
};

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-evidence-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(finish: &str, output_tokens: u32) -> EvidenceObservation {
    let evidence = pin(b"evidence");
    let knowledge = pin(b"knowledge");
    let query = pin(b"query");
    let reflex = pin(b"reflex");
    let tokens = pin(b"output-tokens");
    let text = pin(b"output-text");
    EvidenceObservation {
        evidence_root: evidence.clone(),
        bound_evidence_root: evidence,
        knowledge_image_root: knowledge.clone(),
        bound_knowledge_image_root: knowledge,
        query_receipt_root: query.clone(),
        bound_query_receipt_root: query,
        reflex_receipt_root: reflex.clone(),
        bound_reflex_receipt_root: reflex,
        output_token_root: tokens.clone(),
        bound_output_token_root: tokens,
        output_text_root: text.clone(),
        bound_output_text_root: text,
        receipt_root: pin(b"receipt"),
        chain_root: pin(b"chain"),
        prompt_token_count: 2,
        output_token_count: output_tokens,
        finish_reason: finish.into(),
        assurance: "compatibility".into(),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_output_stays_bound_to_the_chain_evidence() {
    let dir = scratch("bound");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_evidence_output(&plan, &observe("stop", 1)).unwrap();
    assert_eq!(measured.report.validation_result, "verified");
    assert_ne!(measured.report.chain_root, measured.report.receipt_root);
    verify_evidence_output(&measured).unwrap();

    let length = measure_evidence_output(&plan, &observe("length", 0)).unwrap();
    assert_eq!(length.report.output_token_count, 0);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_evidence_output(&slot, &observe("stop", 1)).unwrap();
    assert_eq!(on_slot.report.evidence_root, measured.report.evidence_root);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_unbound_output_issues_no_report() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut knowledge = observe("stop", 1);
    knowledge.bound_knowledge_image_root = pin(b"other-knowledge");
    let err = measure_evidence_output(&plan, &knowledge).unwrap_err();
    assert!(
        err.message
            .contains("the knowledge image does not bind the output"),
        "{err}"
    );

    let mut query = observe("stop", 1);
    query.bound_query_receipt_root = pin(b"other-query");
    let err = measure_evidence_output(&plan, &query).unwrap_err();
    assert!(
        err.message
            .contains("the query receipt does not bind the output"),
        "{err}"
    );

    let mut reflex = observe("stop", 1);
    reflex.bound_reflex_receipt_root = pin(b"other-reflex");
    let err = measure_evidence_output(&plan, &reflex).unwrap_err();
    assert!(
        err.message
            .contains("the reflex receipt does not bind the output"),
        "{err}"
    );

    let mut evidence = observe("stop", 1);
    evidence.bound_evidence_root = pin(b"other-evidence");
    let err = measure_evidence_output(&plan, &evidence).unwrap_err();
    assert!(
        err.message
            .contains("the evidence root does not bind the output"),
        "{err}"
    );

    let mut tokens = observe("stop", 1);
    tokens.bound_output_token_root = pin(b"other-tokens");
    let err = measure_evidence_output(&plan, &tokens).unwrap_err();
    assert!(
        err.message
            .contains("the output token root does not bind the receipt"),
        "{err}"
    );

    let mut repeated = observe("stop", 1);
    repeated.chain_root = repeated.receipt_root.clone();
    let err = measure_evidence_output(&plan, &repeated).unwrap_err();
    assert!(
        err.message.contains("the chain repeats the receipt"),
        "{err}"
    );

    let mut wide = observe("length", 1);
    wide.prompt_token_count = 16;
    let err = measure_evidence_output(&plan, &wide).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContextLimitExceeded, "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_evidence_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_evidence_output(&plan, &observe("stop", 1)).unwrap();
    write_evidence_output_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.evidence-output-report"
    );
    let prompt = b"Review the policy evidence.";
    assert!(!stored.windows(prompt.len()).any(|window| window == prompt));
    let again = write_evidence_output_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-evidence-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_evidence_output_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
