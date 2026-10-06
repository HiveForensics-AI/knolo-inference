use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, MAX_GRAMMAR_SOURCE_BYTES};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_grammar_refusal, reference_engine_build,
    reference_kernel_bundle, verify_grammar_refusal, write_grammar_refusal_report,
    write_synthetic_model, GrammarRefusalObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-grammar-refusal"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-grammar-refusal-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    reason: &str,
    source_bytes: u32,
    opened: bool,
    rooted: bool,
) -> GrammarRefusalObservation {
    GrammarRefusalObservation {
        engine_build_root: engine_root(),
        reason: reason.into(),
        source_bytes,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        source_opened: opened,
        grammar_rooted: rooted,
        mask_applied: false,
        grammar_compiled: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn a_grammar_refusal_records_a_schema_an_automaton_and_a_mask() {
    let dir = scratch("grammar");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let schema = measure_grammar_refusal(&plan, &observe("schema", 0, false, false)).unwrap();
    assert!(!schema.report.source_opened);
    assert!(!schema.report.grammar_compiled);
    verify_grammar_refusal(&schema).unwrap();

    let automaton = measure_grammar_refusal(&plan, &observe("automaton", 32, true, false)).unwrap();
    assert!(automaton.report.source_opened);
    assert!(!automaton.report.grammar_rooted);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let mask = measure_grammar_refusal(&slot, &observe("mask", 16, true, true)).unwrap();
    assert!(mask.report.grammar_rooted);
    assert!(!mask.report.mask_applied);
    assert_eq!(mask.report.code, "CONTRACT_INVALID");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_grammar_refusal_rejects_a_compiled_grammar_and_a_source_past_the_cap() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut compiled = observe("mask", 16, true, true);
    compiled.grammar_compiled = true;
    let err = measure_grammar_refusal(&plan, &compiled).unwrap_err();
    assert!(
        err.message
            .contains("a grammar refusal does not compile the grammar"),
        "{err}"
    );

    let err = measure_grammar_refusal(
        &plan,
        &observe("automaton", MAX_GRAMMAR_SOURCE_BYTES + 1, true, false),
    )
    .unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid);
    assert!(
        err.message
            .contains("grammar source exceeds the record cap"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_grammar_refusal_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_grammar_refusal(&plan, &observe("mask", 8, true, true)).unwrap();
    write_grammar_refusal_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.grammar-refusal-report"
    );
    let again = write_grammar_refusal_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-grammar-refusal-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_grammar_refusal_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
