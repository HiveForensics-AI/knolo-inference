use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, CHAIN_LINK_COUNT};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_receipt_chain, reference_engine_build,
    reference_kernel_bundle, verify_receipt_chain, write_chain_report, write_synthetic_model,
    ChainObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-chain"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn kernel_root() -> infer_contracts::DigestHex {
    reference_kernel_bundle().unwrap().root().unwrap()
}

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-chain-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    source: &infer_engine::VerifiedWeightSource,
    finish: &str,
    output_tokens: u32,
) -> ChainObservation {
    ChainObservation {
        model_runtime_root: source.runtime_root.clone(),
        artifact_root: source.artifact_root.clone(),
        engine_build_root: engine_root(),
        kernel_bundle_root: kernel_root(),
        prompt_token_root: pin(b"prompt-tokens"),
        knowledge_image_root: pin(b"knowledge-image"),
        knowledge_commit_root: pin(b"knowledge-commit"),
        query_receipt_root: pin(b"query"),
        query_receipt_count: 1,
        reflex_receipt_root: pin(b"reflex"),
        reflex_receipt_count: 1,
        receipt_root: pin(b"receipt"),
        effect_root: pin(b"effect"),
        output_token_root: pin(b"output-tokens"),
        output_text_root: pin(b"output-text"),
        prompt_token_count: 2,
        output_token_count: output_tokens,
        finish_reason: finish.into(),
        assurance: "same_build_replayable".into(),
        link_count: CHAIN_LINK_COUNT,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_receipt_chain_links_knowledge_to_the_effect() {
    let dir = scratch("chain");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_receipt_chain(&plan, &observe(&source, "stop", 1)).unwrap();
    assert_eq!(measured.report.link_count, 5);
    assert_eq!(measured.report.validation_result, "verified");
    assert_eq!(measured.report.query_receipt_count, 1);
    assert_eq!(measured.report.reflex_receipt_count, 1);
    assert!(measured.report.extensions.is_empty());
    verify_receipt_chain(&measured).unwrap();

    let length = measure_receipt_chain(&plan, &observe(&source, "length", 0)).unwrap();
    assert_eq!(length.report.output_token_count, 0);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_receipt_chain(&slot, &observe(&source, "stop", 1)).unwrap();
    assert_eq!(on_slot.report.effect_root, measured.report.effect_root);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_receipt_chain_refuses_a_repeated_link() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut repeated = observe(&source, "stop", 1);
    repeated.reflex_receipt_root = repeated.query_receipt_root.clone();
    let err = measure_receipt_chain(&plan, &repeated).unwrap_err();
    assert!(
        err.message.contains("the receipt chain repeats a link"),
        "{err}"
    );

    let mut commit = observe(&source, "stop", 1);
    commit.knowledge_commit_root = commit.knowledge_image_root.clone();
    let err = measure_receipt_chain(&plan, &commit).unwrap_err();
    assert!(
        err.message
            .contains("the knowledge commit repeats the image"),
        "{err}"
    );

    let mut short = observe(&source, "stop", 1);
    short.link_count = 4;
    let err = measure_receipt_chain(&plan, &short).unwrap_err();
    assert!(
        err.message.contains("the receipt chain has five links"),
        "{err}"
    );

    let mut quiet = observe(&source, "stop", 1);
    quiet.query_receipt_count = 0;
    let err = measure_receipt_chain(&plan, &quiet).unwrap_err();
    assert!(
        err.message
            .contains("the receipt chain has no query receipt"),
        "{err}"
    );

    let err = measure_receipt_chain(&plan, &observe(&source, "stop", 0)).unwrap_err();
    assert!(
        err.message.contains("a stop chain has no output tokens"),
        "{err}"
    );

    let mut wide = observe(&source, "length", 1);
    wide.prompt_token_count = 16;
    let err = measure_receipt_chain(&plan, &wide).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContextLimitExceeded, "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_chain_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_receipt_chain(&plan, &observe(&source, "stop", 1)).unwrap();
    write_chain_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.chain-report"
    );
    let prompt = b"Review the policy evidence.";
    assert!(!stored.windows(prompt.len()).any(|window| window == prompt));
    let again = write_chain_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-chain-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_chain_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
