use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, MAX_PROPOSAL_TOKENS};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_speculative, reference_engine_build,
    reference_kernel_bundle, verify_speculative, write_speculative_report, write_synthetic_model,
    SpeculativeObservation,
};

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-speculative"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-speculative-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(reason: &str, proposal_tokens: u32) -> SpeculativeObservation {
    SpeculativeObservation {
        engine_build_root: engine_root(),
        target_root: pin(b"spec-target"),
        proposal_root: pin(b"spec-proposal"),
        reason: reason.into(),
        proposal_tokens,
        accepted_tokens: 0,
        rejected_tokens: 0,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        speculated: false,
        distribution_changed: false,
        cache_affected: false,
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
fn a_speculative_refusal_records_a_draft_an_mtp_head_and_a_plan() {
    let dir = scratch("spec");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let draft = measure_speculative(&plan, &observe("draft", 4)).unwrap();
    assert!(!draft.report.speculated);
    assert!(!draft.report.distribution_changed);
    assert_eq!(draft.report.proposal_tokens, 4);
    verify_speculative(&draft).unwrap();

    let mtp = measure_speculative(&plan, &observe("mtp", MAX_PROPOSAL_TOKENS)).unwrap();
    assert_eq!(mtp.report.proposal_tokens, MAX_PROPOSAL_TOKENS);
    assert_eq!(mtp.report.accepted_tokens, 0);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let refused = measure_speculative(&slot, &observe("plan", 0)).unwrap();
    assert_eq!(refused.report.proposal_tokens, 0);
    assert!(!refused.report.cache_affected);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_speculative_refusal_rejects_a_run_and_a_proposal_past_the_context() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut ran = observe("draft", 4);
    ran.speculated = true;
    let err = measure_speculative(&plan, &ran).unwrap_err();
    assert!(
        err.message
            .contains("a speculative refusal does not speculate"),
        "{err}"
    );

    let err = measure_speculative(&plan, &observe("draft", MAX_PROPOSAL_TOKENS + 1)).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContextLimitExceeded);
    assert!(
        err.message
            .contains("a speculative proposal exceeds the context"),
        "{err}"
    );

    let err = measure_speculative(&plan, &observe("plan", 4)).unwrap_err();
    assert!(
        err.message.contains("a plan refusal carries no proposal"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_speculative_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_speculative(&plan, &observe("mtp", 2)).unwrap();
    write_speculative_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.speculative-report"
    );
    let again = write_speculative_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-speculative-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_speculative_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
