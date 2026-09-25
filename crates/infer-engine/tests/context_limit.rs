use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, MICRO_CONTEXT};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_context_limit, reference_engine_build,
    reference_kernel_bundle, verify_context_limit, write_context_limit_report,
    write_synthetic_model, ContextLimitObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-context-limit"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-context-limit-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(reason: &str, prompt: u32, reserved: u32) -> ContextLimitObservation {
    ContextLimitObservation {
        engine_build_root: engine_root(),
        reason: reason.into(),
        prompt_tokens: prompt,
        reserved_tokens: reserved,
        context_tokens: MICRO_CONTEXT,
        truncated: false,
        code: "CONTEXT_LIMIT_EXCEEDED".into(),
        retryable: false,
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
fn the_context_limit_records_a_long_prompt_a_budget_and_an_overflow() {
    let dir = scratch("limit");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let prompt = measure_context_limit(&plan, &observe("prompt", 17, 0)).unwrap();
    assert_eq!(prompt.report.prompt_tokens, 17);
    assert!(!prompt.report.truncated);
    verify_context_limit(&prompt).unwrap();

    let budget = measure_context_limit(&plan, &observe("budget", 8, 9)).unwrap();
    assert_eq!(budget.report.reserved_tokens, 9);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let overflow = measure_context_limit(&slot, &observe("overflow", u32::MAX, 1)).unwrap();
    assert_eq!(overflow.report.reason, "overflow");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_context_limit_refuses_truncation_and_a_prompt_past_the_record_cap() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut truncated = observe("prompt", 17, 0);
    truncated.truncated = true;
    let err = measure_context_limit(&plan, &truncated).unwrap_err();
    assert!(err.message.contains("tokens are not truncated"), "{err}");

    let err = measure_context_limit(&plan, &observe("prompt", 65, 0)).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContextLimitExceeded, "{err}");
    assert!(
        err.message.contains("prompt tokens exceed the record cap"),
        "{err}"
    );

    let err = measure_context_limit(&plan, &observe("budget", 4, 4)).unwrap_err();
    assert!(
        err.message
            .contains("a budget that fits is not a context limit"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_context_limit_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_context_limit(&plan, &observe("prompt", 17, 0)).unwrap();
    write_context_limit_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.context-limit-report"
    );
    let again = write_context_limit_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-context-limit-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_context_limit_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
