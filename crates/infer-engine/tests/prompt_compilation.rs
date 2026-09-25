use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_prompt_compilation, reference_engine_build,
    reference_kernel_bundle, verify_prompt_compilation, write_prompt_compilation_report,
    write_synthetic_model, PromptCompilationObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-prompt-compilation"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-prompt-compilation-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(failure: &str, tokens: u32, rejected: u32) -> PromptCompilationObservation {
    PromptCompilationObservation {
        engine_build_root: engine_root(),
        failure: failure.into(),
        token_count: tokens,
        rejected_token: rejected,
        code: "PROMPT_COMPILATION_FAILED".into(),
        retryable: false,
        template_rendered: true,
        tokenizer_parsed: true,
        prompt_compiled: false,
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
fn the_prompt_compilation_records_an_empty_prompt_a_bad_token_and_an_oversized_prompt() {
    let dir = scratch("prompt");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let empty = measure_prompt_compilation(&plan, &observe("empty", 0, 0)).unwrap();
    assert!(!empty.report.prompt_compiled);
    assert!(empty.report.template_rendered);
    verify_prompt_compilation(&empty).unwrap();

    let vocab = measure_prompt_compilation(&plan, &observe("vocab", 1, 16)).unwrap();
    assert_eq!(vocab.report.rejected_token, 16);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let size = measure_prompt_compilation(&slot, &observe("size", 0, 0)).unwrap();
    assert_eq!(size.report.failure, "size");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_prompt_compilation_refuses_a_compiled_prompt_and_a_prompt_past_the_context() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut compiled = observe("empty", 0, 0);
    compiled.prompt_compiled = true;
    let err = measure_prompt_compilation(&plan, &compiled).unwrap_err();
    assert!(
        err.message
            .contains("a prompt failure does not compile the prompt"),
        "{err}"
    );

    let err = measure_prompt_compilation(&plan, &observe("vocab", 17, 16)).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContextLimitExceeded, "{err}");
    assert!(
        err.message
            .contains("the prompt compilation is the micro fixture"),
        "{err}"
    );

    let err = measure_prompt_compilation(&plan, &observe("vocab", 1, 15)).unwrap_err();
    assert!(
        err.message
            .contains("a vocabulary failure is outside the micro vocabulary"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_prompt_compilation_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_prompt_compilation(&plan, &observe("empty", 0, 0)).unwrap();
    write_prompt_compilation_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.prompt-compilation-report"
    );
    let again = write_prompt_compilation_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!(
        "knolo-prompt-compilation-not-created-{}",
        std::process::id()
    );
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_prompt_compilation_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
