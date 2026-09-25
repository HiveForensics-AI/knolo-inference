use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_tokenizer_invalid, reference_engine_build,
    reference_kernel_bundle, verify_tokenizer_invalid, write_synthetic_model,
    write_tokenizer_invalid_report, TokenizerInvalidObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-tokenizer-invalid"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-tokenizer-invalid-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(failure: &str, parsed: bool, rendered: bool) -> TokenizerInvalidObservation {
    TokenizerInvalidObservation {
        engine_build_root: engine_root(),
        tokenizer_root: sha256_prefixed(b"tokenizer"),
        failure: failure.into(),
        code: "TOKENIZER_INVALID".into(),
        retryable: false,
        tokenizer_parsed: parsed,
        template_rendered: rendered,
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
fn the_tokenizer_report_records_root_file_special_and_encode() {
    let dir = scratch("tokenizer");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let root = measure_tokenizer_invalid(&plan, &observe("root", false, false)).unwrap();
    assert!(!root.report.tokenizer_parsed);
    assert!(!root.report.template_rendered);
    verify_tokenizer_invalid(&root).unwrap();

    let file = measure_tokenizer_invalid(&plan, &observe("file", false, true)).unwrap();
    assert!(file.report.template_rendered);
    assert!(!file.report.tokenizer_parsed);

    let special = measure_tokenizer_invalid(&plan, &observe("special", true, true)).unwrap();
    assert!(special.report.tokenizer_parsed);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let encode = measure_tokenizer_invalid(&slot, &observe("encode", true, true)).unwrap();
    assert_eq!(encode.report.code, "TOKENIZER_INVALID");
    assert!(!encode.report.prompt_compiled);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_tokenizer_record_refuses_a_parsed_root_and_a_compiled_prompt() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let err = measure_tokenizer_invalid(&plan, &observe("root", true, false)).unwrap_err();
    assert!(
        err.message
            .contains("a root mismatch does not parse the tokenizer"),
        "{err}"
    );

    let err = measure_tokenizer_invalid(&plan, &observe("file", false, false)).unwrap_err();
    assert!(
        err.message
            .contains("a tokenizer file is checked after the template renders"),
        "{err}"
    );

    let mut compiled = observe("encode", true, true);
    compiled.prompt_compiled = true;
    let err = measure_tokenizer_invalid(&plan, &compiled).unwrap_err();
    assert!(
        err.message
            .contains("a tokenizer failure does not compile the prompt"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_tokenizer_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_tokenizer_invalid(&plan, &observe("encode", true, true)).unwrap();
    write_tokenizer_invalid_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.tokenizer-invalid-report"
    );
    let again = write_tokenizer_invalid_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-tokenizer-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_tokenizer_invalid_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
