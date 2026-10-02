use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_redacted_log, reference_engine_build,
    reference_kernel_bundle, verify_redacted_log, write_redaction_report, write_synthetic_model,
    RedactionObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-redaction"),
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
        "knolo-infer-redaction-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(stage: &str) -> RedactionObservation {
    RedactionObservation {
        engine_build_root: engine_root(),
        stage: stage.into(),
        request_id: "req-1".into(),
        prompt_plan_root: pin(b"prompt-plan"),
        receipt_root: pin(b"receipt"),
        format: "json-line".into(),
        redacted: true,
        prompt: "omitted".into(),
        output: "omitted".into(),
        token_ids: "omitted".into(),
        alias: "omitted".into(),
        path: "omitted".into(),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_redacted_log_omits_prompt_output_and_paths() {
    let dir = scratch("log");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let local = measure_redacted_log(&plan, &observe("finalize")).unwrap();
    assert_eq!(local.report.stage, "finalize");
    assert_eq!(local.report.format, "json-line");
    assert!(local.report.redacted);
    assert_eq!(local.report.prompt, "omitted");
    assert_eq!(local.report.output, "omitted");
    assert_eq!(local.report.token_ids, "omitted");
    assert_eq!(local.report.path, "omitted");
    verify_redacted_log(&local).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let prefill = measure_redacted_log(&slot, &observe("prefill")).unwrap();
    assert_eq!(prefill.report.stage, "prefill");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_redacted_log_refuses_prompt_text_and_a_clear_log() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut prompt = observe("api");
    prompt.prompt = "present".into();
    let err = measure_redacted_log(&plan, &prompt).unwrap_err();
    assert!(
        err.message.contains("ordinary logs omit prompt text"),
        "{err}"
    );

    let mut output = observe("decode");
    output.output = "present".into();
    let err = measure_redacted_log(&plan, &output).unwrap_err();
    assert!(
        err.message.contains("ordinary logs omit output text"),
        "{err}"
    );

    let mut clear = observe("admission");
    clear.redacted = false;
    let err = measure_redacted_log(&plan, &clear).unwrap_err();
    assert!(
        err.message.contains("completion logs are redacted"),
        "{err}"
    );

    let stage = observe("cache");
    let err = measure_redacted_log(&plan, &stage).unwrap_err();
    assert!(
        err.message.contains("field stage has an unsupported value"),
        "{err}"
    );

    let mut path = observe("prompt");
    path.path = "present".into();
    let err = measure_redacted_log(&plan, &path).unwrap_err();
    assert!(err.message.contains("ordinary logs omit a path"), "{err}");

    let mut same = observe("finalize");
    same.receipt_root = same.prompt_plan_root.clone();
    let err = measure_redacted_log(&plan, &same).unwrap_err();
    assert!(
        err.message.contains("the receipt repeats the prompt plan"),
        "{err}"
    );

    let measured = measure_redacted_log(&plan, &observe("finalize")).unwrap();
    let mut stored = measured.report.clone();
    stored.alias = "daily".into();
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message.contains("ordinary logs omit the alias"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_redaction_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_redacted_log(&plan, &observe("finalize")).unwrap();
    write_redaction_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.redaction-report"
    );
    let again = write_redaction_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-redaction-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_redaction_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
