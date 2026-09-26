use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_receipt_view, reference_engine_build,
    reference_kernel_bundle, verify_receipt_view, write_studio_report, write_synthetic_model,
    StudioObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-studio"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("knolo-infer-studio-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    source: &infer_engine::VerifiedWeightSource,
    finish: &str,
    output_tokens: u32,
) -> StudioObservation {
    StudioObservation {
        model_image_root: source.image_root.clone(),
        artifact_root: source.artifact_root.clone(),
        engine_build_root: engine_root(),
        receipt_root: pin(b"receipt"),
        prompt_token_root: pin(b"prompt-tokens"),
        output_token_root: pin(b"output-tokens"),
        output_text_root: pin(b"output-text"),
        knowledge_image_root: pin(b"knowledge"),
        evidence_root: pin(b"evidence"),
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
fn the_receipt_view_stores_roots_and_counts() {
    let dir = scratch("view");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_receipt_view(&plan, &observe(&source, "stop", 1)).unwrap();
    assert_eq!(measured.report.finish_reason, "stop");
    assert_eq!(measured.report.prompt_token_count, 2);
    assert_eq!(measured.report.output_token_count, 1);
    assert_eq!(measured.report.validation_result, "recorded");
    assert!(measured.report.extensions.is_empty());
    verify_receipt_view(&measured).unwrap();

    let length = measure_receipt_view(&plan, &observe(&source, "length", 0)).unwrap();
    assert_eq!(length.report.output_token_count, 0);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_receipt_view(&slot, &observe(&source, "stop", 1)).unwrap();
    assert_eq!(on_slot.report.receipt_root, measured.report.receipt_root);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_studio_view_refuses_text_and_a_context_overrun() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let err = measure_receipt_view(&plan, &observe(&source, "stop", 0)).unwrap_err();
    assert!(
        err.message.contains("a stop receipt has no output tokens"),
        "{err}"
    );

    let mut empty = observe(&source, "stop", 1);
    empty.prompt_token_count = 0;
    let err = measure_receipt_view(&plan, &empty).unwrap_err();
    assert!(
        err.message.contains("the studio view has no prompt"),
        "{err}"
    );

    let mut wide = observe(&source, "length", 1);
    wide.prompt_token_count = 16;
    let err = measure_receipt_view(&plan, &wide).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContextLimitExceeded, "{err}");
    assert!(
        err.message.contains("the studio view is the micro fixture"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_studio_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_receipt_view(&plan, &observe(&source, "stop", 1)).unwrap();
    write_studio_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.studio-report"
    );
    let prompt = b"Review the policy evidence.";
    assert!(!stored.windows(prompt.len()).any(|window| window == prompt));
    let again = write_studio_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-studio-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_studio_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
