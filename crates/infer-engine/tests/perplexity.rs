use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    argmax, cpu_placement, greedy_generate, load_verified_micro, micro_kv_layout, perplexity_delta,
    reference_engine_build, reference_kernel_bundle, verify_perplexity_report,
    write_perplexity_report, write_synthetic_model, ArchitectureAdapter, MicroAdapter,
    ReferenceF32Backend, SingleBlockKv,
};

fn reporter_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-perplexity"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-perplexity-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn identical_logits_record_a_zero_delta() {
    let root = reporter_root();
    let reference = [0.0f32, 0.0];
    let measured = perplexity_delta(2, &[0], &reference, &reference, &root).unwrap();
    assert_eq!(measured.report.validation_result, "recorded");
    assert_eq!(measured.report.reference_perplexity_micros, 2_000_000);
    assert_eq!(measured.report.candidate_perplexity_micros, 2_000_000);
    assert_eq!(measured.report.perplexity_delta_micros, 0);
    assert!(measured.report.greedy_parity);
    assert_eq!(measured.report.target_accuracy_millionths, 1_000_000);
    assert_eq!(measured.report.max_abs_logit_delta_millionths, 0);
    assert_eq!(measured.report.token_count, 1);
    verify_perplexity_report(&measured).unwrap();
    assert_eq!(reference, [0.0, 0.0]);
}

#[test]
fn a_shifted_candidate_records_the_perplexity_delta() {
    let root = reporter_root();
    let reference = [0.0f32, 0.0];
    let candidate = [0.0f32, 1.0];
    let measured = perplexity_delta(2, &[0], &reference, &candidate, &root).unwrap();
    assert_eq!(measured.report.reference_perplexity_micros, 2_000_000);
    assert_eq!(measured.report.candidate_perplexity_micros, 3_718_282);
    assert_eq!(measured.report.perplexity_delta_micros, 1_718_282);
    assert!(!measured.report.greedy_parity);
    assert_eq!(measured.report.target_accuracy_millionths, 0);
    assert_eq!(measured.report.max_abs_logit_delta_millionths, 1_000_000);

    let improved = perplexity_delta(2, &[0], &candidate, &reference, &root).unwrap();
    assert_eq!(improved.report.reference_perplexity_micros, 3_718_282);
    assert_eq!(improved.report.candidate_perplexity_micros, 2_000_000);
    assert_eq!(improved.report.perplexity_delta_micros, -1_718_282);
    assert_eq!(improved.report.target_accuracy_millionths, 1_000_000);
    assert!(!improved.report.greedy_parity);
}

#[test]
fn two_rows_use_the_mean_negative_log_likelihood() {
    let root = reporter_root();
    let logits = [0.0f32, 0.0, 0.0, 1.0];
    let measured = perplexity_delta(2, &[0, 0], &logits, &logits, &root).unwrap();
    assert_eq!(measured.report.token_count, 2);
    assert_eq!(measured.report.reference_perplexity_micros, 2_727_006);
    assert_eq!(measured.report.perplexity_delta_micros, 0);
    assert!(measured.report.greedy_parity);
    assert_eq!(measured.report.target_accuracy_millionths, 500_000);
}

#[test]
fn the_micro_model_fixture_scores_its_own_prefill_logits() {
    let dir = scratch("micro");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let placement = cpu_placement(&source).unwrap();
    let mut model = MicroAdapter
        .build(&source, &placement, &ReferenceF32Backend)
        .unwrap();
    let mut kv = SingleBlockKv::new(micro_kv_layout()).unwrap();
    let output = greedy_generate(&mut *model, &mut kv, 1, &[1, 4, 7], 0).unwrap();
    let target = argmax(&output.prefill_logits).unwrap();
    let root = reporter_root();
    let measured = perplexity_delta(
        16,
        &[target],
        &output.prefill_logits,
        &output.prefill_logits,
        &root,
    )
    .unwrap();
    assert_eq!(measured.report.perplexity_delta_micros, 0);
    assert!(measured.report.greedy_parity);
    assert_eq!(measured.report.target_accuracy_millionths, 1_000_000);
    assert_eq!(measured.report.max_abs_logit_delta_millionths, 0);
    assert!(measured.report.reference_perplexity_micros >= 1_000_000);

    let mut shifted = output.prefill_logits.clone();
    let other = if target == 0 { 1 } else { 0 };
    shifted[other as usize] = shifted[target as usize] + 1.0;
    let delta = perplexity_delta(16, &[target], &output.prefill_logits, &shifted, &root).unwrap();
    assert!(delta.report.perplexity_delta_micros > 0, "{delta:?}");
    assert!(!delta.report.greedy_parity);
    assert_eq!(delta.report.target_accuracy_millionths, 0);
    assert!(delta.report.max_abs_logit_delta_millionths >= 1_000_000);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_failed_measurement_issues_no_report_and_does_not_read_a_huge_matrix() {
    let root = reporter_root();
    let zero_vocab = perplexity_delta(0, &[0], &[], &[], &root).unwrap_err();
    assert!(zero_vocab.message.contains("vocab is zero"), "{zero_vocab}");

    let zero_tokens = perplexity_delta(2, &[], &[], &[], &root).unwrap_err();
    assert!(
        zero_tokens.message.contains("token count is zero"),
        "{zero_tokens}"
    );

    let huge = perplexity_delta(8_388_609, &[0], &[], &[], &root).unwrap_err();
    assert_eq!(huge.code, ErrorCode::InsufficientMemory, "{huge}");
    assert!(huge.message.contains("32 MiB"), "{huge}");

    let outside = perplexity_delta(2, &[2], &[0.0, 0.0], &[0.0, 0.0], &root).unwrap_err();
    assert!(outside.message.contains("outside the vocab"), "{outside}");

    let short = perplexity_delta(2, &[0], &[0.0], &[0.0, 0.0], &root).unwrap_err();
    assert!(short.message.contains("reference length"), "{short}");

    let short_candidate = perplexity_delta(2, &[0], &[0.0, 0.0], &[0.0], &root).unwrap_err();
    assert!(
        short_candidate.message.contains("candidate length"),
        "{short_candidate}"
    );

    let nan = perplexity_delta(1, &[0], &[f32::NAN], &[0.0], &root).unwrap_err();
    assert_eq!(nan.code, ErrorCode::ContractInvalid, "{nan}");
    assert!(nan.message.contains("not finite"), "{nan}");

    let nan_candidate = perplexity_delta(1, &[0], &[0.0], &[f32::INFINITY], &root).unwrap_err();
    assert!(
        nan_candidate.message.contains("not finite"),
        "{nan_candidate}"
    );
}

#[test]
fn the_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    let root = reporter_root();
    let measured = perplexity_delta(2, &[0], &[0.0, 0.0], &[0.0, 0.0], &root).unwrap();
    write_perplexity_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.perplexity-report"
    );
    let again = write_perplexity_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");
    assert_eq!(fs::read(dir.join("report.cbor")).unwrap(), stored);

    let outside_name = format!("knolo-ppl-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_perplexity_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());

    std::os::unix::fs::symlink(&outside, dir.join("linked.cbor")).unwrap();
    let linked = write_perplexity_report(&dir, &measured, "linked.cbor").unwrap_err();
    assert!(linked.message.contains("already exists"), "{linked}");
    assert!(!outside.exists());

    let missing = scratch("missing");
    fs::remove_dir_all(&missing).unwrap();
    let err = write_perplexity_report(&missing, &measured, "report.cbor").unwrap_err();
    assert!(err.message.contains("does not exist"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_tampered_matrix_is_not_written() {
    let dir = scratch("tamper");
    let root = reporter_root();
    let mut measured = perplexity_delta(2, &[0], &[0.0, 0.0], &[0.0, 0.0], &root).unwrap();
    measured.reference[0] = 1.0;
    let err = write_perplexity_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(err.message.contains("reference logit root"), "{err}");
    assert!(fs::read_dir(&dir).unwrap().next().is_none());

    measured.reference[0] = 0.0;
    measured.report.validation_result = "matched".into();
    let rejected = write_perplexity_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(rejected.message.contains("validationResult"), "{rejected}");
    assert!(fs::read_dir(&dir).unwrap().next().is_none());
    let _ = fs::remove_dir_all(&dir);
}
