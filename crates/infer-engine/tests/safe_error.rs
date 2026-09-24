use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_safe_error, reference_engine_build,
    reference_kernel_bundle, verify_safe_error, write_safe_error_report, write_synthetic_model,
    SafeErrorObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-safe-error"),
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
        "knolo-infer-safe-error-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(code: &str, retryable: bool, partial: &str) -> SafeErrorObservation {
    SafeErrorObservation {
        engine_build_root: engine_root(),
        code: code.into(),
        message: code.into(),
        retryable,
        request_id: "req-1".into(),
        attempt: 1,
        prompt_present: false,
        secret_present: false,
        partial_receipt: partial.into(),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_safe_error_records_the_stable_code_and_its_retryability() {
    let dir = scratch("error");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let local = measure_safe_error(&plan, &observe("CONTRACT_INVALID", false, "absent")).unwrap();
    assert_eq!(local.report.message, "CONTRACT_INVALID");
    assert!(!local.report.retryable);
    assert!(!local.report.prompt_present);
    assert_eq!(local.report.partial_receipt, "absent");
    verify_safe_error(&local).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let partial = pin(b"partial").to_string();
    let draining = measure_safe_error(&slot, &observe("SERVICE_DRAINING", true, &partial)).unwrap();
    assert!(draining.report.retryable);
    assert_eq!(draining.report.attempt, 1);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_safe_error_refuses_a_prompt_a_secret_and_a_mismatched_retry() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut prompt = observe("CONTRACT_INVALID", false, "absent");
    prompt.prompt_present = true;
    let err = measure_safe_error(&plan, &prompt).unwrap_err();
    assert!(
        err.message.contains("a safe error omits the prompt"),
        "{err}"
    );

    let mut secret = observe("CONTRACT_INVALID", false, "absent");
    secret.secret_present = true;
    let err = measure_safe_error(&plan, &secret).unwrap_err();
    assert!(err.message.contains("a safe error omits secrets"), "{err}");

    let err = measure_safe_error(&plan, &observe("CONTRACT_INVALID", true, "absent")).unwrap_err();
    assert!(
        err.message.contains("retryability follows the stable code"),
        "{err}"
    );

    let mut message = observe("WORKER_LOST", true, "absent");
    message.message = "the prompt was hi".into();
    let err = measure_safe_error(&plan, &message).unwrap_err();
    assert!(
        err.message.contains("the safe message is the stable code"),
        "{err}"
    );

    let err = measure_safe_error(&plan, &observe("NOT_A_CODE", false, "absent")).unwrap_err();
    assert!(
        err.message.contains("field code has an unsupported value"),
        "{err}"
    );

    let mut attempt = observe("REQUEST_CANCELLED", false, "absent");
    attempt.attempt = 2;
    let err = measure_safe_error(&plan, &attempt).unwrap_err();
    assert!(
        err.message.contains("the safe error attempt is one"),
        "{err}"
    );

    let mut ident = observe("CONTRACT_INVALID", false, "absent");
    ident.request_id = "req 1".into();
    let err = measure_safe_error(&plan, &ident).unwrap_err();
    assert!(err.message.contains("the request id is a token"), "{err}");

    let placement = plan.root().unwrap();
    let err = measure_safe_error(
        &plan,
        &observe("CONTRACT_INVALID", false, placement.as_str()),
    )
    .unwrap_err();
    assert!(
        err.message
            .contains("the partial receipt repeats the placement"),
        "{err}"
    );

    let measured =
        measure_safe_error(&plan, &observe("CONTRACT_INVALID", false, "absent")).unwrap();
    let mut stored = measured.report.clone();
    stored.prompt_present = true;
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_safe_error_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured =
        measure_safe_error(&plan, &observe("CONTRACT_INVALID", false, "absent")).unwrap();
    write_safe_error_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.safe-error-report"
    );
    let again = write_safe_error_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-safe-error-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_safe_error_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
