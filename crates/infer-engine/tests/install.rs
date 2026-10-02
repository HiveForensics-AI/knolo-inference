use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, INSTALL_ARTIFACT_COUNT};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_hub_install, reference_engine_build,
    reference_kernel_bundle, verify_hub_install, write_install_report, write_synthetic_model,
    InstallObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-install"),
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
        std::env::temp_dir().join(format!("knolo-infer-install-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(source: &infer_engine::VerifiedWeightSource) -> InstallObservation {
    InstallObservation {
        model_image_present: true,
        weights_present: true,
        tokenizer_present: true,
        template_present: true,
        model_image_root: source.image_root.clone(),
        artifact_root: source.artifact_root.clone(),
        tokenizer_root: pin(b"tokenizer"),
        template_root: pin(b"template"),
        publisher: "knolo".into(),
        license_id: "apache-2.0".into(),
        source_provider: "local".into(),
        artifact_count: INSTALL_ARTIFACT_COUNT,
        engine_build_root: engine_root(),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_hub_install_verifies_every_pinned_artifact() {
    let dir = scratch("install");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_hub_install(&plan, &observe(&source)).unwrap();
    assert_eq!(measured.report.artifact_count, 4);
    assert_eq!(measured.report.source_provider, "local");
    assert_eq!(measured.report.validation_result, "verified");
    verify_hub_install(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_hub_install(&slot, &observe(&source)).unwrap();
    assert_eq!(
        on_slot.report.model_image_root,
        measured.report.model_image_root
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_hub_install_refuses_a_missing_or_remote_artifact() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut remote = observe(&source);
    remote.source_provider = "huggingface".into();
    let err = measure_hub_install(&plan, &remote).unwrap_err();
    assert!(
        err.message
            .contains("the hub install does not download weights"),
        "{err}"
    );

    let mut image = observe(&source);
    image.model_image_present = false;
    let err = measure_hub_install(&plan, &image).unwrap_err();
    assert_eq!(err.code, ErrorCode::ModelArtifactMissing, "{err}");
    assert!(
        err.message
            .contains("the hub install is missing the model image"),
        "{err}"
    );

    let mut weights = observe(&source);
    weights.weights_present = false;
    let err = measure_hub_install(&plan, &weights).unwrap_err();
    assert!(
        err.message
            .contains("the hub install is missing the weight artifact"),
        "{err}"
    );

    let mut tokenizer = observe(&source);
    tokenizer.tokenizer_present = false;
    let err = measure_hub_install(&plan, &tokenizer).unwrap_err();
    assert!(
        err.message
            .contains("the hub install is missing the tokenizer"),
        "{err}"
    );

    let mut template = observe(&source);
    template.template_present = false;
    let err = measure_hub_install(&plan, &template).unwrap_err();
    assert!(
        err.message
            .contains("the hub install is missing the template"),
        "{err}"
    );

    let mut repeated = observe(&source);
    repeated.artifact_root = repeated.model_image_root.clone();
    let err = measure_hub_install(&plan, &repeated).unwrap_err();
    assert!(
        err.message
            .contains("the weight artifact repeats the model image"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_install_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_hub_install(&plan, &observe(&source)).unwrap();
    write_install_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.install-report"
    );
    let weights = b"weight-bytes-are-not-a-field";
    assert!(!stored
        .windows(weights.len())
        .any(|window| window == weights));
    let again = write_install_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-install-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_install_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
