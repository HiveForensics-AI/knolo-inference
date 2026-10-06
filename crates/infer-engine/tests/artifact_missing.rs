use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_artifact_missing, reference_engine_build,
    reference_kernel_bundle, verify_artifact_missing, write_artifact_missing_report,
    write_synthetic_model, ArtifactMissingObservation,
};

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-artifact-missing"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-artifact-missing-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    reason: &str,
    lock: bool,
    alias: bool,
    image: bool,
    weights: bool,
) -> ArtifactMissingObservation {
    ArtifactMissingObservation {
        engine_build_root: engine_root(),
        image_root: pin(b"image"),
        artifact_root: pin(b"artifact"),
        reason: reason.into(),
        code: "MODEL_ARTIFACT_MISSING".into(),
        retryable: false,
        source_provider: "local".into(),
        lock_present: lock,
        alias_pinned: alias,
        image_opened: image,
        weights_opened: weights,
        downloaded: false,
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
fn a_missing_artifact_records_the_lock_the_alias_the_image_and_the_weights() {
    let dir = scratch("missing");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let lock =
        measure_artifact_missing(&plan, &observe("lock", false, false, false, false)).unwrap();
    assert!(!lock.report.lock_present);
    assert!(!lock.report.downloaded);
    verify_artifact_missing(&lock).unwrap();

    let alias =
        measure_artifact_missing(&plan, &observe("alias", true, false, false, false)).unwrap();
    assert!(alias.report.lock_present);
    assert!(!alias.report.alias_pinned);

    let image =
        measure_artifact_missing(&plan, &observe("image", true, true, false, false)).unwrap();
    assert!(image.report.alias_pinned);
    assert!(!image.report.image_opened);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let weights =
        measure_artifact_missing(&slot, &observe("weights", true, true, true, false)).unwrap();
    assert!(weights.report.image_opened);
    assert!(!weights.report.weights_opened);
    assert_eq!(weights.report.code, "MODEL_ARTIFACT_MISSING");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_artifact_refuses_a_download_and_an_opened_weight_file() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut downloaded = observe("lock", false, false, false, false);
    downloaded.downloaded = true;
    let err = measure_artifact_missing(&plan, &downloaded).unwrap_err();
    assert!(err.message.contains("pull does not download"), "{err}");

    let err =
        measure_artifact_missing(&plan, &observe("weights", true, true, true, true)).unwrap_err();
    assert!(
        err.message
            .contains("a missing weight artifact is not opened"),
        "{err}"
    );

    let err =
        measure_artifact_missing(&plan, &observe("image", true, true, true, false)).unwrap_err();
    assert!(
        err.message.contains("a missing image is not opened"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_artifact_missing_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured =
        measure_artifact_missing(&plan, &observe("weights", true, true, true, false)).unwrap();
    write_artifact_missing_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.artifact-missing-report"
    );
    let again = write_artifact_missing_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-artifact-missing-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_artifact_missing_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
