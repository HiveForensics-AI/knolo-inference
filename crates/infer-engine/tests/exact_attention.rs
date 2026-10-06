use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_exact_attention, reference_engine_build,
    reference_kernel_bundle, verify_exact_attention, write_exact_attention_report,
    write_synthetic_model, ExactAttentionObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-exact-attention"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-exact-attention-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe() -> ExactAttentionObservation {
    ExactAttentionObservation {
        engine_build_root: engine_root(),
        reason: "flash".into(),
        requested_tiles: 4,
        selected_tiles: 0,
        code: "UNSUPPORTED_KERNEL".into(),
        retryable: false,
        flash_selected: false,
        tiled: false,
        split: false,
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
fn a_exact_attention_record_is_stored() {
    let dir = scratch("record");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_exact_attention(&plan, &observe()).unwrap();
    assert_eq!(measured.report.requested_tiles, 4);
    assert!(!measured.report.flash_selected);
    verify_exact_attention(&measured).unwrap();

    let mut varied = observe();
    varied.reason = "tile".into();
    let varied = measure_exact_attention(&plan, &varied).unwrap();
    assert!(!varied.report.tiled);
    let mut varied = observe();
    varied.reason = "split".into();
    let varied = measure_exact_attention(&plan, &varied).unwrap();
    assert_eq!(varied.report.selected_tiles, 0);

    let mut slot_obs = observe();
    slot_obs.requested_tiles = 16;
    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_exact_attention(&slot, &slot_obs).unwrap();
    assert_eq!(on_slot.report.requested_tiles, 16);
    assert!(!on_slot.report.split);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_exact_attention_record_rejects_a_bad_observation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let mut bad = observe();
    bad.flash_selected = true;
    let err = measure_exact_attention(&plan, &bad).unwrap_err();
    assert!(
        err.message.contains("a flash kernel is not selected"),
        "{err}"
    );

    let mut bad = observe();
    bad.requested_tiles = 17;
    let err = measure_exact_attention(&plan, &bad).unwrap_err();
    assert!(
        err.message
            .contains("exact attention tiles exceed the record cap"),
        "{err}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_exact_attention_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_exact_attention(&plan, &observe()).unwrap();
    write_exact_attention_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.exact-attention-report"
    );
    let again = write_exact_attention_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-exact-attention-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_exact_attention_report(&dir, &measured, &format!("escape/{outside_name}"))
        .unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
