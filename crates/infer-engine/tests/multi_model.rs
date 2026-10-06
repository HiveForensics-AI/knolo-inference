use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, MAX_DEVICE_COUNT};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_multi_model, reference_engine_build,
    reference_kernel_bundle, verify_multi_model, write_multi_model_report, write_synthetic_model,
    MultiModelObservation,
};

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-multi-model"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-multi-model-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(reason: &str, device_count: u32) -> MultiModelObservation {
    MultiModelObservation {
        engine_build_root: engine_root(),
        resident_root: pin(b"resident-model"),
        incoming_root: pin(b"incoming-model"),
        reason: reason.into(),
        device_count,
        models_loaded: 1,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        second_loaded: false,
        resident_replaced: false,
        tensor_parallel: false,
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
fn a_multi_model_record_keeps_one_resident_model() {
    let dir = scratch("models");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let second = measure_multi_model(&plan, &observe("second", 1)).unwrap();
    assert_eq!(second.report.models_loaded, 1);
    assert!(!second.report.second_loaded);
    verify_multi_model(&second).unwrap();

    let replace = measure_multi_model(&plan, &observe("replace", 1)).unwrap();
    assert!(!replace.report.resident_replaced);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let parallel = measure_multi_model(&slot, &observe("parallel", 2)).unwrap();
    assert_eq!(parallel.report.device_count, 2);
    assert!(!parallel.report.tensor_parallel);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_multi_model_record_rejects_a_second_load_and_a_third_device() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut loaded = observe("second", 1);
    loaded.second_loaded = true;
    let err = measure_multi_model(&plan, &loaded).unwrap_err();
    assert!(
        err.message
            .contains("a multi-model record does not load a second model"),
        "{err}"
    );

    let err = measure_multi_model(&plan, &observe("parallel", MAX_DEVICE_COUNT + 1)).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid);
    assert!(
        err.message.contains("device count exceeds the record cap"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_multi_model_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_multi_model(&plan, &observe("replace", 1)).unwrap();
    write_multi_model_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.multi-model-report"
    );
    let again = write_multi_model_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-multi-model-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_multi_model_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
