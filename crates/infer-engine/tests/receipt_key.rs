use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_receipt_key, reference_engine_build,
    reference_kernel_bundle, verify_receipt_key, write_receipt_key_report, write_synthetic_model,
    ReceiptKeyObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-receipt-key"),
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
        "knolo-infer-receipt-key-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    status: &str,
    key_id: &str,
    bytes: u32,
    rotation: &str,
    previous: &str,
    trusted: bool,
) -> ReceiptKeyObservation {
    ReceiptKeyObservation {
        engine_build_root: engine_root(),
        receipt_root: pin(b"receipt"),
        custody: "host-store".into(),
        key_material_serialized: false,
        key_id: key_id.into(),
        signature_status: status.into(),
        signature_bytes: bytes,
        rotation: rotation.into(),
        previous_key_id: previous.into(),
        trusted_metadata: trusted,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_receipt_key_records_current_rotated_and_unsigned_custody() {
    let dir = scratch("receipt-key");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let current = measure_receipt_key(
        &plan,
        &observe("shape-checked", "local-dev", 64, "current", "", false),
    )
    .unwrap();
    assert_eq!(current.report.custody, "host-store");
    assert!(!current.report.key_material_serialized);
    verify_receipt_key(&current).unwrap();

    let rotated = measure_receipt_key(
        &plan,
        &observe(
            "shape-checked",
            "local-dev",
            64,
            "rotated",
            "local-prev",
            true,
        ),
    )
    .unwrap();
    assert_eq!(rotated.report.previous_key_id, "local-prev");
    assert!(rotated.report.trusted_metadata);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let unsigned = measure_receipt_key(
        &slot,
        &observe("unsigned-local", "", 0, "current", "", false),
    )
    .unwrap();
    assert!(unsigned.report.key_id.is_empty());
    assert_eq!(unsigned.report.signature_bytes, 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_receipt_key_refuses_serialized_material_and_a_bad_rotation() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut material = observe("shape-checked", "local-dev", 64, "current", "", false);
    material.key_material_serialized = true;
    let err = measure_receipt_key(&plan, &material).unwrap_err();
    assert!(
        err.message
            .contains("key material stays out of the receipt"),
        "{err}"
    );

    let mut custody = observe("shape-checked", "local-dev", 64, "current", "", false);
    custody.custody = "process-memory".into();
    let err = measure_receipt_key(&plan, &custody).unwrap_err();
    assert!(
        err.message.contains("receipt keys stay in host storage"),
        "{err}"
    );

    let err = measure_receipt_key(
        &plan,
        &observe("unsigned-local", "", 0, "rotated", "local-prev", true),
    )
    .unwrap_err();
    assert!(
        err.message.contains("an unsigned receipt rotates a key"),
        "{err}"
    );

    let err = measure_receipt_key(
        &plan,
        &observe(
            "shape-checked",
            "local-dev",
            64,
            "rotated",
            "local-dev",
            true,
        ),
    )
    .unwrap_err();
    assert!(
        err.message.contains("rotation repeats the current key"),
        "{err}"
    );

    let err = measure_receipt_key(
        &plan,
        &observe(
            "shape-checked",
            "local-dev",
            64,
            "rotated",
            "local-prev",
            false,
        ),
    )
    .unwrap_err();
    assert!(
        err.message.contains("rotation uses trusted metadata"),
        "{err}"
    );

    let mut repeated = observe("shape-checked", "local-dev", 64, "current", "", false);
    repeated.receipt_root = repeated.engine_build_root.clone();
    let err = measure_receipt_key(&plan, &repeated).unwrap_err();
    assert!(
        err.message.contains("the receipt repeats the engine build"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_receipt_key_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_receipt_key(
        &plan,
        &observe("shape-checked", "local-dev", 64, "current", "", false),
    )
    .unwrap();
    write_receipt_key_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.receipt-key-report"
    );
    let secret = b"ed25519-private-key-material";
    assert!(!stored.windows(secret.len()).any(|window| window == secret));
    let again = write_receipt_key_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-receipt-key-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_receipt_key_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
