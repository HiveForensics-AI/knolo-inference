use std::fs;
use std::path::PathBuf;

use infer_contracts::{
    decode_contract, sha256_prefixed, ErrorCode, ED25519_PUBLIC_KEY_BYTES, ED25519_SCALAR_BYTES,
    ED25519_SIGNATURE_BYTES,
};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_challenge, reference_engine_build,
    reference_kernel_bundle, verify_challenge, write_challenge_report, write_synthetic_model,
    ChallengeObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-challenge"),
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
        "knolo-infer-challenge-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(status: &str, hashed: bool, bytes: u32) -> ChallengeObservation {
    let (public_key, signature, scalar) = if bytes == 0 {
        (0, 0, 0)
    } else {
        (
            ED25519_PUBLIC_KEY_BYTES,
            ED25519_SIGNATURE_BYTES,
            ED25519_SCALAR_BYTES,
        )
    };
    ChallengeObservation {
        engine_build_root: engine_root(),
        release_root: pin(b"release"),
        message_root: pin(b"message"),
        challenge_status: status.into(),
        challenge_hashed: hashed,
        scalar_reduced: false,
        public_key_bytes: public_key,
        signature_bytes: signature,
        scalar_bytes: scalar,
        key_material_present: false,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_challenge_report_records_a_hash_a_rejection_and_an_unsigned_release() {
    let dir = scratch("challenge");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let hashed = measure_challenge(&plan, &observe("hashed", true, 32)).unwrap();
    assert!(hashed.report.challenge_hashed);
    assert!(!hashed.report.scalar_reduced);
    assert_eq!(hashed.report.validation_result, "verified");
    verify_challenge(&hashed).unwrap();

    let rejected = measure_challenge(&plan, &observe("rejected", true, 32)).unwrap();
    assert!(rejected.report.challenge_hashed);
    assert_eq!(rejected.report.validation_result, "recorded");

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let unsigned = measure_challenge(&slot, &observe("unsigned-local", false, 0)).unwrap();
    assert!(!unsigned.report.challenge_hashed);
    assert_eq!(unsigned.report.public_key_bytes, 0);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_challenge_record_refuses_a_reduced_scalar_and_an_unhashed_challenge() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut reduced = observe("hashed", true, 32);
    reduced.scalar_reduced = true;
    let err = measure_challenge(&plan, &reduced).unwrap_err();
    assert!(
        err.message
            .contains("the scalar reduction stays uncomputed"),
        "{err}"
    );

    let err = measure_challenge(&plan, &observe("hashed", false, 32)).unwrap_err();
    assert!(
        err.message.contains("a hashed challenge records the hash"),
        "{err}"
    );

    let err = measure_challenge(&plan, &observe("unsigned-local", true, 0)).unwrap_err();
    assert!(
        err.message
            .contains("an unsigned release hashes the challenge"),
        "{err}"
    );

    let mut material = observe("rejected", true, 32);
    material.key_material_present = true;
    let err = measure_challenge(&plan, &material).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message.contains("key material stays in host storage"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_challenge_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_challenge(&plan, &observe("hashed", true, 32)).unwrap();
    write_challenge_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.challenge-report"
    );
    let again = write_challenge_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-challenge-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_challenge_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
