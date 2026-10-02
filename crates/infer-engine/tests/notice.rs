use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_supply_notice, reference_engine_build,
    reference_kernel_bundle, verify_supply_notice, write_notice_report, write_synthetic_model,
    NoticeObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-notice"),
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
        std::env::temp_dir().join(format!("knolo-infer-notice-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(feature: &str, candle: bool) -> NoticeObservation {
    NoticeObservation {
        engine_build_root: engine_root(),
        notice_root: pin(b"notice"),
        sbom_root: pin(b"sbom"),
        component_count: 1,
        feature_set: feature.into(),
        candle_named: candle,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_notice_records_the_inventory_for_cpu_and_cuda() {
    let dir = scratch("notice");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_supply_notice(&plan, &observe("cpu", false)).unwrap();
    assert_eq!(measured.report.feature_set, "cpu");
    assert!(!measured.report.candle_named);
    assert_eq!(measured.report.validation_result, "recorded");
    assert_eq!(measured.report.component_count, 1);
    verify_supply_notice(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let cuda = measure_supply_notice(&slot, &observe("cuda", true)).unwrap();
    assert!(cuda.report.candle_named);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_notice_refuses_the_wrong_candle_mark() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let err = measure_supply_notice(&plan, &observe("cpu", true)).unwrap_err();
    assert!(err.message.contains("the cpu notice names candle"), "{err}");

    let err = measure_supply_notice(&plan, &observe("cuda", false)).unwrap_err();
    assert!(
        err.message.contains("the cuda notice omits candle"),
        "{err}"
    );

    let mut repeated = observe("cpu", false);
    repeated.sbom_root = repeated.notice_root.clone();
    let err = measure_supply_notice(&plan, &repeated).unwrap_err();
    assert!(err.message.contains("the sbom repeats the notice"), "{err}");

    let mut empty = observe("cpu", false);
    empty.component_count = 0;
    let err = measure_supply_notice(&plan, &empty).unwrap_err();
    assert!(
        err.message.contains("the notice names no component"),
        "{err}"
    );

    let mut wide = observe("cpu", false);
    wide.component_count = 257;
    let err = measure_supply_notice(&plan, &wide).unwrap_err();
    assert!(
        err.message.contains("the notice inventory is too large"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_notice_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_supply_notice(&plan, &observe("cpu", false)).unwrap();
    write_notice_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.notice-report"
    );
    let notice = b"Candle is not named on cpu.";
    assert!(!stored.windows(notice.len()).any(|window| window == notice));
    let again = write_notice_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-notice-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_notice_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
