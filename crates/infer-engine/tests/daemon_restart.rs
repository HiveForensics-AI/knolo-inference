use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_daemon_restart, reference_engine_build,
    reference_kernel_bundle, verify_daemon_restart, write_restart_report, write_synthetic_model,
    RestartObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-restart"),
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
        std::env::temp_dir().join(format!("knolo-infer-restart-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(reason: &str) -> RestartObservation {
    let stale = reason != "clean-exit";
    RestartObservation {
        engine_build_root: engine_root(),
        previous_owner_root: pin(b"previous-owner"),
        incoming_owner_root: pin(b"incoming-owner"),
        reason: reason.into(),
        lock_replaced: stale,
        journals_sealed: stale,
        listener_up: true,
        worker_restart_count: 0,
        process_spawned: false,
        second_worker: false,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_restart_report_replaces_a_stale_owner_and_records_a_clean_exit() {
    let dir = scratch("restart");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let stale = measure_daemon_restart(&plan, &observe("stale-lock")).unwrap();
    assert!(stale.report.lock_replaced);
    assert!(stale.report.journals_sealed);
    assert_eq!(stale.report.worker_restart_count, 0);
    assert!(!stale.report.process_spawned);
    verify_daemon_restart(&stale).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let clean = measure_daemon_restart(&slot, &observe("clean-exit")).unwrap();
    assert!(!clean.report.lock_replaced);
    assert!(!clean.report.journals_sealed);
    assert!(clean.report.listener_up);

    let killed = measure_daemon_restart(&plan, &observe("killed")).unwrap();
    assert!(killed.report.lock_replaced);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_restart_report_refuses_a_spawn_and_a_repeated_owner() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut spawned = observe("stale-lock");
    spawned.process_spawned = true;
    let err = measure_daemon_restart(&plan, &spawned).unwrap_err();
    assert!(
        err.message
            .contains("a restart record does not spawn a process"),
        "{err}"
    );

    let mut clean = observe("clean-exit");
    clean.lock_replaced = true;
    let err = measure_daemon_restart(&plan, &clean).unwrap_err();
    assert!(
        err.message
            .contains("a clean exit does not replace a held lock"),
        "{err}"
    );

    let mut same = observe("killed");
    same.incoming_owner_root = same.previous_owner_root.clone();
    let err = measure_daemon_restart(&plan, &same).unwrap_err();
    assert!(
        err.message
            .contains("the incoming owner repeats the previous owner"),
        "{err}"
    );

    let mut counted = observe("stale-lock");
    counted.worker_restart_count = 1;
    let err = measure_daemon_restart(&plan, &counted).unwrap_err();
    assert!(
        err.message
            .contains("a new process starts the restart counter at zero"),
        "{err}"
    );

    let measured = measure_daemon_restart(&plan, &observe("stale-lock")).unwrap();
    let mut stored = measured.report.clone();
    stored.journals_sealed = false;
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message.contains("a stale owner seals open journals"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_restart_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_daemon_restart(&plan, &observe("clean-exit")).unwrap();
    write_restart_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.restart-report"
    );
    let again = write_restart_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-restart-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_restart_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
