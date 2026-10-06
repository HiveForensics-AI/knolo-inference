use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_disconnect, reference_engine_build,
    reference_kernel_bundle, verify_disconnect, write_disconnect_report, write_synthetic_model,
    DisconnectObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-disconnect"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-disconnect-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(stage: &str, prompt: u32, output: u32) -> DisconnectObservation {
    DisconnectObservation {
        engine_build_root: engine_root(),
        request_id: "req-1".into(),
        stage: stage.into(),
        outcome: "cancelled".into(),
        code: "REQUEST_CANCELLED".into(),
        listener_up: true,
        socket_closed: false,
        prompt_tokens: prompt,
        output_tokens: output,
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_disconnect_report_records_a_queued_cancel_and_a_decode_cancel() {
    let dir = scratch("disconnect");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let queued = measure_disconnect(&plan, &observe("queue", 4, 0)).unwrap();
    assert!(queued.report.listener_up);
    assert!(!queued.report.socket_closed);
    assert_eq!(queued.report.output_tokens, 0);
    assert_eq!(queued.report.validation_result, "recorded");
    verify_disconnect(&queued).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let decode = measure_disconnect(&slot, &observe("decode", 4, 2)).unwrap();
    assert_eq!(decode.report.output_tokens, 2);
    assert_eq!(decode.report.code, "REQUEST_CANCELLED");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_disconnect_report_refuses_a_closed_socket_and_an_oversized_prompt() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut closed = observe("stream", 4, 1);
    closed.socket_closed = true;
    let err = measure_disconnect(&plan, &closed).unwrap_err();
    assert!(
        err.message
            .contains("a disconnect record does not close the socket"),
        "{err}"
    );

    let mut down = observe("prefill", 4, 0);
    down.listener_up = false;
    let err = measure_disconnect(&plan, &down).unwrap_err();
    assert!(err.message.contains("the listener stays up"), "{err}");

    let err = measure_disconnect(&plan, &observe("queue", 4, 1)).unwrap_err();
    assert!(
        err.message.contains("a queued disconnect has no output"),
        "{err}"
    );

    let err = measure_disconnect(&plan, &observe("decode", 16, 1)).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContextLimitExceeded, "{err}");
    assert!(
        err.message.contains("the disconnect is the micro fixture"),
        "{err}"
    );

    let err = measure_disconnect(&plan, &observe("admission", 4, 0)).unwrap_err();
    assert!(
        err.message.contains("field stage has an unsupported value"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_disconnect_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_disconnect(&plan, &observe("queue", 4, 0)).unwrap();
    write_disconnect_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.disconnect-report"
    );
    let again = write_disconnect_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-disconnect-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_disconnect_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
