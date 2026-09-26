use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, MAX_TOOL_NAME_BYTES};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_tool_refusal, reference_engine_build,
    reference_kernel_bundle, verify_tool_refusal, write_synthetic_model, write_tool_refusal_report,
    ToolRefusalObservation,
};

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-tool-refusal"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-tool-refusal-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    reason: &str,
    name_bytes: u32,
    accepted: bool,
    generated: bool,
) -> ToolRefusalObservation {
    ToolRefusalObservation {
        engine_build_root: engine_root(),
        object_root: pin(b"tool-object"),
        reason: reason.into(),
        name_bytes,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        name_accepted: accepted,
        object_generated: generated,
        tool_executed: false,
        authority_checked: false,
        budget_checked: false,
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
fn a_tool_refusal_records_a_name_an_object_and_an_execution() {
    let dir = scratch("tool");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let name = measure_tool_refusal(&plan, &observe("name", 0, false, false)).unwrap();
    assert!(!name.report.name_accepted);
    assert!(!name.report.tool_executed);
    verify_tool_refusal(&name).unwrap();

    let object = measure_tool_refusal(&plan, &observe("object", 8, true, false)).unwrap();
    assert!(object.report.name_accepted);
    assert!(!object.report.object_generated);

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let execute = measure_tool_refusal(&slot, &observe("execute", 12, true, true)).unwrap();
    assert!(execute.report.object_generated);
    assert!(!execute.report.authority_checked);
    assert!(!execute.report.budget_checked);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_tool_refusal_rejects_execution_and_a_name_past_the_cap() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut executed = observe("execute", 8, true, true);
    executed.tool_executed = true;
    let err = measure_tool_refusal(&plan, &executed).unwrap_err();
    assert!(
        err.message
            .contains("a tool refusal does not execute the tool"),
        "{err}"
    );

    let err = measure_tool_refusal(
        &plan,
        &observe("execute", MAX_TOOL_NAME_BYTES + 1, true, true),
    )
    .unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid);
    assert!(
        err.message.contains("tool name exceeds the record cap"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_tool_refusal_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_tool_refusal(&plan, &observe("object", 4, true, false)).unwrap();
    write_tool_refusal_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.tool-refusal-report"
    );
    let again = write_tool_refusal_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-tool-refusal-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_tool_refusal_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
