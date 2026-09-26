use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_agent_effect, reference_engine_build,
    reference_kernel_bundle, verify_agent_effect, write_agent_effect_report, write_synthetic_model,
    AgentEffectObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-agent-effect"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-agent-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(source: &infer_engine::VerifiedWeightSource) -> AgentEffectObservation {
    let runtime = pin(b"runtime");
    let artifact = source.artifact_root.clone();
    let knowledge = pin(b"knowledge");
    AgentEffectObservation {
        model_image_root: source.image_root.clone(),
        artifact_root: artifact.clone(),
        engine_build_root: engine_root(),
        model_runtime_root: runtime.clone(),
        required_model_runtime_root: runtime,
        required_artifact_root: artifact,
        knowledge_image_root: knowledge.clone(),
        required_knowledge_image_root: knowledge,
        backend: "reference-f32".into(),
        assurance: "same_build_replayable".into(),
        required_assurance: "same_build_replayable".into(),
        execution_mode: "pinned".into(),
        allowed_execution_modes: vec!["isolated-replay".into(), "pinned".into()],
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        prompt_tokens: 2,
        output_tokens: 1,
        max_prompt_tokens: 8,
        max_output_tokens: 4,
        receipt_present: true,
    }
}

#[test]
fn a_final_native_receipt_is_allowed() {
    let dir = scratch("allow");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_agent_effect(&plan, &observe(&source)).unwrap();
    assert_eq!(measured.report.decision, "allow");
    assert_eq!(measured.report.reason, "accepted");
    assert_eq!(measured.report.validation_result, "recorded");
    verify_agent_effect(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let mut cuda = observe(&source);
    cuda.backend = "candle-cuda".into();
    let on_slot = measure_agent_effect(&slot, &cuda).unwrap();
    assert_eq!(on_slot.report.decision, "allow");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn policy_denials_are_recorded_and_a_bad_input_is_not() {
    let dir = scratch("deny");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut streamed = observe(&source);
    streamed.receipt_present = false;
    let measured = measure_agent_effect(&plan, &streamed).unwrap();
    assert_eq!(measured.report.decision, "deny");
    assert_eq!(measured.report.reason, "stream-not-authorization");

    let mut backend = observe(&source);
    backend.backend = "llama.cpp".into();
    let measured = measure_agent_effect(&plan, &backend).unwrap();
    assert_eq!(measured.report.reason, "unverified-backend");

    let mut runtime = observe(&source);
    runtime.model_runtime_root = pin(b"other-runtime");
    let measured = measure_agent_effect(&plan, &runtime).unwrap();
    assert_eq!(measured.report.reason, "model-runtime-rejected");

    let mut mode = observe(&source);
    mode.allowed_execution_modes = vec!["isolated-replay".into()];
    let measured = measure_agent_effect(&plan, &mode).unwrap();
    assert_eq!(measured.report.reason, "execution-mode-rejected");

    let mut budget = observe(&source);
    budget.prompt_tokens = 9;
    let measured = measure_agent_effect(&plan, &budget).unwrap();
    assert_eq!(measured.report.reason, "prompt-budget-exceeded");

    let mut assurance = observe(&source);
    assurance.assurance = "compatibility".into();
    let measured = measure_agent_effect(&plan, &assurance).unwrap();
    assert_eq!(measured.report.reason, "assurance-rejected");

    let mut wide = observe(&source);
    wide.required_assurance = "compatibility".into();
    wide.assurance = "same_build_replayable".into();
    let measured = measure_agent_effect(&plan, &wide).unwrap();
    assert_eq!(measured.report.decision, "allow");

    let mut over = observe(&source);
    over.prompt_tokens = 16;
    over.output_tokens = 1;
    over.max_prompt_tokens = 16;
    over.max_output_tokens = 16;
    let err = measure_agent_effect(&plan, &over).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContextLimitExceeded, "{err}");
    assert!(
        err.message
            .contains("the agent effect is the micro fixture"),
        "{err}"
    );

    let mut throughput = observe(&source);
    throughput.execution_mode = "throughput".into();
    let err = measure_agent_effect(&plan, &throughput).unwrap_err();
    assert_eq!(err.code, ErrorCode::BackendNotAllowed, "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_agent_effect_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_agent_effect(&plan, &observe(&source)).unwrap();
    write_agent_effect_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.agent-effect-report"
    );
    let again = write_agent_effect_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-agent-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_agent_effect_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
