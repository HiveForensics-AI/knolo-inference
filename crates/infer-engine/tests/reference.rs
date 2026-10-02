use std::fs;
use std::path::PathBuf;

use infer_contracts::{
    decode_contract, sha256_prefixed, ErrorCode, BACKEND_REPORTED, LLAMA_FAMILY, LLAMA_RUNTIME,
    MISTRAL_FAMILY, MISTRAL_RUNTIME, SIDECAR_ARTIFACT_VERIFIED, VLLM_FAMILY, VLLM_RUNTIME,
};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_llama_comparison, measure_mistral_comparison,
    measure_vllm_comparison, reference_engine_build, reference_kernel_bundle,
    verify_llama_comparison, write_llama_report, write_synthetic_model, ReferenceObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-reference"),
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
        "knolo-infer-reference-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    source: &infer_engine::VerifiedWeightSource,
    runtime: &str,
    family: &str,
    class: &str,
) -> ReferenceObservation {
    let artifact = source.artifact_root.clone();
    let tokenizer = pin(b"tokenizer");
    let template = pin(b"template");
    let sampler = pin(b"sampler");
    let hardware = pin(b"hardware");
    let prompts = pin(b"prompts");
    let outputs = pin(b"outputs");
    ReferenceObservation {
        model_image_root: source.image_root.clone(),
        artifact_root: artifact.clone(),
        engine_build_root: engine_root(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        quantization: "f32".into(),
        context_tokens: 16,
        tokenizer_root: tokenizer.clone(),
        template_root: template.clone(),
        sampler_root: sampler.clone(),
        hardware_probe_root: hardware.clone(),
        prompt_distribution_root: prompts.clone(),
        output_distribution_root: outputs.clone(),
        reference_runtime: runtime.into(),
        reference_family: family.into(),
        reference_build_root: pin(b"reference-build"),
        reference_artifact_root: artifact,
        reference_quantization: "f32".into(),
        reference_tokenizer_root: tokenizer,
        reference_template_root: template,
        reference_sampler_root: sampler,
        reference_hardware_root: hardware,
        reference_prompt_distribution_root: prompts,
        reference_output_distribution_root: outputs,
        reference_verification_class: class.into(),
    }
}

#[test]
fn a_cold_llama_comparison_records_the_same_artifact_without_spawning() {
    let dir = scratch("llama");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let observed = observe(
        &source,
        LLAMA_RUNTIME,
        LLAMA_FAMILY,
        SIDECAR_ARTIFACT_VERIFIED,
    );
    let measured = measure_llama_comparison(&plan, &observed).unwrap();
    assert_eq!(measured.report.reference_runtime, "llama.cpp");
    assert_eq!(measured.report.reference_family, "gguf");
    assert_eq!(
        measured.report.reference_verification_class,
        "sidecar-artifact-verified"
    );
    assert_eq!(
        measured.report.reference_artifact_root,
        measured.report.artifact_root
    );
    assert_ne!(
        measured.report.reference_build_root,
        measured.report.engine_build_root
    );
    assert_eq!(measured.report.validation_result, "recorded");
    verify_llama_comparison(&measured).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let on_slot = measure_llama_comparison(&slot, &observed).unwrap();
    assert_eq!(on_slot.report.reference_family, "gguf");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_failed_llama_comparison_issues_no_report() {
    let dir = scratch("llama-reject");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let root = observe(
        &source,
        LLAMA_RUNTIME,
        LLAMA_FAMILY,
        SIDECAR_ARTIFACT_VERIFIED,
    );

    let mut native = root.clone();
    native.reference_verification_class = "native-verified".into();
    let err = measure_llama_comparison(&plan, &native).unwrap_err();
    assert!(
        err.message
            .contains("a compatibility backend is not native-verified"),
        "{err}"
    );

    let mut same_build = root.clone();
    same_build.reference_build_root = same_build.engine_build_root.clone();
    let err = measure_llama_comparison(&plan, &same_build).unwrap_err();
    assert!(
        err.message
            .contains("the reference build is the engine build"),
        "{err}"
    );

    let mut artifact = root.clone();
    artifact.reference_artifact_root = pin(b"other-artifact");
    let err = measure_llama_comparison(&plan, &artifact).unwrap_err();
    assert!(
        err.message.contains("artifact root does not match"),
        "{err}"
    );

    let mut family = root.clone();
    family.reference_family = "safetensors".into();
    let err = measure_llama_comparison(&plan, &family).unwrap_err();
    assert!(
        err.message.contains("llama.cpp comparison is a gguf pin"),
        "{err}"
    );

    let mut mode = root.clone();
    mode.execution_mode = "throughput".into();
    let err = measure_llama_comparison(&plan, &mode).unwrap_err();
    assert_eq!(err.code, ErrorCode::BackendNotAllowed, "{err}");

    let mut graphs = plan.clone();
    graphs.graph_capture_mode = "decode-buckets".into();
    let err = measure_llama_comparison(&graphs, &root).unwrap_err();
    assert!(err.message.contains("graph capture is off"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn vllm_and_mistral_record_their_own_pins() {
    let dir = scratch("other-runtimes");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let vllm = observe(
        &source,
        VLLM_RUNTIME,
        VLLM_FAMILY,
        SIDECAR_ARTIFACT_VERIFIED,
    );
    let measured = measure_vllm_comparison(&plan, &vllm).unwrap();
    assert_eq!(measured.report.reference_runtime, "vllm");
    assert_eq!(measured.report.reference_family, "safetensors");
    assert_eq!(measured.report.quantization, "f32");

    let mut gguf = vllm.clone();
    gguf.reference_family = "gguf".into();
    let err = measure_vllm_comparison(&plan, &gguf).unwrap_err();
    assert!(
        err.message.contains("vllm comparison is a safetensors pin"),
        "{err}"
    );

    let mistral = observe(&source, MISTRAL_RUNTIME, MISTRAL_FAMILY, BACKEND_REPORTED);
    let measured = measure_mistral_comparison(&plan, &mistral).unwrap();
    assert_eq!(measured.report.reference_runtime, "mistral.rs");
    assert_eq!(measured.report.reference_family, "rust");
    assert_eq!(
        measured.report.reference_verification_class,
        "backend-reported"
    );

    let mut sidecar = mistral.clone();
    sidecar.reference_verification_class = SIDECAR_ARTIFACT_VERIFIED.into();
    let err = measure_mistral_comparison(&plan, &sidecar).unwrap_err();
    assert!(
        err.message
            .contains("field referenceVerificationClass has an unsupported value"),
        "{err}"
    );

    let slot = infer_engine::cuda_placement(&source).unwrap();
    assert!(measure_vllm_comparison(&slot, &vllm).is_ok());
    assert!(measure_mistral_comparison(&slot, &mistral).is_ok());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_llama_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let observed = observe(
        &source,
        LLAMA_RUNTIME,
        LLAMA_FAMILY,
        SIDECAR_ARTIFACT_VERIFIED,
    );
    let measured = measure_llama_comparison(&plan, &observed).unwrap();
    write_llama_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.llama-report"
    );
    let again = write_llama_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-llama-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_llama_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
