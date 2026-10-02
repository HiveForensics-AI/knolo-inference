use std::fs;
use std::path::PathBuf;

use infer_contracts::{
    decode_contract, output_token_root, prompt_token_root, sha256_prefixed, ErrorCode,
};
use infer_engine::{
    cpu_placement, cuda_placement, greedy_generate, load_verified_micro, measure_micro_throughput,
    micro_kv_layout, reference_engine_build, reference_kernel_bundle, verify_micro_throughput,
    write_synthetic_model, write_throughput_report, ArchitectureAdapter, ExecutableModel,
    MicroAdapter, ReferenceF32Backend, SingleBlockKv, ThroughputObservation, FIXTURE_CASES,
    MAX_CONTEXT, VOCAB,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(
        sha256_prefixed(b"knolo-infer-throughput"),
        bundle.root().unwrap(),
    )
    .unwrap();
    build.root().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-throughput-{}-{name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(
    source: &infer_engine::VerifiedWeightSource,
    prompt: &[u32],
    output: &[u32],
    prefill_nanos: u64,
    decode_nanos: u64,
) -> ThroughputObservation {
    ThroughputObservation {
        model_image_root: source.image_root.clone(),
        artifact_root: source.artifact_root.clone(),
        engine_build_root: engine_root(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        prefill_tokens: u32::try_from(prompt.len()).unwrap(),
        decode_tokens: u32::try_from(output.len()).unwrap(),
        prompt_tokens: prompt.to_vec(),
        output_tokens: output.to_vec(),
        prefill_nanos,
        decode_nanos,
    }
}

fn oracle(source: &infer_engine::VerifiedWeightSource) -> Box<dyn ExecutableModel> {
    let placement = cpu_placement(source).unwrap();
    MicroAdapter
        .build(source, &placement, &ReferenceF32Backend)
        .unwrap()
}

#[test]
fn the_micro_fixture_records_prefill_decode_and_request_throughput() {
    let dir = scratch("fixture");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    assert_eq!(plan.graph_capture_mode, "off");
    assert_eq!(plan.context_reservation_tokens, MAX_CONTEXT);
    let mut model = oracle(&source);

    for fixture in FIXTURE_CASES {
        let mut kv = SingleBlockKv::new(micro_kv_layout()).unwrap();
        let output = greedy_generate(
            model.as_mut(),
            &mut kv,
            1,
            fixture.prompt,
            fixture.new_tokens,
        )
        .unwrap();
        let mut again = SingleBlockKv::new(micro_kv_layout()).unwrap();
        let second = greedy_generate(
            model.as_mut(),
            &mut again,
            1,
            fixture.prompt,
            fixture.new_tokens,
        )
        .unwrap();
        assert_eq!(second.tokens, output.tokens);
        assert_eq!(output.tokens.len(), fixture.new_tokens as usize);

        let observed = observe(
            &source,
            fixture.prompt,
            &output.tokens,
            1_000_000_000,
            1_000_000_000,
        );
        let measured = measure_micro_throughput(&plan, &observed).unwrap();
        assert_eq!(measured.report.validation_result, "recorded");
        assert_eq!(measured.report.execution_mode, "isolated-replay");
        assert_eq!(measured.report.cache_policy, "off");
        assert_eq!(measured.report.warm_state, "cold");
        assert_eq!(measured.report.concurrency, 1);
        assert_eq!(measured.report.run_count, 1);
        assert_eq!(measured.report.request_count, 1);
        assert_eq!(
            measured.report.prefill_tokens,
            u32::try_from(fixture.prompt.len()).unwrap()
        );
        assert_eq!(measured.report.decode_tokens, fixture.new_tokens);
        assert_eq!(
            measured.report.prompt_token_root,
            prompt_token_root(fixture.prompt).unwrap()
        );
        assert_eq!(
            measured.report.output_token_root,
            output_token_root(&output.tokens).unwrap()
        );
        assert_eq!(measured.report.model_image_root, source.image_root);
        assert_eq!(measured.report.artifact_root, source.artifact_root);
        assert_eq!(measured.report.placement_root, plan.root().unwrap());
        let prefill = u64::from(measured.report.prefill_tokens) * 1_000_000;
        let decode = u64::from(measured.report.decode_tokens) * 1_000_000;
        assert_eq!(measured.report.prefill_tokens_per_second_micros, prefill);
        assert_eq!(measured.report.decode_tokens_per_second_micros, decode);
        assert_eq!(measured.report.requests_per_second_micros, 500_000);
        verify_micro_throughput(&measured).unwrap();

        let mut pinned = observed.clone();
        pinned.execution_mode = "pinned".into();
        let pinned = measure_micro_throughput(&plan, &pinned).unwrap();
        assert_eq!(
            pinned.report.output_token_root,
            measured.report.output_token_root
        );

        let cuda = cuda_placement(&source).unwrap();
        let on_slot = measure_micro_throughput(&cuda, &observed).unwrap();
        assert_eq!(on_slot.report.placement_root, cuda.root().unwrap());
        assert_eq!(on_slot.report.decode_tokens, measured.report.decode_tokens);
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_failed_measurement_issues_no_report() {
    let dir = scratch("reject");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let prompt = FIXTURE_CASES[0].prompt;
    let output = [13u32, 6, 5, 7];
    let root = observe(&source, prompt, &output, 1_000_000_000, 1_000_000_000);

    let mut mode = root.clone();
    mode.execution_mode = "throughput".into();
    let err = measure_micro_throughput(&plan, &mode).unwrap_err();
    assert_eq!(err.code, ErrorCode::BackendNotAllowed, "{err}");
    assert!(
        err.message
            .contains("throughput execution mode is not enabled"),
        "{err}"
    );

    let mut other = root.clone();
    other.execution_mode = "batch".into();
    let err = measure_micro_throughput(&plan, &other).unwrap_err();
    assert!(err.message.contains("executionMode"), "{err}");

    let mut cache = root.clone();
    cache.cache_policy = "prefix".into();
    let err = measure_micro_throughput(&plan, &cache).unwrap_err();
    assert!(err.message.contains("prefix cache is off"), "{err}");

    let mut warm = root.clone();
    warm.warm_state = "warm".into();
    let err = measure_micro_throughput(&plan, &warm).unwrap_err();
    assert!(err.message.contains("warm state is cold"), "{err}");

    let mut wide = root.clone();
    wide.concurrency = 2;
    let err = measure_micro_throughput(&plan, &wide).unwrap_err();
    assert!(err.message.contains("concurrency is one"), "{err}");

    let mut runs = root.clone();
    runs.run_count = 2;
    let err = measure_micro_throughput(&plan, &runs).unwrap_err();
    assert!(err.message.contains("run count is one"), "{err}");

    let mut requests = root.clone();
    requests.request_count = 2;
    let err = measure_micro_throughput(&plan, &requests).unwrap_err();
    assert!(err.message.contains("request count is one"), "{err}");

    let mut zero_prefill = root.clone();
    zero_prefill.prefill_tokens = 0;
    zero_prefill.prompt_tokens.clear();
    let err = measure_micro_throughput(&plan, &zero_prefill).unwrap_err();
    assert!(err.message.contains("prefill token count is zero"), "{err}");

    let mut mismatch = root.clone();
    mismatch.prefill_tokens = 2;
    let err = measure_micro_throughput(&plan, &mismatch).unwrap_err();
    assert!(
        err.message.contains("prefill token count does not match"),
        "{err}"
    );

    let mut over = root.clone();
    over.prompt_tokens = vec![1; MAX_CONTEXT as usize];
    over.prefill_tokens = MAX_CONTEXT;
    over.output_tokens = vec![1];
    over.decode_tokens = 1;
    let err = measure_micro_throughput(&plan, &over).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContextLimitExceeded, "{err}");
    assert!(err.message.contains("does not fit"), "{err}");

    let mut vocab = root.clone();
    vocab.prompt_tokens = vec![VOCAB as u32];
    vocab.prefill_tokens = 1;
    vocab.output_tokens = vec![1];
    vocab.decode_tokens = 1;
    let err = measure_micro_throughput(&plan, &vocab).unwrap_err();
    assert!(
        err.message.contains("outside the micro vocabulary"),
        "{err}"
    );

    let mut zero_time = root.clone();
    zero_time.prefill_nanos = 0;
    let err = measure_micro_throughput(&plan, &zero_time).unwrap_err();
    assert!(err.message.contains("prefill duration is zero"), "{err}");

    let mut overflow = root.clone();
    overflow.prefill_nanos = u64::MAX;
    overflow.decode_nanos = 1;
    let err = measure_micro_throughput(&plan, &overflow).unwrap_err();
    assert!(
        err.message.contains("throughput duration overflows"),
        "{err}"
    );

    let mut graphs = plan.clone();
    graphs.graph_capture_mode = "decode-buckets".into();
    let err = measure_micro_throughput(&graphs, &root).unwrap_err();
    assert!(err.message.contains("graph capture is off"), "{err}");

    let mut rejected = plan.clone();
    rejected.rejection_reason = Some("INSUFFICIENT_MEMORY".into());
    let err = measure_micro_throughput(&rejected, &root).unwrap_err();
    assert_eq!(err.code, ErrorCode::PlacementUnsatisfiable, "{err}");
    assert!(err.message.contains("already rejected"), "{err}");

    let mut wide_context = plan.clone();
    wide_context.context_reservation_tokens = 32;
    let err = measure_micro_throughput(&wide_context, &root).unwrap_err();
    assert!(
        err.message
            .contains("throughput report is the micro fixture"),
        "{err}"
    );

    let floor = observe(&source, &[1], &[11, 11], 3, 2);
    let floor = measure_micro_throughput(&plan, &floor).unwrap();
    assert_eq!(
        floor.report.prefill_tokens_per_second_micros,
        333_333_333_333_333
    );
    assert_eq!(
        floor.report.decode_tokens_per_second_micros,
        1_000_000_000_000_000
    );
    assert_eq!(floor.report.requests_per_second_micros, 200_000_000_000_000);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let observed = observe(
        &source,
        &[1, 4, 7],
        &[13, 6, 5, 7],
        1_000_000_000,
        1_000_000_000,
    );
    let measured = measure_micro_throughput(&plan, &observed).unwrap();
    write_throughput_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.throughput-report"
    );
    let again = write_throughput_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");
    assert_eq!(fs::read(dir.join("report.cbor")).unwrap(), stored);

    let outside_name = format!("knolo-throughput-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_throughput_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());

    std::os::unix::fs::symlink(&outside, dir.join("linked.cbor")).unwrap();
    let linked = write_throughput_report(&dir, &measured, "linked.cbor").unwrap_err();
    assert!(linked.message.contains("already exists"), "{linked}");
    assert!(!outside.exists());

    let missing = scratch("missing");
    fs::remove_dir_all(&missing).unwrap();
    let err = write_throughput_report(&missing, &measured, "report.cbor").unwrap_err();
    assert!(err.message.contains("does not exist"), "{err}");

    let mut changed = measured.clone();
    changed.report.prefill_nanos = 2;
    let err = verify_micro_throughput(&changed).unwrap_err();
    assert!(err.message.contains("does not match"), "{err}");
    let _ = fs::remove_dir_all(&dir);
}
