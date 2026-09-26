use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use infer_contracts::*;

fn ext() -> BTreeMap<String, CborValue> {
    BTreeMap::new()
}

fn raw(bytes: &[u8]) -> DigestHex {
    sha256_prefixed(bytes)
}

fn roundtrip(bytes: &[u8], kind: &str) {
    let contract = decode_contract(bytes).unwrap_or_else(|err| panic!("{kind}: {err}"));
    assert_eq!(contract.kind(), kind);
    assert_eq!(contract.to_bytes().unwrap(), bytes);
}

fn with_extra(bytes: &[u8]) -> Vec<u8> {
    let mut value = decode_canonical(bytes).unwrap();
    let CborValue::Map(entries) = &mut value else {
        panic!("map");
    };
    entries.push(("unknownField".into(), CborValue::Integer(1)));
    entries.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
    value.to_bytes()
}

fn bump_version(bytes: &[u8]) -> Vec<u8> {
    let CborValue::Map(entries) = decode_canonical(bytes).unwrap() else {
        panic!("map");
    };
    let entries = entries
        .into_iter()
        .map(|(key, value)| {
            if key == "version" {
                (key, CborValue::Integer(2))
            } else {
                (key, value)
            }
        })
        .collect();
    CborValue::Map(entries).to_bytes()
}

struct Row {
    name: &'static str,
    ok: bool,
    error: &'static str,
    hex: String,
    domain: &'static str,
    payload_hex: String,
    digest: String,
}

fn row(name: &'static str, bytes: &[u8], domain: &'static str, payload: &[u8]) -> Row {
    Row {
        name,
        ok: true,
        error: "",
        hex: encode_hex(bytes),
        domain,
        payload_hex: encode_hex(payload),
        digest: digest_bytes(domain, payload).unwrap().to_string(),
    }
}

fn bad(name: &'static str, bytes: &[u8], error: &'static str) -> Row {
    let err = decode_contract(bytes).expect_err(name);
    assert_eq!(err.code.as_str(), error, "{name}");
    Row {
        name,
        ok: false,
        error,
        hex: encode_hex(bytes),
        domain: "",
        payload_hex: String::new(),
        digest: String::new(),
    }
}

#[test]
fn domains_are_sorted_and_errors_are_stable() {
    for window in INFER_DOMAINS.windows(2) {
        assert!(window[0].as_bytes() < window[1].as_bytes());
    }
    assert_eq!(ErrorCode::ALL.len(), 29);
    assert!(ErrorCode::RequestTimeout.retryable());
    assert!(ErrorCode::ServiceDraining.retryable());
    assert!(!ErrorCode::ServiceUnloaded.retryable());
    assert!(!ErrorCode::CanonicalCborInvalid.retryable());
    assert_eq!(
        digest_bytes("infer-conformance", &CborValue::Null.to_bytes())
            .unwrap()
            .as_str()
            .len(),
        "sha256-".len() + 64
    );
}

#[test]
#[allow(clippy::type_complexity)]
fn contracts_roundtrip_reject_and_export_vectors() {
    let tokenizer = EmbeddedArtifactV1::new("infer-tokenizer", b"tok-v1".to_vec()).unwrap();
    let template = EmbeddedArtifactV1::new("infer-template", b"{{ m }}".to_vec()).unwrap();
    let defaults = FixedPointSamplerV1 {
        temperature_micros: 0,
        top_p_millionths: 1_000_000,
        min_p_millionths: 0,
        repetition_penalty_micros: 0,
        presence_penalty_micros: 0,
        frequency_penalty_micros: 0,
        top_k: 0,
        max_output_tokens: 16,
    };
    let image = ModelImageV1 {
        name: "micro".into(),
        variant: "f32".into(),
        architecture: ArchitectureRefV1 {
            family: "micro".into(),
            adapter: "knolo.micro.v1".into(),
        },
        format: "safetensors".into(),
        files: vec![ArtifactFileV1 {
            path: "weights.safetensors".into(),
            size_bytes: 16,
            sha256: raw(b"weights"),
        }],
        tokenizer: tokenizer.clone(),
        template: template.clone(),
        special_tokens: SpecialTokensV1 {
            bos: Some(1),
            eos: Some(2),
            pad: None,
            unk: None,
            additional: BTreeMap::new(),
        },
        generation_defaults: defaults.clone(),
        capabilities: vec!["text-generation".into()],
        license: LicenseV1 {
            id: "test".into(),
            acceptance_required: false,
        },
        sources: vec![],
        tensor_inventory: vec![TensorSpecV1 {
            name: "w".into(),
            shape: vec![2, 2],
            dtype: "f32".into(),
        }],
        precisions: vec!["f32".into()],
        requirements: ResourceRequirementsV1 {
            minimum_ram_bytes: 1024,
            minimum_vram_bytes: 0,
        },
        placement_hints: None,
        extensions: ext(),
        signatures: vec![],
    };
    let image_bytes = image.to_bytes().unwrap();
    roundtrip(&image_bytes, "knolo.infer.model-image");
    let unsigned_root = image.image_root().unwrap();
    let mut signed = image.clone();
    signed.signatures.push(SignatureV1 {
        algorithm: "ed25519".into(),
        key_id: "local-dev".into(),
        signature: vec![7; 64],
    });
    assert_eq!(signed.image_root().unwrap(), unsigned_root);
    assert_ne!(signed.to_bytes().unwrap(), image_bytes);
    let mut tampered = image.clone();
    tampered.tokenizer.bytes[0] ^= 0xff;
    assert!(tampered.to_bytes().is_err());
    let mut flipped_template = image.clone();
    flipped_template.template.bytes[0] ^= 0xff;
    assert!(flipped_template.to_bytes().is_err());

    let artifacts = ModelArtifactSetV1 {
        files: image.files.clone(),
        extensions: ext(),
    };
    assert_eq!(
        artifacts.artifact_root().unwrap(),
        image.artifact_root().unwrap()
    );

    let kernel = KernelBundleDescriptorV1 {
        cuda_architectures: vec![],
        cuda_toolkit_version: "none".into(),
        source_root: raw(b"kernel-source"),
        compiler_flags: vec![],
        build_mode: "cpu".into(),
        code_object_root: raw(b"kernel-code"),
        jit: None,
        extensions: ext(),
    };
    let engine = EngineBuildDescriptorV1 {
        binary_sha256: raw(b"knolo-infer"),
        source_commit: "unknown".into(),
        cargo_lock_root: raw(b"cargo-lock"),
        rustc_version: "1.98.1".into(),
        target_triple: "x86_64-unknown-linux-gnu".into(),
        build_profile: "debug".into(),
        feature_set: vec![],
        tensor_backend: "candle-cpu".into(),
        tensor_backend_version: "0.8.4".into(),
        kernel_bundle_root: kernel.root().unwrap(),
        extensions: ext(),
    };
    let hardware = HardwareProbeV1 {
        cpu_architecture: "x86_64".into(),
        cpu_model_class: "dev-class".into(),
        physical_cores: 8,
        logical_cores: 16,
        ram_total_bytes: 4096,
        ram_available_bytes: 2048,
        numa_node_count: Some(1),
        gpus: vec![],
        storage_class: "ssd".into(),
        storage_available_bytes: 8192,
        kernel_bundles: vec![kernel.root().unwrap()],
        extensions: ext(),
    };
    let placement = PlacementPlanV1 {
        model_runtime_root: image.runtime_root().unwrap(),
        devices: vec!["cpu".into()],
        tensor_groups: vec![TensorGroupV1 {
            name: "weights".into(),
            device: "cpu".into(),
            compute_precision: "f32".into(),
            storage_precision: "f32".into(),
        }],
        context_reservation_tokens: 32,
        kv_block_size: 16,
        kv_precision: "f32".into(),
        workspace_bytes: 10,
        graph_capture_mode: "off".into(),
        safety_margin_bytes: 1,
        expected_weight_bytes: 100,
        expected_kv_bytes: 20,
        expected_workspace_bytes: 10,
        expected_staging_bytes: 0,
        expected_overhead_bytes: 0,
        expected_total_bytes: 131,
        rejection_reason: None,
        extensions: ext(),
    };
    let prompt = PromptInputV1 {
        messages: vec![ChatMessageV1 {
            role: "user".into(),
            content: "Hello".into(),
        }],
        tools_root: None,
        evidence: None,
        system_policy: None,
        extensions: ext(),
    };
    let evidence = EvidenceBindingV1 {
        knowledge_image_root: Some(raw(b"knowledge")),
        knowledge_commit_root: None,
        query_receipt_ids: vec![raw(b"query")],
        reflex_receipt_ids: vec![],
        context_root: raw(b"context"),
        ordered_evidence_ids: vec!["block-1".into()],
        extensions: ext(),
    };
    let plan = PromptPlanV1 {
        messages_root: prompt.messages_root().unwrap(),
        tools_root: None,
        evidence_root: Some(evidence.root().unwrap()),
        template_root: image.template.root.clone(),
        tokenizer_root: image.tokenizer.root.clone(),
        special_token_plan_root: image.special_tokens_root().unwrap(),
        rendered_text: "Hello".into(),
        token_ids: vec![1, 2],
        truncation: TruncationV1 {
            strategy: "none".into(),
            original_token_count: 2,
            final_token_count: 2,
            removed_ids: vec![],
            preserved_sections: vec!["system".into()],
            max_context_tokens: 32,
            reserved_generation_tokens: 4,
        },
        extensions: ext(),
    };
    let sampler = SamplerPlanV1 {
        settings: defaults,
        rng: "none".into(),
        seed: None,
        stream: None,
        tie_break: "lowest-token-id".into(),
        eos_token_ids: vec![2],
        stop_string_roots: vec![],
        extensions: ext(),
    };
    let grammar = GrammarPlanV1 {
        format: "none".into(),
        source_root: raw(b"no-grammar"),
        enabled: false,
        extensions: ext(),
    };
    let limits = LimitsV1 {
        max_output_tokens: 16,
        max_prompt_tokens: 32,
        deadline_unix_micros: None,
    };
    let intent = InferenceIntentV1 {
        model_runtime_root: image.runtime_root().unwrap(),
        prompt_plan_root: plan.root().unwrap(),
        sampler_plan_root: sampler.root().unwrap(),
        grammar_plan_root: Some(grammar.root().unwrap()),
        evidence_binding_root: Some(evidence.root().unwrap()),
        requested_mode: "pinned".into(),
        limits_root: limits.root().unwrap(),
        extensions: ext(),
    };
    let execution = ExecutionPlanV1 {
        intent_root: intent.root().unwrap(),
        placement_root: placement.root().unwrap(),
        kernel_plan_root: kernel.root().unwrap(),
        scheduling_mode: "isolated".into(),
        prefill_chunk_tokens: 8,
        kv_block_size: 16,
        prefix_cache_enabled: false,
        prefix_cache_namespace: "local".into(),
        speculative_plan_root: None,
        receipt_policy: "durable-stream".into(),
        mode: "pinned".into(),
        extensions: ext(),
    };
    let event = InferenceEventV1 {
        request_id: "req-1".into(),
        attempt: 0,
        name: "accepted".into(),
        previous_event_root: None,
        payload_root: raw(b"accepted"),
        extensions: ext(),
    };
    let mut receipt = InferenceReceiptV1 {
        receipt_id: raw(b"placeholder"),
        intent_root: intent.root().unwrap(),
        model: ModelReceiptBindingV1 {
            model_image_root: image.image_root().unwrap(),
            artifact_root: image.artifact_root().unwrap(),
            model_runtime_root: image.runtime_root().unwrap(),
            architecture_adapter_id: "knolo.micro.v1".into(),
            architecture_adapter_root: image.architecture_adapter_root().unwrap(),
            config_root: image.config_root().unwrap(),
            tokenizer_root: image.tokenizer.root.clone(),
            template_root: image.template.root.clone(),
            storage_precision: "f32".into(),
            compute_precision: "f32".into(),
        },
        engine: EngineReceiptBindingV1 {
            engine_build_root: engine.root().unwrap(),
            backend: "native".into(),
            backend_version: "0.1.0".into(),
            binary_sha256: engine.binary_sha256.clone(),
            kernel_bundle_root: kernel.root().unwrap(),
            kernel_plan_root: kernel.root().unwrap(),
        },
        hardware: HardwareReceiptBindingV1 {
            hardware_root: hardware.root().unwrap(),
            gpu_model: None,
            compute_capability: None,
            device_slot: "cpu".into(),
        },
        placement: PlacementReceiptBindingV1 {
            placement_root: placement.root().unwrap(),
            kv_block_size: 16,
            kv_precision: "f32".into(),
        },
        prompt: PromptReceiptBindingV1 {
            messages_root: prompt.messages_root().unwrap(),
            tools_root: None,
            evidence_root: Some(evidence.root().unwrap()),
            rendered_text_root: plan.rendered_text_root().unwrap(),
            token_id_root: plan.token_id_root().unwrap(),
            prompt_token_count: 2,
            truncation_root: plan.truncation.root().unwrap(),
        },
        knowledge: Some(evidence.clone()),
        sampler: SamplerReceiptBindingV1 {
            sampler_plan_root: sampler.root().unwrap(),
            temperature_micros: 0,
            seed: None,
            rng: "none".into(),
        },
        execution: ExecutionReceiptBindingV1 {
            execution_plan_root: execution.root().unwrap(),
            scheduling_mode: "isolated".into(),
            prefill_chunk_tokens: 8,
            kv_block_size: 16,
            prefix_cache_hit_tokens: 0,
            batch_trace_root: raw(b"batch"),
            speculative_plan_root: None,
            event_trace_root: event.root().unwrap(),
            attempt: 0,
        },
        output: OutputReceiptBindingV1 {
            output_token_root: output_token_root(&[7]).unwrap(),
            output_text_root: output_text_root("ok").unwrap(),
            token_count: 1,
            finish_reason: "stop".into(),
            structured_output_root: None,
        },
        timing: TimingReceiptV1 {
            queue_micros: 1,
            prefill_micros: 2,
            decode_micros: 3,
            total_micros: 6,
        },
        assurance: "same_build_replayable".into(),
        previous_receipt_root: None,
        extensions: ext(),
        signatures: vec![],
    };
    receipt.receipt_id = receipt.computed_id().unwrap();
    let replay = ReplayCheckReceiptV1 {
        receipt_root: receipt.receipt_id.clone(),
        engine_build_root: engine.root().unwrap(),
        placement_root: placement.root().unwrap(),
        output_token_root: receipt.output.output_token_root.clone(),
        matched: true,
        assurance: "exact_replay_verified".into(),
        mismatches: vec![],
        extensions: ext(),
    };
    let conversion = ConversionReceiptV1 {
        source_artifact_root: raw(b"source-artifact"),
        converter_build_root: engine.root().unwrap(),
        conversion_config_root: raw(b"conversion-config"),
        destination_artifact_root: raw(b"destination-artifact"),
        validation_result: "matched".into(),
        extensions: ext(),
    };
    let perplexity = PerplexityReportV1 {
        reference_logit_root: raw(b"reference-logits"),
        candidate_logit_root: raw(b"candidate-logits"),
        target_token_root: output_token_root(&[0]).unwrap(),
        reporter_build_root: engine.root().unwrap(),
        token_count: 1,
        reference_perplexity_micros: 3_718_282,
        candidate_perplexity_micros: 2_000_000,
        perplexity_delta_micros: -1_718_282,
        greedy_parity: false,
        target_accuracy_millionths: 1_000_000,
        max_abs_logit_delta_millionths: 1_000_000,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let memory = MemoryEstimateReportV1 {
        placement_root: raw(b"placement-plan"),
        estimator_build_root: engine.root().unwrap(),
        declared_weight_bytes: 100,
        declared_kv_bytes: 40,
        declared_workspace_bytes: 10,
        declared_staging_bytes: 0,
        declared_overhead_bytes: 0,
        declared_margin_bytes: 1,
        declared_total_bytes: 151,
        measured_weight_bytes: 100,
        measured_kv_bytes: 40,
        measured_workspace_bytes: 10,
        measured_staging_bytes: 0,
        measured_overhead_bytes: 0,
        measured_total_bytes: 150,
        headroom_bytes: 1,
        validation_result: "within-bounds".into(),
        extensions: ext(),
    };
    let throughput = ThroughputReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        prompt_token_root: raw(b"prompt-tokens"),
        output_token_root: output_token_root(&[13, 6, 5, 7]).unwrap(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        prefill_tokens: 3,
        decode_tokens: 4,
        request_count: 1,
        prefill_nanos: 1_000_000_000,
        decode_nanos: 1_000_000_000,
        prefill_tokens_per_second_micros: 3_000_000,
        decode_tokens_per_second_micros: 4_000_000,
        requests_per_second_micros: 500_000,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let latency = LatencyReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        prompt_token_root: raw(b"prompt-tokens"),
        output_token_root: output_token_root(&[13, 6, 5, 7]).unwrap(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        prefill_tokens: 3,
        decode_tokens: 4,
        request_count: 1,
        prefill_nanos: 1_000_000_000,
        decode_nanos: 1_000_000_000,
        time_to_first_token_nanos: 1_000_000_000,
        time_per_output_token_nanos: 250_000_000,
        request_latency_nanos: 2_000_000_000,
        latency_p50_nanos: 2_000_000_000,
        latency_p95_nanos: 2_000_000_000,
        latency_p99_nanos: 2_000_000_000,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let fuzz = FuzzReportV1 {
        engine_build_root: engine.root().unwrap(),
        corpus_root: raw(b"fuzz-corpus"),
        target_count: 9,
        seed_count: 9,
        mutation_count: 54,
        rejected_count: 54,
        distinguished_count: 0,
        accepted_count: 0,
        validation_result: "fail-closed".into(),
        extensions: ext(),
    };
    let cancellation = CancellationReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        prompt_token_root: raw(b"prompt-tokens"),
        output_token_root: output_token_root(&[]).unwrap(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        cancel_stage: "prefill".into(),
        prefill_tokens: 3,
        decode_tokens: 0,
        requested_nanos: 1_000_000_000,
        terminal_nanos: 1_500_000_000,
        cancellation_latency_nanos: 500_000_000,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let finalization = FinalizationReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        prompt_token_root: raw(b"prompt-tokens"),
        output_token_root: output_token_root(&[13]).unwrap(),
        receipt_root: raw(b"stored-receipt"),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        finish_reason: "stop".into(),
        prefill_tokens: 3,
        decode_tokens: 1,
        terminal_nanos: 5_000,
        finalized_nanos: 5_180,
        finalization_latency_nanos: 180,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let overhead = OverheadReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        prompt_token_root: raw(b"prompt-tokens"),
        output_token_root: output_token_root(&[]).unwrap(),
        receipt_root: raw(b"stored-receipt"),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        finish_reason: "length".into(),
        prefill_tokens: 3,
        decode_tokens: 0,
        accepted_write_nanos: 80,
        receipt_write_nanos: 15,
        receipt_overhead_nanos: 95,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let swap = SwapReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        resident_model_image_root: raw(b"resident-image"),
        resident_artifact_root: raw(b"resident-artifact"),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        unload_nanos: 50,
        incoming_verification_nanos: 30,
        incoming_load_nanos: 80,
        model_swap_nanos: 160,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let verification = VerificationReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        verified_bytes: 4096,
        model_verification_nanos: 40,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let load = LoadReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        model_load_nanos: 80,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let peak = PeakReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        device: "cpu".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        peak_ram_bytes: 4096,
        peak_vram_bytes: 0,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let kv = KvReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        page_total: MICRO_KV_PAGES,
        page_size_tokens: MICRO_KV_PAGE_TOKENS,
        peak_pages: 1,
        peak_tokens: 16,
        utilization_millionths: 125_000,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let prefix = PrefixReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        lookup_count: 0,
        hit_count: 0,
        miss_count: 0,
        reused_tokens: 0,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let artifact = raw(b"artifact");
    let tokenizer = raw(b"tokenizer");
    let template = raw(b"template");
    let sampler_root = raw(b"sampler");
    let hardware_root = raw(b"hardware");
    let prompts = raw(b"prompts");
    let outputs = raw(b"outputs");
    let reference_build = raw(b"reference-build");
    let llama = LlamaReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: artifact.clone(),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
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
        sampler_root: sampler_root.clone(),
        hardware_probe_root: hardware_root.clone(),
        prompt_distribution_root: prompts.clone(),
        output_distribution_root: outputs.clone(),
        reference_runtime: "llama.cpp".into(),
        reference_family: "gguf".into(),
        reference_build_root: reference_build.clone(),
        reference_artifact_root: artifact.clone(),
        reference_quantization: "f32".into(),
        reference_tokenizer_root: tokenizer.clone(),
        reference_template_root: template.clone(),
        reference_sampler_root: sampler_root.clone(),
        reference_hardware_root: hardware_root.clone(),
        reference_prompt_distribution_root: prompts.clone(),
        reference_output_distribution_root: outputs.clone(),
        reference_verification_class: "sidecar-artifact-verified".into(),
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let vllm = VllmReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: artifact.clone(),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        quantization: "f32".into(),
        context_tokens: 16,
        tokenizer_root: tokenizer.clone(),
        template_root: template.clone(),
        sampler_root: sampler_root.clone(),
        hardware_probe_root: hardware_root.clone(),
        prompt_distribution_root: prompts.clone(),
        output_distribution_root: outputs.clone(),
        reference_runtime: "vllm".into(),
        reference_family: "safetensors".into(),
        reference_build_root: reference_build.clone(),
        reference_artifact_root: artifact.clone(),
        reference_quantization: "f32".into(),
        reference_tokenizer_root: tokenizer.clone(),
        reference_template_root: template.clone(),
        reference_sampler_root: sampler_root.clone(),
        reference_hardware_root: hardware_root.clone(),
        reference_prompt_distribution_root: prompts.clone(),
        reference_output_distribution_root: outputs.clone(),
        reference_verification_class: "sidecar-artifact-verified".into(),
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let mistral = MistralReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: artifact.clone(),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
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
        sampler_root: sampler_root.clone(),
        hardware_probe_root: hardware_root.clone(),
        prompt_distribution_root: prompts.clone(),
        output_distribution_root: outputs.clone(),
        reference_runtime: "mistral.rs".into(),
        reference_family: "rust".into(),
        reference_build_root: reference_build,
        reference_artifact_root: artifact,
        reference_quantization: "f32".into(),
        reference_tokenizer_root: tokenizer,
        reference_template_root: template,
        reference_sampler_root: sampler_root,
        reference_hardware_root: hardware_root,
        reference_prompt_distribution_root: prompts,
        reference_output_distribution_root: outputs,
        reference_verification_class: "backend-reported".into(),
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let recipe = RecipeReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        adapter_id: "knolo.micro.v1".into(),
        conformance_root: raw(b"conformance"),
        security_root: raw(b"security"),
        stability_root: raw(b"stability"),
        benchmark_root: raw(b"benchmark"),
        support_level: "experimental".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let composition = CompositionReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        evidence_root: raw(b"evidence"),
        knowledge_image_root: raw(b"knowledge-image"),
        knowledge_commit_root: raw(b"knowledge-commit"),
        context_root: raw(b"context"),
        query_receipt_count: 1,
        reflex_receipt_count: 1,
        ordered_evidence_count: 1,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let runtime = raw(b"runtime");
    let agent_artifact = raw(b"artifact");
    let knowledge = raw(b"knowledge");
    let agent = AgentEffectReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: agent_artifact.clone(),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        model_runtime_root: runtime.clone(),
        required_model_runtime_root: runtime,
        required_artifact_root: agent_artifact,
        knowledge_image_root: knowledge.clone(),
        required_knowledge_image_root: knowledge,
        backend: "reference-f32".into(),
        assurance: "same_build_replayable".into(),
        required_assurance: "same_build_replayable".into(),
        execution_mode: "isolated-replay".into(),
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
        decision: "allow".into(),
        reason: "accepted".into(),
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let hub = HubReportV1 {
        publisher: "knolo".into(),
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        adapter_id: "knolo.micro.v1".into(),
        quantization: "f32".into(),
        license_id: "apache-2.0".into(),
        source_provider: "local".into(),
        conformance_root: raw(b"conformance"),
        benchmark_root: raw(b"benchmark"),
        support_level: "experimental".into(),
        greedy_token_parity: true,
        distribution: "active".into(),
        native_supported: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let studio = StudioReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        receipt_root: raw(b"receipt"),
        prompt_token_root: raw(b"prompt"),
        output_token_root: raw(b"output"),
        output_text_root: raw(b"output-text"),
        knowledge_image_root: raw(b"knowledge"),
        evidence_root: raw(b"evidence"),
        prompt_token_count: 2,
        output_token_count: 1,
        finish_reason: "stop".into(),
        assurance: "compatibility".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let chain = ChainReportV1 {
        model_runtime_root: raw(b"runtime"),
        artifact_root: raw(b"artifact"),
        engine_build_root: engine.root().unwrap(),
        kernel_bundle_root: raw(b"kernel"),
        placement_root: raw(b"placement-plan"),
        prompt_token_root: raw(b"prompt"),
        knowledge_image_root: raw(b"knowledge-image"),
        knowledge_commit_root: raw(b"knowledge-commit"),
        query_receipt_root: raw(b"query"),
        query_receipt_count: 1,
        reflex_receipt_root: raw(b"reflex"),
        reflex_receipt_count: 1,
        receipt_root: raw(b"receipt"),
        effect_root: raw(b"effect"),
        output_token_root: raw(b"output"),
        output_text_root: raw(b"output-text"),
        prompt_token_count: 2,
        output_token_count: 1,
        finish_reason: "stop".into(),
        assurance: "compatibility".into(),
        link_count: 5,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let evidence_output = EvidenceOutputReportV1 {
        evidence_root: raw(b"evidence"),
        knowledge_image_root: raw(b"knowledge-image"),
        query_receipt_root: raw(b"query"),
        reflex_receipt_root: raw(b"reflex"),
        receipt_root: raw(b"receipt"),
        chain_root: raw(b"chain"),
        output_token_root: raw(b"output"),
        output_text_root: raw(b"output-text"),
        prompt_token_count: 2,
        output_token_count: 1,
        finish_reason: "stop".into(),
        assurance: "compatibility".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let install = InstallReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        tokenizer_root: raw(b"tokenizer"),
        template_root: raw(b"template"),
        publisher: "knolo".into(),
        license_id: "apache-2.0".into(),
        source_provider: "local".into(),
        artifact_count: 4,
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let notice = NoticeReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        notice_root: raw(b"notice"),
        sbom_root: raw(b"sbom"),
        component_count: 1,
        feature_set: "cpu".into(),
        candle_named: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let release = ReleaseReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        notice_root: raw(b"notice"),
        sbom_root: raw(b"sbom"),
        binary_set_root: raw(b"binaries"),
        signature_status: "unsigned-local".into(),
        signature_count: 0,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let binary = BinaryReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        supervisor_name: "knolo-infer".into(),
        supervisor_hash: raw(b"supervisor"),
        supervisor_bytes: 1024,
        worker_name: "knolo-infer-worker".into(),
        worker_hash: raw(b"worker"),
        worker_bytes: 2048,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let reproducible = ReproducibleReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        lock_root: raw(b"cargo-lock"),
        source_root: raw(b"instructions"),
        feature_set: "cpu".into(),
        build_profile: "debug".into(),
        instruction_count: 4,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let signature = SignatureReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        signature_status: "shape-checked".into(),
        signature_count: 1,
        key_id: "local-dev".into(),
        signature_bytes: 64,
        key_verified: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let host_key = HostKeyReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        host_key_root: raw(b"host-key"),
        signature_status: "matched".into(),
        key_id: "local-dev".into(),
        signature_bytes: 64,
        key_verified: true,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let receipt_key = ReceiptKeyReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        receipt_root: raw(b"receipt"),
        custody: "host-store".into(),
        key_material_serialized: false,
        key_id: "local-dev".into(),
        signature_status: "shape-checked".into(),
        signature_bytes: 64,
        rotation: "rotated".into(),
        previous_key_id: "local-prev".into(),
        trusted_metadata: true,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let sandbox = SandboxReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        user_class: "unprivileged".into(),
        network: "none".into(),
        model_cas: "read-only".into(),
        scratch: "worker-only".into(),
        seccomp: "deferred".into(),
        memory_limit_bytes: 1024,
        process_group: "isolated".into(),
        parent_death: "socket-eof".into(),
        shared_memory_bytes: 0,
        arguments: "direct-array".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let api = ApiReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        tenant_root: raw(b"tenant"),
        bind: "localhost".into(),
        remote_explicit: false,
        auth: "hook".into(),
        body_limit_bytes: 1024,
        rate_per_minute: 60,
        concurrency_limit: 1,
        prompt_log: "omitted".into(),
        metrics_labels: "counts".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let cache_channel = CacheChannelReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        tenant_root: raw(b"tenant"),
        project_root: raw(b"project"),
        sharing_policy: "isolated".into(),
        cross_tenant: false,
        existence_disclosure: "hidden".into(),
        metrics_scope: "aggregate".into(),
        prefix_allocated: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let equation = EquationReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        message_root: raw(b"message"),
        equation_status: "accepted".into(),
        equation_evaluated: true,
        public_key_bytes: ED25519_PUBLIC_KEY_BYTES,
        signature_bytes: ED25519_SIGNATURE_BYTES,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let safe_error = SafeErrorReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        code: "CONTRACT_INVALID".into(),
        message: "CONTRACT_INVALID".into(),
        retryable: false,
        request_id: "req-1".into(),
        attempt: 1,
        prompt_present: false,
        secret_present: false,
        partial_receipt: "absent".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let redaction = RedactionReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        stage: "finalize".into(),
        request_id: "req-1".into(),
        prompt_plan_root: raw(b"prompt-plan"),
        receipt_root: raw(b"receipt"),
        format: "json-line".into(),
        redacted: true,
        prompt: "omitted".into(),
        output: "omitted".into(),
        token_ids: "omitted".into(),
        alias: "omitted".into(),
        path: "omitted".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let curve = CurveReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        message_root: raw(b"message"),
        curve_status: "on-curve".into(),
        curve_computed: true,
        public_key_bytes: ED25519_PUBLIC_KEY_BYTES,
        signature_bytes: ED25519_SIGNATURE_BYTES,
        point_checked: true,
        base_multiplied: false,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let rollback = RollbackReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        previous_image_root: raw(b"previous-image"),
        incoming_image_root: raw(b"incoming-image"),
        previous_artifact_root: raw(b"previous-artifact"),
        incoming_artifact_root: raw(b"incoming-artifact"),
        reason: "pin-mismatch".into(),
        lockfile_mutated: false,
        core_lock_touched: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let disconnect = DisconnectReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        request_id: "req-1".into(),
        stage: "queue".into(),
        outcome: "cancelled".into(),
        code: "REQUEST_CANCELLED".into(),
        listener_up: true,
        socket_closed: false,
        prompt_tokens: 4,
        output_tokens: 0,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let receipt_store = ReceiptStoreReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        request_id: "req-1".into(),
        code: "RECEIPT_PERSIST_FAILED".into(),
        retryable: true,
        receipt_stored: false,
        journal_event: "failed".into(),
        partial_receipt: "absent".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let base = BaseReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        message_root: raw(b"message"),
        base_status: "multiplied".into(),
        base_multiplied: true,
        public_key_bytes: ED25519_PUBLIC_KEY_BYTES,
        signature_bytes: ED25519_SIGNATURE_BYTES,
        scalar_bytes: ED25519_SCALAR_BYTES,
        point_added: false,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let disk = DiskReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        request_id: "req-1".into(),
        store: "journal".into(),
        free_bytes: 0,
        needed_bytes: 4096,
        file_written: false,
        space_reclaimed: false,
        listener_up: true,
        code: "RECEIPT_PERSIST_FAILED".into(),
        retryable: true,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let unload = UnloadReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        request_id: "req-1".into(),
        queued_requests: 2,
        admitted_requests: 1,
        queued_started: false,
        admitted_finished: true,
        lifecycle: "unloading".into(),
        worker_exited: false,
        code: "SERVICE_UNLOADED".into(),
        retryable: false,
        listener_up: true,
        model_reloaded: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let restart = RestartReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        previous_owner_root: raw(b"previous-owner"),
        incoming_owner_root: raw(b"incoming-owner"),
        reason: "stale-lock".into(),
        lock_replaced: true,
        journals_sealed: true,
        listener_up: true,
        worker_restart_count: 0,
        process_spawned: false,
        second_worker: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let point = PointReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        message_root: raw(b"message"),
        point_status: "added".into(),
        point_added: true,
        public_key_bytes: ED25519_PUBLIC_KEY_BYTES,
        signature_bytes: ED25519_SIGNATURE_BYTES,
        scalar_bytes: ED25519_SCALAR_BYTES,
        points_equal: false,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let duplicate = DuplicateReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        request_id: "req-1".into(),
        occupant_root: raw(b"occupant"),
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        duplicate_started: false,
        occupant_kept: true,
        second_journal: false,
        listener_up: true,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let concurrent_load = ConcurrentLoadReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        phase: "ready".into(),
        lifecycle: "serving".into(),
        inflight: 0,
        worker_started: false,
        second_worker: false,
        worker_replaced: false,
        restart_count: 0,
        listener_up: true,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let eviction = EvictionReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        load_requests: 2,
        evicted_pages: 0,
        evicted_tokens: 0,
        active_evicted: false,
        prefix_allocated: false,
        listener_up: true,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let equality = EqualityReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        message_root: raw(b"message"),
        equality_status: "equal".into(),
        points_compared: true,
        points_equal: true,
        challenge_hashed: false,
        public_key_bytes: ED25519_PUBLIC_KEY_BYTES,
        signature_bytes: ED25519_SIGNATURE_BYTES,
        scalar_bytes: ED25519_SCALAR_BYTES,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let oom = OomReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        device: "slot-0".into(),
        free_bytes: 0,
        needed_bytes: 4096,
        code: "CUDA_OOM".into(),
        retryable: true,
        receipt_stored: false,
        listener_up: true,
        supervisor_exited: false,
        cpu_fallback: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let fault = FaultReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        device: "slot-0".into(),
        fault_class: "kernel".into(),
        code: "CUDA_FAULT".into(),
        retryable: false,
        receipt_stored: false,
        listener_up: true,
        supervisor_exited: false,
        cpu_fallback: false,
        graph_captured: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let timeout = TimeoutReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        request_id: "req-1".into(),
        timeout_stage: "completion".into(),
        waited_nanos: 1_000,
        code: "REQUEST_TIMEOUT".into(),
        retryable: true,
        receipt_stored: false,
        listener_up: true,
        worker_lost: false,
        http_status: HTTP_TIMEOUT_STATUS,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let worker_start = WorkerStartReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        start_failure: "not-ready".into(),
        code: "WORKER_START_FAILED".into(),
        retryable: true,
        receipt_stored: false,
        listener_up: true,
        process_spawned: true,
        worker_ready: false,
        restart_count: 0,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let challenge = ChallengeReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        message_root: raw(b"message"),
        challenge_status: "hashed".into(),
        challenge_hashed: true,
        scalar_reduced: false,
        public_key_bytes: ED25519_PUBLIC_KEY_BYTES,
        signature_bytes: ED25519_SIGNATURE_BYTES,
        scalar_bytes: ED25519_SCALAR_BYTES,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let replay_environment = ReplayEnvironmentReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        receipt_root: raw(b"receipt"),
        mismatched_field: "engine".into(),
        code: "REPLAY_ENVIRONMENT_MISMATCH".into(),
        retryable: false,
        check_stored: false,
        forward_ran: false,
        output_compared: false,
        assurance: "incomplete".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let replay_output = ReplayOutputReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        receipt_root: raw(b"receipt"),
        candidate_output_root: raw(b"candidate"),
        code: "REPLAY_OUTPUT_MISMATCH".into(),
        retryable: false,
        check_stored: false,
        forward_ran: true,
        environment_matched: true,
        assurance: "incomplete".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let worker_lost = WorkerLostReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        request_id: "req-1".into(),
        code: "WORKER_LOST".into(),
        retryable: true,
        receipt_stored: false,
        listener_up: true,
        supervisor_exited: false,
        http_status: HTTP_SERVICE_UNAVAILABLE,
        journal_sealed: true,
        restart_counted: true,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let draining = DrainingReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        lifecycle: "draining".into(),
        code: "SERVICE_DRAINING".into(),
        retryable: true,
        receipt_stored: false,
        body_parsed: false,
        listener_up: true,
        worker_loaded: true,
        restart_counted: false,
        http_status: HTTP_SERVICE_UNAVAILABLE,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let scalar = ScalarReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        message_root: raw(b"message"),
        scalar_status: "reduced".into(),
        scalar_reduced: true,
        public_multiplied: false,
        public_key_bytes: ED25519_PUBLIC_KEY_BYTES,
        signature_bytes: ED25519_SIGNATURE_BYTES,
        scalar_bytes: ED25519_SCALAR_BYTES,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let digest_mismatch = DigestMismatchReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        artifact_root: raw(b"artifact"),
        mismatch: "digest".into(),
        code: "MODEL_DIGEST_MISMATCH".into(),
        retryable: false,
        header_parsed: false,
        body_read: true,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let tokenizer_invalid = TokenizerInvalidReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        tokenizer_root: raw(b"tokenizer"),
        failure: "encode".into(),
        code: "TOKENIZER_INVALID".into(),
        retryable: false,
        tokenizer_parsed: true,
        template_rendered: true,
        prompt_compiled: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let template_invalid = TemplateInvalidReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        template_root: raw(b"template"),
        failure: "grammar".into(),
        code: "TEMPLATE_INVALID".into(),
        retryable: false,
        template_rendered: false,
        tokenizer_parsed: false,
        prompt_compiled: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let architecture = ArchitectureReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        rejected_adapter: "knolo.llama.v1".into(),
        code: "UNSUPPORTED_ARCHITECTURE".into(),
        retryable: false,
        weights_opened: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let public_key = PublicReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        message_root: raw(b"message"),
        public_status: "multiplied".into(),
        public_multiplied: true,
        signature_checked: false,
        public_key_bytes: ED25519_PUBLIC_KEY_BYTES,
        signature_bytes: ED25519_SIGNATURE_BYTES,
        scalar_bytes: ED25519_SCALAR_BYTES,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let quantization = QuantizationReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        artifact_root: raw(b"artifact"),
        reason: "dtype".into(),
        code: "UNSUPPORTED_QUANTIZATION".into(),
        retryable: false,
        weights_opened: true,
        payload_read: true,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let kernel_report = KernelReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "feature".into(),
        code: "UNSUPPORTED_KERNEL".into(),
        retryable: false,
        cuda_requested: true,
        kernel_selected: false,
        device_opened: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let placement_refusal = PlacementRefusalReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "device".into(),
        code: "PLACEMENT_UNSATISFIABLE".into(),
        retryable: false,
        probe_reached: true,
        slot_visible: false,
        device_opened: false,
        cpu_fallback: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let memory_refusal = MemoryRefusalReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "pool".into(),
        free_bytes: 0,
        needed_bytes: 4096,
        code: "INSUFFICIENT_MEMORY".into(),
        retryable: true,
        resident_full: true,
        queue_held: false,
        allocated: false,
        forward_ran: false,
        receipt_stored: false,
        listener_up: true,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let signature_check = SignatureCheckReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        message_root: raw(b"message"),
        check_status: "checked".into(),
        signature_checked: true,
        cofactor_cleared: false,
        public_key_bytes: ED25519_PUBLIC_KEY_BYTES,
        signature_bytes: ED25519_SIGNATURE_BYTES,
        scalar_bytes: ED25519_SCALAR_BYTES,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let context_limit = ContextLimitReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "prompt".into(),
        prompt_tokens: 17,
        reserved_tokens: 0,
        context_tokens: MICRO_CONTEXT,
        truncated: false,
        code: "CONTEXT_LIMIT_EXCEEDED".into(),
        retryable: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let prompt_compilation = PromptCompilationReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        failure: "empty".into(),
        token_count: 0,
        rejected_token: 0,
        code: "PROMPT_COMPILATION_FAILED".into(),
        retryable: false,
        template_rendered: true,
        tokenizer_parsed: true,
        prompt_compiled: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let image_invalid = ImageInvalidReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        image_root: raw(b"image"),
        reason: "format".into(),
        code: "MODEL_IMAGE_INVALID".into(),
        retryable: false,
        image_parsed: true,
        weights_opened: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let image_signature = ImageSignatureReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        image_root: raw(b"image"),
        reason: "length".into(),
        signature_count: 1,
        signature_bytes: 32,
        code: "MODEL_IMAGE_SIGNATURE_INVALID".into(),
        retryable: false,
        algorithm_accepted: true,
        weights_opened: false,
        forward_ran: false,
        receipt_stored: false,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let cofactor = CofactorReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        message_root: raw(b"message"),
        clear_status: "cleared".into(),
        cofactor_cleared: true,
        receipt_signed: false,
        public_key_bytes: ED25519_PUBLIC_KEY_BYTES,
        signature_bytes: ED25519_SIGNATURE_BYTES,
        scalar_bytes: ED25519_SCALAR_BYTES,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let artifact_missing = ArtifactMissingReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        image_root: raw(b"image"),
        artifact_root: raw(b"artifact"),
        reason: "weights".into(),
        code: "MODEL_ARTIFACT_MISSING".into(),
        retryable: false,
        source_provider: "local".into(),
        lock_present: true,
        alias_pinned: true,
        image_opened: true,
        weights_opened: false,
        downloaded: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let receipt_required = ReceiptRequiredReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        receipt_root: raw(b"receipt"),
        reason: "file".into(),
        code: "RECEIPT_REQUIRED".into(),
        retryable: false,
        http_status: RECEIPT_HTTP_STATUS,
        receipt_read: false,
        request_bound: false,
        journal_opened: false,
        event_count: 0,
        listener_up: true,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let backend_report = BackendReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        surface: "run".into(),
        requested_mode: "throughput".into(),
        code: "BACKEND_NOT_ALLOWED".into(),
        retryable: false,
        backend_selected: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let digest_invalid = DigestInvalidReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "length".into(),
        hex_length: 32,
        domain: "none".into(),
        code: "DIGEST_INVALID".into(),
        retryable: false,
        prefix_accepted: true,
        length_accepted: false,
        alphabet_accepted: false,
        domain_accepted: false,
        hashed: false,
        file_opened: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let receipt_sign = ReceiptSignReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        message_root: raw(b"message"),
        sign_status: "signed".into(),
        receipt_signed: true,
        receipt_verified: false,
        public_key_bytes: ED25519_PUBLIC_KEY_BYTES,
        signature_bytes: ED25519_SIGNATURE_BYTES,
        scalar_bytes: ED25519_SCALAR_BYTES,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let canonical_cbor = CanonicalCborReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "order".into(),
        major: 5,
        additional_info: 2,
        code: "CANONICAL_CBOR_INVALID".into(),
        retryable: false,
        argument_read: true,
        value_accepted: false,
        keys_ordered: false,
        reencoded: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let contract_invalid = ContractInvalidReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "unknown".into(),
        field_bytes: 12,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        field_present: true,
        type_accepted: false,
        value_accepted: false,
        decoded: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let grammar_refusal = GrammarRefusalReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "mask".into(),
        source_bytes: 16,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        source_opened: true,
        grammar_rooted: true,
        mask_applied: false,
        grammar_compiled: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let receipt_verify = ReceiptVerifyReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"verify-release"),
        message_root: raw(b"verify-message"),
        verify_status: "verified".into(),
        receipt_verified: true,
        domain_separated: false,
        public_key_bytes: ED25519_PUBLIC_KEY_BYTES,
        signature_bytes: ED25519_SIGNATURE_BYTES,
        scalar_bytes: ED25519_SCALAR_BYTES,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let speculative = SpeculativeReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        target_root: raw(b"spec-target"),
        proposal_root: raw(b"spec-proposal"),
        reason: "draft".into(),
        proposal_tokens: 4,
        accepted_tokens: 0,
        rejected_tokens: 0,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        speculated: false,
        distribution_changed: false,
        cache_affected: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let graph = GraphReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "shape".into(),
        device: "slot-0".into(),
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        graph_captured: false,
        captured_nodes: 0,
        workspace_bytes: 0,
        shape_buckets: 0,
        kernel_repeated: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let multi_model = MultiModelReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        resident_root: raw(b"resident-model"),
        incoming_root: raw(b"incoming-model"),
        reason: "parallel".into(),
        device_count: 2,
        models_loaded: 1,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        second_loaded: false,
        resident_replaced: false,
        tensor_parallel: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let secondary = SecondaryReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        primary_root: raw(b"primary-model"),
        secondary_root: raw(b"secondary-model"),
        reason: "overflow".into(),
        secondary_slot: "slot-1".into(),
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        model_named: true,
        embeddings_requested: false,
        overflow_requested: true,
        service_started: false,
        embeddings_ran: false,
        overflow_placed: false,
        device_opened: false,
        tensor_parallel: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let tool_refusal = ToolRefusalReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        object_root: raw(b"tool-object"),
        reason: "execute".into(),
        name_bytes: 8,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        name_accepted: true,
        object_generated: true,
        tool_executed: false,
        authority_checked: false,
        budget_checked: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let conformance = ModelConformanceReceiptV1 {
        model_runtime_root: image.runtime_root().unwrap(),
        engine_build_root: engine.root().unwrap(),
        adapter_id: "knolo.micro.v1".into(),
        logit_abs_tolerance_millionths: 1_000,
        greedy_token_parity: true,
        support_level: "conformant".into(),
        notes_root: raw(b"notes"),
        extensions: ext(),
    };
    let domain_sep_report = DomainReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        domain_root: raw(b"domain-prefix"),
        message_root: raw(b"domain-message"),
        separate_status: "separated".into(),
        domain_separated: true,
        payload_hashed: false,
        public_key_bytes: 32,
        signature_bytes: 64,
        scalar_bytes: 32,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    let attention_report = AttentionReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "window".into(),
        window_tokens: 4,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        attention_exact: true,
        approximated: false,
        window_applied: false,
        sparsity_applied: false,
        kv_quantized: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let moe_report = MoeReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "router".into(),
        requested_experts: 4,
        selected_experts: 0,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        routed: false,
        shared_used: false,
        tie_broken: false,
        grouped: false,
        expert_placed: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let expert_placement_report = ExpertPlacementReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "asymmetric".into(),
        device_count: 2,
        experts_placed: 0,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        cpu_offload: false,
        resident_moved: false,
        tensor_parallel: false,
        automatic_fallback: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let grouped_kernel_report = GroupedKernelReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "gemm".into(),
        group_count: 4,
        code: "UNSUPPORTED_KERNEL".into(),
        retryable: false,
        kernel_selected: false,
        grouped: false,
        routing_ran: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let router_report = RouterReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reference_root: raw(b"router-reference"),
        candidate_root: raw(b"router-candidate"),
        reason: "logits".into(),
        sample_count: 4,
        compared: false,
        parity: false,
        load_recorded: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let glm_report = GlmReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "inventory".into(),
        rejected_adapter: "knolo.glm.v1".into(),
        code: "UNSUPPORTED_ARCHITECTURE".into(),
        retryable: false,
        weights_opened: true,
        inventory_read: false,
        recipe_blessed: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let workstation_report = WorkstationReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        profile_root: raw(b"workstation-profile"),
        benchmark_root: raw(b"workstation-benchmark"),
        reason: "benchmark".into(),
        benchmark_count: 4,
        blessed: false,
        benchmark_recorded: false,
        fallback_selected: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let mixed_placement_report = MixedPlacementReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "mixed".into(),
        device_count: 2,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        single_recorded: false,
        mixed_selected: false,
        automatic_fallback: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let capacity_report = CapacityReportV1 {
        engine_build_root: engine.root().unwrap(),
        placement_root: raw(b"placement-plan"),
        reason: "overflow".into(),
        requested_tokens: 4,
        capacity_tokens: 0,
        code: "CONTRACT_INVALID".into(),
        retryable: false,
        overflowed: false,
        dropped: false,
        balanced: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };

    let items: Vec<(&str, &str, Vec<u8>, Vec<u8>, &str)> = vec![
        (
            "model-image",
            "infer-model-image",
            image.to_bytes().unwrap(),
            image.content_cbor().unwrap().to_bytes(),
            "knolo.infer.model-image",
        ),
        (
            "model-artifact",
            "infer-model-artifact",
            artifacts.to_bytes().unwrap(),
            artifact_payload(&artifacts),
            "knolo.infer.model-artifact-set",
        ),
        (
            "engine-build",
            "infer-engine-build",
            engine.to_bytes().unwrap(),
            engine.to_cbor().unwrap().to_bytes(),
            "knolo.infer.engine-build",
        ),
        (
            "kernel-bundle",
            "infer-kernel-bundle",
            kernel.to_bytes().unwrap(),
            kernel.to_cbor().unwrap().to_bytes(),
            "knolo.infer.kernel-bundle",
        ),
        (
            "hardware-probe",
            "infer-hardware",
            hardware.to_bytes().unwrap(),
            hardware.to_cbor().unwrap().to_bytes(),
            "knolo.infer.hardware-probe",
        ),
        (
            "placement-plan",
            "infer-placement",
            placement.to_bytes().unwrap(),
            placement.to_cbor().unwrap().to_bytes(),
            "knolo.infer.placement-plan",
        ),
        (
            "prompt-input",
            "infer-prompt-input",
            prompt.to_bytes().unwrap(),
            prompt.to_cbor().unwrap().to_bytes(),
            "knolo.infer.prompt-input",
        ),
        (
            "prompt-plan",
            "infer-prompt-plan",
            plan.to_bytes().unwrap(),
            plan.to_cbor().unwrap().to_bytes(),
            "knolo.infer.prompt-plan",
        ),
        (
            "evidence",
            "infer-evidence",
            evidence.to_bytes().unwrap(),
            evidence.to_cbor().unwrap().to_bytes(),
            "knolo.infer.evidence-binding",
        ),
        (
            "sampler-plan",
            "infer-sampler",
            sampler.to_bytes().unwrap(),
            sampler.to_cbor().unwrap().to_bytes(),
            "knolo.infer.sampler-plan",
        ),
        (
            "grammar-plan",
            "infer-grammar",
            grammar.to_bytes().unwrap(),
            grammar.to_cbor().unwrap().to_bytes(),
            "knolo.infer.grammar-plan",
        ),
        (
            "intent",
            "infer-request-intent",
            intent.to_bytes().unwrap(),
            intent.to_cbor().unwrap().to_bytes(),
            "knolo.infer.inference-intent",
        ),
        (
            "execution-plan",
            "infer-execution-plan",
            execution.to_bytes().unwrap(),
            execution.to_cbor().unwrap().to_bytes(),
            "knolo.infer.execution-plan",
        ),
        (
            "event",
            "infer-execution-event",
            event.to_bytes().unwrap(),
            event_payload(&event),
            "knolo.infer.inference-event",
        ),
        (
            "receipt",
            "infer-receipt",
            receipt.to_bytes().unwrap(),
            receipt.content_cbor().unwrap().to_bytes(),
            "knolo.infer.inference-receipt",
        ),
        (
            "replay-check",
            "infer-replay-check",
            replay.to_bytes().unwrap(),
            replay.to_cbor().unwrap().to_bytes(),
            "knolo.infer.replay-check",
        ),
        (
            "conformance",
            "infer-conformance",
            conformance.to_bytes().unwrap(),
            conformance.to_cbor().unwrap().to_bytes(),
            "knolo.infer.model-conformance",
        ),
        (
            "conversion-receipt",
            "infer-conversion",
            conversion.to_bytes().unwrap(),
            conversion.to_cbor().unwrap().to_bytes(),
            "knolo.infer.conversion-receipt",
        ),
        (
            "perplexity-report",
            "infer-perplexity",
            perplexity.to_bytes().unwrap(),
            perplexity.to_cbor().unwrap().to_bytes(),
            "knolo.infer.perplexity-report",
        ),
        (
            "memory-estimate",
            "infer-memory",
            memory.to_bytes().unwrap(),
            memory.to_cbor().unwrap().to_bytes(),
            "knolo.infer.memory-estimate",
        ),
        (
            "throughput-report",
            "infer-throughput",
            throughput.to_bytes().unwrap(),
            throughput.to_cbor().unwrap().to_bytes(),
            "knolo.infer.throughput-report",
        ),
        (
            "latency-report",
            "infer-latency",
            latency.to_bytes().unwrap(),
            latency.to_cbor().unwrap().to_bytes(),
            "knolo.infer.latency-report",
        ),
        (
            "fuzz-report",
            "infer-fuzz",
            fuzz.to_bytes().unwrap(),
            fuzz.to_cbor().unwrap().to_bytes(),
            "knolo.infer.fuzz-report",
        ),
        (
            "cancellation-report",
            "infer-cancellation",
            cancellation.to_bytes().unwrap(),
            cancellation.to_cbor().unwrap().to_bytes(),
            "knolo.infer.cancellation-report",
        ),
        (
            "finalization-report",
            "infer-finalization",
            finalization.to_bytes().unwrap(),
            finalization.to_cbor().unwrap().to_bytes(),
            "knolo.infer.finalization-report",
        ),
        (
            "overhead-report",
            "infer-overhead",
            overhead.to_bytes().unwrap(),
            overhead.to_cbor().unwrap().to_bytes(),
            "knolo.infer.overhead-report",
        ),
        (
            "swap-report",
            "infer-swap",
            swap.to_bytes().unwrap(),
            swap.to_cbor().unwrap().to_bytes(),
            "knolo.infer.swap-report",
        ),
        (
            "verification-report",
            "infer-verification",
            verification.to_bytes().unwrap(),
            verification.to_cbor().unwrap().to_bytes(),
            "knolo.infer.verification-report",
        ),
        (
            "load-report",
            "infer-load",
            load.to_bytes().unwrap(),
            load.to_cbor().unwrap().to_bytes(),
            "knolo.infer.load-report",
        ),
        (
            "peak-report",
            "infer-peak",
            peak.to_bytes().unwrap(),
            peak.to_cbor().unwrap().to_bytes(),
            "knolo.infer.peak-report",
        ),
        (
            "kv-report",
            "infer-kv",
            kv.to_bytes().unwrap(),
            kv.to_cbor().unwrap().to_bytes(),
            "knolo.infer.kv-report",
        ),
        (
            "prefix-report",
            "infer-prefix",
            prefix.to_bytes().unwrap(),
            prefix.to_cbor().unwrap().to_bytes(),
            "knolo.infer.prefix-report",
        ),
        (
            "llama-report",
            "infer-llama",
            llama.to_bytes().unwrap(),
            llama.to_cbor().unwrap().to_bytes(),
            "knolo.infer.llama-report",
        ),
        (
            "vllm-report",
            "infer-vllm",
            vllm.to_bytes().unwrap(),
            vllm.to_cbor().unwrap().to_bytes(),
            "knolo.infer.vllm-report",
        ),
        (
            "mistral-report",
            "infer-mistral",
            mistral.to_bytes().unwrap(),
            mistral.to_cbor().unwrap().to_bytes(),
            "knolo.infer.mistral-report",
        ),
        (
            "recipe-report",
            "infer-recipe",
            recipe.to_bytes().unwrap(),
            recipe.to_cbor().unwrap().to_bytes(),
            "knolo.infer.recipe-report",
        ),
        (
            "composition-report",
            "infer-composition",
            composition.to_bytes().unwrap(),
            composition.to_cbor().unwrap().to_bytes(),
            "knolo.infer.composition-report",
        ),
        (
            "agent-effect-report",
            "infer-agent-effect",
            agent.to_bytes().unwrap(),
            agent.to_cbor().unwrap().to_bytes(),
            "knolo.infer.agent-effect-report",
        ),
        (
            "hub-report",
            "infer-hub",
            hub.to_bytes().unwrap(),
            hub.to_cbor().unwrap().to_bytes(),
            "knolo.infer.hub-report",
        ),
        (
            "studio-report",
            "infer-studio",
            studio.to_bytes().unwrap(),
            studio.to_cbor().unwrap().to_bytes(),
            "knolo.infer.studio-report",
        ),
        (
            "chain-report",
            "infer-chain",
            chain.to_bytes().unwrap(),
            chain.to_cbor().unwrap().to_bytes(),
            "knolo.infer.chain-report",
        ),
        (
            "evidence-output-report",
            "infer-evidence-output",
            evidence_output.to_bytes().unwrap(),
            evidence_output.to_cbor().unwrap().to_bytes(),
            "knolo.infer.evidence-output-report",
        ),
        (
            "install-report",
            "infer-install",
            install.to_bytes().unwrap(),
            install.to_cbor().unwrap().to_bytes(),
            "knolo.infer.install-report",
        ),
        (
            "notice-report",
            "infer-notice",
            notice.to_bytes().unwrap(),
            notice.to_cbor().unwrap().to_bytes(),
            "knolo.infer.notice-report",
        ),
        (
            "release-report",
            "infer-release",
            release.to_bytes().unwrap(),
            release.to_cbor().unwrap().to_bytes(),
            "knolo.infer.release-report",
        ),
        (
            "binary-report",
            "infer-binary",
            binary.to_bytes().unwrap(),
            binary.to_cbor().unwrap().to_bytes(),
            "knolo.infer.binary-report",
        ),
        (
            "reproducible-report",
            "infer-reproducible",
            reproducible.to_bytes().unwrap(),
            reproducible.to_cbor().unwrap().to_bytes(),
            "knolo.infer.reproducible-report",
        ),
        (
            "signature-report",
            "infer-signature",
            signature.to_bytes().unwrap(),
            signature.to_cbor().unwrap().to_bytes(),
            "knolo.infer.signature-report",
        ),
        (
            "host-key-report",
            "infer-host-key",
            host_key.to_bytes().unwrap(),
            host_key.to_cbor().unwrap().to_bytes(),
            "knolo.infer.host-key-report",
        ),
        (
            "receipt-key-report",
            "infer-receipt-key",
            receipt_key.to_bytes().unwrap(),
            receipt_key.to_cbor().unwrap().to_bytes(),
            "knolo.infer.receipt-key-report",
        ),
        (
            "sandbox-report",
            "infer-sandbox",
            sandbox.to_bytes().unwrap(),
            sandbox.to_cbor().unwrap().to_bytes(),
            "knolo.infer.sandbox-report",
        ),
        (
            "api-report",
            "infer-api",
            api.to_bytes().unwrap(),
            api.to_cbor().unwrap().to_bytes(),
            "knolo.infer.api-report",
        ),
        (
            "cache-channel-report",
            "infer-cache-channel",
            cache_channel.to_bytes().unwrap(),
            cache_channel.to_cbor().unwrap().to_bytes(),
            "knolo.infer.cache-channel-report",
        ),
        (
            "equation-report",
            "infer-equation",
            equation.to_bytes().unwrap(),
            equation.to_cbor().unwrap().to_bytes(),
            "knolo.infer.equation-report",
        ),
        (
            "safe-error-report",
            "infer-safe-error",
            safe_error.to_bytes().unwrap(),
            safe_error.to_cbor().unwrap().to_bytes(),
            "knolo.infer.safe-error-report",
        ),
        (
            "redaction-report",
            "infer-redaction",
            redaction.to_bytes().unwrap(),
            redaction.to_cbor().unwrap().to_bytes(),
            "knolo.infer.redaction-report",
        ),
        (
            "curve-report",
            "infer-curve",
            curve.to_bytes().unwrap(),
            curve.to_cbor().unwrap().to_bytes(),
            "knolo.infer.curve-report",
        ),
        (
            "rollback-report",
            "infer-rollback",
            rollback.to_bytes().unwrap(),
            rollback.to_cbor().unwrap().to_bytes(),
            "knolo.infer.rollback-report",
        ),
        (
            "disconnect-report",
            "infer-disconnect",
            disconnect.to_bytes().unwrap(),
            disconnect.to_cbor().unwrap().to_bytes(),
            "knolo.infer.disconnect-report",
        ),
        (
            "receipt-store-report",
            "infer-receipt-store",
            receipt_store.to_bytes().unwrap(),
            receipt_store.to_cbor().unwrap().to_bytes(),
            "knolo.infer.receipt-store-report",
        ),
        (
            "base-report",
            "infer-base",
            base.to_bytes().unwrap(),
            base.to_cbor().unwrap().to_bytes(),
            "knolo.infer.base-report",
        ),
        (
            "disk-report",
            "infer-disk",
            disk.to_bytes().unwrap(),
            disk.to_cbor().unwrap().to_bytes(),
            "knolo.infer.disk-report",
        ),
        (
            "unload-report",
            "infer-unload",
            unload.to_bytes().unwrap(),
            unload.to_cbor().unwrap().to_bytes(),
            "knolo.infer.unload-report",
        ),
        (
            "restart-report",
            "infer-restart",
            restart.to_bytes().unwrap(),
            restart.to_cbor().unwrap().to_bytes(),
            "knolo.infer.restart-report",
        ),
        (
            "point-report",
            "infer-point",
            point.to_bytes().unwrap(),
            point.to_cbor().unwrap().to_bytes(),
            "knolo.infer.point-report",
        ),
        (
            "duplicate-report",
            "infer-duplicate",
            duplicate.to_bytes().unwrap(),
            duplicate.to_cbor().unwrap().to_bytes(),
            "knolo.infer.duplicate-report",
        ),
        (
            "concurrent-load-report",
            "infer-concurrent-load",
            concurrent_load.to_bytes().unwrap(),
            concurrent_load.to_cbor().unwrap().to_bytes(),
            "knolo.infer.concurrent-load-report",
        ),
        (
            "eviction-report",
            "infer-eviction",
            eviction.to_bytes().unwrap(),
            eviction.to_cbor().unwrap().to_bytes(),
            "knolo.infer.eviction-report",
        ),
        (
            "equality-report",
            "infer-equality",
            equality.to_bytes().unwrap(),
            equality.to_cbor().unwrap().to_bytes(),
            "knolo.infer.equality-report",
        ),
        (
            "oom-report",
            "infer-oom",
            oom.to_bytes().unwrap(),
            oom.to_cbor().unwrap().to_bytes(),
            "knolo.infer.oom-report",
        ),
        (
            "fault-report",
            "infer-fault",
            fault.to_bytes().unwrap(),
            fault.to_cbor().unwrap().to_bytes(),
            "knolo.infer.fault-report",
        ),
        (
            "timeout-report",
            "infer-timeout",
            timeout.to_bytes().unwrap(),
            timeout.to_cbor().unwrap().to_bytes(),
            "knolo.infer.timeout-report",
        ),
        (
            "worker-start-report",
            "infer-worker-start",
            worker_start.to_bytes().unwrap(),
            worker_start.to_cbor().unwrap().to_bytes(),
            "knolo.infer.worker-start-report",
        ),
        (
            "challenge-report",
            "infer-challenge",
            challenge.to_bytes().unwrap(),
            challenge.to_cbor().unwrap().to_bytes(),
            "knolo.infer.challenge-report",
        ),
        (
            "replay-environment-report",
            "infer-replay-environment",
            replay_environment.to_bytes().unwrap(),
            replay_environment.to_cbor().unwrap().to_bytes(),
            "knolo.infer.replay-environment-report",
        ),
        (
            "replay-output-report",
            "infer-replay-output",
            replay_output.to_bytes().unwrap(),
            replay_output.to_cbor().unwrap().to_bytes(),
            "knolo.infer.replay-output-report",
        ),
        (
            "worker-lost-report",
            "infer-worker-lost",
            worker_lost.to_bytes().unwrap(),
            worker_lost.to_cbor().unwrap().to_bytes(),
            "knolo.infer.worker-lost-report",
        ),
        (
            "draining-report",
            "infer-draining",
            draining.to_bytes().unwrap(),
            draining.to_cbor().unwrap().to_bytes(),
            "knolo.infer.draining-report",
        ),
        (
            "scalar-report",
            "infer-scalar",
            scalar.to_bytes().unwrap(),
            scalar.to_cbor().unwrap().to_bytes(),
            "knolo.infer.scalar-report",
        ),
        (
            "digest-mismatch-report",
            "infer-digest-mismatch",
            digest_mismatch.to_bytes().unwrap(),
            digest_mismatch.to_cbor().unwrap().to_bytes(),
            "knolo.infer.digest-mismatch-report",
        ),
        (
            "tokenizer-invalid-report",
            "infer-tokenizer-invalid",
            tokenizer_invalid.to_bytes().unwrap(),
            tokenizer_invalid.to_cbor().unwrap().to_bytes(),
            "knolo.infer.tokenizer-invalid-report",
        ),
        (
            "template-invalid-report",
            "infer-template-invalid",
            template_invalid.to_bytes().unwrap(),
            template_invalid.to_cbor().unwrap().to_bytes(),
            "knolo.infer.template-invalid-report",
        ),
        (
            "architecture-report",
            "infer-architecture",
            architecture.to_bytes().unwrap(),
            architecture.to_cbor().unwrap().to_bytes(),
            "knolo.infer.architecture-report",
        ),
        (
            "public-report",
            "infer-public",
            public_key.to_bytes().unwrap(),
            public_key.to_cbor().unwrap().to_bytes(),
            "knolo.infer.public-report",
        ),
        (
            "quantization-report",
            "infer-quantization",
            quantization.to_bytes().unwrap(),
            quantization.to_cbor().unwrap().to_bytes(),
            "knolo.infer.quantization-report",
        ),
        (
            "kernel-report",
            "infer-kernel",
            kernel_report.to_bytes().unwrap(),
            kernel_report.to_cbor().unwrap().to_bytes(),
            "knolo.infer.kernel-report",
        ),
        (
            "placement-refusal-report",
            "infer-placement-refusal",
            placement_refusal.to_bytes().unwrap(),
            placement_refusal.to_cbor().unwrap().to_bytes(),
            "knolo.infer.placement-refusal-report",
        ),
        (
            "memory-refusal-report",
            "infer-memory-refusal",
            memory_refusal.to_bytes().unwrap(),
            memory_refusal.to_cbor().unwrap().to_bytes(),
            "knolo.infer.memory-refusal-report",
        ),
        (
            "signature-check-report",
            "infer-signature-check",
            signature_check.to_bytes().unwrap(),
            signature_check.to_cbor().unwrap().to_bytes(),
            "knolo.infer.signature-check-report",
        ),
        (
            "context-limit-report",
            "infer-context-limit",
            context_limit.to_bytes().unwrap(),
            context_limit.to_cbor().unwrap().to_bytes(),
            "knolo.infer.context-limit-report",
        ),
        (
            "prompt-compilation-report",
            "infer-prompt-compilation",
            prompt_compilation.to_bytes().unwrap(),
            prompt_compilation.to_cbor().unwrap().to_bytes(),
            "knolo.infer.prompt-compilation-report",
        ),
        (
            "image-invalid-report",
            "infer-image-invalid",
            image_invalid.to_bytes().unwrap(),
            image_invalid.to_cbor().unwrap().to_bytes(),
            "knolo.infer.image-invalid-report",
        ),
        (
            "image-signature-report",
            "infer-image-signature",
            image_signature.to_bytes().unwrap(),
            image_signature.to_cbor().unwrap().to_bytes(),
            "knolo.infer.image-signature-report",
        ),
        (
            "cofactor-report",
            "infer-cofactor",
            cofactor.to_bytes().unwrap(),
            cofactor.to_cbor().unwrap().to_bytes(),
            "knolo.infer.cofactor-report",
        ),
        (
            "artifact-missing-report",
            "infer-artifact-missing",
            artifact_missing.to_bytes().unwrap(),
            artifact_missing.to_cbor().unwrap().to_bytes(),
            "knolo.infer.artifact-missing-report",
        ),
        (
            "receipt-required-report",
            "infer-receipt-required",
            receipt_required.to_bytes().unwrap(),
            receipt_required.to_cbor().unwrap().to_bytes(),
            "knolo.infer.receipt-required-report",
        ),
        (
            "backend-report",
            "infer-backend",
            backend_report.to_bytes().unwrap(),
            backend_report.to_cbor().unwrap().to_bytes(),
            "knolo.infer.backend-report",
        ),
        (
            "digest-invalid-report",
            "infer-digest-invalid",
            digest_invalid.to_bytes().unwrap(),
            digest_invalid.to_cbor().unwrap().to_bytes(),
            "knolo.infer.digest-invalid-report",
        ),
        (
            "receipt-sign-report",
            "infer-receipt-sign",
            receipt_sign.to_bytes().unwrap(),
            receipt_sign.to_cbor().unwrap().to_bytes(),
            "knolo.infer.receipt-sign-report",
        ),
        (
            "canonical-cbor-report",
            "infer-canonical-cbor",
            canonical_cbor.to_bytes().unwrap(),
            canonical_cbor.to_cbor().unwrap().to_bytes(),
            "knolo.infer.canonical-cbor-report",
        ),
        (
            "contract-invalid-report",
            "infer-contract-invalid",
            contract_invalid.to_bytes().unwrap(),
            contract_invalid.to_cbor().unwrap().to_bytes(),
            "knolo.infer.contract-invalid-report",
        ),
        (
            "grammar-refusal-report",
            "infer-grammar-refusal",
            grammar_refusal.to_bytes().unwrap(),
            grammar_refusal.to_cbor().unwrap().to_bytes(),
            "knolo.infer.grammar-refusal-report",
        ),
        (
            "tool-refusal-report",
            "infer-tool-refusal",
            tool_refusal.to_bytes().unwrap(),
            tool_refusal.to_cbor().unwrap().to_bytes(),
            "knolo.infer.tool-refusal-report",
        ),
        (
            "receipt-verify-report",
            "infer-receipt-verify",
            receipt_verify.to_bytes().unwrap(),
            receipt_verify.to_cbor().unwrap().to_bytes(),
            "knolo.infer.receipt-verify-report",
        ),
        (
            "speculative-report",
            "infer-speculative",
            speculative.to_bytes().unwrap(),
            speculative.to_cbor().unwrap().to_bytes(),
            "knolo.infer.speculative-report",
        ),
        (
            "graph-report",
            "infer-graph",
            graph.to_bytes().unwrap(),
            graph.to_cbor().unwrap().to_bytes(),
            "knolo.infer.graph-report",
        ),
        (
            "multi-model-report",
            "infer-multi-model",
            multi_model.to_bytes().unwrap(),
            multi_model.to_cbor().unwrap().to_bytes(),
            "knolo.infer.multi-model-report",
        ),
        (
            "secondary-report",
            "infer-secondary",
            secondary.to_bytes().unwrap(),
            secondary.to_cbor().unwrap().to_bytes(),
            "knolo.infer.secondary-report",
        ),
        (
            "domain-report",
            "infer-domain",
            domain_sep_report.to_bytes().unwrap(),
            domain_sep_report.to_cbor().unwrap().to_bytes(),
            "knolo.infer.domain-report",
        ),
        (
            "attention-report",
            "infer-attention",
            attention_report.to_bytes().unwrap(),
            attention_report.to_cbor().unwrap().to_bytes(),
            "knolo.infer.attention-report",
        ),
        (
            "moe-report",
            "infer-moe",
            moe_report.to_bytes().unwrap(),
            moe_report.to_cbor().unwrap().to_bytes(),
            "knolo.infer.moe-report",
        ),
        (
            "expert-placement-report",
            "infer-expert-placement",
            expert_placement_report.to_bytes().unwrap(),
            expert_placement_report.to_cbor().unwrap().to_bytes(),
            "knolo.infer.expert-placement-report",
        ),
        (
            "grouped-kernel-report",
            "infer-grouped-kernel",
            grouped_kernel_report.to_bytes().unwrap(),
            grouped_kernel_report.to_cbor().unwrap().to_bytes(),
            "knolo.infer.grouped-kernel-report",
        ),
        (
            "router-report",
            "infer-router",
            router_report.to_bytes().unwrap(),
            router_report.to_cbor().unwrap().to_bytes(),
            "knolo.infer.router-report",
        ),
        (
            "glm-report",
            "infer-glm",
            glm_report.to_bytes().unwrap(),
            glm_report.to_cbor().unwrap().to_bytes(),
            "knolo.infer.glm-report",
        ),
        (
            "workstation-report",
            "infer-workstation",
            workstation_report.to_bytes().unwrap(),
            workstation_report.to_cbor().unwrap().to_bytes(),
            "knolo.infer.workstation-report",
        ),
        (
            "mixed-placement-report",
            "infer-mixed-placement",
            mixed_placement_report.to_bytes().unwrap(),
            mixed_placement_report.to_cbor().unwrap().to_bytes(),
            "knolo.infer.mixed-placement-report",
        ),
        (
            "capacity-report",
            "infer-capacity",
            capacity_report.to_bytes().unwrap(),
            capacity_report.to_cbor().unwrap().to_bytes(),
            "knolo.infer.capacity-report",
        ),
    ];

    let mut rows = Vec::new();
    for (name, domain, bytes, payload, kind) in &items {
        roundtrip(bytes, kind);
        let root = digest_bytes(domain, payload).unwrap();
        assert_eq!(root.as_str().len(), 71, "{name}");
        rows.push(row(name, bytes, domain, payload));
    }
    let grammar_bytes = grammar.to_bytes().unwrap();
    let conversion_bytes = conversion.to_bytes().unwrap();
    let perplexity_bytes = perplexity.to_bytes().unwrap();
    let memory_bytes = memory.to_bytes().unwrap();
    let throughput_bytes = throughput.to_bytes().unwrap();
    let latency_bytes = latency.to_bytes().unwrap();
    let fuzz_bytes = fuzz.to_bytes().unwrap();
    let cancellation_bytes = cancellation.to_bytes().unwrap();
    let finalization_bytes = finalization.to_bytes().unwrap();
    let overhead_bytes = overhead.to_bytes().unwrap();
    let swap_bytes = swap.to_bytes().unwrap();
    let verification_bytes = verification.to_bytes().unwrap();
    let load_bytes = load.to_bytes().unwrap();
    let peak_bytes = peak.to_bytes().unwrap();
    let kv_bytes = kv.to_bytes().unwrap();
    let prefix_bytes = prefix.to_bytes().unwrap();
    let composition_bytes = composition.to_bytes().unwrap();
    let agent_bytes = agent.to_bytes().unwrap();
    let hub_bytes = hub.to_bytes().unwrap();
    let studio_bytes = studio.to_bytes().unwrap();
    let chain_bytes = chain.to_bytes().unwrap();
    let evidence_bytes = evidence_output.to_bytes().unwrap();
    let install_bytes = install.to_bytes().unwrap();
    let notice_bytes = notice.to_bytes().unwrap();
    let release_bytes = release.to_bytes().unwrap();
    let binary_bytes = binary.to_bytes().unwrap();
    let reproducible_bytes = reproducible.to_bytes().unwrap();
    let signature_bytes = signature.to_bytes().unwrap();
    let host_key_bytes = host_key.to_bytes().unwrap();
    let receipt_key_bytes = receipt_key.to_bytes().unwrap();
    let sandbox_bytes = sandbox.to_bytes().unwrap();
    let api_bytes = api.to_bytes().unwrap();
    let cache_channel_bytes = cache_channel.to_bytes().unwrap();
    let equation_bytes = equation.to_bytes().unwrap();
    let safe_error_bytes = safe_error.to_bytes().unwrap();
    let redaction_bytes = redaction.to_bytes().unwrap();
    let curve_bytes = curve.to_bytes().unwrap();
    let rollback_bytes = rollback.to_bytes().unwrap();
    let disconnect_bytes = disconnect.to_bytes().unwrap();
    let receipt_store_bytes = receipt_store.to_bytes().unwrap();
    let base_bytes = base.to_bytes().unwrap();
    let disk_bytes = disk.to_bytes().unwrap();
    let unload_bytes = unload.to_bytes().unwrap();
    let restart_bytes = restart.to_bytes().unwrap();
    let point_bytes = point.to_bytes().unwrap();
    let duplicate_bytes = duplicate.to_bytes().unwrap();
    let concurrent_load_bytes = concurrent_load.to_bytes().unwrap();
    let eviction_bytes = eviction.to_bytes().unwrap();
    let equality_bytes = equality.to_bytes().unwrap();
    let oom_bytes = oom.to_bytes().unwrap();
    let fault_bytes = fault.to_bytes().unwrap();
    let timeout_bytes = timeout.to_bytes().unwrap();
    let worker_start_bytes = worker_start.to_bytes().unwrap();
    let challenge_bytes = challenge.to_bytes().unwrap();
    let replay_environment_bytes = replay_environment.to_bytes().unwrap();
    let replay_output_bytes = replay_output.to_bytes().unwrap();
    let worker_lost_bytes = worker_lost.to_bytes().unwrap();
    let draining_bytes = draining.to_bytes().unwrap();
    let scalar_bytes = scalar.to_bytes().unwrap();
    let digest_mismatch_bytes = digest_mismatch.to_bytes().unwrap();
    let tokenizer_invalid_bytes = tokenizer_invalid.to_bytes().unwrap();
    let template_invalid_bytes = template_invalid.to_bytes().unwrap();
    let architecture_bytes = architecture.to_bytes().unwrap();
    rows.push(bad(
        "architecture-unknown-field",
        &with_extra(&architecture_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "architecture-bad-version",
        &bump_version(&architecture_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "template-invalid-unknown-field",
        &with_extra(&template_invalid_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "template-invalid-bad-version",
        &bump_version(&template_invalid_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "tokenizer-invalid-unknown-field",
        &with_extra(&tokenizer_invalid_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "tokenizer-invalid-bad-version",
        &bump_version(&tokenizer_invalid_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "digest-mismatch-unknown-field",
        &with_extra(&digest_mismatch_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "digest-mismatch-bad-version",
        &bump_version(&digest_mismatch_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "scalar-unknown-field",
        &with_extra(&scalar_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "scalar-bad-version",
        &bump_version(&scalar_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "draining-unknown-field",
        &with_extra(&draining_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "draining-bad-version",
        &bump_version(&draining_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "worker-lost-unknown-field",
        &with_extra(&worker_lost_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "worker-lost-bad-version",
        &bump_version(&worker_lost_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "replay-output-unknown-field",
        &with_extra(&replay_output_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "replay-output-bad-version",
        &bump_version(&replay_output_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "replay-environment-unknown-field",
        &with_extra(&replay_environment_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "replay-environment-bad-version",
        &bump_version(&replay_environment_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "challenge-unknown-field",
        &with_extra(&challenge_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "challenge-bad-version",
        &bump_version(&challenge_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "worker-start-unknown-field",
        &with_extra(&worker_start_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "worker-start-bad-version",
        &bump_version(&worker_start_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "timeout-unknown-field",
        &with_extra(&timeout_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "timeout-bad-version",
        &bump_version(&timeout_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "fault-unknown-field",
        &with_extra(&fault_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "fault-bad-version",
        &bump_version(&fault_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "oom-unknown-field",
        &with_extra(&oom_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "oom-bad-version",
        &bump_version(&oom_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "equality-unknown-field",
        &with_extra(&equality_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "equality-bad-version",
        &bump_version(&equality_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "eviction-unknown-field",
        &with_extra(&eviction_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "eviction-bad-version",
        &bump_version(&eviction_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "concurrent-load-unknown-field",
        &with_extra(&concurrent_load_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "concurrent-load-bad-version",
        &bump_version(&concurrent_load_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "duplicate-unknown-field",
        &with_extra(&duplicate_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "duplicate-bad-version",
        &bump_version(&duplicate_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "point-unknown-field",
        &with_extra(&point_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "point-bad-version",
        &bump_version(&point_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "restart-unknown-field",
        &with_extra(&restart_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "restart-bad-version",
        &bump_version(&restart_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "unload-unknown-field",
        &with_extra(&unload_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "unload-bad-version",
        &bump_version(&unload_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "disk-unknown-field",
        &with_extra(&disk_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "disk-bad-version",
        &bump_version(&disk_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "base-unknown-field",
        &with_extra(&base_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "base-bad-version",
        &bump_version(&base_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "receipt-store-unknown-field",
        &with_extra(&receipt_store_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "receipt-store-bad-version",
        &bump_version(&receipt_store_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "disconnect-unknown-field",
        &with_extra(&disconnect_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "disconnect-bad-version",
        &bump_version(&disconnect_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "rollback-unknown-field",
        &with_extra(&rollback_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "rollback-bad-version",
        &bump_version(&rollback_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "curve-unknown-field",
        &with_extra(&curve_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "curve-bad-version",
        &bump_version(&curve_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "redaction-unknown-field",
        &with_extra(&redaction_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "redaction-bad-version",
        &bump_version(&redaction_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "safe-error-unknown-field",
        &with_extra(&safe_error_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "safe-error-bad-version",
        &bump_version(&safe_error_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "equation-unknown-field",
        &with_extra(&equation_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "equation-bad-version",
        &bump_version(&equation_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "cache-channel-unknown-field",
        &with_extra(&cache_channel_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "cache-channel-bad-version",
        &bump_version(&cache_channel_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "api-unknown-field",
        &with_extra(&api_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "api-bad-version",
        &bump_version(&api_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "sandbox-unknown-field",
        &with_extra(&sandbox_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "sandbox-bad-version",
        &bump_version(&sandbox_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "receipt-key-unknown-field",
        &with_extra(&receipt_key_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "receipt-key-bad-version",
        &bump_version(&receipt_key_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "host-key-unknown-field",
        &with_extra(&host_key_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "host-key-bad-version",
        &bump_version(&host_key_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "signature-unknown-field",
        &with_extra(&signature_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "signature-bad-version",
        &bump_version(&signature_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "reproducible-unknown-field",
        &with_extra(&reproducible_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "reproducible-bad-version",
        &bump_version(&reproducible_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "binary-unknown-field",
        &with_extra(&binary_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "binary-bad-version",
        &bump_version(&binary_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "release-unknown-field",
        &with_extra(&release_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "release-bad-version",
        &bump_version(&release_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "notice-unknown-field",
        &with_extra(&notice_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "notice-bad-version",
        &bump_version(&notice_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "install-unknown-field",
        &with_extra(&install_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "install-bad-version",
        &bump_version(&install_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "evidence-output-unknown-field",
        &with_extra(&evidence_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "evidence-output-bad-version",
        &bump_version(&evidence_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "chain-unknown-field",
        &with_extra(&chain_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "chain-bad-version",
        &bump_version(&chain_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "studio-unknown-field",
        &with_extra(&studio_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "studio-bad-version",
        &bump_version(&studio_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "hub-unknown-field",
        &with_extra(&hub_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "hub-bad-version",
        &bump_version(&hub_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "agent-effect-unknown-field",
        &with_extra(&agent_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "agent-effect-bad-version",
        &bump_version(&agent_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "composition-unknown-field",
        &with_extra(&composition_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "composition-bad-version",
        &bump_version(&composition_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "prefix-unknown-field",
        &with_extra(&prefix_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "prefix-bad-version",
        &bump_version(&prefix_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "kv-unknown-field",
        &with_extra(&kv_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "kv-bad-version",
        &bump_version(&kv_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "peak-unknown-field",
        &with_extra(&peak_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "peak-bad-version",
        &bump_version(&peak_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "load-unknown-field",
        &with_extra(&load_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "load-bad-version",
        &bump_version(&load_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "verification-unknown-field",
        &with_extra(&verification_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "verification-bad-version",
        &bump_version(&verification_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "swap-unknown-field",
        &with_extra(&swap_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "swap-bad-version",
        &bump_version(&swap_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "overhead-unknown-field",
        &with_extra(&overhead_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "overhead-bad-version",
        &bump_version(&overhead_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "finalization-unknown-field",
        &with_extra(&finalization_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "finalization-bad-version",
        &bump_version(&finalization_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "cancellation-unknown-field",
        &with_extra(&cancellation_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "cancellation-bad-version",
        &bump_version(&cancellation_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "fuzz-unknown-field",
        &with_extra(&fuzz_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "fuzz-bad-version",
        &bump_version(&fuzz_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "latency-unknown-field",
        &with_extra(&latency_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "latency-bad-version",
        &bump_version(&latency_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "throughput-unknown-field",
        &with_extra(&throughput_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "throughput-bad-version",
        &bump_version(&throughput_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "memory-unknown-field",
        &with_extra(&memory_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "memory-bad-version",
        &bump_version(&memory_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "perplexity-unknown-field",
        &with_extra(&perplexity_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "perplexity-bad-version",
        &bump_version(&perplexity_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "conversion-unknown-field",
        &with_extra(&conversion_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "conversion-bad-version",
        &bump_version(&conversion_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "grammar-unknown-field",
        &with_extra(&grammar_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad(
        "grammar-bad-version",
        &bump_version(&grammar_bytes),
        "CONTRACT_INVALID",
    ));
    rows.push(bad("truncated-cbor", &[0x18], "CANONICAL_CBOR_INVALID"));
    rows.push(bad(
        "nonminimal-int",
        &[0x18, 0x00],
        "CANONICAL_CBOR_INVALID",
    ));
    rows.push(bad(
        "indefinite-array",
        &[0x9f, 0xff],
        "CANONICAL_CBOR_INVALID",
    ));

    let primitives = [
        ("null", CborValue::Null),
        ("false", CborValue::Bool(false)),
        ("true", CborValue::Bool(true)),
        ("int-0", CborValue::Integer(0)),
        ("int-24", CborValue::Integer(24)),
        ("int-neg-25", CborValue::Integer(-25)),
        ("text", CborValue::Text("knolo".into())),
        ("bytes", CborValue::Bytes(vec![0, 255])),
    ];
    for (name, value) in primitives {
        let bytes = value.to_bytes();
        assert_eq!(decode_canonical(&bytes).unwrap(), value);
        rows.push(row(name, &bytes, "infer-conformance", &bytes));
    }

    write_vectors(&rows);
}

#[test]
fn memory_estimate_rejects_a_result_other_than_within_bounds() {
    let mut report = MemoryEstimateReportV1 {
        placement_root: raw(b"placement-plan"),
        estimator_build_root: raw(b"estimator"),
        declared_weight_bytes: 100,
        declared_kv_bytes: 40,
        declared_workspace_bytes: 10,
        declared_staging_bytes: 0,
        declared_overhead_bytes: 0,
        declared_margin_bytes: 1,
        declared_total_bytes: 151,
        measured_weight_bytes: 100,
        measured_kv_bytes: 40,
        measured_workspace_bytes: 10,
        measured_staging_bytes: 0,
        measured_overhead_bytes: 0,
        measured_total_bytes: 150,
        headroom_bytes: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    let err = report.to_bytes().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(err.message.contains("validationResult"), "{err}");
    report.validation_result = "within-bounds".into();
    report.measured_total_bytes = 149;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("measured bytes do not match"), "{err}");
    report.measured_total_bytes = 150;
    report.declared_total_bytes = 150;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("declared bytes do not match"), "{err}");
    report.declared_total_bytes = 151;
    report.measured_kv_bytes = 41;
    report.measured_total_bytes = 151;
    report.headroom_bytes = 0;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("measured kv bytes exceed"), "{err}");
    report.measured_kv_bytes = 40;
    report.measured_total_bytes = 150;
    report.headroom_bytes = 0;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("memory headroom does not match"),
        "{err}"
    );
}

#[test]
fn throughput_report_rejects_a_rate_that_does_not_match() {
    let mut report = ThroughputReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        prompt_token_root: raw(b"prompt-tokens"),
        output_token_root: raw(b"output-tokens"),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        prefill_tokens: 1,
        decode_tokens: 2,
        request_count: 1,
        prefill_nanos: 3,
        decode_nanos: 2,
        prefill_tokens_per_second_micros: tokens_per_second_micros(1, 3).unwrap(),
        decode_tokens_per_second_micros: tokens_per_second_micros(2, 2).unwrap(),
        requests_per_second_micros: tokens_per_second_micros(1, 5).unwrap(),
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    assert_eq!(report.prefill_tokens_per_second_micros, 333_333_333_333_333);
    assert_eq!(report.requests_per_second_micros, 200_000_000_000_000);
    report.validation_result = "within-bounds".into();
    let err = report.to_bytes().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(err.message.contains("validationResult"), "{err}");
    report.validation_result = "recorded".into();
    report.execution_mode = "throughput".into();
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("executionMode"), "{err}");
    report.execution_mode = "pinned".into();
    report.prefill_tokens_per_second_micros += 1;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("prefill throughput does not match"),
        "{err}"
    );
    report.prefill_tokens_per_second_micros -= 1;
    report.prefill_tokens = 16;
    report.decode_tokens = 1;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("micro prompt does not fit"), "{err}");
}

#[test]
fn latency_report_rejects_a_duration_that_does_not_match() {
    let mut report = LatencyReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        prompt_token_root: raw(b"prompt-tokens"),
        output_token_root: raw(b"output-tokens"),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        prefill_tokens: 1,
        decode_tokens: 2,
        request_count: 1,
        prefill_nanos: 3,
        decode_nanos: 5,
        time_to_first_token_nanos: 3,
        time_per_output_token_nanos: 2,
        request_latency_nanos: 8,
        latency_p50_nanos: 8,
        latency_p95_nanos: 8,
        latency_p99_nanos: 8,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    assert_eq!(
        time_per_output_token_nanos(2, 5).unwrap(),
        report.time_per_output_token_nanos
    );
    assert_eq!(request_latency_nanos(3, 5).unwrap(), 8);
    report.validation_result = "within-bounds".into();
    let err = report.to_bytes().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(err.message.contains("validationResult"), "{err}");
    report.validation_result = "recorded".into();
    report.execution_mode = "throughput".into();
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("executionMode"), "{err}");
    report.execution_mode = "pinned".into();
    report.time_to_first_token_nanos += 1;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("time to first token does not match"),
        "{err}"
    );
    report.time_to_first_token_nanos -= 1;
    report.time_per_output_token_nanos = 3;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("time per output token does not match"),
        "{err}"
    );
    report.time_per_output_token_nanos = 2;
    report.latency_p95_nanos = 7;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("latency p95 does not match"), "{err}");
    report.latency_p95_nanos = 8;
    report.decode_nanos = 1;
    report.time_per_output_token_nanos = 0;
    report.request_latency_nanos = 4;
    report.latency_p50_nanos = 4;
    report.latency_p95_nanos = 4;
    report.latency_p99_nanos = 4;
    report.to_bytes().unwrap();
    report.prefill_tokens = 16;
    report.decode_tokens = 1;
    report.time_per_output_token_nanos = 1;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("micro prompt does not fit"), "{err}");
}

#[test]
fn fuzz_report_rejects_an_accepted_corruption() {
    let mut report = FuzzReportV1 {
        engine_build_root: raw(b"engine"),
        corpus_root: raw(b"corpus"),
        target_count: 9,
        seed_count: 9,
        mutation_count: 54,
        rejected_count: 53,
        distinguished_count: 0,
        accepted_count: 1,
        validation_result: "fail-closed".into(),
        extensions: ext(),
    };
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("corruption was accepted"), "{err}");
    report.accepted_count = 0;
    report.rejected_count = 53;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("fuzz case count does not match"),
        "{err}"
    );
    report.rejected_count = 54;
    report.validation_result = "recorded".into();
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("validationResult"), "{err}");
}

#[test]
fn cancellation_report_rejects_a_latency_that_does_not_match() {
    let mut report = CancellationReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        prompt_token_root: raw(b"prompt"),
        output_token_root: raw(b"output"),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        cancel_stage: "decode".into(),
        prefill_tokens: 1,
        decode_tokens: 2,
        requested_nanos: 4,
        terminal_nanos: 9,
        cancellation_latency_nanos: 5,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    assert_eq!(
        cancellation_latency_nanos(4, 9).unwrap(),
        report.cancellation_latency_nanos
    );
    report.to_bytes().unwrap();
    report.cancellation_latency_nanos = 4;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("cancellation latency does not match"),
        "{err}"
    );
    report.cancellation_latency_nanos = 5;
    report.cancel_stage = "prefill".into();
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("prefill cancel has output tokens"),
        "{err}"
    );
    report.cancel_stage = "later".into();
    report.decode_tokens = 0;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("cancelStage"), "{err}");
}

#[test]
fn finalization_report_rejects_a_latency_that_does_not_match() {
    let mut report = FinalizationReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        prompt_token_root: raw(b"prompt"),
        output_token_root: raw(b"output"),
        receipt_root: raw(b"receipt"),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        finish_reason: "stop".into(),
        prefill_tokens: 1,
        decode_tokens: 1,
        terminal_nanos: 5_000,
        finalized_nanos: 5_180,
        finalization_latency_nanos: 180,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    assert_eq!(
        receipt_finalization_nanos(5_000, 5_180).unwrap(),
        report.finalization_latency_nanos
    );
    report.to_bytes().unwrap();
    report.finalization_latency_nanos = 179;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("receipt finalization latency does not match"),
        "{err}"
    );
    report.finalization_latency_nanos = 180;
    report.finish_reason = "length".into();
    report.decode_tokens = 0;
    report.to_bytes().unwrap();
    report.finish_reason = "stop".into();
    report.decode_tokens = 0;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("a stop receipt has no output tokens"),
        "{err}"
    );
}

#[test]
fn overhead_report_rejects_a_sum_that_does_not_match() {
    let mut report = OverheadReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        prompt_token_root: raw(b"prompt"),
        output_token_root: raw(b"output"),
        receipt_root: raw(b"receipt"),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        finish_reason: "length".into(),
        prefill_tokens: 2,
        decode_tokens: 0,
        accepted_write_nanos: 120,
        receipt_write_nanos: 40,
        receipt_overhead_nanos: 160,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    assert_eq!(
        receipt_overhead_nanos(120, 40).unwrap(),
        report.receipt_overhead_nanos
    );
    report.to_bytes().unwrap();
    report.receipt_overhead_nanos = 159;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("receipt overhead does not match"),
        "{err}"
    );
    report.receipt_overhead_nanos = 160;
    report.accepted_write_nanos = 0;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("accepted write time is zero"), "{err}");
    report.accepted_write_nanos = u64::MAX;
    report.receipt_write_nanos = 1;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("receipt overhead overflows"), "{err}");
}

#[test]
fn swap_report_rejects_a_sum_that_does_not_match() {
    let mut report = SwapReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        resident_model_image_root: raw(b"resident-image"),
        resident_artifact_root: raw(b"resident-artifact"),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        unload_nanos: 50,
        incoming_verification_nanos: 30,
        incoming_load_nanos: 80,
        model_swap_nanos: 160,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    assert_eq!(
        model_swap_nanos(50, 30, 80).unwrap(),
        report.model_swap_nanos
    );
    report.to_bytes().unwrap();
    report.model_swap_nanos = 159;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("model swap time does not match"),
        "{err}"
    );
    report.model_swap_nanos = 160;
    report.resident_model_image_root = report.model_image_root.clone();
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("a swap replaces a different model image"),
        "{err}"
    );
    report.resident_model_image_root = raw(b"resident-image");
    report.unload_nanos = u64::MAX;
    report.incoming_verification_nanos = 1;
    report.incoming_load_nanos = 1;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("model swap time overflows"), "{err}");
}

#[test]
fn verification_report_rejects_a_count_above_the_parser_cap() {
    let mut report = VerificationReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        verified_bytes: 4096,
        model_verification_nanos: 40,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    report.to_bytes().unwrap();
    report.verified_bytes = 0;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("verified byte count is zero"), "{err}");
    report.verified_bytes = MAX_VERIFIED_BYTES + 1;
    let err = report.to_bytes().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message.contains("verified bytes exceed the parser cap"),
        "{err}"
    );
}

#[test]
fn peak_report_rejects_a_host_vram_count_and_a_cap_breach() {
    let mut report = PeakReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        device: "cpu".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        peak_ram_bytes: 4096,
        peak_vram_bytes: 0,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    report.to_bytes().unwrap();
    report.peak_ram_bytes = 0;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("peak ram is zero"), "{err}");
    report.peak_ram_bytes = MAX_PEAK_BYTES + 1;
    let err = report.to_bytes().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(err.message.contains("peak ram exceeds 64 MiB"), "{err}");
    report.peak_ram_bytes = 4096;
    report.peak_vram_bytes = 16;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("cpu placement has no device memory"),
        "{err}"
    );
    report.device = "slot-0".into();
    report.peak_vram_bytes = 0;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("slot-0 placement records device memory"),
        "{err}"
    );
    report.peak_vram_bytes = MAX_PEAK_BYTES + 1;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("peak vram exceeds 64 MiB"), "{err}");
}

#[test]
fn kv_report_rejects_a_utilization_that_does_not_match() {
    let mut report = KvReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        page_total: MICRO_KV_PAGES,
        page_size_tokens: MICRO_KV_PAGE_TOKENS,
        peak_pages: 1,
        peak_tokens: 16,
        utilization_millionths: kv_utilization_millionths(16, 1).unwrap(),
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    report.to_bytes().unwrap();
    report.utilization_millionths = 1;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("kv utilization does not match"),
        "{err}"
    );
    report.utilization_millionths = 125_000;
    report.peak_tokens = 0;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("kv peak tokens are zero"), "{err}");
    report.peak_tokens = 17;
    let err = report.to_bytes().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message
            .contains("kv peak tokens exceed the micro context"),
        "{err}"
    );
    report.peak_tokens = 16;
    report.peak_pages = 2;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("the micro fixture occupies one page"),
        "{err}"
    );
    report.peak_pages = 1;
    report.page_total = 7;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("the kv report is the micro fixture"),
        "{err}"
    );
}

#[test]
fn prefix_report_rejects_a_nonzero_count_while_the_cache_is_off() {
    let mut report = PrefixReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        lookup_count: 0,
        hit_count: 0,
        miss_count: 0,
        reused_tokens: 0,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    report.to_bytes().unwrap();
    report.lookup_count = 1;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("prefix lookups are zero while the cache is off"),
        "{err}"
    );
    report.lookup_count = 0;
    report.reused_tokens = 1;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("reused tokens are zero while the cache is off"),
        "{err}"
    );
}

#[test]
fn llama_report_rejects_a_native_badge_and_a_repeated_build() {
    let mut report = sample_llama();
    report.to_bytes().unwrap();
    report.reference_verification_class = "native-verified".into();
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("a compatibility backend is not native-verified"),
        "{err}"
    );
    report.reference_verification_class = "sidecar-artifact-verified".into();
    report.reference_build_root = report.engine_build_root.clone();
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("the reference build is the engine build"),
        "{err}"
    );
    report.reference_build_root = raw(b"reference-build");
    report.context_tokens = 32;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("the llama comparison is the micro fixture"),
        "{err}"
    );
}

#[test]
fn recipe_report_rejects_a_blessed_mark_that_repeats_a_receipt() {
    let mut report = RecipeReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        adapter_id: "knolo.micro.v1".into(),
        conformance_root: raw(b"conformance"),
        security_root: raw(b"security"),
        stability_root: raw(b"stability"),
        benchmark_root: raw(b"benchmark"),
        support_level: "blessed".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    report.to_bytes().unwrap();
    report.benchmark_root = report.security_root.clone();
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("blessed recipe repeats a receipt"),
        "{err}"
    );
    report.benchmark_root = raw(b"benchmark");
    report.adapter_id = "knolo.llama.v1".into();
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("the recipe report is the micro fixture"),
        "{err}"
    );
}

#[test]
fn product_reports_reject_a_decision_or_mark_that_does_not_match() {
    let mut agent = AgentEffectReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        model_runtime_root: raw(b"runtime"),
        required_model_runtime_root: raw(b"runtime"),
        required_artifact_root: raw(b"artifact"),
        knowledge_image_root: raw(b"knowledge"),
        required_knowledge_image_root: raw(b"knowledge"),
        backend: "reference-f32".into(),
        assurance: "same_build_replayable".into(),
        required_assurance: "same_build_replayable".into(),
        execution_mode: "isolated-replay".into(),
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
        decision: "allow".into(),
        reason: "accepted".into(),
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    agent.to_bytes().unwrap();
    agent.decision = "deny".into();
    agent.reason = "accepted".into();
    let err = agent.to_bytes().unwrap_err();
    assert!(
        err.message.contains("agent effect decision does not match"),
        "{err}"
    );

    let mut hub = HubReportV1 {
        publisher: "knolo".into(),
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        adapter_id: "knolo.micro.v1".into(),
        quantization: "f32".into(),
        license_id: "apache-2.0".into(),
        source_provider: "local".into(),
        conformance_root: raw(b"conformance"),
        benchmark_root: raw(b"benchmark"),
        support_level: "blessed".into(),
        greedy_token_parity: true,
        distribution: "active".into(),
        native_supported: true,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    hub.to_bytes().unwrap();
    hub.native_supported = false;
    let err = hub.to_bytes().unwrap_err();
    assert!(
        err.message.contains("native support does not match"),
        "{err}"
    );
    hub.native_supported = true;
    hub.distribution = "yanked".into();
    let err = hub.to_bytes().unwrap_err();
    assert!(
        err.message.contains("native support does not match"),
        "{err}"
    );
    hub.greedy_token_parity = false;
    let err = hub.to_bytes().unwrap_err();
    assert!(
        err.message.contains("fast execution is not blessed"),
        "{err}"
    );

    let mut studio = StudioReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        receipt_root: raw(b"receipt"),
        prompt_token_root: raw(b"prompt"),
        output_token_root: raw(b"output"),
        output_text_root: raw(b"output-text"),
        knowledge_image_root: raw(b"knowledge"),
        evidence_root: raw(b"evidence"),
        prompt_token_count: 2,
        output_token_count: 1,
        finish_reason: "stop".into(),
        assurance: "compatibility".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    studio.to_bytes().unwrap();
    studio.extensions.insert(
        "knolo.prompt".into(),
        CborValue::Text("secret prompt".into()),
    );
    let err = studio.to_bytes().unwrap_err();
    assert!(
        err.message.contains("the studio view extensions are empty"),
        "{err}"
    );

    let mut chain = ChainReportV1 {
        model_runtime_root: raw(b"runtime"),
        artifact_root: raw(b"artifact"),
        engine_build_root: raw(b"engine"),
        kernel_bundle_root: raw(b"kernel"),
        placement_root: raw(b"placement"),
        prompt_token_root: raw(b"prompt"),
        knowledge_image_root: raw(b"knowledge-image"),
        knowledge_commit_root: raw(b"knowledge-commit"),
        query_receipt_root: raw(b"query"),
        query_receipt_count: 1,
        reflex_receipt_root: raw(b"reflex"),
        reflex_receipt_count: 1,
        receipt_root: raw(b"receipt"),
        effect_root: raw(b"effect"),
        output_token_root: raw(b"output"),
        output_text_root: raw(b"output-text"),
        prompt_token_count: 2,
        output_token_count: 1,
        finish_reason: "stop".into(),
        assurance: "compatibility".into(),
        link_count: 5,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    chain.to_bytes().unwrap();
    chain.query_receipt_root = chain.reflex_receipt_root.clone();
    let err = chain.to_bytes().unwrap_err();
    assert!(
        err.message.contains("the receipt chain repeats a link"),
        "{err}"
    );

    let mut evidence = EvidenceOutputReportV1 {
        evidence_root: raw(b"evidence"),
        knowledge_image_root: raw(b"knowledge-image"),
        query_receipt_root: raw(b"query"),
        reflex_receipt_root: raw(b"reflex"),
        receipt_root: raw(b"receipt"),
        chain_root: raw(b"chain"),
        output_token_root: raw(b"output"),
        output_text_root: raw(b"output-text"),
        prompt_token_count: 2,
        output_token_count: 1,
        finish_reason: "stop".into(),
        assurance: "compatibility".into(),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    evidence.to_bytes().unwrap();
    evidence.chain_root = evidence.receipt_root.clone();
    let err = evidence.to_bytes().unwrap_err();
    assert!(
        err.message.contains("the chain repeats the receipt"),
        "{err}"
    );

    let mut install = InstallReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        tokenizer_root: raw(b"tokenizer"),
        template_root: raw(b"template"),
        publisher: "knolo".into(),
        license_id: "apache-2.0".into(),
        source_provider: "local".into(),
        artifact_count: 4,
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    install.to_bytes().unwrap();
    install.source_provider = "huggingface".into();
    let err = install.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("the hub install does not download weights"),
        "{err}"
    );

    let mut notice = NoticeReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        notice_root: raw(b"notice"),
        sbom_root: raw(b"sbom"),
        component_count: 1,
        feature_set: "cpu".into(),
        candle_named: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    notice.to_bytes().unwrap();
    notice.candle_named = true;
    let err = notice.to_bytes().unwrap_err();
    assert!(err.message.contains("the cpu notice names candle"), "{err}");

    let mut stored = BinaryReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        supervisor_name: "knolo-infer".into(),
        supervisor_hash: raw(b"supervisor"),
        supervisor_bytes: 1024,
        worker_name: "knolo-infer-worker".into(),
        worker_hash: raw(b"worker"),
        worker_bytes: 2048,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    stored.to_bytes().unwrap();
    stored.worker_bytes = MAX_BINARY_BYTES + 1;
    let err = stored.to_bytes().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message.contains("the worker binary exceeds 64 MiB"),
        "{err}"
    );
}

fn sample_llama() -> LlamaReportV1 {
    let artifact = raw(b"artifact");
    let tokenizer = raw(b"tokenizer");
    let template = raw(b"template");
    let sampler_root = raw(b"sampler");
    let hardware_root = raw(b"hardware");
    let prompts = raw(b"prompts");
    let outputs = raw(b"outputs");
    LlamaReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: artifact.clone(),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
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
        sampler_root: sampler_root.clone(),
        hardware_probe_root: hardware_root.clone(),
        prompt_distribution_root: prompts.clone(),
        output_distribution_root: outputs.clone(),
        reference_runtime: "llama.cpp".into(),
        reference_family: "gguf".into(),
        reference_build_root: raw(b"reference-build"),
        reference_artifact_root: artifact,
        reference_quantization: "f32".into(),
        reference_tokenizer_root: tokenizer,
        reference_template_root: template,
        reference_sampler_root: sampler_root,
        reference_hardware_root: hardware_root,
        reference_prompt_distribution_root: prompts,
        reference_output_distribution_root: outputs,
        reference_verification_class: "sidecar-artifact-verified".into(),
        validation_result: "recorded".into(),
        extensions: ext(),
    }
}

#[test]
fn load_report_rejects_a_zero_duration() {
    let mut report = LoadReportV1 {
        model_image_root: raw(b"model-image"),
        artifact_root: raw(b"artifact"),
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement"),
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        model_load_nanos: 80,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    report.to_bytes().unwrap();
    report.model_load_nanos = 0;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("model load time is zero"), "{err}");
}

#[test]
fn perplexity_report_rejects_a_result_other_than_recorded() {
    let mut report = PerplexityReportV1 {
        reference_logit_root: raw(b"reference-logits"),
        candidate_logit_root: raw(b"candidate-logits"),
        target_token_root: raw(b"targets"),
        reporter_build_root: raw(b"reporter"),
        token_count: 1,
        reference_perplexity_micros: 3_718_282,
        candidate_perplexity_micros: 2_000_000,
        perplexity_delta_micros: -1_718_282,
        greedy_parity: false,
        target_accuracy_millionths: 1_000_000,
        max_abs_logit_delta_millionths: 1_000_000,
        validation_result: "matched".into(),
        extensions: ext(),
    };
    let err = report.to_bytes().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(err.message.contains("validationResult"), "{err}");
    report.validation_result = "recorded".into();
    report.perplexity_delta_micros = 0;
    let err = report.to_bytes().unwrap_err();
    assert!(
        err.message.contains("perplexity delta does not match"),
        "{err}"
    );
    report.perplexity_delta_micros = -1_718_282;
    report.token_count = 0;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("token count is zero"), "{err}");
    report.token_count = 1;
    report.target_accuracy_millionths = 1_000_001;
    let err = report.to_bytes().unwrap_err();
    assert!(err.message.contains("target accuracy"), "{err}");
}

#[test]
fn preflight_reports_reject_a_crossed_flag() {
    let mut scalar = ScalarReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        message_root: raw(b"message"),
        scalar_status: "reduced".into(),
        scalar_reduced: true,
        public_multiplied: false,
        public_key_bytes: ED25519_PUBLIC_KEY_BYTES,
        signature_bytes: ED25519_SIGNATURE_BYTES,
        scalar_bytes: ED25519_SCALAR_BYTES,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    scalar.to_bytes().unwrap();
    scalar.public_multiplied = true;
    let err = scalar.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("the public-key multiplication stays uncomputed"),
        "{err}"
    );

    let mut mismatch = DigestMismatchReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        artifact_root: raw(b"artifact"),
        mismatch: "size".into(),
        code: "MODEL_DIGEST_MISMATCH".into(),
        retryable: false,
        header_parsed: false,
        body_read: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    mismatch.to_bytes().unwrap();
    mismatch.body_read = true;
    let err = mismatch.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("a size mismatch does not read the body"),
        "{err}"
    );

    let mut tokenizer = TokenizerInvalidReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        tokenizer_root: raw(b"tokenizer"),
        failure: "root".into(),
        code: "TOKENIZER_INVALID".into(),
        retryable: false,
        tokenizer_parsed: false,
        template_rendered: false,
        prompt_compiled: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    tokenizer.to_bytes().unwrap();
    tokenizer.tokenizer_parsed = true;
    let err = tokenizer.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("a root mismatch does not parse the tokenizer"),
        "{err}"
    );

    let mut template = TemplateInvalidReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        template_root: raw(b"template"),
        failure: "cap".into(),
        code: "TEMPLATE_INVALID".into(),
        retryable: false,
        template_rendered: false,
        tokenizer_parsed: false,
        prompt_compiled: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    template.to_bytes().unwrap();
    template.template_rendered = true;
    let err = template.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("a template failure does not render the prompt"),
        "{err}"
    );

    let mut architecture = ArchitectureReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        rejected_adapter: "knolo.llama.v1".into(),
        code: "UNSUPPORTED_ARCHITECTURE".into(),
        retryable: false,
        weights_opened: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    architecture.to_bytes().unwrap();
    architecture.rejected_adapter = "knolo.micro.v1".into();
    let err = architecture.to_bytes().unwrap_err();
    assert!(
        err.message.contains("the micro adapter is compiled in"),
        "{err}"
    );

    let mut public_key = PublicReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        message_root: raw(b"message"),
        public_status: "multiplied".into(),
        public_multiplied: true,
        signature_checked: false,
        public_key_bytes: ED25519_PUBLIC_KEY_BYTES,
        signature_bytes: ED25519_SIGNATURE_BYTES,
        scalar_bytes: ED25519_SCALAR_BYTES,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    public_key.to_bytes().unwrap();
    public_key.signature_checked = true;
    let err = public_key.to_bytes().unwrap_err();
    assert!(
        err.message.contains("the signature check stays uncomputed"),
        "{err}"
    );

    let mut quantization = QuantizationReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        artifact_root: raw(b"artifact"),
        reason: "precision".into(),
        code: "UNSUPPORTED_QUANTIZATION".into(),
        retryable: false,
        weights_opened: false,
        payload_read: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    quantization.to_bytes().unwrap();
    quantization.weights_opened = true;
    let err = quantization.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("a precision refusal does not open weights"),
        "{err}"
    );

    let mut kernel = KernelReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        reason: "backend".into(),
        code: "UNSUPPORTED_KERNEL".into(),
        retryable: false,
        cuda_requested: false,
        kernel_selected: false,
        device_opened: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    kernel.to_bytes().unwrap();
    kernel.cuda_requested = true;
    let err = kernel.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("a foreign backend does not request cuda"),
        "{err}"
    );

    let mut placement = PlacementRefusalReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        reason: "device".into(),
        code: "PLACEMENT_UNSATISFIABLE".into(),
        retryable: false,
        probe_reached: true,
        slot_visible: false,
        device_opened: false,
        cpu_fallback: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    placement.to_bytes().unwrap();
    placement.slot_visible = true;
    let err = placement.to_bytes().unwrap_err();
    assert!(
        err.message.contains("a missing device is not visible"),
        "{err}"
    );

    let mut memory = MemoryRefusalReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        reason: "pool".into(),
        free_bytes: 0,
        needed_bytes: MAX_PEAK_BYTES,
        code: "INSUFFICIENT_MEMORY".into(),
        retryable: true,
        resident_full: true,
        queue_held: false,
        allocated: false,
        forward_ran: false,
        receipt_stored: false,
        listener_up: true,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    memory.to_bytes().unwrap();
    memory.needed_bytes = MAX_PEAK_BYTES + 1;
    let err = memory.to_bytes().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(err.message.contains("needed bytes exceed 64 MiB"), "{err}");

    let mut signature_check = SignatureCheckReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        release_root: raw(b"release"),
        message_root: raw(b"message"),
        check_status: "checked".into(),
        signature_checked: true,
        cofactor_cleared: false,
        public_key_bytes: ED25519_PUBLIC_KEY_BYTES,
        signature_bytes: ED25519_SIGNATURE_BYTES,
        scalar_bytes: ED25519_SCALAR_BYTES,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "verified".into(),
        extensions: ext(),
    };
    signature_check.to_bytes().unwrap();
    signature_check.cofactor_cleared = true;
    let err = signature_check.to_bytes().unwrap_err();
    assert!(
        err.message.contains("the cofactor stays uncleared"),
        "{err}"
    );

    let mut context_limit = ContextLimitReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        reason: "prompt".into(),
        prompt_tokens: 17,
        reserved_tokens: 0,
        context_tokens: MICRO_CONTEXT,
        truncated: false,
        code: "CONTEXT_LIMIT_EXCEEDED".into(),
        retryable: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    context_limit.to_bytes().unwrap();
    context_limit.truncated = true;
    let err = context_limit.to_bytes().unwrap_err();
    assert!(err.message.contains("tokens are not truncated"), "{err}");

    let mut prompt_compilation = PromptCompilationReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        failure: "empty".into(),
        token_count: 0,
        rejected_token: 0,
        code: "PROMPT_COMPILATION_FAILED".into(),
        retryable: false,
        template_rendered: true,
        tokenizer_parsed: true,
        prompt_compiled: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    prompt_compilation.to_bytes().unwrap();
    prompt_compilation.prompt_compiled = true;
    let err = prompt_compilation.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("a prompt failure does not compile the prompt"),
        "{err}"
    );

    let mut image_invalid = ImageInvalidReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        image_root: raw(b"image"),
        reason: "empty".into(),
        code: "MODEL_IMAGE_INVALID".into(),
        retryable: false,
        image_parsed: false,
        weights_opened: false,
        forward_ran: false,
        receipt_stored: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    image_invalid.to_bytes().unwrap();
    image_invalid.image_parsed = true;
    let err = image_invalid.to_bytes().unwrap_err();
    assert!(
        err.message.contains("an empty image is not parsed"),
        "{err}"
    );

    let mut image_signature = ImageSignatureReportV1 {
        engine_build_root: raw(b"engine"),
        placement_root: raw(b"placement-plan"),
        image_root: raw(b"image"),
        reason: "algorithm".into(),
        signature_count: 1,
        signature_bytes: 0,
        code: "MODEL_IMAGE_SIGNATURE_INVALID".into(),
        retryable: false,
        algorithm_accepted: false,
        weights_opened: false,
        forward_ran: false,
        receipt_stored: false,
        key_material_present: false,
        execution_mode: "isolated-replay".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
        validation_result: "recorded".into(),
        extensions: ext(),
    };
    image_signature.to_bytes().unwrap();
    image_signature.algorithm_accepted = true;
    let err = image_signature.to_bytes().unwrap_err();
    assert!(
        err.message
            .contains("an algorithm refusal does not accept the algorithm"),
        "{err}"
    );
}

#[test]
fn conversion_receipt_rejects_a_result_other_than_matched() {
    let mut receipt = ConversionReceiptV1 {
        source_artifact_root: raw(b"source-artifact"),
        converter_build_root: raw(b"converter"),
        conversion_config_root: raw(b"conversion-config"),
        destination_artifact_root: raw(b"destination-artifact"),
        validation_result: "rejected".into(),
        extensions: ext(),
    };
    let err = receipt.to_bytes().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(err.message.contains("validationResult"), "{err}");
    receipt.validation_result = "matched".into();
    let bytes = replace_text(&receipt.to_bytes().unwrap(), "validationResult", "rejected");
    let err = decode_contract(&bytes).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
}

fn replace_text(bytes: &[u8], key: &str, value: &str) -> Vec<u8> {
    let CborValue::Map(entries) = decode_canonical(bytes).unwrap() else {
        panic!("map");
    };
    let entries = entries
        .into_iter()
        .map(|(entry_key, entry_value)| {
            if entry_key == key {
                (entry_key, CborValue::Text(value.into()))
            } else {
                (entry_key, entry_value)
            }
        })
        .collect();
    CborValue::Map(entries).to_bytes()
}

fn artifact_payload(artifacts: &ModelArtifactSetV1) -> Vec<u8> {
    let value = decode_canonical(&artifacts.to_bytes().unwrap()).unwrap();
    let files = value
        .as_map()
        .unwrap()
        .iter()
        .find(|entry| entry.0 == "files")
        .unwrap()
        .1
        .clone();
    let mut payload = BTreeMap::new();
    payload.insert("files".into(), files);
    CborValue::map(payload).to_bytes()
}

fn event_payload(event: &InferenceEventV1) -> Vec<u8> {
    let mut payload = BTreeMap::new();
    payload.insert("event".into(), event.to_cbor().unwrap());
    payload.insert(
        "previousEventRoot".into(),
        match &event.previous_event_root {
            Some(root) => CborValue::Text(root.as_str().to_string()),
            None => CborValue::Null,
        },
    );
    CborValue::map(payload).to_bytes()
}

fn write_vectors(rows: &[Row]) {
    let mut out = String::from("{\n  \"vectors\": [\n");
    for (index, row) in rows.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"name\": \"{}\",\n", row.name));
        out.push_str(&format!("      \"ok\": {},\n", row.ok));
        out.push_str(&format!("      \"error\": \"{}\",\n", row.error));
        out.push_str(&format!("      \"domain\": \"{}\",\n", row.domain));
        out.push_str(&format!("      \"hex\": \"{}\",\n", row.hex));
        out.push_str(&format!("      \"payloadHex\": \"{}\",\n", row.payload_hex));
        out.push_str(&format!("      \"digest\": \"{}\"\n", row.digest));
        out.push_str("    }");
        if index + 1 != rows.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n}\n");
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/contracts/vectors.json");
    fs::write(&path, out).unwrap();
}
