use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use infer_contracts::{ErrorCode, FixedPointSamplerV1, InferFailure, SamplerPlanV1};
use infer_engine::{
    generate_samples, load_verified_micro, write_synthetic_model, ArchitectureAdapter,
    CpuScheduler, ExecutableModel, KvStore, MicroAdapter, PagedKv, ReferenceF32Backend, ScheduleOp,
    ScheduleRequest, SchedulerConfig, ServiceClass, SingleBlockKv, BLOCK_SIZE, CPU_KV_PAGE_POOL,
    FIXTURE_CASES, MAX_CONTEXT,
};

static TEMP: AtomicU64 = AtomicU64::new(0);

fn code(err: InferFailure) -> ErrorCode {
    err.code
}

fn scratch() -> PathBuf {
    let n = TEMP.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("knolo-infer-schedule-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn oracle() -> (String, Box<dyn ExecutableModel>) {
    let dir = scratch();
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let placement = infer_engine::cpu_placement(&source).unwrap();
    let model = MicroAdapter
        .build(&source, &placement, &ReferenceF32Backend)
        .unwrap();
    (source.runtime_root.as_str().to_string(), model)
}

fn config(root: &str, pages: u32, chunk: u32) -> SchedulerConfig {
    SchedulerConfig::new(root, MAX_CONTEXT, BLOCK_SIZE, pages, chunk).unwrap()
}

fn greedy(max_output: u32) -> SamplerPlanV1 {
    SamplerPlanV1 {
        settings: FixedPointSamplerV1 {
            temperature_micros: 0,
            top_p_millionths: 1_000_000,
            min_p_millionths: 0,
            repetition_penalty_micros: 1_000_000,
            presence_penalty_micros: 0,
            frequency_penalty_micros: 0,
            top_k: 0,
            max_output_tokens: max_output,
        },
        rng: "none".into(),
        seed: None,
        stream: None,
        tie_break: "lowest-token-id".into(),
        eos_token_ids: vec![2],
        stop_string_roots: Vec::new(),
        extensions: BTreeMap::new(),
    }
}

fn greedy_open(max_output: u32) -> SamplerPlanV1 {
    let mut plan = greedy(max_output);
    plan.eos_token_ids.clear();
    plan
}

fn request(
    root: &str,
    id: &str,
    prompt: &[u32],
    sampler: SamplerPlanV1,
    class: ServiceClass,
) -> ScheduleRequest {
    ScheduleRequest {
        request_id: id.into(),
        runtime_root: root.into(),
        prompt: prompt.to_vec(),
        sampler,
        class,
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn admission_refuses_context_pages_and_a_foreign_runtime() {
    let scheduler_root = "runtime-a";
    let mut scheduler = CpuScheduler::new(config(scheduler_root, 1, 4));
    let long = vec![1u32; 16];
    let err = scheduler
        .submit(request(
            scheduler_root,
            "too-long",
            &long,
            greedy(1),
            ServiceClass::Standard,
        ))
        .unwrap_err();
    assert_eq!(code(err), ErrorCode::ContextLimitExceeded);
    assert_eq!(
        code(
            scheduler
                .submit(request(
                    scheduler_root,
                    "empty",
                    &[],
                    greedy(1),
                    ServiceClass::Standard,
                ))
                .unwrap_err()
        ),
        ErrorCode::PromptCompilationFailed
    );
    assert_eq!(
        code(
            scheduler
                .submit(request(
                    "runtime-b",
                    "other-model",
                    &[1],
                    greedy(1),
                    ServiceClass::Standard,
                ))
                .unwrap_err()
        ),
        ErrorCode::ContractInvalid
    );
    assert_eq!(
        code(
            scheduler
                .submit(request(
                    scheduler_root,
                    "bad id",
                    &[1],
                    greedy(1),
                    ServiceClass::Standard,
                ))
                .unwrap_err()
        ),
        ErrorCode::ContractInvalid
    );

    let mut wide =
        CpuScheduler::new(SchedulerConfig::new(scheduler_root, 32, BLOCK_SIZE, 1, 4).unwrap());
    assert_eq!(
        code(
            wide.submit(request(
                scheduler_root,
                "two-pages",
                &long,
                greedy_open(1),
                ServiceClass::Batch,
            ))
            .unwrap_err()
        ),
        ErrorCode::InsufficientMemory
    );

    let held = scheduler
        .submit(request(
            scheduler_root,
            "held",
            &[1, 4],
            greedy(1),
            ServiceClass::Interactive,
        ))
        .unwrap();
    assert_eq!(held, 1);
    assert_eq!(
        code(
            scheduler
                .submit(request(
                    scheduler_root,
                    "held",
                    &[1],
                    greedy(1),
                    ServiceClass::Interactive,
                ))
                .unwrap_err()
        ),
        ErrorCode::ContractInvalid
    );
    assert_eq!(
        code(
            scheduler
                .submit(request(
                    scheduler_root,
                    "second",
                    &[1],
                    greedy(1),
                    ServiceClass::Background,
                ))
                .unwrap_err()
        ),
        ErrorCode::InsufficientMemory
    );
    assert_eq!(
        code(SchedulerConfig::new(scheduler_root, MAX_CONTEXT, BLOCK_SIZE, 1, 0).unwrap_err()),
        ErrorCode::ContractInvalid
    );
}

#[test]
fn chunked_prefill_matches_one_shot_logits_and_isolated_tokens() {
    let (root, mut model) = oracle();
    let prompt = [1u32, 4, 7, 5, 8, 3, 1, 4];
    let mut together = CpuScheduler::new(config(&root, CPU_KV_PAGE_POOL, 3));
    together
        .submit(request(
            &root,
            "chunked",
            &prompt,
            greedy_open(2),
            ServiceClass::Standard,
        ))
        .unwrap();
    let mut kv = PagedKv::new(infer_engine::micro_kv_layout(), CPU_KV_PAGE_POOL).unwrap();
    let mut steps = Vec::new();
    while let Some(step) = together.step(model.as_mut(), &mut kv).unwrap() {
        steps.push(step);
    }
    let prefill_tokens: Vec<u32> = steps
        .iter()
        .filter_map(|step| match step.ops.as_slice() {
            [ScheduleOp::Prefill { tokens, .. }] => Some(*tokens),
            _ => None,
        })
        .collect();
    assert_eq!(prefill_tokens, vec![3, 3, 2]);
    let scheduled = &together.results()[0];
    assert_eq!(scheduled.finish_reason, "length");

    let mut alone = SingleBlockKv::new(infer_engine::micro_kv_layout()).unwrap();
    let sampled =
        generate_samples(model.as_mut(), &mut alone, 1, &prompt, &greedy_open(2)).unwrap();
    assert_eq!(scheduled.tokens, sampled.tokens);
    assert_eq!(
        bits(&scheduled.prefill_logits),
        bits_of_prefill(&mut model, &prompt)
    );
}

#[test]
fn a_batch_matches_the_same_requests_run_alone() {
    let (root, mut model) = oracle();
    let mut batched = CpuScheduler::new(config(&root, CPU_KV_PAGE_POOL, 2));
    for (index, fixture) in FIXTURE_CASES.iter().enumerate() {
        batched
            .submit(request(
                &root,
                &format!("req-{index}"),
                fixture.prompt,
                greedy(fixture.new_tokens),
                ServiceClass::Interactive,
            ))
            .unwrap();
    }
    let mut kv = PagedKv::new(infer_engine::micro_kv_layout(), CPU_KV_PAGE_POOL).unwrap();
    let results = batched.run_until_idle(model.as_mut(), &mut kv).unwrap();
    assert_eq!(results.len(), FIXTURE_CASES.len());
    for (result, fixture) in results.iter().zip(FIXTURE_CASES) {
        let mut isolated = CpuScheduler::new(config(&root, CPU_KV_PAGE_POOL, 2));
        isolated
            .submit(request(
                &root,
                "alone",
                fixture.prompt,
                greedy(fixture.new_tokens),
                ServiceClass::Interactive,
            ))
            .unwrap();
        let mut alone_kv = PagedKv::new(infer_engine::micro_kv_layout(), CPU_KV_PAGE_POOL).unwrap();
        let alone = isolated
            .run_until_idle(model.as_mut(), &mut alone_kv)
            .unwrap();
        assert_eq!(result.tokens, alone[0].tokens);
        assert_eq!(result.finish_reason, alone[0].finish_reason);
        assert_eq!(bits(&result.prefill_logits), bits(&alone[0].prefill_logits));

        let mut reference = SingleBlockKv::new(infer_engine::micro_kv_layout()).unwrap();
        let sampled = generate_samples(
            model.as_mut(),
            &mut reference,
            1,
            fixture.prompt,
            &greedy(fixture.new_tokens),
        )
        .unwrap();
        assert_eq!(result.tokens, sampled.tokens);
        assert_eq!(result.finish_reason, sampled.finish_reason);
    }
}

#[test]
fn seeded_draws_stay_with_their_stream_when_requests_share_a_pool() {
    let (root, mut model) = oracle();
    let prompt = FIXTURE_CASES[2].prompt;
    let mut left = greedy(3);
    left.settings.temperature_micros = 1_000_000;
    left.rng = "philox-4x32-v1".into();
    left.seed = Some(42);
    left.stream = Some(1);
    left.eos_token_ids.clear();
    let mut right = left.clone();
    right.stream = Some(2);

    let mut shared = CpuScheduler::new(config(&root, CPU_KV_PAGE_POOL, 4));
    shared
        .submit(request(
            &root,
            "left",
            prompt,
            left.clone(),
            ServiceClass::Standard,
        ))
        .unwrap();
    shared
        .submit(request(
            &root,
            "right",
            prompt,
            right.clone(),
            ServiceClass::Standard,
        ))
        .unwrap();
    let mut kv = PagedKv::new(infer_engine::micro_kv_layout(), CPU_KV_PAGE_POOL).unwrap();
    let results = shared.run_until_idle(model.as_mut(), &mut kv).unwrap();

    for (result, plan) in results.iter().zip([&left, &right]) {
        let mut alone = SingleBlockKv::new(infer_engine::micro_kv_layout()).unwrap();
        let sampled = generate_samples(model.as_mut(), &mut alone, 1, prompt, plan).unwrap();
        assert_eq!(result.tokens, sampled.tokens);
    }
}

#[test]
fn interactive_prefill_yields_to_a_waiting_background_then_batches_decode() {
    let (root, mut model) = oracle();
    let mut scheduler = CpuScheduler::new(config(&root, CPU_KV_PAGE_POOL, 4));
    scheduler
        .submit(request(
            &root,
            "fast",
            &[1, 4, 7, 5, 8, 3, 1, 4],
            greedy_open(2),
            ServiceClass::Interactive,
        ))
        .unwrap();
    scheduler
        .submit(request(
            &root,
            "slow",
            &[1, 4, 7, 5],
            greedy_open(1),
            ServiceClass::Background,
        ))
        .unwrap();
    let mut kv = PagedKv::new(infer_engine::micro_kv_layout(), CPU_KV_PAGE_POOL).unwrap();
    let mut trace = Vec::new();
    while let Some(step) = scheduler.step(model.as_mut(), &mut kv).unwrap() {
        trace.push(step);
    }
    assert_eq!(
        trace[0].ops,
        vec![ScheduleOp::Prefill {
            sequence_id: 1,
            tokens: 4
        }]
    );
    assert_eq!(
        trace[1].ops,
        vec![ScheduleOp::Prefill {
            sequence_id: 2,
            tokens: 4
        }]
    );
    assert_eq!(
        trace[2].ops,
        vec![ScheduleOp::Prefill {
            sequence_id: 1,
            tokens: 4
        }]
    );
    assert_eq!(trace[3].batch, vec![1, 2]);
    assert!(matches!(
        trace[3].ops[0],
        ScheduleOp::Decode { sequence_id: 1, .. }
    ));
    assert!(matches!(
        trace[3].ops[1],
        ScheduleOp::Decode { sequence_id: 2, .. }
    ));
    assert_eq!(trace[4].batch, vec![1]);
    assert_eq!(scheduler.results()[0].finish_reason, "length");
    assert_eq!(scheduler.results()[1].finish_reason, "length");
}

#[test]
fn cancel_releases_the_page_and_the_other_request_still_matches() {
    let (root, mut model) = oracle();
    let prompt = FIXTURE_CASES[0].prompt;
    let mut scheduler = CpuScheduler::new(config(&root, 1, 4));
    scheduler
        .submit(request(
            &root,
            "keep",
            prompt,
            greedy(FIXTURE_CASES[0].new_tokens),
            ServiceClass::Interactive,
        ))
        .unwrap();
    let mut kv = PagedKv::new(infer_engine::micro_kv_layout(), 1).unwrap();
    let first = scheduler.step(model.as_mut(), &mut kv).unwrap().unwrap();
    assert!(matches!(first.ops[0], ScheduleOp::Prefill { .. }));
    assert!(kv.token_len(1).unwrap() > 0);
    assert_eq!(
        code(
            scheduler
                .submit(request(
                    &root,
                    "extra",
                    &[1],
                    greedy(1),
                    ServiceClass::Standard,
                ))
                .unwrap_err()
        ),
        ErrorCode::InsufficientMemory
    );
    assert!(kv.token_len(1).is_ok());
    scheduler.cancel("keep").unwrap();
    assert_eq!(
        code(scheduler.cancel("keep").unwrap_err()),
        ErrorCode::ContractInvalid
    );
    let cancelled = scheduler.step(model.as_mut(), &mut kv).unwrap().unwrap();
    assert_eq!(cancelled.ops, vec![ScheduleOp::Cancel { sequence_id: 1 }]);
    assert_eq!(
        code(kv.token_len(1).unwrap_err()),
        ErrorCode::ContractInvalid
    );
    assert_eq!(scheduler.results()[0].finish_reason, "cancelled");
    assert!(scheduler.results()[0].tokens.is_empty());

    scheduler
        .submit(request(
            &root,
            "next",
            prompt,
            greedy(FIXTURE_CASES[0].new_tokens),
            ServiceClass::Standard,
        ))
        .unwrap();
    let results = scheduler.run_until_idle(model.as_mut(), &mut kv).unwrap();
    let mut alone = SingleBlockKv::new(infer_engine::micro_kv_layout()).unwrap();
    let sampled = generate_samples(
        model.as_mut(),
        &mut alone,
        7,
        prompt,
        &greedy(FIXTURE_CASES[0].new_tokens),
    )
    .unwrap();
    assert_eq!(results[1].tokens, sampled.tokens);
    assert_eq!(results[1].finish_reason, sampled.finish_reason);
}

#[test]
fn one_bad_prompt_fails_closed_and_the_other_request_finishes() {
    let (root, mut model) = oracle();
    let mut scheduler = CpuScheduler::new(config(&root, CPU_KV_PAGE_POOL, 4));
    scheduler
        .submit(request(
            &root,
            "bad",
            &[99],
            greedy(1),
            ServiceClass::Background,
        ))
        .unwrap();
    scheduler
        .submit(request(
            &root,
            "good",
            FIXTURE_CASES[1].prompt,
            greedy(FIXTURE_CASES[1].new_tokens),
            ServiceClass::Interactive,
        ))
        .unwrap();
    let mut kv = PagedKv::new(infer_engine::micro_kv_layout(), CPU_KV_PAGE_POOL).unwrap();
    let results = scheduler.run_until_idle(model.as_mut(), &mut kv).unwrap();
    assert_eq!(results[0].finish_reason, "error");
    assert_eq!(
        results[0].error_code,
        Some(ErrorCode::PromptCompilationFailed)
    );
    assert!(results[0].tokens.is_empty());
    let mut alone = SingleBlockKv::new(infer_engine::micro_kv_layout()).unwrap();
    let sampled = generate_samples(
        model.as_mut(),
        &mut alone,
        1,
        FIXTURE_CASES[1].prompt,
        &greedy(FIXTURE_CASES[1].new_tokens),
    )
    .unwrap();
    assert_eq!(results[1].tokens, sampled.tokens);
    assert_eq!(
        kv.token_len(1).unwrap_err().code,
        ErrorCode::ContractInvalid
    );
}

#[test]
fn admission_counters_do_not_change_the_run() {
    let (root, mut model) = oracle();
    let mut scheduler = CpuScheduler::new(config(&root, 1, 4));
    let mut kv = PagedKv::new(infer_engine::micro_kv_layout(), 1).unwrap();
    let wide = scheduler
        .submit(request(
            &root,
            "wide",
            &[1; 17],
            greedy(1),
            ServiceClass::Interactive,
        ))
        .unwrap_err();
    assert_eq!(code(wide), ErrorCode::ContextLimitExceeded);
    let empty = scheduler
        .submit(request(&root, "empty", &[], greedy(1), ServiceClass::Batch))
        .unwrap_err();
    assert_eq!(code(empty), ErrorCode::PromptCompilationFailed);
    scheduler
        .submit(request(
            &root,
            "one",
            &[1],
            greedy_open(1),
            ServiceClass::Interactive,
        ))
        .unwrap();
    let blocked = scheduler
        .submit(request(
            &root,
            "two",
            &[1],
            greedy(1),
            ServiceClass::Background,
        ))
        .unwrap_err();
    assert_eq!(code(blocked), ErrorCode::InsufficientMemory);
    let queued = scheduler.counters();
    assert_eq!(queued.rejected_context, 1);
    assert_eq!(queued.rejected_prompt, 1);
    assert_eq!(queued.rejected_memory, 1);
    assert_eq!(queued.oom, 1);
    assert_eq!(queued.queue_interactive, 1);
    assert_eq!(queued.queue_background, 0);
    assert_eq!(queued.active_sequences, 1);
    assert_eq!(queued.prefix_lookups, 0);
    assert_eq!(queued.prefix_hits, 0);
    assert_eq!(queued.prefix_reused_tokens, 0);
    let census = kv.census();
    assert_eq!(census.total, 1);
    assert_eq!(census.free, 1);
    assert_eq!(census.pinned, 0);
    scheduler.run_until_idle(&mut *model, &mut kv).unwrap();
    let done = scheduler.counters();
    assert!(done.prefill_tokens >= 1);
    assert_eq!(done.decode_tokens, 1);
    assert!(done.iterations >= 1);
    assert_eq!(done.iteration_buckets.iter().sum::<u64>(), done.iterations);
    assert_eq!(done.prefix_reused_tokens, 0);
    assert_eq!(done.queue_interactive, 0);
    assert_eq!(done.active_sequences, 0);
    let census = kv.census();
    assert_eq!(census.free, census.total);
    assert_eq!(census.pinned, 0);
}

#[test]
fn a_finished_slot_is_dropped_and_the_page_pool_stays_fixed() {
    let (root, mut model) = oracle();
    let mut scheduler = CpuScheduler::new(config(&root, CPU_KV_PAGE_POOL, 4));
    let mut kv = PagedKv::new(infer_engine::micro_kv_layout(), CPU_KV_PAGE_POOL).unwrap();
    let resident = kv.resident_bytes();
    assert!(resident > 0);
    assert!(resident <= 64 * 1024 * 1024);

    scheduler
        .submit(request(
            &root,
            "held",
            &[1],
            greedy(2),
            ServiceClass::Interactive,
        ))
        .unwrap();
    let duplicate = scheduler
        .submit(request(
            &root,
            "held",
            &[1],
            greedy(2),
            ServiceClass::Interactive,
        ))
        .unwrap_err();
    assert_eq!(code(duplicate), ErrorCode::ContractInvalid);
    let early = scheduler.reap("held").unwrap_err();
    assert_eq!(code(early), ErrorCode::ContractInvalid);
    assert_eq!(scheduler.retained(), 1);
    let first = scheduler
        .run_until_idle(&mut *model, &mut kv)
        .unwrap()
        .into_iter()
        .find(|result| result.request_id == "held")
        .unwrap();
    let tokens = first.tokens.clone();
    assert!(!tokens.is_empty());
    scheduler.reap("held").unwrap();
    assert_eq!(scheduler.retained(), 0);
    assert_eq!(
        code(scheduler.reap("held").unwrap_err()),
        ErrorCode::ContractInvalid
    );

    for n in 0..32 {
        let id = format!("soak-{n}");
        scheduler
            .submit(request(
                &root,
                &id,
                &[1],
                greedy(2),
                ServiceClass::Interactive,
            ))
            .unwrap();
        let results = scheduler.run_until_idle(&mut *model, &mut kv).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tokens, tokens);
        assert_eq!(scheduler.retained(), 1);
        scheduler.reap(&id).unwrap();
        assert_eq!(scheduler.retained(), 0);
        assert_eq!(kv.census().total, CPU_KV_PAGE_POOL);
        assert_eq!(kv.census().free, CPU_KV_PAGE_POOL);
        assert_eq!(kv.census().pinned, 0);
        assert_eq!(kv.resident_bytes(), resident);
    }

    scheduler
        .submit(request(
            &root,
            "held",
            &[1],
            greedy(2),
            ServiceClass::Interactive,
        ))
        .unwrap();
    let again = scheduler.run_until_idle(&mut *model, &mut kv).unwrap();
    assert_eq!(again[0].tokens, tokens);
    scheduler.reap("held").unwrap();
    assert_eq!(scheduler.retained(), 0);
    assert_eq!(scheduler.counters().active_sequences, 0);
    assert_eq!(kv.resident_bytes(), resident);
}

fn bits_of_prefill(model: &mut Box<dyn ExecutableModel>, prompt: &[u32]) -> Vec<u32> {
    let mut kv = SingleBlockKv::new(infer_engine::micro_kv_layout()).unwrap();
    kv.begin_sequence(1).unwrap();
    let output = model
        .prefill(
            &infer_engine::PrefillBatch {
                sequence_id: 1,
                token_ids: prompt.to_vec(),
            },
            &mut kv,
        )
        .unwrap();
    bits(&output.logits)
}
