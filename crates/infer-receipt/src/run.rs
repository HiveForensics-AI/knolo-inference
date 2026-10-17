//! One pinned run. The second pass is what makes `same_build_replayable`
//! true. `exact_replay_verified` is written only by `replay_pinned`.
//!
//! Without the `cuda` feature the placement device is `cpu`. With that
//! feature the device is `slot-0` and the kernel bundle names `candle-cuda`.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use infer_artifact::{
    hash_current_executable, read_lockfile, resolve_lock_relative, sign_receipt_id, write_atomic,
};
use infer_contracts::{
    fail, output_text_root, output_token_root, CborValue, ChatMessageV1, DigestHex,
    EngineBuildDescriptorV1, EngineReceiptBindingV1, ErrorCode, EvidenceBindingV1, ExecutionPlanV1,
    ExecutionReceiptBindingV1, HardwareReceiptBindingV1, InferFailure, InferenceIntentV1,
    InferenceReceiptV1, KernelBundleDescriptorV1, LimitsV1, ModelReceiptBindingV1,
    OutputReceiptBindingV1, PlacementPlanV1, PlacementReceiptBindingV1, PromptReceiptBindingV1,
    ReplayCheckReceiptV1, SamplerPlanV1, SamplerReceiptBindingV1, TimingReceiptV1,
};
#[cfg(not(feature = "cuda"))]
use infer_engine::{
    accept_cpu_placement, cpu_kernel_bundle, cpu_kernel_plan_root, cpu_placement,
    host_engine_build, llama_cpu_placement,
};
#[cfg(feature = "cuda")]
use infer_engine::{
    accept_cuda_placement, cuda_engine_build, cuda_kernel_bundle, cuda_kernel_plan_root,
    cuda_placement, llama_cuda_placement, require_cuda_slot0,
};
use infer_engine::{
    accept_llama_placement, adapter_vocab, admit_probed_placement, generate_samples_with,
    load_sampler_plan, load_verified_model, open_paged_kv, probe_machine, Journal,
    VerifiedWeightSource, LLAMA_ADAPTER_ID,
};
#[cfg(not(feature = "cuda"))]
use infer_native::CandleCpuBackend;
#[cfg(feature = "cuda")]
use infer_native::CandleCudaBackend;
use infer_native::{cpu_adapter_by_id, CANDLE_CPU_VERSION};
use infer_prompt::{compile_model_prompt, decode_tokens, parse_tokenizer};

use crate::verify::request_id_of;

#[derive(Clone)]
pub struct RunOptions {
    pub alias: String,
    pub prompt: String,
    pub mode: String,
    pub lock_path: PathBuf,
    pub home: PathBuf,
    pub weights_dir: Option<PathBuf>,
    pub max_output_tokens: Option<u32>,
    pub temperature_micros: Option<u32>,
    pub seed: Option<u64>,
    pub stream: Option<u64>,
    pub evidence: Option<EvidenceBindingV1>,
    pub signing_key: Option<[u8; 32]>,
}

pub struct PlacementSummary {
    pub placement_root: DigestHex,
    pub device_slot: String,
    pub kv_block_size: u32,
    pub expected_total_bytes: u64,
}

pub struct RunOutput {
    pub receipt: InferenceReceiptV1,
    pub receipt_bytes: Vec<u8>,
    pub output_text: String,
    pub output_tokens: Vec<u32>,
    pub prompt_tokens: Vec<u32>,
    pub request_id: String,
    pub journal_dir: PathBuf,
}

pub struct ReplayOptions {
    pub receipt_bytes: Vec<u8>,
    pub alias: String,
    pub prompt: String,
    pub lock_path: PathBuf,
    pub home: PathBuf,
    pub weights_dir: Option<PathBuf>,
}

pub fn plan_pinned(
    alias: &str,
    lock_path: &Path,
    weights_dir: Option<&Path>,
) -> Result<PlacementSummary, InferFailure> {
    let source = open_pinned(alias, lock_path, weights_dir)?;
    let selected = select_host(&source)?;
    let _hardware = probe_machine(&selected.bundle.root()?)?;
    Ok(PlacementSummary {
        placement_root: selected.placement.root()?,
        device_slot: selected.device_slot,
        kv_block_size: selected.placement.kv_block_size,
        expected_total_bytes: selected.placement.expected_total_bytes,
    })
}

pub fn run_pinned(options: &RunOptions) -> Result<RunOutput, InferFailure> {
    check_mode(&options.mode)?;
    let source = open_pinned(
        &options.alias,
        &options.lock_path,
        options.weights_dir.as_deref(),
    )?;
    let selected = select_host(&source)?;
    let messages = vec![ChatMessageV1 {
        role: "user".into(),
        content: options.prompt.clone(),
    }];
    execute_turn(&source, &selected, &messages, options, &mut |_| {})
}

/// One loaded model and the turns sent so far. `open` loads and places once.
/// A turn that fails to compile is not appended.
pub struct ChatSession {
    source: VerifiedWeightSource,
    selected: HostSelection,
    messages: Vec<ChatMessageV1>,
    loads: u32,
    last_prompt_tokens: Vec<u32>,
}

impl ChatSession {
    pub fn open(options: &RunOptions) -> Result<Self, InferFailure> {
        check_mode(&options.mode)?;
        let source = open_pinned(
            &options.alias,
            &options.lock_path,
            options.weights_dir.as_deref(),
        )?;
        let selected = select_host(&source)?;
        Ok(Self {
            source,
            selected,
            messages: Vec::new(),
            loads: 1,
            last_prompt_tokens: Vec::new(),
        })
    }

    pub fn messages(&self) -> &[ChatMessageV1] {
        &self.messages
    }

    pub fn loads(&self) -> u32 {
        self.loads
    }

    pub fn last_prompt_tokens(&self) -> &[u32] {
        &self.last_prompt_tokens
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        match self
            .source
            .image
            .special_tokens
            .eos
            .or(self.source.gguf_eos)
        {
            Some(id) => vec![id],
            None => Vec::new(),
        }
    }

    pub fn tokenizer(&self) -> Result<infer_prompt::MicroTokenizer, InferFailure> {
        parse_tokenizer(
            &self.source.image.tokenizer.bytes,
            model_vocab(&self.source)?,
        )
    }

    pub fn turn(
        &mut self,
        options: &RunOptions,
        line: &str,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<RunOutput, InferFailure> {
        check_mode(&options.mode)?;
        let mut pending = self.messages.clone();
        pending.push(ChatMessageV1 {
            role: "user".into(),
            content: line.to_string(),
        });
        let output = execute_turn(&self.source, &self.selected, &pending, options, on_token)?;
        self.last_prompt_tokens.clone_from(&output.prompt_tokens);
        self.messages.push(ChatMessageV1 {
            role: "user".into(),
            content: line.to_string(),
        });
        self.messages.push(ChatMessageV1 {
            role: "assistant".into(),
            content: output.output_text.clone(),
        });
        Ok(output)
    }
}

fn execute_turn(
    source: &VerifiedWeightSource,
    selected: &HostSelection,
    messages: &[ChatMessageV1],
    options: &RunOptions,
    on_token: &mut dyn FnMut(u32),
) -> Result<RunOutput, InferFailure> {
    let placement = &selected.placement;
    let sampler = sampler_from_image(source, options)?;
    let compiled = compile_model_prompt(
        &source.image,
        messages.to_vec(),
        model_vocab(source)?,
        placement.context_reservation_tokens,
        sampler.settings.max_output_tokens,
    )?;
    let limits = LimitsV1 {
        max_output_tokens: sampler.settings.max_output_tokens,
        max_prompt_tokens: placement.context_reservation_tokens,
        deadline_unix_micros: None,
    };
    let intent = InferenceIntentV1 {
        model_runtime_root: source.runtime_root.clone(),
        prompt_plan_root: compiled.plan.root()?,
        sampler_plan_root: sampler.root()?,
        grammar_plan_root: None,
        evidence_binding_root: options
            .evidence
            .as_ref()
            .map(|binding| binding.root())
            .transpose()?,
        requested_mode: options.mode.clone(),
        limits_root: limits.root()?,
        extensions: BTreeMap::new(),
    };
    let kernel_plan_root = selected.kernel_plan_root.clone();
    let execution = ExecutionPlanV1 {
        intent_root: intent.root()?,
        placement_root: placement.root()?,
        kernel_plan_root: kernel_plan_root.clone(),
        scheduling_mode: "isolated".into(),
        prefill_chunk_tokens: compiled.plan.token_ids.len() as u32,
        kv_block_size: placement.kv_block_size,
        prefix_cache_enabled: false,
        prefix_cache_namespace: "off".into(),
        speculative_plan_root: None,
        receipt_policy: "atomic-verified".into(),
        mode: options.mode.clone(),
        extensions: BTreeMap::new(),
    };
    execution.validate()?;
    prepare_home(&options.home)?;
    let request_id = new_request_id()?;
    let mut journal = Journal::create(&options.home, &request_id)?;
    journal.write_sampler(&sampler)?;
    journal.append("accepted", intent.root()?)?;
    let generated = match forward(
        source,
        placement,
        &sampler,
        &compiled.plan.token_ids,
        on_token,
    ) {
        Ok(generated) => generated,
        Err(err) => return Err(note_failure(&mut journal, &request_id, err)),
    };
    let again = match forward(
        source,
        placement,
        &sampler,
        &compiled.plan.token_ids,
        &mut |_| {},
    ) {
        Ok(again) => again,
        Err(err) => return Err(note_failure(&mut journal, &request_id, err)),
    };
    if again.tokens != generated.tokens {
        return Err(note_failure(
            &mut journal,
            &request_id,
            fail(
                ErrorCode::ReplayOutputMismatch,
                "same-build rerun did not reproduce token ids",
            ),
        ));
    }
    let tokenizer = parse_tokenizer(&source.image.tokenizer.bytes, model_vocab(source)?)?;
    let output_text = decode_tokens(
        &tokenizer,
        visible_tokens(&generated.tokens, &sampler.eos_token_ids),
    )?;
    journal.append("prefill", compiled.plan.token_id_root()?)?;
    let mut rolling = Vec::new();
    for token in &generated.tokens {
        rolling.push(*token);
        journal.append("token", output_token_root(&rolling)?)?;
    }
    let completed = journal.append("completed", output_token_root(&generated.tokens)?)?;
    let engine = engine_for(selected.bundle.root()?)?;
    let hardware = probe_machine(&selected.bundle.root()?)?;
    let mut extensions = BTreeMap::new();
    extensions.insert(
        "knolo.request-id".into(),
        CborValue::Text(request_id.clone()),
    );
    let mut receipt = InferenceReceiptV1 {
        receipt_id: compiled.plan.token_id_root()?,
        intent_root: intent.root()?,
        model: model_binding(source)?,
        engine: EngineReceiptBindingV1 {
            engine_build_root: engine.root()?,
            backend: "native".into(),
            backend_version: CANDLE_CPU_VERSION.into(),
            binary_sha256: engine.binary_sha256.clone(),
            kernel_bundle_root: selected.bundle.root()?,
            kernel_plan_root,
        },
        hardware: HardwareReceiptBindingV1 {
            hardware_root: hardware.root()?,
            gpu_model: hardware.gpus.first().map(|gpu| gpu.model.clone()),
            compute_capability: hardware
                .gpus
                .first()
                .map(|gpu| gpu.compute_capability.clone()),
            device_slot: selected.device_slot.clone(),
        },
        placement: PlacementReceiptBindingV1 {
            placement_root: placement.root()?,
            kv_block_size: placement.kv_block_size,
            kv_precision: placement.kv_precision.clone(),
        },
        knowledge: options.evidence.clone(),
        prompt: PromptReceiptBindingV1 {
            messages_root: compiled.plan.messages_root.clone(),
            tools_root: compiled.plan.tools_root.clone(),
            evidence_root: compiled.plan.evidence_root.clone(),
            rendered_text_root: compiled.plan.rendered_text_root()?,
            token_id_root: compiled.plan.token_id_root()?,
            prompt_token_count: compiled.plan.token_ids.len() as u32,
            truncation_root: compiled.plan.truncation.root()?,
        },
        sampler: SamplerReceiptBindingV1 {
            sampler_plan_root: sampler.root()?,
            temperature_micros: sampler.settings.temperature_micros,
            seed: sampler.seed,
            rng: sampler.rng.clone(),
        },
        execution: ExecutionReceiptBindingV1 {
            execution_plan_root: execution.root()?,
            scheduling_mode: execution.scheduling_mode.clone(),
            prefill_chunk_tokens: execution.prefill_chunk_tokens,
            kv_block_size: execution.kv_block_size,
            prefix_cache_hit_tokens: 0,
            batch_trace_root: batch_trace(
                compiled.plan.token_ids.len() as u32,
                generated.tokens.len() as u32,
            )?,
            speculative_plan_root: None,
            event_trace_root: completed,
            attempt: 1,
        },
        output: OutputReceiptBindingV1 {
            output_token_root: output_token_root(&generated.tokens)?,
            output_text_root: output_text_root(&output_text)?,
            token_count: generated.tokens.len() as u32,
            finish_reason: generated.finish_reason,
            structured_output_root: None,
        },
        timing: TimingReceiptV1 {
            queue_micros: 0,
            prefill_micros: generated.prefill_micros,
            decode_micros: generated.decode_micros,
            total_micros: generated
                .prefill_micros
                .checked_add(generated.decode_micros)
                .ok_or_else(|| fail(ErrorCode::ContractInvalid, "timing total overflows"))?,
        },
        assurance: "same_build_replayable".into(),
        previous_receipt_root: None,
        extensions,
        signatures: Vec::new(),
    };
    receipt.receipt_id = receipt.computed_id()?;
    if let Some(seed) = options.signing_key {
        receipt.signatures = vec![sign_receipt_id(&seed, receipt.receipt_id.as_str())?];
    }
    let receipt_bytes = store_receipt(&options.home, &receipt)?;
    Ok(RunOutput {
        output_text,
        output_tokens: generated.tokens,
        prompt_tokens: compiled.plan.token_ids,
        request_id,
        journal_dir: journal.dir().to_path_buf(),
        receipt,
        receipt_bytes,
    })
}

pub fn replay_pinned(options: &ReplayOptions) -> Result<ReplayCheckReceiptV1, InferFailure> {
    let original = crate::verify::verify_receipt_bytes(&options.receipt_bytes)?;
    let request_id = request_id_of(&original)?;
    let source = open_pinned(
        &options.alias,
        &options.lock_path,
        options.weights_dir.as_deref(),
    )?;
    let selected = select_host(&source)?;
    let placement = &selected.placement;
    let sampler = load_sampler_plan(&options.home, &request_id)?;
    let compiled = compile_model_prompt(
        &source.image,
        vec![ChatMessageV1 {
            role: "user".into(),
            content: options.prompt.clone(),
        }],
        model_vocab(&source)?,
        placement.context_reservation_tokens,
        sampler.settings.max_output_tokens,
    )?;
    let engine = engine_for(selected.bundle.root()?)?;
    let mut mismatches = Vec::new();
    if compiled.plan.token_id_root()? != original.prompt.token_id_root
        || sampler.root()? != original.sampler.sampler_plan_root
        || source.runtime_root != original.model.model_runtime_root
        || engine.root()? != original.engine.engine_build_root
        || placement.root()? != original.placement.placement_root
    {
        mismatches.push("REPLAY_ENVIRONMENT_MISMATCH".into());
    }
    let output_root = if mismatches.is_empty() {
        let generated = forward(
            &source,
            placement,
            &sampler,
            &compiled.plan.token_ids,
            &mut |_| {},
        )?;
        let root = output_token_root(&generated.tokens)?;
        if root != original.output.output_token_root {
            mismatches.push("REPLAY_OUTPUT_MISMATCH".into());
        }
        root
    } else {
        original.output.output_token_root.clone()
    };
    mismatches.sort();
    mismatches.dedup();
    let matched = mismatches.is_empty();
    let check = ReplayCheckReceiptV1 {
        receipt_root: original.receipt_id.clone(),
        engine_build_root: engine.root()?,
        placement_root: placement.root()?,
        output_token_root: output_root,
        matched,
        assurance: if matched {
            "exact_replay_verified".into()
        } else {
            "incomplete".into()
        },
        mismatches,
        extensions: BTreeMap::new(),
    };
    check.validate()?;
    if !matched {
        let code = if check
            .mismatches
            .iter()
            .any(|item| item == "REPLAY_ENVIRONMENT_MISMATCH")
        {
            ErrorCode::ReplayEnvironmentMismatch
        } else {
            ErrorCode::ReplayOutputMismatch
        };
        return Err(fail(code, "replay did not verify the original receipt"));
    }
    Ok(check)
}

struct HostSelection {
    placement: PlacementPlanV1,
    kernel_plan_root: DigestHex,
    bundle: KernelBundleDescriptorV1,
    device_slot: String,
}

fn select_host(source: &VerifiedWeightSource) -> Result<HostSelection, InferFailure> {
    #[cfg(feature = "cuda")]
    {
        let slot = require_cuda_slot0()?;
        let placement = if source.image.architecture.adapter == LLAMA_ADAPTER_ID {
            llama_cuda_placement(source)?
        } else {
            cuda_placement(source)?
        };
        if source.image.architecture.adapter == LLAMA_ADAPTER_ID {
            accept_llama_placement(source, &placement)?;
        } else {
            accept_cuda_placement(source, &placement)?;
        }
        admit_probed_placement(&placement)?;
        let bundle = cuda_kernel_bundle(
            CANDLE_CPU_VERSION,
            &slot.toolkit_version,
            &slot.architecture,
        )?;
        Ok(HostSelection {
            placement,
            kernel_plan_root: cuda_kernel_plan_root()?,
            bundle,
            device_slot: "slot-0".into(),
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let placement = if source.image.architecture.adapter == LLAMA_ADAPTER_ID {
            llama_cpu_placement(source)?
        } else {
            cpu_placement(source)?
        };
        if source.image.architecture.adapter == LLAMA_ADAPTER_ID {
            accept_llama_placement(source, &placement)?;
        } else {
            accept_cpu_placement(source, &placement)?;
        }
        admit_probed_placement(&placement)?;
        let bundle = cpu_kernel_bundle(CANDLE_CPU_VERSION)?;
        Ok(HostSelection {
            placement,
            kernel_plan_root: cpu_kernel_plan_root()?,
            bundle,
            device_slot: "cpu".into(),
        })
    }
}

fn engine_for(bundle_root: DigestHex) -> Result<EngineBuildDescriptorV1, InferFailure> {
    let binary = hash_current_executable()?;
    #[cfg(feature = "cuda")]
    {
        cuda_engine_build(binary, CANDLE_CPU_VERSION, bundle_root)
    }
    #[cfg(not(feature = "cuda"))]
    {
        host_engine_build(binary, CANDLE_CPU_VERSION, bundle_root)
    }
}

fn model_vocab(source: &VerifiedWeightSource) -> Result<u32, InferFailure> {
    if let Some(shape) = source.llama {
        return u32::try_from(shape.vocab).map_err(|_| {
            fail(
                ErrorCode::ModelImageInvalid,
                "llama vocabulary does not fit u32",
            )
        });
    }
    adapter_vocab(&source.image.architecture.adapter)
}

fn forward(
    source: &VerifiedWeightSource,
    placement: &PlacementPlanV1,
    sampler: &SamplerPlanV1,
    prompt: &[u32],
    on_token: &mut dyn FnMut(u32),
) -> Result<infer_engine::SampledOutput, InferFailure> {
    let adapter = cpu_adapter_by_id(&source.image.architecture.adapter)?;
    let mut model = {
        #[cfg(feature = "cuda")]
        {
            let backend = CandleCudaBackend;
            adapter.build(source, placement, &backend)?
        }
        #[cfg(not(feature = "cuda"))]
        {
            let backend = CandleCpuBackend;
            adapter.build(source, placement, &backend)?
        }
    };
    let mut kv = open_paged_kv(source, model.kv_layout())?;
    generate_samples_with(model.as_mut(), &mut kv, 1, prompt, sampler, on_token)
}

fn note_failure(journal: &mut Journal, request_id: &str, err: InferFailure) -> InferFailure {
    if let Ok(payload) = trace_text(err.code.as_str()) {
        let _ = journal.append("failed", payload);
    }
    let mut err = err;
    err.request_id = Some(request_id.to_string());
    if !err.message.contains("request ") {
        err.message = format!("request {request_id}: {}", err.message);
    }
    err
}

fn model_binding(source: &VerifiedWeightSource) -> Result<ModelReceiptBindingV1, InferFailure> {
    Ok(ModelReceiptBindingV1 {
        model_image_root: source.image_root.clone(),
        artifact_root: source.artifact_root.clone(),
        model_runtime_root: source.runtime_root.clone(),
        architecture_adapter_id: source.image.architecture.adapter.clone(),
        architecture_adapter_root: source.image.architecture_adapter_root()?,
        config_root: source.image.config_root()?,
        tokenizer_root: source.image.tokenizer.root.clone(),
        template_root: source.image.template.root.clone(),
        storage_precision: source
            .image
            .precisions
            .first()
            .cloned()
            .unwrap_or_else(|| "f32".into()),
        compute_precision: "f32".into(),
    })
}

/// The stop token stays in the receipt ids. The displayed reply ends before it.
fn visible_tokens<'a>(tokens: &'a [u32], eos: &[u32]) -> &'a [u32] {
    let mut end = tokens.len();
    while end > 0 && eos.contains(&tokens[end - 1]) {
        end -= 1;
    }
    &tokens[..end]
}

fn sampler_from_image(
    source: &VerifiedWeightSource,
    options: &RunOptions,
) -> Result<SamplerPlanV1, InferFailure> {
    let mut settings = source.image.generation_defaults.clone();
    if let Some(max_output_tokens) = options.max_output_tokens {
        settings.max_output_tokens = max_output_tokens;
    }
    if let Some(temperature) = options.temperature_micros {
        settings.temperature_micros = temperature;
    }
    settings.validate()?;
    let greedy = settings.temperature_micros == 0;
    if greedy && (options.seed.is_some() || options.stream.is_some()) {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "greedy sampler cannot carry an RNG seed",
        ));
    }
    if !greedy && options.seed.is_none() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "sampled generation requires --seed",
        ));
    }
    let mut eos = Vec::new();
    if let Some(id) = source.image.special_tokens.eos.or(source.gguf_eos) {
        eos.push(id);
    }
    eos.sort_unstable();
    eos.dedup();
    let plan = SamplerPlanV1 {
        settings,
        rng: if greedy {
            "none".into()
        } else {
            "philox-4x32-v1".into()
        },
        seed: if greedy { None } else { options.seed },
        stream: if greedy {
            None
        } else {
            Some(options.stream.unwrap_or(0))
        },
        tie_break: "lowest-token-id".into(),
        eos_token_ids: eos,
        stop_string_roots: Vec::new(),
        extensions: BTreeMap::new(),
    };
    plan.validate()?;
    Ok(plan)
}

fn open_pinned(
    alias: &str,
    lock_path: &Path,
    weights_dir: Option<&Path>,
) -> Result<VerifiedWeightSource, InferFailure> {
    let lock = read_lockfile(lock_path)?
        .ok_or_else(|| fail(ErrorCode::ModelArtifactMissing, "infer lockfile is missing"))?;
    let pin = lock
        .models
        .get(alias)
        .ok_or_else(|| fail(ErrorCode::ModelArtifactMissing, "alias is not pinned"))?;
    let kmodel = resolve_lock_relative(lock_path, &pin.model_image_path);
    let weights = match weights_dir {
        Some(dir) => dir.to_path_buf(),
        None => kmodel
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
    };
    let source = load_verified_model(&kmodel, &weights)?;
    if source.image_root.as_str() != pin.model_image_root
        || source.artifact_root.as_str() != pin.artifact_root
    {
        return Err(fail(
            ErrorCode::ModelDigestMismatch,
            "pinned roots do not match the model image",
        ));
    }
    Ok(source)
}

fn check_mode(mode: &str) -> Result<(), InferFailure> {
    match mode {
        "pinned" | "isolated-replay" => Ok(()),
        "throughput" => Err(fail(
            ErrorCode::BackendNotAllowed,
            "throughput mode is not available on the cpu reference",
        )),
        _ => Err(fail(
            ErrorCode::ContractInvalid,
            "mode must be pinned, isolated-replay, or throughput",
        )),
    }
}

fn prepare_home(home: &Path) -> Result<(), InferFailure> {
    fs::create_dir_all(home.join("journals")).map_err(io_receipt)?;
    fs::create_dir_all(home.join("receipts").join("sha256")).map_err(io_receipt)?;
    Ok(())
}

fn store_receipt(home: &Path, receipt: &InferenceReceiptV1) -> Result<Vec<u8>, InferFailure> {
    let bytes = receipt.to_bytes()?;
    let hex = receipt
        .receipt_id
        .as_str()
        .strip_prefix("sha256-")
        .ok_or_else(|| {
            fail(
                ErrorCode::DigestInvalid,
                "receipt id is not a sha256 digest",
            )
        })?;
    let dir = home.join("receipts").join("sha256").join(hex);
    fs::create_dir_all(&dir).map_err(io_receipt)?;
    write_atomic(
        &dir.join("receipt.cbor"),
        &bytes,
        ErrorCode::ReceiptPersistFailed,
    )?;
    Ok(bytes)
}

fn batch_trace(prompt_tokens: u32, decode_steps: u32) -> Result<DigestHex, InferFailure> {
    let mut map = BTreeMap::new();
    map.insert(
        "decodeSteps".into(),
        CborValue::Integer(i128::from(decode_steps)),
    );
    map.insert(
        "promptTokens".into(),
        CborValue::Integer(i128::from(prompt_tokens)),
    );
    map.insert("schedulingMode".into(), CborValue::Text("isolated".into()));
    infer_contracts::digest_value("infer-execution-trace", &CborValue::map(map))
}

fn trace_text(text: &str) -> Result<DigestHex, InferFailure> {
    infer_contracts::digest_value("infer-execution-trace", &CborValue::Text(text.to_string()))
}

fn new_request_id() -> Result<String, InferFailure> {
    let mut bytes = [0u8; 16];
    let mut file = File::open("/dev/urandom").map_err(io_receipt)?;
    file.read_exact(&mut bytes).map_err(io_receipt)?;
    let mut id = String::from("r");
    for byte in bytes {
        id.push_str(&format!("{byte:02x}"));
    }
    Ok(id)
}

fn io_receipt(err: std::io::Error) -> InferFailure {
    fail(
        ErrorCode::ReceiptPersistFailed,
        format!("receipt store: {err}"),
    )
}

#[cfg(test)]
mod chat_session {
    use std::fs;

    use infer_artifact::{
        compile_manifest, new_lockfile, pin_alias, verify_image, write_lockfile, write_model_image,
    };
    use infer_contracts::ErrorCode;
    use infer_engine::{cargo_lock_root, write_synthetic_model};
    use infer_prompt::{encode_text, parse_tokenizer};

    use super::{ChatSession, RunOptions};

    #[test]
    fn two_turns_keep_the_text_and_load_once_and_a_long_line_is_dropped() {
        let dir =
            std::env::temp_dir().join(format!("knolo-infer-chat-session-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        write_synthetic_model(&dir).unwrap();
        fs::write(
            dir.join("template.jinja"),
            "{% for message in messages %}{{ message.content }}{% endfor %}",
        )
        .unwrap();
        let compiled = compile_manifest(&dir.join("manifest.json")).unwrap();
        write_model_image(&dir.join("micro.kmodel"), &compiled.bytes).unwrap();
        let bytes = fs::read(dir.join("micro.kmodel")).unwrap();
        let verification = verify_image(&bytes).unwrap();
        let mut lock = new_lockfile(&cargo_lock_root().unwrap());
        pin_alias(&mut lock, "micro", "micro.kmodel", &verification.image).unwrap();
        write_lockfile(&dir.join("knolo.infer.lock.json"), &lock).unwrap();

        let options = RunOptions {
            alias: "micro".into(),
            prompt: String::new(),
            mode: "pinned".into(),
            lock_path: dir.join("knolo.infer.lock.json"),
            home: dir.join("home"),
            weights_dir: Some(dir.clone()),
            max_output_tokens: Some(2),
            temperature_micros: None,
            seed: None,
            stream: None,
            evidence: None,
            signing_key: None,
        };
        let mut session = ChatSession::open(&options).unwrap();
        assert_eq!(session.loads(), 1);
        let first = session.turn(&options, "h", &mut |_| {}).unwrap();
        let assistant = first.output_text.clone();
        let second = session.turn(&options, "t", &mut |_| {}).unwrap();
        let tokenizer = parse_tokenizer(&verification.image.tokenizer.bytes, 16).unwrap();
        let mut expected = vec![verification.image.special_tokens.bos.unwrap()];
        expected.extend(encode_text(&tokenizer, &format!("h{assistant}t")).unwrap());
        assert_eq!(session.last_prompt_tokens(), expected.as_slice());
        assert!(session
            .last_prompt_tokens()
            .windows(encode_text(&tokenizer, "h").unwrap().len())
            .any(|window| window == encode_text(&tokenizer, "h").unwrap()));
        let assistant_ids = encode_text(&tokenizer, &assistant).unwrap();
        if !assistant_ids.is_empty() {
            assert!(session
                .last_prompt_tokens()
                .windows(assistant_ids.len())
                .any(|window| window == assistant_ids));
        }
        assert_eq!(session.messages().len(), 4);
        assert_eq!(session.messages()[0].content, "h");
        assert_eq!(session.messages()[1].role, "assistant");
        assert_eq!(session.messages()[1].content, assistant);
        assert_eq!(session.messages()[2].content, "t");
        assert_eq!(session.messages()[3].content, second.output_text);
        assert_eq!(session.loads(), 1);
        let kept = session.messages().to_vec();
        let kept_prompt = session.last_prompt_tokens().to_vec();
        let err = match session.turn(&options, &"a".repeat(20), &mut |_| {}) {
            Ok(_) => panic!("a line past the context was kept"),
            Err(err) => err,
        };
        assert_eq!(err.code, ErrorCode::ContextLimitExceeded);
        assert_eq!(session.messages(), kept.as_slice());
        assert_eq!(session.last_prompt_tokens(), kept_prompt.as_slice());
        assert_eq!(session.loads(), 1);
        let _ = fs::remove_dir_all(&dir);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_run {
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::{
        cuda_engine_build, cuda_kernel_bundle, cuda_kernel_plan_root, replay_pinned, run_pinned,
        sampler_from_image, ReplayOptions, RunOptions, CANDLE_CPU_VERSION,
    };
    use infer_artifact::{
        hash_current_executable, new_lockfile, pin_alias, verify_image, write_lockfile,
    };
    use infer_contracts::ChatMessageV1;
    use infer_engine::{
        cargo_lock_root, cpu_placement, generate_samples, load_verified_micro, micro_kv_layout,
        require_cuda_slot0, write_synthetic_model, ArchitectureAdapter, MicroAdapter, PagedKv,
        ReferenceF32Backend, CPU_KV_PAGE_POOL, VOCAB,
    };
    use infer_prompt::compile_model_prompt;

    struct CwdGuard(PathBuf);

    impl Drop for CwdGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.0);
        }
    }

    #[test]
    fn cuda_run_names_slot0_and_matches_oracle() {
        let dir = std::env::temp_dir().join(format!("knolo-infer-cuda-run-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        write_synthetic_model(&dir).unwrap();
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        let _cwd = CwdGuard(previous);
        run_case(&dir);
        let _ = fs::remove_dir_all(&dir);
    }

    fn run_case(dir: &Path) {
        let bytes = fs::read(dir.join("micro.kmodel")).unwrap();
        let verification = verify_image(&bytes).unwrap();
        let mut lock = new_lockfile(&cargo_lock_root().unwrap());
        pin_alias(&mut lock, "micro", "micro.kmodel", &verification.image).unwrap();
        write_lockfile(&dir.join("knolo.infer.lock.json"), &lock).unwrap();
        let home = dir.join("home");
        let options = RunOptions {
            alias: "micro".into(),
            prompt: "hi".into(),
            mode: "pinned".into(),
            lock_path: dir.join("knolo.infer.lock.json"),
            home: home.clone(),
            weights_dir: Some(dir.to_path_buf()),
            max_output_tokens: None,
            temperature_micros: None,
            seed: None,
            stream: None,
            evidence: None,
            signing_key: None,
        };
        let output = run_pinned(&options).expect("cuda run");
        let slot = require_cuda_slot0().unwrap();
        let bundle = cuda_kernel_bundle(
            CANDLE_CPU_VERSION,
            &slot.toolkit_version,
            &slot.architecture,
        )
        .unwrap();
        assert_eq!(output.receipt.hardware.device_slot, "slot-0");
        assert_eq!(output.receipt.placement.placement_root, {
            infer_engine::cuda_placement(
                &load_verified_micro(&dir.join("micro.kmodel"), dir).unwrap(),
            )
            .unwrap()
            .root()
            .unwrap()
        });
        assert_eq!(bundle.build_mode, "cuda");
        assert_eq!(bundle.cuda_toolkit_version, slot.toolkit_version);
        assert_eq!(bundle.cuda_architectures.len(), 1);
        assert_eq!(bundle.cuda_architectures[0], slot.architecture);
        assert!(bundle.compiler_flags.is_empty());
        assert!(bundle.jit.is_none());
        assert_eq!(
            output.receipt.engine.kernel_bundle_root,
            bundle.root().unwrap()
        );
        assert_eq!(
            output.receipt.engine.kernel_plan_root,
            cuda_kernel_plan_root().unwrap()
        );
        let engine = cuda_engine_build(
            hash_current_executable().unwrap(),
            CANDLE_CPU_VERSION,
            bundle.root().unwrap(),
        )
        .unwrap();
        assert_eq!(engine.tensor_backend, "candle-cuda");
        assert_eq!(engine.feature_set, ["cuda".to_string()]);
        assert_eq!(
            output.receipt.engine.engine_build_root,
            engine.root().unwrap()
        );
        assert_eq!(output.receipt.engine.backend, "native");
        assert_eq!(output.receipt.assurance, "same_build_replayable");

        let source = load_verified_micro(&dir.join("micro.kmodel"), dir).unwrap();
        let placement = cpu_placement(&source).unwrap();
        let sampler = sampler_from_image(&source, &options).unwrap();
        let compiled = compile_model_prompt(
            &source.image,
            vec![ChatMessageV1 {
                role: "user".into(),
                content: "hi".into(),
            }],
            VOCAB as u32,
            placement.context_reservation_tokens,
            sampler.settings.max_output_tokens,
        )
        .unwrap();
        let mut oracle = MicroAdapter
            .build(&source, &placement, &ReferenceF32Backend)
            .unwrap();
        let mut kv = PagedKv::new(micro_kv_layout(), CPU_KV_PAGE_POOL).unwrap();
        let generated = generate_samples(
            oracle.as_mut(),
            &mut kv,
            1,
            &compiled.plan.token_ids,
            &sampler,
        )
        .unwrap();
        assert_eq!(output.output_tokens, generated.tokens);

        let check = replay_pinned(&ReplayOptions {
            receipt_bytes: output.receipt_bytes,
            alias: "micro".into(),
            prompt: "hi".into(),
            lock_path: dir.join("knolo.infer.lock.json"),
            home,
            weights_dir: Some(dir.to_path_buf()),
        })
        .unwrap();
        assert!(check.matched);
        assert_eq!(check.assurance, "exact_replay_verified");
    }
}
