//! Serve-path journal. `accepted` is fsynced before the worker is asked to run.
//!
//! Without the `cuda` feature the worker binary is the reference oracle, so
//! the engine build says `reference-f32` and the device is `cpu`. With that
//! feature the engine build says `candle-cuda` and the device is `slot-0`.
//! Assurance stays `compatibility`: this path does not rerun the sequence.
//! `knolo-infer run` is still the path that sets `same_build_replayable`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use infer_artifact::{hash_regular_file, write_atomic};
use infer_contracts::{
    decode_canonical, digest_value, fail, output_text_root, output_token_root, CborValue,
    DigestHex, EngineBuildDescriptorV1, EngineReceiptBindingV1, ErrorCode, ExecutionPlanV1,
    ExecutionReceiptBindingV1, HardwareReceiptBindingV1, InferFailure, InferenceIntentV1,
    InferenceReceiptV1, KernelBundleDescriptorV1, LimitsV1, ModelReceiptBindingV1,
    OutputReceiptBindingV1, PlacementPlanV1, PlacementReceiptBindingV1, PromptPlanV1,
    PromptReceiptBindingV1, SamplerPlanV1, SamplerReceiptBindingV1, TimingReceiptV1,
};
#[cfg(not(feature = "cuda"))]
use infer_engine::{
    cpu_placement_bytes, reference_engine_build, reference_kernel_bundle,
    reference_kernel_plan_root,
};
#[cfg(feature = "cuda")]
use infer_engine::{
    cuda_engine_build, cuda_kernel_bundle, cuda_kernel_plan_root, cuda_placement_bytes,
    require_cuda_slot0,
};
use infer_engine::{probe_machine, Journal, BLOCK_SIZE};
#[cfg(feature = "cuda")]
use infer_native::CANDLE_CPU_VERSION;

use crate::model::PinnedModel;
use crate::worker::PREFILL_CHUNK_TOKENS;

const ASSURANCE: &str = "compatibility";

pub(crate) struct ServeIdentity {
    placement_root: DigestHex,
    kv_block_size: u32,
    kv_precision: String,
    context_tokens: u32,
    engine: EngineReceiptBindingV1,
    hardware: HardwareReceiptBindingV1,
    kernel_plan_root: DigestHex,
}

pub(crate) struct RequestDraft<'a> {
    pub request_id: &'a str,
    pub prompt: &'a PromptPlanV1,
    pub sampler: &'a SamplerPlanV1,
    pub class: &'a str,
    pub openai: bool,
    pub stream: bool,
}

pub(crate) struct Published {
    pub receipt_id: String,
}

pub(crate) struct OpenedRequest {
    journal: Journal,
    sealed: bool,
    started: Instant,
    request_id: String,
    class: String,
    intent_root: DigestHex,
    execution: ExecutionPlanV1,
    model: ModelReceiptBindingV1,
    engine: EngineReceiptBindingV1,
    hardware: HardwareReceiptBindingV1,
    placement: PlacementReceiptBindingV1,
    prompt: PromptReceiptBindingV1,
    sampler: SamplerReceiptBindingV1,
}

pub(crate) fn prepare_serve(
    home: &Path,
    worker_bin: &Path,
    pinned: &PinnedModel,
) -> Result<ServeIdentity, InferFailure> {
    let runtime = DigestHex::parse(&pinned.runtime_root)?;
    let (placement, bundle, kernel_plan_root, backend_version, device_slot) =
        serve_target(runtime, pinned.weight_bytes)?;
    if placement.kv_block_size != BLOCK_SIZE {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "serve placement does not use the micro KV block",
        ));
    }
    fs::create_dir_all(home.join("journals")).map_err(io_receipt)?;
    fs::create_dir_all(home.join("receipts").join("sha256")).map_err(io_receipt)?;
    let binary = hash_regular_file(worker_bin, ErrorCode::WorkerStartFailed)?;
    let bundle_root = bundle.root()?;
    let engine = serve_engine(binary.sha256, bundle_root.clone())?;
    let hardware = probe_machine(&bundle_root)?;
    Ok(ServeIdentity {
        context_tokens: placement.context_reservation_tokens,
        engine: EngineReceiptBindingV1 {
            engine_build_root: engine.root()?,
            backend: "native".into(),
            backend_version,
            binary_sha256: engine.binary_sha256.clone(),
            kernel_bundle_root: bundle_root,
            kernel_plan_root: kernel_plan_root.clone(),
        },
        hardware: HardwareReceiptBindingV1 {
            hardware_root: hardware.root()?,
            gpu_model: hardware.gpus.first().map(|gpu| gpu.model.clone()),
            compute_capability: hardware
                .gpus
                .first()
                .map(|gpu| gpu.compute_capability.clone()),
            device_slot,
        },
        kernel_plan_root,
        kv_block_size: placement.kv_block_size,
        kv_precision: placement.kv_precision.clone(),
        placement_root: placement.root()?,
    })
}

#[cfg(not(feature = "cuda"))]
fn serve_target(
    runtime: DigestHex,
    weight_bytes: u64,
) -> Result<
    (
        PlacementPlanV1,
        KernelBundleDescriptorV1,
        DigestHex,
        String,
        String,
    ),
    InferFailure,
> {
    let placement = cpu_placement_bytes(runtime, weight_bytes)?;
    let bundle = reference_kernel_bundle()?;
    let plan = reference_kernel_plan_root()?;
    Ok((
        placement,
        bundle,
        plan,
        "reference-f32".into(),
        "cpu".into(),
    ))
}

#[cfg(feature = "cuda")]
fn serve_target(
    runtime: DigestHex,
    weight_bytes: u64,
) -> Result<
    (
        PlacementPlanV1,
        KernelBundleDescriptorV1,
        DigestHex,
        String,
        String,
    ),
    InferFailure,
> {
    let slot = require_cuda_slot0()?;
    let placement = cuda_placement_bytes(runtime, weight_bytes)?;
    let bundle = cuda_kernel_bundle(
        CANDLE_CPU_VERSION,
        &slot.toolkit_version,
        &slot.architecture,
    )?;
    let plan = cuda_kernel_plan_root()?;
    Ok((
        placement,
        bundle,
        plan,
        CANDLE_CPU_VERSION.into(),
        "slot-0".into(),
    ))
}

#[cfg(not(feature = "cuda"))]
fn serve_engine(
    binary: DigestHex,
    bundle_root: DigestHex,
) -> Result<EngineBuildDescriptorV1, InferFailure> {
    reference_engine_build(binary, bundle_root)
}

#[cfg(feature = "cuda")]
fn serve_engine(
    binary: DigestHex,
    bundle_root: DigestHex,
) -> Result<EngineBuildDescriptorV1, InferFailure> {
    cuda_engine_build(binary, CANDLE_CPU_VERSION, bundle_root)
}

pub(crate) fn open_request(
    home: &Path,
    identity: &ServeIdentity,
    pinned: &PinnedModel,
    draft: &RequestDraft<'_>,
) -> Result<OpenedRequest, InferFailure> {
    let limits = LimitsV1 {
        max_output_tokens: draft.sampler.settings.max_output_tokens,
        max_prompt_tokens: identity.context_tokens,
        deadline_unix_micros: None,
    };
    let runtime = DigestHex::parse(&pinned.runtime_root)?;
    let intent = InferenceIntentV1 {
        model_runtime_root: runtime.clone(),
        prompt_plan_root: draft.prompt.root()?,
        sampler_plan_root: draft.sampler.root()?,
        grammar_plan_root: None,
        evidence_binding_root: None,
        requested_mode: "pinned".into(),
        limits_root: limits.root()?,
        extensions: BTreeMap::new(),
    };
    let execution = ExecutionPlanV1 {
        intent_root: intent.root()?,
        placement_root: identity.placement_root.clone(),
        kernel_plan_root: identity.kernel_plan_root.clone(),
        scheduling_mode: "continuous".into(),
        prefill_chunk_tokens: PREFILL_CHUNK_TOKENS,
        kv_block_size: identity.kv_block_size,
        prefix_cache_enabled: false,
        prefix_cache_namespace: "off".into(),
        speculative_plan_root: None,
        receipt_policy: policy(draft.openai, draft.stream).into(),
        mode: "pinned".into(),
        extensions: BTreeMap::new(),
    };
    execution.validate()?;
    let mut journal = Journal::create(home, draft.request_id)?;
    if let Err(err) = write_sidecars(&journal, draft.sampler, &execution) {
        let _ = fs::remove_dir_all(journal.dir());
        return Err(err);
    }
    if let Err(err) = journal.append("accepted", intent.root()?) {
        let _ = fs::remove_dir_all(journal.dir());
        return Err(err);
    }
    Ok(OpenedRequest {
        class: draft.class.to_string(),
        engine: identity.engine.clone(),
        execution,
        hardware: identity.hardware.clone(),
        intent_root: intent.root()?,
        journal,
        model: model_binding(pinned, runtime)?,
        placement: PlacementReceiptBindingV1 {
            placement_root: identity.placement_root.clone(),
            kv_block_size: identity.kv_block_size,
            kv_precision: identity.kv_precision.clone(),
        },
        prompt: PromptReceiptBindingV1 {
            messages_root: draft.prompt.messages_root.clone(),
            tools_root: draft.prompt.tools_root.clone(),
            evidence_root: draft.prompt.evidence_root.clone(),
            rendered_text_root: draft.prompt.rendered_text_root()?,
            token_id_root: draft.prompt.token_id_root()?,
            prompt_token_count: u32::try_from(draft.prompt.token_ids.len()).map_err(|_| {
                fail(
                    ErrorCode::ContextLimitExceeded,
                    "prompt does not fit in the reserved context",
                )
            })?,
            truncation_root: draft.prompt.truncation.root()?,
        },
        request_id: draft.request_id.to_string(),
        sampler: SamplerReceiptBindingV1 {
            sampler_plan_root: draft.sampler.root()?,
            temperature_micros: draft.sampler.settings.temperature_micros,
            seed: draft.sampler.seed,
            rng: draft.sampler.rng.clone(),
        },
        sealed: false,
        started: Instant::now(),
    })
}

impl OpenedRequest {
    pub(crate) fn publish(
        &mut self,
        tokens: &[u32],
        text: &str,
        finish: &str,
        home: &Path,
    ) -> Result<Published, InferFailure> {
        if self.sealed {
            return Err(fail(
                ErrorCode::ReceiptPersistFailed,
                "request journal is already sealed",
            ));
        }
        match self.write_receipt(tokens, text, finish, home) {
            Ok(published) => {
                self.sealed = true;
                Ok(published)
            }
            Err(err) => Err(self.seal_failed(err)),
        }
    }

    pub(crate) fn seal_cancelled(&mut self) -> Result<(), InferFailure> {
        if self.sealed {
            return Ok(());
        }
        self.journal.append("cancelled", trace_text("cancelled")?)?;
        self.sealed = true;
        Ok(())
    }

    pub(crate) fn seal_failed(&mut self, err: InferFailure) -> InferFailure {
        if !self.sealed {
            let payload = trace_text(err.code.as_str()).or_else(|_| trace_text("failed"));
            if let Ok(payload) = payload {
                if self.journal.append("failed", payload).is_ok() {
                    self.sealed = true;
                }
            }
        }
        err
    }

    fn write_receipt(
        &mut self,
        tokens: &[u32],
        text: &str,
        finish: &str,
        home: &Path,
    ) -> Result<Published, InferFailure> {
        if finish != "stop" && finish != "length" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "only a finished completion publishes a receipt",
            ));
        }
        self.journal
            .append("prefill", self.prompt.token_id_root.clone())?;
        let mut rolling = Vec::new();
        for token in tokens {
            rolling.push(*token);
            self.journal.append("token", output_token_root(&rolling)?)?;
        }
        let completed = self
            .journal
            .append("completed", output_token_root(tokens)?)?;
        let token_count = u32::try_from(tokens.len())
            .map_err(|_| fail(ErrorCode::ContractInvalid, "output token count overflows"))?;
        let elapsed = u64::try_from(self.started.elapsed().as_micros()).unwrap_or(u64::MAX);
        let mut extensions = BTreeMap::new();
        extensions.insert(
            "knolo.request-id".into(),
            CborValue::Text(self.request_id.clone()),
        );
        let mut receipt = InferenceReceiptV1 {
            receipt_id: self.prompt.token_id_root.clone(),
            intent_root: self.intent_root.clone(),
            model: self.model.clone(),
            engine: self.engine.clone(),
            hardware: self.hardware.clone(),
            placement: self.placement.clone(),
            prompt: self.prompt.clone(),
            knowledge: None,
            sampler: self.sampler.clone(),
            execution: ExecutionReceiptBindingV1 {
                execution_plan_root: self.execution.root()?,
                scheduling_mode: self.execution.scheduling_mode.clone(),
                prefill_chunk_tokens: self.execution.prefill_chunk_tokens,
                kv_block_size: self.execution.kv_block_size,
                prefix_cache_hit_tokens: 0,
                batch_trace_root: batch_trace(
                    self.prompt.prompt_token_count,
                    token_count,
                    &self.class,
                )?,
                speculative_plan_root: None,
                event_trace_root: completed,
                attempt: 1,
            },
            output: OutputReceiptBindingV1 {
                output_token_root: output_token_root(tokens)?,
                output_text_root: output_text_root(text)?,
                token_count,
                finish_reason: finish.to_string(),
                structured_output_root: None,
            },
            timing: TimingReceiptV1 {
                queue_micros: 0,
                prefill_micros: 0,
                decode_micros: elapsed,
                total_micros: elapsed,
            },
            assurance: ASSURANCE.into(),
            previous_receipt_root: None,
            extensions,
            signatures: Vec::new(),
        };
        receipt.receipt_id = receipt.computed_id()?;
        store_receipt(home, &receipt)?;
        Ok(Published {
            receipt_id: receipt.receipt_id.to_string(),
        })
    }
}

impl Drop for OpenedRequest {
    fn drop(&mut self) {
        if self.sealed {
            return;
        }
        let Ok(payload) = trace_text("unfinished") else {
            return;
        };
        if self.journal.append("failed", payload).is_ok() {
            self.sealed = true;
        }
    }
}

pub(crate) fn read_receipt(home: &Path, digest: &str) -> Result<Vec<u8>, InferFailure> {
    if digest.contains('/') || digest.contains('\\') {
        return Err(fail(ErrorCode::DigestInvalid, "receipt digest is invalid"));
    }
    let parsed = DigestHex::parse(digest)?;
    let hex = parsed
        .as_str()
        .strip_prefix("sha256-")
        .ok_or_else(|| fail(ErrorCode::DigestInvalid, "receipt digest is invalid"))?;
    let path = home
        .join("receipts")
        .join("sha256")
        .join(hex)
        .join("receipt.cbor");
    let bytes =
        fs::read(&path).map_err(|_| fail(ErrorCode::ReceiptRequired, "receipt is missing"))?;
    let receipt = infer_contracts::InferenceReceiptV1::from_cbor(&decode_canonical(&bytes)?)?;
    if receipt.to_bytes()? != bytes || receipt.receipt_id.as_str() != parsed.as_str() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "stored receipt bytes are not canonical",
        ));
    }
    let request_id = match receipt.extensions.get("knolo.request-id") {
        Some(CborValue::Text(id)) => id.clone(),
        _ => String::new(),
    };
    Ok(serde_json::to_vec(&serde_json::json!({
        "assurance": receipt.assurance,
        "receiptRoot": receipt.receipt_id.as_str(),
        "requestId": request_id,
    }))
    .expect("receipt json"))
}

fn policy(openai: bool, stream: bool) -> &'static str {
    if openai {
        "compatibility"
    } else if stream {
        "durable-stream"
    } else {
        "atomic-verified"
    }
}

fn write_sidecars(
    journal: &Journal,
    sampler: &SamplerPlanV1,
    execution: &ExecutionPlanV1,
) -> Result<(), InferFailure> {
    journal.write_sampler(sampler)?;
    write_atomic(
        &journal.dir().join("execution-plan.cbor"),
        &execution.to_bytes()?,
        ErrorCode::ReceiptPersistFailed,
    )
}

fn model_binding(
    pinned: &PinnedModel,
    runtime: DigestHex,
) -> Result<ModelReceiptBindingV1, InferFailure> {
    Ok(ModelReceiptBindingV1 {
        model_image_root: pinned.image_root.clone(),
        artifact_root: pinned.artifact_root.clone(),
        model_runtime_root: runtime,
        architecture_adapter_id: pinned.image.architecture.adapter.clone(),
        architecture_adapter_root: pinned.image.architecture_adapter_root()?,
        config_root: pinned.image.config_root()?,
        tokenizer_root: pinned.image.tokenizer.root.clone(),
        template_root: pinned.image.template.root.clone(),
        storage_precision: "f32".into(),
        compute_precision: "f32".into(),
    })
}

fn store_receipt(home: &Path, receipt: &InferenceReceiptV1) -> Result<(), InferFailure> {
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
    )
}

fn batch_trace(
    prompt_tokens: u32,
    decode_steps: u32,
    class: &str,
) -> Result<DigestHex, InferFailure> {
    let mut map = BTreeMap::new();
    map.insert(
        "decodeSteps".into(),
        CborValue::Integer(i128::from(decode_steps)),
    );
    map.insert(
        "promptTokens".into(),
        CborValue::Integer(i128::from(prompt_tokens)),
    );
    map.insert(
        "schedulingMode".into(),
        CborValue::Text("continuous".into()),
    );
    map.insert("serviceClass".into(), CborValue::Text(class.to_string()));
    digest_value("infer-execution-trace", &CborValue::map(map))
}

fn trace_text(text: &str) -> Result<DigestHex, InferFailure> {
    digest_value("infer-execution-trace", &CborValue::Text(text.to_string()))
}

fn io_receipt(err: std::io::Error) -> InferFailure {
    fail(
        ErrorCode::ReceiptPersistFailed,
        format!("receipt store: {err}"),
    )
}
