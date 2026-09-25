mod admission;
mod assurance;
mod boundary;
mod census;
mod common;
mod compare;
mod continuity;
mod demo;
mod execution;
mod exposure;
mod gate;
mod incident;
mod lifecycle;
mod model;
mod preflight;
mod product;
mod prompt;
mod receipt;
mod refusal;
mod release;
mod resilience;
mod runtime;

pub use admission::{
    ConcurrentLoadReportV1, DuplicateReportV1, EvictionReportV1, PointReportV1,
    MAX_ADMISSION_REQUESTS,
};
pub use assurance::{
    ChallengeReportV1, DrainingReportV1, ReplayEnvironmentReportV1, ReplayOutputReportV1,
    WorkerLostReportV1, HTTP_SERVICE_UNAVAILABLE,
};
pub use boundary::{
    ApiReportV1, HostKeyReportV1, ReceiptKeyReportV1, SandboxReportV1, MAX_API_BODY, MAX_API_RATE,
    MAX_SANDBOX_MEMORY, MAX_SHARED_MEMORY,
};
pub use census::{
    kv_utilization_millionths, KvReportV1, PeakReportV1, PrefixReportV1, MAX_PEAK_BYTES,
    MICRO_KV_PAGES, MICRO_KV_PAGE_TOKENS,
};
pub use common::{
    artifact_root, validate_relative_path, ArtifactFileV1, EmbeddedArtifactV1, FixedPointSamplerV1,
    SignatureV1, SpecialTokensV1, MAX_EMBEDDED_BYTES,
};
pub use compare::{
    LlamaReportV1, MistralReportV1, RecipeReportV1, VllmReportV1, BACKEND_REPORTED, LLAMA_FAMILY,
    LLAMA_RUNTIME, MICRO_COMPARISON_CONTEXT, MICRO_COMPARISON_QUANTIZATION, MISTRAL_FAMILY,
    MISTRAL_RUNTIME, NATIVE_VERIFIED, RECIPE_ADAPTER, SIDECAR_ARTIFACT_VERIFIED, VLLM_FAMILY,
    VLLM_RUNTIME,
};
pub use continuity::{CurveReportV1, DisconnectReportV1, ReceiptStoreReportV1, RollbackReportV1};
pub use demo::{
    ChainReportV1, EvidenceOutputReportV1, InstallReportV1, NoticeReportV1, CHAIN_LINK_COUNT,
    INSTALL_ARTIFACT_COUNT,
};
pub use execution::{
    ExecutionPlanV1, GrammarPlanV1, InferenceEventV1, InferenceIntentV1, LimitsV1, SamplerPlanV1,
    SAMPLER_ORDER_V1,
};
pub use exposure::{
    CacheChannelReportV1, EquationReportV1, RedactionReportV1, SafeErrorReportV1,
    ED25519_PUBLIC_KEY_BYTES, ED25519_SIGNATURE_BYTES,
};
pub use gate::{
    ContextLimitReportV1, ImageInvalidReportV1, ImageSignatureReportV1, PromptCompilationReportV1,
    SignatureCheckReportV1, MAX_CONTEXT_PROMPT, MAX_REJECTED_TOKEN, MAX_SIGNATURE_RECORD_BYTES,
    MAX_SIGNATURE_RECORD_COUNT,
};
pub use incident::{
    EqualityReportV1, FaultReportV1, OomReportV1, TimeoutReportV1, WorkerStartReportV1,
    HTTP_TIMEOUT_STATUS,
};
pub use lifecycle::{
    model_swap_nanos, LoadReportV1, SwapReportV1, VerificationReportV1, MAX_VERIFIED_BYTES,
};
pub use model::{
    ArchitectureRefV1, LicenseV1, ModelArtifactSetV1, ModelImageV1, PlacementHintsV1,
    ResourceRequirementsV1, SourceHintV1, TensorSpecV1, MODEL_IMAGE_KIND,
};
pub use preflight::{
    ArchitectureReportV1, DigestMismatchReportV1, ScalarReportV1, TemplateInvalidReportV1,
    TokenizerInvalidReportV1,
};
pub use product::{
    AgentEffectReportV1, CompositionReportV1, HubReportV1, StudioReportV1, LOCAL_WEIGHT_SOURCE,
    MICRO_ADAPTER, MICRO_CONTEXT,
};
pub use prompt::{
    prompt_token_root, ChatMessageV1, EvidenceBindingV1, PromptInputV1, PromptPlanV1, TruncationV1,
};
pub use receipt::{
    cancellation_latency_nanos, conversion_config_root, logit_root, output_text_root,
    output_token_root, receipt_finalization_nanos, receipt_overhead_nanos, request_latency_nanos,
    time_per_output_token_nanos, tokens_per_second_micros, CancellationReportV1,
    ConversionReceiptV1, EngineReceiptBindingV1, ExecutionReceiptBindingV1, FinalizationReportV1,
    FuzzReportV1, HardwareReceiptBindingV1, InferenceReceiptV1, LatencyReportV1,
    MemoryEstimateReportV1, ModelConformanceReceiptV1, ModelReceiptBindingV1,
    OutputReceiptBindingV1, OverheadReportV1, PerplexityReportV1, PlacementReceiptBindingV1,
    PromptReceiptBindingV1, ReplayCheckReceiptV1, SamplerReceiptBindingV1, ThroughputReportV1,
    TimingReceiptV1, FUZZ_MUTATIONS_PER_SEED, FUZZ_MUTATION_COUNT, FUZZ_SEED_COUNT, FUZZ_TARGETS,
};
pub use refusal::{
    KernelReportV1, MemoryRefusalReportV1, PlacementRefusalReportV1, PublicReportV1,
    QuantizationReportV1,
};
pub use release::{
    BinaryReportV1, ReleaseReportV1, ReproducibleReportV1, SignatureReportV1, MAX_BINARY_BYTES,
    MAX_BUILD_INSTRUCTIONS, SUPERVISOR_BINARY, WORKER_BINARY,
};
pub use resilience::{
    BaseReportV1, DiskReportV1, RestartReportV1, UnloadReportV1, ED25519_SCALAR_BYTES,
    MAX_DISK_NEEDED_BYTES, MAX_UNLOAD_QUEUE,
};
pub use runtime::{
    EngineBuildDescriptorV1, GpuProbeV1, HardwareProbeV1, JitKernelV1, KernelBundleDescriptorV1,
    PlacementPlanV1, TensorGroupV1,
};

use crate::cbor::{decode_canonical, CborValue};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::Fields;

use admission::{CONCURRENT_LOAD_KIND, DUPLICATE_KIND, EVICTION_KIND, POINT_KIND};
use assurance::{
    CHALLENGE_KIND, DRAINING_KIND, REPLAY_ENVIRONMENT_KIND, REPLAY_OUTPUT_KIND, WORKER_LOST_KIND,
};
use boundary::{API_KIND, HOST_KEY_KIND, RECEIPT_KEY_KIND, SANDBOX_KIND};
use census::{KV_KIND, PEAK_KIND, PREFIX_KIND};
use compare::{LLAMA_KIND, MISTRAL_KIND, RECIPE_KIND, VLLM_KIND};
use continuity::{CURVE_KIND, DISCONNECT_KIND, RECEIPT_STORE_KIND, ROLLBACK_KIND};
use demo::{CHAIN_KIND, EVIDENCE_OUTPUT_KIND, INSTALL_KIND, NOTICE_KIND};
use execution::{
    EXECUTION_PLAN_KIND, GRAMMAR_PLAN_KIND, INFERENCE_EVENT_KIND, INFERENCE_INTENT_KIND,
    SAMPLER_PLAN_KIND,
};
use exposure::{CACHE_CHANNEL_KIND, EQUATION_KIND, REDACTION_KIND, SAFE_ERROR_KIND};
use gate::{
    CONTEXT_LIMIT_KIND, IMAGE_INVALID_KIND, IMAGE_SIGNATURE_KIND, PROMPT_COMPILATION_KIND,
    SIGNATURE_CHECK_KIND,
};
use incident::{EQUALITY_KIND, FAULT_KIND, OOM_KIND, TIMEOUT_KIND, WORKER_START_KIND};
use lifecycle::{LOAD_KIND, SWAP_KIND, VERIFICATION_KIND};
use model::MODEL_ARTIFACT_KIND;
use preflight::{
    ARCHITECTURE_KIND, DIGEST_MISMATCH_KIND, SCALAR_KIND, TEMPLATE_INVALID_KIND,
    TOKENIZER_INVALID_KIND,
};
use product::{AGENT_EFFECT_KIND, COMPOSITION_KIND, HUB_KIND, STUDIO_KIND};
use prompt::{EVIDENCE_KIND, PROMPT_INPUT_KIND, PROMPT_PLAN_KIND};
use receipt::{
    CANCELLATION_KIND, CONFORMANCE_KIND, CONVERSION_KIND, FINALIZATION_KIND, FUZZ_KIND,
    LATENCY_KIND, MEMORY_KIND, OVERHEAD_KIND, PERPLEXITY_KIND, RECEIPT_KIND, REPLAY_KIND,
    THROUGHPUT_KIND,
};
use refusal::{
    KERNEL_KIND, MEMORY_REFUSAL_KIND, PLACEMENT_REFUSAL_KIND, PUBLIC_KIND, QUANTIZATION_KIND,
};
use release::{BINARY_KIND, RELEASE_KIND, REPRODUCIBLE_KIND, SIGNATURE_KIND};
use resilience::{BASE_KIND, DISK_KIND, RESTART_KIND, UNLOAD_KIND};
use runtime::{ENGINE_BUILD_KIND, HARDWARE_PROBE_KIND, KERNEL_BUNDLE_KIND, PLACEMENT_PLAN_KIND};

/// Decoded documents are large by design. Callers move one contract at a time.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Contract {
    ModelImage(ModelImageV1),
    ModelArtifactSet(ModelArtifactSetV1),
    EngineBuild(EngineBuildDescriptorV1),
    KernelBundle(KernelBundleDescriptorV1),
    HardwareProbe(HardwareProbeV1),
    PlacementPlan(PlacementPlanV1),
    PromptInput(PromptInputV1),
    PromptPlan(PromptPlanV1),
    EvidenceBinding(EvidenceBindingV1),
    SamplerPlan(SamplerPlanV1),
    GrammarPlan(GrammarPlanV1),
    InferenceIntent(InferenceIntentV1),
    ExecutionPlan(ExecutionPlanV1),
    InferenceEvent(InferenceEventV1),
    InferenceReceipt(InferenceReceiptV1),
    ReplayCheck(ReplayCheckReceiptV1),
    ModelConformance(ModelConformanceReceiptV1),
    ConversionReceipt(ConversionReceiptV1),
    PerplexityReport(PerplexityReportV1),
    MemoryEstimate(MemoryEstimateReportV1),
    ThroughputReport(ThroughputReportV1),
    LatencyReport(LatencyReportV1),
    FuzzReport(FuzzReportV1),
    CancellationReport(CancellationReportV1),
    FinalizationReport(FinalizationReportV1),
    OverheadReport(OverheadReportV1),
    SwapReport(SwapReportV1),
    VerificationReport(VerificationReportV1),
    LoadReport(LoadReportV1),
    PeakReport(PeakReportV1),
    KvReport(KvReportV1),
    PrefixReport(PrefixReportV1),
    LlamaReport(LlamaReportV1),
    VllmReport(VllmReportV1),
    MistralReport(MistralReportV1),
    RecipeReport(RecipeReportV1),
    CompositionReport(CompositionReportV1),
    AgentEffectReport(AgentEffectReportV1),
    HubReport(HubReportV1),
    StudioReport(StudioReportV1),
    ChainReport(ChainReportV1),
    EvidenceOutputReport(EvidenceOutputReportV1),
    InstallReport(InstallReportV1),
    NoticeReport(NoticeReportV1),
    ReleaseReport(ReleaseReportV1),
    BinaryReport(BinaryReportV1),
    ReproducibleReport(ReproducibleReportV1),
    SignatureReport(SignatureReportV1),
    HostKeyReport(HostKeyReportV1),
    ReceiptKeyReport(ReceiptKeyReportV1),
    SandboxReport(SandboxReportV1),
    ApiReport(ApiReportV1),
    CacheChannelReport(CacheChannelReportV1),
    EquationReport(EquationReportV1),
    SafeErrorReport(SafeErrorReportV1),
    RedactionReport(RedactionReportV1),
    CurveReport(CurveReportV1),
    RollbackReport(RollbackReportV1),
    DisconnectReport(DisconnectReportV1),
    ReceiptStoreReport(ReceiptStoreReportV1),
    BaseReport(BaseReportV1),
    DiskReport(DiskReportV1),
    UnloadReport(UnloadReportV1),
    RestartReport(RestartReportV1),
    PointReport(PointReportV1),
    DuplicateReport(DuplicateReportV1),
    ConcurrentLoadReport(ConcurrentLoadReportV1),
    EvictionReport(EvictionReportV1),
    EqualityReport(EqualityReportV1),
    OomReport(OomReportV1),
    FaultReport(FaultReportV1),
    TimeoutReport(TimeoutReportV1),
    WorkerStartReport(WorkerStartReportV1),
    ChallengeReport(ChallengeReportV1),
    ReplayEnvironmentReport(ReplayEnvironmentReportV1),
    ReplayOutputReport(ReplayOutputReportV1),
    WorkerLostReport(WorkerLostReportV1),
    DrainingReport(DrainingReportV1),
    ScalarReport(ScalarReportV1),
    DigestMismatchReport(DigestMismatchReportV1),
    TokenizerInvalidReport(TokenizerInvalidReportV1),
    TemplateInvalidReport(TemplateInvalidReportV1),
    ArchitectureReport(ArchitectureReportV1),
    PublicReport(PublicReportV1),
    QuantizationReport(QuantizationReportV1),
    KernelReport(KernelReportV1),
    PlacementRefusalReport(PlacementRefusalReportV1),
    MemoryRefusalReport(MemoryRefusalReportV1),
    SignatureCheckReport(SignatureCheckReportV1),
    ContextLimitReport(ContextLimitReportV1),
    PromptCompilationReport(PromptCompilationReportV1),
    ImageInvalidReport(ImageInvalidReportV1),
    ImageSignatureReport(ImageSignatureReportV1),
}

impl Contract {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, InferFailure> {
        Self::from_cbor(&decode_canonical(bytes)?)
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let fields = Fields::parse(value)?;
        let kind = fields.peek_text("kind")?;
        match kind.as_str() {
            MODEL_IMAGE_KIND => Ok(Self::ModelImage(ModelImageV1::from_cbor(value)?)),
            MODEL_ARTIFACT_KIND => Ok(Self::ModelArtifactSet(ModelArtifactSetV1::from_cbor(
                value,
            )?)),
            ENGINE_BUILD_KIND => Ok(Self::EngineBuild(EngineBuildDescriptorV1::from_cbor(
                value,
            )?)),
            KERNEL_BUNDLE_KIND => Ok(Self::KernelBundle(KernelBundleDescriptorV1::from_cbor(
                value,
            )?)),
            HARDWARE_PROBE_KIND => Ok(Self::HardwareProbe(HardwareProbeV1::from_cbor(value)?)),
            PLACEMENT_PLAN_KIND => Ok(Self::PlacementPlan(PlacementPlanV1::from_cbor(value)?)),
            PROMPT_INPUT_KIND => Ok(Self::PromptInput(PromptInputV1::from_cbor(value)?)),
            PROMPT_PLAN_KIND => Ok(Self::PromptPlan(PromptPlanV1::from_cbor(value)?)),
            EVIDENCE_KIND => Ok(Self::EvidenceBinding(EvidenceBindingV1::from_cbor(value)?)),
            SAMPLER_PLAN_KIND => Ok(Self::SamplerPlan(SamplerPlanV1::from_cbor(value)?)),
            GRAMMAR_PLAN_KIND => Ok(Self::GrammarPlan(GrammarPlanV1::from_cbor(value)?)),
            INFERENCE_INTENT_KIND => {
                Ok(Self::InferenceIntent(InferenceIntentV1::from_cbor(value)?))
            }
            EXECUTION_PLAN_KIND => Ok(Self::ExecutionPlan(ExecutionPlanV1::from_cbor(value)?)),
            INFERENCE_EVENT_KIND => Ok(Self::InferenceEvent(InferenceEventV1::from_cbor(value)?)),
            RECEIPT_KIND => Ok(Self::InferenceReceipt(InferenceReceiptV1::from_cbor(
                value,
            )?)),
            REPLAY_KIND => Ok(Self::ReplayCheck(ReplayCheckReceiptV1::from_cbor(value)?)),
            CONFORMANCE_KIND => Ok(Self::ModelConformance(
                ModelConformanceReceiptV1::from_cbor(value)?,
            )),
            CONVERSION_KIND => Ok(Self::ConversionReceipt(ConversionReceiptV1::from_cbor(
                value,
            )?)),
            PERPLEXITY_KIND => Ok(Self::PerplexityReport(PerplexityReportV1::from_cbor(
                value,
            )?)),
            MEMORY_KIND => Ok(Self::MemoryEstimate(MemoryEstimateReportV1::from_cbor(
                value,
            )?)),
            THROUGHPUT_KIND => Ok(Self::ThroughputReport(ThroughputReportV1::from_cbor(
                value,
            )?)),
            LATENCY_KIND => Ok(Self::LatencyReport(LatencyReportV1::from_cbor(value)?)),
            FUZZ_KIND => Ok(Self::FuzzReport(FuzzReportV1::from_cbor(value)?)),
            CANCELLATION_KIND => Ok(Self::CancellationReport(CancellationReportV1::from_cbor(
                value,
            )?)),
            FINALIZATION_KIND => Ok(Self::FinalizationReport(FinalizationReportV1::from_cbor(
                value,
            )?)),
            OVERHEAD_KIND => Ok(Self::OverheadReport(OverheadReportV1::from_cbor(value)?)),
            SWAP_KIND => Ok(Self::SwapReport(SwapReportV1::from_cbor(value)?)),
            VERIFICATION_KIND => Ok(Self::VerificationReport(VerificationReportV1::from_cbor(
                value,
            )?)),
            LOAD_KIND => Ok(Self::LoadReport(LoadReportV1::from_cbor(value)?)),
            PEAK_KIND => Ok(Self::PeakReport(PeakReportV1::from_cbor(value)?)),
            KV_KIND => Ok(Self::KvReport(KvReportV1::from_cbor(value)?)),
            PREFIX_KIND => Ok(Self::PrefixReport(PrefixReportV1::from_cbor(value)?)),
            LLAMA_KIND => Ok(Self::LlamaReport(LlamaReportV1::from_cbor(value)?)),
            VLLM_KIND => Ok(Self::VllmReport(VllmReportV1::from_cbor(value)?)),
            MISTRAL_KIND => Ok(Self::MistralReport(MistralReportV1::from_cbor(value)?)),
            RECIPE_KIND => Ok(Self::RecipeReport(RecipeReportV1::from_cbor(value)?)),
            COMPOSITION_KIND => Ok(Self::CompositionReport(CompositionReportV1::from_cbor(
                value,
            )?)),
            AGENT_EFFECT_KIND => Ok(Self::AgentEffectReport(AgentEffectReportV1::from_cbor(
                value,
            )?)),
            HUB_KIND => Ok(Self::HubReport(HubReportV1::from_cbor(value)?)),
            STUDIO_KIND => Ok(Self::StudioReport(StudioReportV1::from_cbor(value)?)),
            CHAIN_KIND => Ok(Self::ChainReport(ChainReportV1::from_cbor(value)?)),
            EVIDENCE_OUTPUT_KIND => Ok(Self::EvidenceOutputReport(
                EvidenceOutputReportV1::from_cbor(value)?,
            )),
            INSTALL_KIND => Ok(Self::InstallReport(InstallReportV1::from_cbor(value)?)),
            NOTICE_KIND => Ok(Self::NoticeReport(NoticeReportV1::from_cbor(value)?)),
            RELEASE_KIND => Ok(Self::ReleaseReport(ReleaseReportV1::from_cbor(value)?)),
            BINARY_KIND => Ok(Self::BinaryReport(BinaryReportV1::from_cbor(value)?)),
            REPRODUCIBLE_KIND => Ok(Self::ReproducibleReport(ReproducibleReportV1::from_cbor(
                value,
            )?)),
            SIGNATURE_KIND => Ok(Self::SignatureReport(SignatureReportV1::from_cbor(value)?)),
            HOST_KEY_KIND => Ok(Self::HostKeyReport(HostKeyReportV1::from_cbor(value)?)),
            RECEIPT_KEY_KIND => Ok(Self::ReceiptKeyReport(ReceiptKeyReportV1::from_cbor(
                value,
            )?)),
            SANDBOX_KIND => Ok(Self::SandboxReport(SandboxReportV1::from_cbor(value)?)),
            API_KIND => Ok(Self::ApiReport(ApiReportV1::from_cbor(value)?)),
            CACHE_CHANNEL_KIND => Ok(Self::CacheChannelReport(CacheChannelReportV1::from_cbor(
                value,
            )?)),
            EQUATION_KIND => Ok(Self::EquationReport(EquationReportV1::from_cbor(value)?)),
            SAFE_ERROR_KIND => Ok(Self::SafeErrorReport(SafeErrorReportV1::from_cbor(value)?)),
            REDACTION_KIND => Ok(Self::RedactionReport(RedactionReportV1::from_cbor(value)?)),
            CURVE_KIND => Ok(Self::CurveReport(CurveReportV1::from_cbor(value)?)),
            ROLLBACK_KIND => Ok(Self::RollbackReport(RollbackReportV1::from_cbor(value)?)),
            DISCONNECT_KIND => Ok(Self::DisconnectReport(DisconnectReportV1::from_cbor(
                value,
            )?)),
            RECEIPT_STORE_KIND => Ok(Self::ReceiptStoreReport(ReceiptStoreReportV1::from_cbor(
                value,
            )?)),
            BASE_KIND => Ok(Self::BaseReport(BaseReportV1::from_cbor(value)?)),
            DISK_KIND => Ok(Self::DiskReport(DiskReportV1::from_cbor(value)?)),
            UNLOAD_KIND => Ok(Self::UnloadReport(UnloadReportV1::from_cbor(value)?)),
            RESTART_KIND => Ok(Self::RestartReport(RestartReportV1::from_cbor(value)?)),
            POINT_KIND => Ok(Self::PointReport(PointReportV1::from_cbor(value)?)),
            DUPLICATE_KIND => Ok(Self::DuplicateReport(DuplicateReportV1::from_cbor(value)?)),
            CONCURRENT_LOAD_KIND => Ok(Self::ConcurrentLoadReport(
                ConcurrentLoadReportV1::from_cbor(value)?,
            )),
            EVICTION_KIND => Ok(Self::EvictionReport(EvictionReportV1::from_cbor(value)?)),
            EQUALITY_KIND => Ok(Self::EqualityReport(EqualityReportV1::from_cbor(value)?)),
            OOM_KIND => Ok(Self::OomReport(OomReportV1::from_cbor(value)?)),
            FAULT_KIND => Ok(Self::FaultReport(FaultReportV1::from_cbor(value)?)),
            TIMEOUT_KIND => Ok(Self::TimeoutReport(TimeoutReportV1::from_cbor(value)?)),
            WORKER_START_KIND => Ok(Self::WorkerStartReport(WorkerStartReportV1::from_cbor(
                value,
            )?)),
            CHALLENGE_KIND => Ok(Self::ChallengeReport(ChallengeReportV1::from_cbor(value)?)),
            REPLAY_ENVIRONMENT_KIND => Ok(Self::ReplayEnvironmentReport(
                ReplayEnvironmentReportV1::from_cbor(value)?,
            )),
            REPLAY_OUTPUT_KIND => Ok(Self::ReplayOutputReport(ReplayOutputReportV1::from_cbor(
                value,
            )?)),
            WORKER_LOST_KIND => Ok(Self::WorkerLostReport(WorkerLostReportV1::from_cbor(
                value,
            )?)),
            DRAINING_KIND => Ok(Self::DrainingReport(DrainingReportV1::from_cbor(value)?)),
            SCALAR_KIND => Ok(Self::ScalarReport(ScalarReportV1::from_cbor(value)?)),
            DIGEST_MISMATCH_KIND => Ok(Self::DigestMismatchReport(
                DigestMismatchReportV1::from_cbor(value)?,
            )),
            TOKENIZER_INVALID_KIND => Ok(Self::TokenizerInvalidReport(
                TokenizerInvalidReportV1::from_cbor(value)?,
            )),
            TEMPLATE_INVALID_KIND => Ok(Self::TemplateInvalidReport(
                TemplateInvalidReportV1::from_cbor(value)?,
            )),
            ARCHITECTURE_KIND => Ok(Self::ArchitectureReport(ArchitectureReportV1::from_cbor(
                value,
            )?)),
            PUBLIC_KIND => Ok(Self::PublicReport(PublicReportV1::from_cbor(value)?)),
            QUANTIZATION_KIND => Ok(Self::QuantizationReport(QuantizationReportV1::from_cbor(
                value,
            )?)),
            KERNEL_KIND => Ok(Self::KernelReport(KernelReportV1::from_cbor(value)?)),
            PLACEMENT_REFUSAL_KIND => Ok(Self::PlacementRefusalReport(
                PlacementRefusalReportV1::from_cbor(value)?,
            )),
            MEMORY_REFUSAL_KIND => Ok(Self::MemoryRefusalReport(MemoryRefusalReportV1::from_cbor(
                value,
            )?)),
            SIGNATURE_CHECK_KIND => Ok(Self::SignatureCheckReport(
                SignatureCheckReportV1::from_cbor(value)?,
            )),
            CONTEXT_LIMIT_KIND => Ok(Self::ContextLimitReport(ContextLimitReportV1::from_cbor(
                value,
            )?)),
            PROMPT_COMPILATION_KIND => Ok(Self::PromptCompilationReport(
                PromptCompilationReportV1::from_cbor(value)?,
            )),
            IMAGE_INVALID_KIND => Ok(Self::ImageInvalidReport(ImageInvalidReportV1::from_cbor(
                value,
            )?)),
            IMAGE_SIGNATURE_KIND => Ok(Self::ImageSignatureReport(
                ImageSignatureReportV1::from_cbor(value)?,
            )),
            _ => Err(fail(ErrorCode::ContractInvalid, "unexpected contract kind")),
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::ModelImage(_) => MODEL_IMAGE_KIND,
            Self::ModelArtifactSet(_) => MODEL_ARTIFACT_KIND,
            Self::EngineBuild(_) => ENGINE_BUILD_KIND,
            Self::KernelBundle(_) => KERNEL_BUNDLE_KIND,
            Self::HardwareProbe(_) => HARDWARE_PROBE_KIND,
            Self::PlacementPlan(_) => PLACEMENT_PLAN_KIND,
            Self::PromptInput(_) => PROMPT_INPUT_KIND,
            Self::PromptPlan(_) => PROMPT_PLAN_KIND,
            Self::EvidenceBinding(_) => EVIDENCE_KIND,
            Self::SamplerPlan(_) => SAMPLER_PLAN_KIND,
            Self::GrammarPlan(_) => GRAMMAR_PLAN_KIND,
            Self::InferenceIntent(_) => INFERENCE_INTENT_KIND,
            Self::ExecutionPlan(_) => EXECUTION_PLAN_KIND,
            Self::InferenceEvent(_) => INFERENCE_EVENT_KIND,
            Self::InferenceReceipt(_) => RECEIPT_KIND,
            Self::ReplayCheck(_) => REPLAY_KIND,
            Self::ModelConformance(_) => CONFORMANCE_KIND,
            Self::ConversionReceipt(_) => CONVERSION_KIND,
            Self::PerplexityReport(_) => PERPLEXITY_KIND,
            Self::MemoryEstimate(_) => MEMORY_KIND,
            Self::ThroughputReport(_) => THROUGHPUT_KIND,
            Self::LatencyReport(_) => LATENCY_KIND,
            Self::FuzzReport(_) => FUZZ_KIND,
            Self::CancellationReport(_) => CANCELLATION_KIND,
            Self::FinalizationReport(_) => FINALIZATION_KIND,
            Self::OverheadReport(_) => OVERHEAD_KIND,
            Self::SwapReport(_) => SWAP_KIND,
            Self::VerificationReport(_) => VERIFICATION_KIND,
            Self::LoadReport(_) => LOAD_KIND,
            Self::PeakReport(_) => PEAK_KIND,
            Self::KvReport(_) => KV_KIND,
            Self::PrefixReport(_) => PREFIX_KIND,
            Self::LlamaReport(_) => LLAMA_KIND,
            Self::VllmReport(_) => VLLM_KIND,
            Self::MistralReport(_) => MISTRAL_KIND,
            Self::RecipeReport(_) => RECIPE_KIND,
            Self::CompositionReport(_) => COMPOSITION_KIND,
            Self::AgentEffectReport(_) => AGENT_EFFECT_KIND,
            Self::HubReport(_) => HUB_KIND,
            Self::StudioReport(_) => STUDIO_KIND,
            Self::ChainReport(_) => CHAIN_KIND,
            Self::EvidenceOutputReport(_) => EVIDENCE_OUTPUT_KIND,
            Self::InstallReport(_) => INSTALL_KIND,
            Self::NoticeReport(_) => NOTICE_KIND,
            Self::ReleaseReport(_) => RELEASE_KIND,
            Self::BinaryReport(_) => BINARY_KIND,
            Self::ReproducibleReport(_) => REPRODUCIBLE_KIND,
            Self::SignatureReport(_) => SIGNATURE_KIND,
            Self::HostKeyReport(_) => HOST_KEY_KIND,
            Self::ReceiptKeyReport(_) => RECEIPT_KEY_KIND,
            Self::SandboxReport(_) => SANDBOX_KIND,
            Self::ApiReport(_) => API_KIND,
            Self::CacheChannelReport(_) => CACHE_CHANNEL_KIND,
            Self::EquationReport(_) => EQUATION_KIND,
            Self::SafeErrorReport(_) => SAFE_ERROR_KIND,
            Self::RedactionReport(_) => REDACTION_KIND,
            Self::CurveReport(_) => CURVE_KIND,
            Self::RollbackReport(_) => ROLLBACK_KIND,
            Self::DisconnectReport(_) => DISCONNECT_KIND,
            Self::ReceiptStoreReport(_) => RECEIPT_STORE_KIND,
            Self::BaseReport(_) => BASE_KIND,
            Self::DiskReport(_) => DISK_KIND,
            Self::UnloadReport(_) => UNLOAD_KIND,
            Self::RestartReport(_) => RESTART_KIND,
            Self::PointReport(_) => POINT_KIND,
            Self::DuplicateReport(_) => DUPLICATE_KIND,
            Self::ConcurrentLoadReport(_) => CONCURRENT_LOAD_KIND,
            Self::EvictionReport(_) => EVICTION_KIND,
            Self::EqualityReport(_) => EQUALITY_KIND,
            Self::OomReport(_) => OOM_KIND,
            Self::FaultReport(_) => FAULT_KIND,
            Self::TimeoutReport(_) => TIMEOUT_KIND,
            Self::WorkerStartReport(_) => WORKER_START_KIND,
            Self::ChallengeReport(_) => CHALLENGE_KIND,
            Self::ReplayEnvironmentReport(_) => REPLAY_ENVIRONMENT_KIND,
            Self::ReplayOutputReport(_) => REPLAY_OUTPUT_KIND,
            Self::WorkerLostReport(_) => WORKER_LOST_KIND,
            Self::DrainingReport(_) => DRAINING_KIND,
            Self::ScalarReport(_) => SCALAR_KIND,
            Self::DigestMismatchReport(_) => DIGEST_MISMATCH_KIND,
            Self::TokenizerInvalidReport(_) => TOKENIZER_INVALID_KIND,
            Self::TemplateInvalidReport(_) => TEMPLATE_INVALID_KIND,
            Self::ArchitectureReport(_) => ARCHITECTURE_KIND,
            Self::PublicReport(_) => PUBLIC_KIND,
            Self::QuantizationReport(_) => QUANTIZATION_KIND,
            Self::KernelReport(_) => KERNEL_KIND,
            Self::PlacementRefusalReport(_) => PLACEMENT_REFUSAL_KIND,
            Self::MemoryRefusalReport(_) => MEMORY_REFUSAL_KIND,
            Self::SignatureCheckReport(_) => SIGNATURE_CHECK_KIND,
            Self::ContextLimitReport(_) => CONTEXT_LIMIT_KIND,
            Self::PromptCompilationReport(_) => PROMPT_COMPILATION_KIND,
            Self::ImageInvalidReport(_) => IMAGE_INVALID_KIND,
            Self::ImageSignatureReport(_) => IMAGE_SIGNATURE_KIND,
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        match self {
            Self::ModelImage(v) => v.to_bytes(),
            Self::ModelArtifactSet(v) => v.to_bytes(),
            Self::EngineBuild(v) => v.to_bytes(),
            Self::KernelBundle(v) => v.to_bytes(),
            Self::HardwareProbe(v) => v.to_bytes(),
            Self::PlacementPlan(v) => v.to_bytes(),
            Self::PromptInput(v) => v.to_bytes(),
            Self::PromptPlan(v) => v.to_bytes(),
            Self::EvidenceBinding(v) => v.to_bytes(),
            Self::SamplerPlan(v) => v.to_bytes(),
            Self::GrammarPlan(v) => v.to_bytes(),
            Self::InferenceIntent(v) => v.to_bytes(),
            Self::ExecutionPlan(v) => v.to_bytes(),
            Self::InferenceEvent(v) => v.to_bytes(),
            Self::InferenceReceipt(v) => v.to_bytes(),
            Self::ReplayCheck(v) => v.to_bytes(),
            Self::ModelConformance(v) => v.to_bytes(),
            Self::ConversionReceipt(v) => v.to_bytes(),
            Self::PerplexityReport(v) => v.to_bytes(),
            Self::MemoryEstimate(v) => v.to_bytes(),
            Self::ThroughputReport(v) => v.to_bytes(),
            Self::LatencyReport(v) => v.to_bytes(),
            Self::FuzzReport(v) => v.to_bytes(),
            Self::CancellationReport(v) => v.to_bytes(),
            Self::FinalizationReport(v) => v.to_bytes(),
            Self::OverheadReport(v) => v.to_bytes(),
            Self::SwapReport(v) => v.to_bytes(),
            Self::VerificationReport(v) => v.to_bytes(),
            Self::LoadReport(v) => v.to_bytes(),
            Self::PeakReport(v) => v.to_bytes(),
            Self::KvReport(v) => v.to_bytes(),
            Self::PrefixReport(v) => v.to_bytes(),
            Self::LlamaReport(v) => v.to_bytes(),
            Self::VllmReport(v) => v.to_bytes(),
            Self::MistralReport(v) => v.to_bytes(),
            Self::RecipeReport(v) => v.to_bytes(),
            Self::CompositionReport(v) => v.to_bytes(),
            Self::AgentEffectReport(v) => v.to_bytes(),
            Self::HubReport(v) => v.to_bytes(),
            Self::StudioReport(v) => v.to_bytes(),
            Self::ChainReport(v) => v.to_bytes(),
            Self::EvidenceOutputReport(v) => v.to_bytes(),
            Self::InstallReport(v) => v.to_bytes(),
            Self::NoticeReport(v) => v.to_bytes(),
            Self::ReleaseReport(v) => v.to_bytes(),
            Self::BinaryReport(v) => v.to_bytes(),
            Self::ReproducibleReport(v) => v.to_bytes(),
            Self::SignatureReport(v) => v.to_bytes(),
            Self::HostKeyReport(v) => v.to_bytes(),
            Self::ReceiptKeyReport(v) => v.to_bytes(),
            Self::SandboxReport(v) => v.to_bytes(),
            Self::ApiReport(v) => v.to_bytes(),
            Self::CacheChannelReport(v) => v.to_bytes(),
            Self::EquationReport(v) => v.to_bytes(),
            Self::SafeErrorReport(v) => v.to_bytes(),
            Self::RedactionReport(v) => v.to_bytes(),
            Self::CurveReport(v) => v.to_bytes(),
            Self::RollbackReport(v) => v.to_bytes(),
            Self::DisconnectReport(v) => v.to_bytes(),
            Self::ReceiptStoreReport(v) => v.to_bytes(),
            Self::BaseReport(v) => v.to_bytes(),
            Self::DiskReport(v) => v.to_bytes(),
            Self::UnloadReport(v) => v.to_bytes(),
            Self::RestartReport(v) => v.to_bytes(),
            Self::PointReport(v) => v.to_bytes(),
            Self::DuplicateReport(v) => v.to_bytes(),
            Self::ConcurrentLoadReport(v) => v.to_bytes(),
            Self::EvictionReport(v) => v.to_bytes(),
            Self::EqualityReport(v) => v.to_bytes(),
            Self::OomReport(v) => v.to_bytes(),
            Self::FaultReport(v) => v.to_bytes(),
            Self::TimeoutReport(v) => v.to_bytes(),
            Self::WorkerStartReport(v) => v.to_bytes(),
            Self::ChallengeReport(v) => v.to_bytes(),
            Self::ReplayEnvironmentReport(v) => v.to_bytes(),
            Self::ReplayOutputReport(v) => v.to_bytes(),
            Self::WorkerLostReport(v) => v.to_bytes(),
            Self::DrainingReport(v) => v.to_bytes(),
            Self::ScalarReport(v) => v.to_bytes(),
            Self::DigestMismatchReport(v) => v.to_bytes(),
            Self::TokenizerInvalidReport(v) => v.to_bytes(),
            Self::TemplateInvalidReport(v) => v.to_bytes(),
            Self::ArchitectureReport(v) => v.to_bytes(),
            Self::PublicReport(v) => v.to_bytes(),
            Self::QuantizationReport(v) => v.to_bytes(),
            Self::KernelReport(v) => v.to_bytes(),
            Self::PlacementRefusalReport(v) => v.to_bytes(),
            Self::MemoryRefusalReport(v) => v.to_bytes(),
            Self::SignatureCheckReport(v) => v.to_bytes(),
            Self::ContextLimitReport(v) => v.to_bytes(),
            Self::PromptCompilationReport(v) => v.to_bytes(),
            Self::ImageInvalidReport(v) => v.to_bytes(),
            Self::ImageSignatureReport(v) => v.to_bytes(),
        }
    }
}
