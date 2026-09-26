//! CPU reference engine for the Knolo micro-transformer.
//!
//! The oracle is plain f32. Candle stays in `infer-native`. The continuous
//! scheduler admits several sequences onto one KV pool. GGUF `F16`, `Q8_0`,
//! `Q4_K`, `Q5_K`, and `Q6_K` payloads dequant to finite `f32` without
//! writing a new weight file. `quant_gemm` multiplies an allowlisted GGUF
//! payload on the CPU without allocating the expanded matrix. An explicit
//! conversion writes a new `f32` artifact and a conversion receipt. Dequant
//! and `quant_gemm` do not call it. `perplexity_delta` records the
//! perplexity and accuracy delta of two `f32` logit matrices. It does not
//! run a model. `measure_placement_memory` records measured bytes that stay
//! within a placement plan. It does not allocate the page pool.
//! `measure_micro_throughput` records prefill, decode, and request
//! throughput for one cold run of the micro fixture. It does not run the
//! model. `measure_micro_latency` records time to first token, time per
//! output token, and the p50, p95, and p99 of that one request. It does not
//! run the model. `measure_corruption_fuzz` mutates one seed for each of nine
//! parsers and records that none kept the seed identity.
//! `measure_cancellation_latency` records how long one cancel took to become
//! terminal. `measure_receipt_finalization` records how long a stored receipt
//! took to become durable. `measure_receipt_overhead` records the accepted
//! write plus the receipt write. `measure_model_swap` records unload plus
//! incoming verification plus incoming load. `measure_model_verification`
//! records the check of one cold micro image. `measure_model_load` records
//! the load after that check. `measure_peak_memory` records peak host RAM
//! and peak device VRAM. `measure_kv_utilization` records occupied pages
//! and tokens for the eight-page pool. `measure_prefix_reuse` records that
//! the cache is off and every reuse count is zero.
//! `measure_llama_comparison`, `measure_vllm_comparison`, and
//! `measure_mistral_comparison` record a pinned reference runtime on the same
//! micro-fixture artifact. They do not spawn that runtime.
//! `measure_recipe_status` records the micro fixture as experimental,
//! conformant, or blessed. `measure_evidence_composition` records a
//! host-supplied Knowledge Image binding. `measure_agent_effect` records
//! whether policy accepts one final receipt. `measure_hub_record` records
//! `.kmodel` metadata without weight bytes. `measure_receipt_view` records
//! the roots a panel may show. `measure_release_manifest` records the notice,
//! SBOM, and binary-set roots. `measure_binary_inventory` records the
//! supervisor and worker hashes. `measure_reproducible_build` records the
//! lock and the instruction source. `measure_signature_gate` records an
//! unsigned local release or one shape-checked Ed25519 block.
//! `measure_host_key` records a host-supplied match against a key id.
//! `measure_receipt_key` records that the signing key stays in host storage.
//! `measure_sandbox_profile` records the worker profile. `measure_api_boundary`
//! records the bind and the limits. `measure_cache_channel` records an isolated
//! prefix policy. `measure_signature_equation` records a host-supplied equation
//! status. `measure_safe_error` records one stable error without the prompt.
//! `measure_redacted_log` records one redacted completion line.
//! `measure_curve` records a host-supplied Ed25519 curve result.
//! `measure_rollback` records a pin rollback. `measure_disconnect` records a
//! client disconnect. `measure_receipt_store` records a receipt-store failure.
//! `measure_base` records a host-supplied base-point multiplication.
//! `measure_disk` records a full disk. `measure_queued_unload` records an
//! unload while requests are queued. `measure_daemon_restart` records a new
//! daemon owner. `measure_point` records a host-supplied point addition.
//! `measure_duplicate` records a duplicate request id.
//! `measure_concurrent_load` records one concurrent load.
//! `measure_prefix_eviction` records zero prefix eviction under load.
//! `measure_point_equality` records a host-supplied point comparison.
//! `measure_cuda_oom` records a CUDA out-of-memory result.
//! `measure_cuda_fault` records a CUDA kernel or device fault.
//! `measure_timeout` records a request timeout.
//! `measure_worker_start` records a worker that did not become ready.
//! `measure_challenge` records a host-supplied challenge hash.
//! `measure_replay_environment` records a replay environment mismatch.
//! `measure_replay_output` records a replay output mismatch.
//! `measure_worker_lost` records a ready worker that exited.
//! `measure_draining` records a completion refused while draining.
//! `measure_scalar` records a host-supplied scalar reduction.
//! `measure_digest_mismatch` records a weight size or digest mismatch.
//! `measure_tokenizer_invalid` records a tokenizer the compiler refused.
//! `measure_template_invalid` records a template the compiler refused.
//! `measure_architecture` records an adapter that is not compiled in.
//! `measure_public` records a host-supplied public-key multiplication.
//! `measure_quantization` records an unsupported weight precision.
//! `measure_kernel` records a backend that is not selected.
//! `measure_placement_refusal` records an unsatisfiable placement.
//! `measure_memory_refusal` records a host bound that was exceeded.
//! `measure_signature_check` records a host-supplied signature check.
//! `measure_context_limit` records a prompt that does not fit.
//! `measure_prompt_compilation` records a prompt the compiler refused.
//! `measure_image_invalid` records a model image the compiler refused.
//! `measure_image_signature` records a signature block the compiler refused.
//! `measure_cofactor` records a host-supplied cofactor clear.
//! `measure_artifact_missing` records a pinned artifact that was not found.
//! `measure_receipt_required` records a receipt the verifier could not use.
//! `measure_backend` records a throughput mode the reference refused.
//! `measure_digest_invalid` records a digest the parser refused.
//! `measure_receipt_sign` records a host-supplied receipt signature.
//! `measure_canonical_cbor` records a canonical document the decoder refused.
//! `measure_contract_invalid` records a contract field the decoder refused.
//! `measure_grammar_refusal` records a grammar the compiler did not build.
//! `measure_tool_refusal` records a tool call the engine did not execute.
//! `measure_receipt_verify` records a host-supplied receipt verification.
//! `measure_speculative` records a speculative request the engine did not run.
//! `measure_cuda_graph` records a CUDA graph that was not captured.
//! `measure_multi_model` records a second model the worker did not load.
//! `measure_secondary` records a secondary service the worker did not start.
//! `measure_domain` records the domain report.
//! `measure_attention` records the attention report.
//! `measure_moe` records the moe report.
//! `measure_expert_placement` records the expert-placement report.
//! `measure_grouped_kernel` records the grouped-kernel report.
//! `measure_router` records the router report.
//! `measure_glm` records the glm report.
//! `measure_workstation` records the workstation report.
//! `measure_mixed_placement` records the mixed-placement report.
//! `measure_capacity` records the capacity report.
//! None of them runs the model. This crate does
//! not start a server. A
//! CUDA placement plan names `slot-0`. `infer-native` selects that kernel.
//! The journal and the host probe live here so the default worker binary
//! does not link Candle.

mod agent_effect;
mod api;
mod architecture;
mod artifact_missing;
mod attention;
mod backend;
mod base;
mod binary;
mod cache_channel;
mod cancellation;
mod canonical_cbor;
mod capacity;
mod chain;
mod challenge;
mod cofactor;
mod composition;
mod concurrent_load;
mod context_limit;
mod contract_invalid;
mod conversion;
mod curve;
mod daemon_restart;
mod dequant;
mod digest_invalid;
mod digest_mismatch;
mod disconnect;
mod disk;
mod domain_sep;
mod draining;
mod duplicate;
mod equality;
mod equation;
mod eviction;
mod evidence;
mod expert_placement;
mod fault;
mod finalization;
mod fuzz;
mod glm;
mod grammar_refusal;
mod graph;
mod greedy;
mod grouped_kernel;
mod hardware;
mod host_key;
mod hub;
mod identity;
mod image_invalid;
mod image_signature;
mod install;
mod journal;
mod kernel;
mod kv;
mod latency;
mod load;
mod memory;
mod memory_refusal;
mod micro;
mod mixed_placement;
mod moe;
mod multi_model;
mod notice;
mod oom;
mod overhead;
mod paged;
mod peak;
mod perplexity;
mod philox;
mod placement_refusal;
mod point;
mod prefix;
mod prompt_compilation;
mod public_key;
mod quant_gemm;
mod quantization;
mod queued_unload;
mod receipt_key;
mod receipt_required;
mod receipt_sign;
mod receipt_store;
mod receipt_verify;
mod recipe;
mod redaction;
mod reference;
mod release;
mod replay_environment;
mod replay_output;
mod report_io;
mod reproducible;
mod rollback;
mod router;
mod safe_error;
mod sample;
mod sandbox;
mod scalar;
mod schedule;
mod secondary;
mod signature;
mod signature_check;
mod speculative;
mod studio;
mod swap;
mod template_invalid;
mod throughput;
mod timeout;
mod tokenizer_invalid;
mod tool_refusal;
mod traits;
mod utilization;
mod verification;
mod worker_lost;
mod worker_start;
mod workstation;

pub use agent_effect::{
    measure_agent_effect, verify_agent_effect, write_agent_effect_report, AgentEffectObservation,
    MicroAgentEffect,
};
pub use api::{
    measure_api_boundary, verify_api_boundary, write_api_report, ApiObservation, MicroApi,
};
pub use architecture::{
    measure_architecture, verify_architecture, write_architecture_report, ArchitectureObservation,
    MicroArchitecture,
};
pub use artifact_missing::{
    measure_artifact_missing, verify_artifact_missing, write_artifact_missing_report,
    ArtifactMissingObservation, MicroArtifactMissing,
};
pub use attention::{
    measure_attention, verify_attention, write_attention_report, AttentionObservation,
    MicroAttention,
};
pub use backend::{
    measure_backend, verify_backend, write_backend_report, BackendObservation, MicroBackend,
};
pub use base::{measure_base, verify_base, write_base_report, BaseObservation, MicroBase};
pub use binary::{
    measure_binary_inventory, verify_binary_inventory, write_binary_report, BinaryObservation,
    MicroBinary,
};
pub use cache_channel::{
    measure_cache_channel, verify_cache_channel, write_cache_channel_report,
    CacheChannelObservation, MicroCacheChannel,
};
pub use cancellation::{
    measure_cancellation_latency, verify_cancellation_latency, write_cancellation_report,
    CancellationObservation, MicroCancellation,
};
pub use canonical_cbor::{
    measure_canonical_cbor, verify_canonical_cbor, write_canonical_cbor_report,
    CanonicalCborObservation, MicroCanonicalCbor,
};
pub use capacity::{
    measure_capacity, verify_capacity, write_capacity_report, CapacityObservation, MicroCapacity,
};
pub use chain::{
    measure_receipt_chain, verify_receipt_chain, write_chain_report, ChainObservation, MicroChain,
};
pub use challenge::{
    measure_challenge, verify_challenge, write_challenge_report, ChallengeObservation,
    MicroChallenge,
};
pub use cofactor::{
    measure_cofactor, verify_cofactor, write_cofactor_report, CofactorObservation, MicroCofactor,
};
pub use composition::{
    measure_evidence_composition, verify_evidence_composition, write_composition_report,
    CompositionObservation, MicroComposition,
};
pub use concurrent_load::{
    measure_concurrent_load, verify_concurrent_load, write_concurrent_load_report,
    ConcurrentLoadObservation, MicroConcurrentLoad,
};
pub use context_limit::{
    measure_context_limit, verify_context_limit, write_context_limit_report,
    ContextLimitObservation, MicroContextLimit,
};
pub use contract_invalid::{
    measure_contract_invalid, verify_contract_invalid, write_contract_invalid_report,
    ContractInvalidObservation, MicroContractInvalid,
};
pub use conversion::{
    convert_gguf_tensor, verify_gguf_conversion, write_gguf_conversion, GgufConversion,
};
pub use curve::{measure_curve, verify_curve, write_curve_report, CurveObservation, MicroCurve};
pub use daemon_restart::{
    measure_daemon_restart, verify_daemon_restart, write_restart_report, MicroRestart,
    RestartObservation,
};
pub use dequant::{dequant_gguf, dequant_output_bytes, MAX_DEQUANT_BYTES};
pub use digest_invalid::{
    measure_digest_invalid, verify_digest_invalid, write_digest_invalid_report,
    DigestInvalidObservation, MicroDigestInvalid,
};
pub use digest_mismatch::{
    measure_digest_mismatch, verify_digest_mismatch, write_digest_mismatch_report,
    DigestMismatchObservation, MicroDigestMismatch,
};
pub use disconnect::{
    measure_disconnect, verify_disconnect, write_disconnect_report, DisconnectObservation,
    MicroDisconnect,
};
pub use disk::{measure_disk, verify_disk, write_disk_report, DiskObservation, MicroDisk};
pub use domain_sep::{
    measure_domain, verify_domain, write_domain_report, DomainObservation, MicroDomain,
};
pub use draining::{
    measure_draining, verify_draining, write_draining_report, DrainingObservation, MicroDraining,
};
pub use duplicate::{
    measure_duplicate, verify_duplicate, write_duplicate_report, DuplicateObservation,
    MicroDuplicate,
};
pub use equality::{
    measure_point_equality, verify_point_equality, write_equality_report, EqualityObservation,
    MicroEquality,
};
pub use equation::{
    measure_signature_equation, verify_signature_equation, write_equation_report,
    EquationObservation, MicroEquation,
};
pub use eviction::{
    measure_prefix_eviction, verify_prefix_eviction, write_eviction_report, EvictionObservation,
    MicroEviction,
};
pub use evidence::{
    measure_evidence_output, verify_evidence_output, write_evidence_output_report,
    EvidenceObservation, MicroEvidenceOutput,
};
pub use expert_placement::{
    measure_expert_placement, verify_expert_placement, write_expert_placement_report,
    ExpertPlacementObservation, MicroExpertPlacement,
};
pub use fault::{
    measure_cuda_fault, verify_cuda_fault, write_fault_report, FaultObservation, MicroFault,
};
pub use finalization::{
    measure_receipt_finalization, verify_receipt_finalization, write_finalization_report,
    FinalizationObservation, MicroFinalization,
};
pub use fuzz::{
    fuzz_mutations, measure_corruption_fuzz, verify_corruption_fuzz, write_fuzz_report,
    CorruptionFuzz, CorruptionObservation, CorruptionProbe, FuzzSeed, MAX_FUZZ_SEED_BYTES,
};
pub use glm::{measure_glm, verify_glm, write_glm_report, GlmObservation, MicroGlm};
pub use grammar_refusal::{
    measure_grammar_refusal, verify_grammar_refusal, write_grammar_refusal_report,
    GrammarRefusalObservation, MicroGrammarRefusal,
};
pub use graph::{
    measure_cuda_graph, verify_cuda_graph, write_cuda_graph_report, GraphObservation, MicroGraph,
};
pub use greedy::{argmax, greedy_generate, logit_margin, GreedyOutput};
pub use grouped_kernel::{
    measure_grouped_kernel, verify_grouped_kernel, write_grouped_kernel_report,
    GroupedKernelObservation, MicroGroupedKernel,
};
pub use hardware::{default_home, probe_machine, require_cuda_slot0, CudaSlot0};
pub use host_key::{
    measure_host_key, verify_host_key, write_host_key_report, HostKeyObservation, MicroHostKey,
};
pub use hub::{measure_hub_record, verify_hub_record, write_hub_report, HubObservation, MicroHub};
pub use identity::{
    cargo_lock_root, cpu_kernel_bundle, cpu_kernel_plan_root, cuda_engine_build,
    cuda_kernel_bundle, cuda_kernel_plan_root, host_engine_build, reference_engine_build,
    reference_kernel_bundle, reference_kernel_plan_root,
};
pub use image_invalid::{
    measure_image_invalid, verify_image_invalid, write_image_invalid_report,
    ImageInvalidObservation, MicroImageInvalid,
};
pub use image_signature::{
    measure_image_signature, verify_image_signature, write_image_signature_report,
    ImageSignatureObservation, MicroImageSignature,
};
pub use install::{
    measure_hub_install, verify_hub_install, write_install_report, InstallObservation, MicroInstall,
};
pub use journal::{load_sampler_plan, recover_open_journals, verify_journal, Journal};
pub use kernel::{
    measure_kernel, verify_kernel, write_kernel_report, KernelObservation, MicroKernel,
};
pub use kv::SingleBlockKv;
pub use latency::{
    measure_micro_latency, verify_micro_latency, write_latency_report, LatencyObservation,
    MicroLatency,
};
pub use load::{
    measure_model_load, verify_model_load, write_load_report, LoadObservation, MicroLoad,
};
pub use memory::{
    measure_placement_memory, planned_kv_bytes, verify_memory_estimate, write_memory_estimate,
    MemoryEstimate, MemoryObservation,
};
pub use memory_refusal::{
    measure_memory_refusal, verify_memory_refusal, write_memory_refusal_report,
    MemoryRefusalObservation, MicroMemoryRefusal,
};
pub use micro::*;
pub use mixed_placement::{
    measure_mixed_placement, verify_mixed_placement, write_mixed_placement_report,
    MicroMixedPlacement, MixedPlacementObservation,
};
pub use moe::{measure_moe, verify_moe, write_moe_report, MicroMoe, MoeObservation};
pub use multi_model::{
    measure_multi_model, verify_multi_model, write_multi_model_report, MicroMultiModel,
    MultiModelObservation,
};
pub use notice::{
    measure_supply_notice, verify_supply_notice, write_notice_report, MicroNotice,
    NoticeObservation,
};
pub use oom::{measure_cuda_oom, verify_cuda_oom, write_oom_report, MicroOom, OomObservation};
pub use overhead::{
    measure_receipt_overhead, verify_receipt_overhead, write_overhead_report, MicroOverhead,
    OverheadObservation,
};
pub use paged::{KvCensus, PagedKv, CPU_KV_PAGE_POOL};
pub use peak::{
    measure_peak_memory, verify_peak_memory, write_peak_report, MicroPeak, PeakObservation,
};
pub use perplexity::{
    perplexity_delta, verify_perplexity_report, write_perplexity_report, PerplexityMeasurement,
};
pub use philox::{philox4x32_10, philox_u32, philox_unit};
pub use placement_refusal::{
    measure_placement_refusal, verify_placement_refusal, write_placement_refusal_report,
    MicroPlacementRefusal, PlacementRefusalObservation,
};
pub use point::{measure_point, verify_point, write_point_report, MicroPoint, PointObservation};
pub use prefix::{
    measure_prefix_reuse, verify_prefix_reuse, write_prefix_report, MicroPrefix, PrefixObservation,
};
pub use prompt_compilation::{
    measure_prompt_compilation, verify_prompt_compilation, write_prompt_compilation_report,
    MicroPromptCompilation, PromptCompilationObservation,
};
pub use public_key::{
    measure_public, verify_public, write_public_report, MicroPublic, PublicObservation,
};
pub use quant_gemm::{quant_gemm, quant_gemm_output_bytes};
pub use quantization::{
    measure_quantization, verify_quantization, write_quantization_report, MicroQuantization,
    QuantizationObservation,
};
pub use queued_unload::{
    measure_queued_unload, verify_queued_unload, write_unload_report, MicroUnload,
    UnloadObservation,
};
pub use receipt_key::{
    measure_receipt_key, verify_receipt_key, write_receipt_key_report, MicroReceiptKey,
    ReceiptKeyObservation,
};
pub use receipt_required::{
    measure_receipt_required, verify_receipt_required, write_receipt_required_report,
    MicroReceiptRequired, ReceiptRequiredObservation,
};
pub use receipt_sign::{
    measure_receipt_sign, verify_receipt_sign, write_receipt_sign_report, MicroReceiptSign,
    ReceiptSignObservation,
};
pub use receipt_store::{
    measure_receipt_store, verify_receipt_store, write_receipt_store_report, MicroReceiptStore,
    ReceiptStoreObservation,
};
pub use receipt_verify::{
    measure_receipt_verify, verify_receipt_verify, write_receipt_verify_report, MicroReceiptVerify,
    ReceiptVerifyObservation,
};
pub use recipe::{
    measure_recipe_status, verify_recipe_status, write_recipe_report, MicroRecipe,
    RecipeObservation,
};
pub use redaction::{
    measure_redacted_log, verify_redacted_log, write_redaction_report, MicroRedaction,
    RedactionObservation,
};
pub use reference::{
    measure_llama_comparison, measure_mistral_comparison, measure_vllm_comparison,
    verify_llama_comparison, verify_mistral_comparison, verify_vllm_comparison, write_llama_report,
    write_mistral_report, write_vllm_report, MicroLlama, MicroMistral, MicroVllm,
    ReferenceObservation,
};
pub use release::{
    measure_release_manifest, verify_release_manifest, write_release_report, MicroRelease,
    ReleaseObservation,
};
pub use replay_environment::{
    measure_replay_environment, verify_replay_environment, write_replay_environment_report,
    MicroReplayEnvironment, ReplayEnvironmentObservation,
};
pub use replay_output::{
    measure_replay_output, verify_replay_output, write_replay_output_report, MicroReplayOutput,
    ReplayOutputObservation,
};
pub use reproducible::{
    measure_reproducible_build, verify_reproducible_build, write_reproducible_report,
    MicroReproducible, ReproducibleObservation,
};
pub use rollback::{
    measure_rollback, verify_rollback, write_rollback_report, MicroRollback, RollbackObservation,
};
pub use router::{
    measure_router, verify_router, write_router_report, MicroRouter, RouterObservation,
};
pub use safe_error::{
    measure_safe_error, verify_safe_error, write_safe_error_report, MicroSafeError,
    SafeErrorObservation,
};
pub use sample::{generate_samples, sample_token, SampledOutput};
pub use sandbox::{
    measure_sandbox_profile, verify_sandbox_profile, write_sandbox_report, MicroSandbox,
    SandboxObservation,
};
pub use scalar::{
    measure_scalar, verify_scalar, write_scalar_report, MicroScalar, ScalarObservation,
};
pub use schedule::{
    CpuScheduler, ScheduleOp, ScheduleRequest, ScheduleResult, ScheduleStep, SchedulerConfig,
    SchedulerCounters, ServiceClass, ITERATION_BOUND_NANOS, ITERATION_BUCKETS,
};
pub use secondary::{
    measure_secondary, verify_secondary, write_secondary_report, MicroSecondary,
    SecondaryObservation,
};
pub use signature::{
    measure_signature_gate, verify_signature_gate, write_signature_report, MicroSignature,
    SignatureObservation,
};
pub use signature_check::{
    measure_signature_check, verify_signature_check, write_signature_check_report,
    MicroSignatureCheck, SignatureCheckObservation,
};
pub use speculative::{
    measure_speculative, verify_speculative, write_speculative_report, MicroSpeculative,
    SpeculativeObservation,
};
pub use studio::{
    measure_receipt_view, verify_receipt_view, write_studio_report, MicroStudio, StudioObservation,
};
pub use swap::{
    measure_model_swap, verify_model_swap, write_swap_report, MicroSwap, SwapObservation,
};
pub use template_invalid::{
    measure_template_invalid, verify_template_invalid, write_template_invalid_report,
    MicroTemplateInvalid, TemplateInvalidObservation,
};
pub use throughput::{
    measure_micro_throughput, verify_micro_throughput, write_throughput_report, MicroThroughput,
    ThroughputObservation,
};
pub use timeout::{
    measure_timeout, verify_timeout, write_timeout_report, MicroTimeout, TimeoutObservation,
};
pub use tokenizer_invalid::{
    measure_tokenizer_invalid, verify_tokenizer_invalid, write_tokenizer_invalid_report,
    MicroTokenizerInvalid, TokenizerInvalidObservation,
};
pub use tool_refusal::{
    measure_tool_refusal, verify_tool_refusal, write_tool_refusal_report, MicroToolRefusal,
    ToolRefusalObservation,
};
pub use traits::*;
pub use utilization::{
    measure_kv_utilization, verify_kv_utilization, write_kv_report, KvObservation,
    MicroKvUtilization,
};
pub use verification::{
    measure_model_verification, verify_model_verification, write_verification_report,
    MicroVerification, VerificationObservation,
};
pub use worker_lost::{
    measure_worker_lost, verify_worker_lost, write_worker_lost_report, MicroWorkerLost,
    WorkerLostObservation,
};
pub use worker_start::{
    measure_worker_start, verify_worker_start, write_worker_start_report, MicroWorkerStart,
    WorkerStartObservation,
};
pub use workstation::{
    measure_workstation, verify_workstation, write_workstation_report, MicroWorkstation,
    WorkstationObservation,
};
