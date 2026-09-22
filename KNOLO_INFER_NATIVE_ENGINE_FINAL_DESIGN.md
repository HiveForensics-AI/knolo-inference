# Knolo Infer Native Engine
## Final Software Design and Agent Build Handoff

**Document status:** Architecture baseline for implementation  
**Version:** 1.0  
**Date:** 2026-09-18  
**Product:** Knolo Infer  
**Recommended repository:** `HiveForensics-AI/knolo-infer`  
**Primary implementation:** Rust  
**Application SDK and integrations:** TypeScript  
**Knolo-owned license target:** Apache-2.0  
**Public binaries:** `knolo-inferd`, `knolo-infer-worker`, `knolo infer`  
**Public TypeScript package:** `@knolo/infer`

---

## 0. Executive decision

Build **Knolo Infer** as a real, first-party inference runtime, not as a renamed copy of Ollama, llama.cpp, mistral.rs, vLLM, or another model server.

The native engine must:

1. load and verify model artifacts itself;
2. compile pinned tokenizer, template, model, placement, sampler, and kernel plans;
3. execute transformer prefill and decode in a Knolo-owned worker process;
4. own admission control, continuous batching, KV-cache paging, prefix reuse, cancellation, and model lifecycle;
5. emit a receipt binding the generated output to the exact model bytes, Knowledge Image roots, Reflex/query receipts, rendered prompt tokens, engine build, hardware placement, execution plan, and output bytes;
6. fail closed when a pinned artifact or required receipt cannot be verified;
7. keep model inference outside `@knolo/core`;
8. keep tools, network side effects, and agent policy outside the engine;
9. use established tensor and GPU primitives without pretending Knolo authored CUDA, BLAS, or third-party runtimes;
10. support explicit compatibility backends while making the Knolo native backend the product’s architectural center.

The public product is **Knolo Infer**. The native runtime inside it is referred to in this document as the **Knolo Native Engine**.

The design principle is:

> **Build engines around a verifiable execution contract, not the execution contract around an engine.**

---

# 1. Why this belongs in the Knolo product family

Knolo’s current architecture deliberately separates deterministic knowledge from probabilistic model execution:

- `@knolo/core` owns portable Knowledge Images, evidence identity, roots, validation, deterministic retrieval, query receipts, bounded runtime state, and verification.
- `@knolo/reflex` owns deterministic behavior/context selection, budgeted rendering, and selection receipts.
- Knolo Agents owns governed control flow, authority, checkpoints, host effects, and replayable events.
- Host applications own credentials, networking, model inference, external effects, and deployment.

Knolo Infer fills the host-owned **model execution** boundary without moving inference into Core.

The resulting product stack is:

```text
KNOLO CORE
Portable and verifiable knowledge
        ↓
KNOLO REFLEX
Bounded and receipted behavior/context selection
        ↓
KNOLO INFER
Pinned and receipted model execution
        ↓
KNOLO AGENTS
Governed action, tools, checkpoints, and human review
        ↓
KNOLO HUB
Verifiable distribution of packs, model images, and recipes
```

The full execution story becomes:

```text
Knowledge Image root
        │
        ▼
Query receipt
        │
        ▼
Reflex selection receipt
        │
        ▼
Prompt/token plan
        │
        ▼
Inference receipt
        │
        ▼
Agent effect event
```

This is the differentiator. The product is not merely “a server that returns tokens.” It is a host that can prove which knowledge, model, prompt representation, configuration, and runtime produced those tokens.

---

# 2. Definition of “our own inference engine”

Knolo Infer qualifies as its own engine only when the native path satisfies all of the following:

| Requirement | Native Knolo behavior |
|---|---|
| Weight loading | Knolo parses and loads verified weight files directly |
| Model graph | Knolo-owned Rust architecture adapters build the model |
| Tokenization | Knolo loads a pinned tokenizer and performs tokenization |
| Chat rendering | Knolo compiles the pinned chat template under a bounded renderer |
| KV cache | Knolo allocates, pages, reuses, evicts, and accounts for KV memory |
| Scheduling | Knolo performs admission, prefill scheduling, decode batching, fairness, and cancellation |
| Sampling | Knolo applies a fixed, receipted logit-processing and sampling pipeline |
| Execution | Knolo invokes tensor operations and GPU kernels in-process |
| Placement | Knolo creates and verifies the CPU/GPU/tensor placement plan |
| Receipts | Knolo commits model, prompt, plan, knowledge, execution, and output identity |
| Lifecycle | Knolo owns load, warm-up, health, drain, unload, restart, and failure recovery |
| API | Knolo exposes its native and compatibility interfaces |

Using Candle, CUDA, cuBLAS, FlashAttention-compatible kernels, Hugging Face tokenizers, or safetensors does **not** make the product a wrapper. Those are implementation primitives.

The following does **not** count as the native engine:

```text
HTTP request → llama.cpp server → return output
HTTP request → Ollama → return output
HTTP request → mistral.rs → return output
```

Those remain compatibility backends.

---

# 3. Research conclusions translated into design choices

The architecture incorporates the strongest established serving techniques while keeping the Knolo-specific value above the tensor layer.

## 3.1 Paged KV memory is mandatory

PagedAttention demonstrated that KV-cache memory can be managed in fixed-size blocks rather than requiring large contiguous allocations. The practical consequences for Knolo are:

- fixed-size KV blocks;
- per-sequence logical-to-physical block tables;
- copy-on-write sharing for common prefixes;
- bounded allocation and explicit eviction;
- predictable memory accounting before request admission.

Knolo should not begin with one contiguous KV tensor per request and retrofit paging later. The page manager is a foundation component.

## 3.2 Continuous batching is mandatory for the server path

Iteration-level scheduling lets the decode scheduler add and remove sequences between token steps. Knolo must separate:

- request admission;
- prompt compilation;
- chunked prefill;
- decode-ready queues;
- one-token decode iterations;
- output streaming.

A “one request, one model call” loop is acceptable only for the CPU reference implementation.

## 3.3 Prefix caching must be exact and tenant-scoped

SGLang’s prefix reuse work validates the value of caching shared prompt prefixes. Knolo’s implementation must add stronger identity rules:

```text
prefix cache key =
model runtime root
+ tokenizer root
+ template root
+ exact token-ID prefix root
+ positional configuration root
+ execution precision root
+ cache namespace
```

Cross-tenant cache sharing is disabled by default. A cache hit must never be inferred from text similarity.

## 3.4 Exact attention kernels should be optimized, not approximated silently

FlashAttention research shows that exact attention can be made substantially more memory-efficient through IO-aware tiling and better work partitioning. Knolo may use established exact kernels, but any approximation, sliding-window modification, sparsity rule, or quantized KV representation must be explicit in the execution plan and receipt.

## 3.5 Speculative decoding is valuable but not part of the first correctness gate

Speculative sampling can reduce latency while preserving the target distribution under its defined algorithm. It introduces another model, acceptance logic, cache coordination, and more receipt fields. Implement it only after the ordinary target-only decode path is conformant.

## 3.6 CUDA Graphs belong after shape plans stabilize

CUDA Graphs can reduce CPU launch overhead by capturing reusable execution graphs. They are useful for stable decode shape buckets, but graph identity, kernel identity, workspace addresses, and shape constraints must become part of the execution plan. Do not build the first correctness implementation around graph capture.

## 3.7 The official GLM configurations require staged support

Current official model metadata makes the implementation order clear:

- GLM-4.7-Flash is a roughly 30B-total, 3B-active mixture-of-experts architecture. It is a realistic native MoE target after the dense path works.
- GLM-5.3-Flash is a much larger multimodal MoE model with hybrid sparse and linear-attention layers, mHC-related configuration, many routed experts, and an MTP layer. It requires a separate research adapter rather than being treated as “GLM-4.7 with more weights.”

Therefore:

```text
dense reference
→ dense CUDA
→ quantized dense
→ ordinary MoE
→ GLM-4.7-Flash adapter
→ GLM-5.3-Flash research adapter
```

Compatibility backends may support models before native support is certified, but backend selection must be explicit.

---

# 4. Goals and non-goals

## 4.1 Goals

1. Local-first inference on Linux workstations.
2. CPU and NVIDIA CUDA as the first supported execution targets.
3. Rust-native runtime with no Python process in the serving path.
4. Direct support for safetensors and GGUF model artifacts.
5. Exact artifact verification before model load.
6. Reproducible prompt compilation and token identity.
7. Paged KV cache, prefix sharing, continuous batching, and chunked prefill.
8. Content-addressed `.kmodel` Model Images that do not contain weight blobs.
9. Native OpenAI-compatible serving.
10. First-class Knolo receipts and evidence bindings.
11. Strict no-silent-fallback behavior.
12. Explicit support levels per model architecture and execution backend.
13. Bounded memory, bounded inputs, and fail-closed parsing.
14. Independent Rust and TypeScript contract verification.
15. Clean integration with Core, Reflex, Agents, Hub, and a future Studio panel.

## 4.2 Non-goals for the first production release

- Training or fine-tuning.
- LoRA management.
- A desktop chat product.
- A model social network or broad model zoo.
- Hosting large weight files on Knolo Hub.
- Executing Python from model repositories.
- Executing tools inside the inference server.
- Replacing `@knolo/core`.
- Requiring a vector database.
- Distributed multi-node inference.
- Arbitrary remote-code model architectures.
- Automatic model conversion without a new content-addressed artifact.
- Claiming that all model output is deterministic.
- Shipping GLM-5.3 native support before the lower-level adapters are conformant.
- Reimplementing CUDA GEMM from first principles.

---

# 5. High-level architecture

```text
                                     ┌─────────────────────────────┐
                                     │ Application / IDE / Agent   │
                                     └──────────────┬──────────────┘
                                                    │
                         OpenAI API or Knolo native │
                                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           knolo-inferd                                   │
│                                                                          │
│  API Gateway         Auth / Limits          Receipt Store                │
│       │                    │                      ▲                       │
│       ▼                    ▼                      │                       │
│  Request Normalizer → Prompt Compiler → Request Journal                  │
│                              │                                           │
│                              ▼                                           │
│                     Admission / Router                                   │
│                              │                                           │
│                 Model Manager / Placement Planner                        │
│                              │                                           │
│                     Worker Supervisor                                    │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │ local canonical-CBOR IPC
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        knolo-infer-worker                                │
│                one loaded model runtime per worker                       │
│                                                                          │
│  Scheduler ──► Paged KV Manager ──► Native Model Executor                │
│      │                                      │                            │
│      │                                      ▼                            │
│      │                         Tensor Backend / Kernel Registry           │
│      │                                      │                            │
│      ▼                                      ▼                            │
│  Sampler / Grammar                    CPU / CUDA devices                  │
│      │                                                                   │
│      └──────────── execution events / token chunks ──────────────────────┘
                               │
                               ▼
                        Final receipt builder
```

## 5.1 Process boundary

Use two process classes:

### `knolo-inferd`

The supervisor owns:

- network API;
- authentication and authorization hooks;
- model download and artifact verification;
- `.kmodel` verification;
- lockfile resolution;
- prompt rendering and tokenization;
- model lifecycle;
- placement planning;
- worker spawning and recovery;
- receipt journaling and final signing;
- metrics and safe logs.

### `knolo-infer-worker`

A worker owns:

- one model runtime identity;
- GPU/CPU contexts;
- weights;
- model graph;
- KV page pool;
- scheduler;
- kernel plans;
- decode loops;
- sampling;
- worker-local performance counters.

A worker has no external network access. It receives already verified local paths or file handles and a compiled request plan. A model crash, CUDA OOM, or kernel fault must not terminate the supervisor.

## 5.2 IPC

Use a local Unix domain socket on Linux and a named pipe equivalent on Windows.

Protocol:

```text
u32 big-endian frame length
canonical CBOR payload
```

Properties:

- versioned message kinds;
- bounded frame lengths;
- no duplicate keys;
- no floats in rooted contract payloads;
- request and worker instance IDs;
- cancellation messages;
- heartbeats;
- explicit backpressure;
- no raw credentials.

---

# 6. Recommended repository layout

```text
knolo-infer/
├── Cargo.toml
├── Cargo.lock
├── package.json
├── LICENSE
├── NOTICE
├── SECURITY.md
├── README.md
│
├── spec/
│   ├── KIP-INFER-0001-canonical-contracts.md
│   ├── KIP-INFER-0002-model-image.md
│   ├── KIP-INFER-0003-artifact-root.md
│   ├── KIP-INFER-0004-placement-plan.md
│   ├── KIP-INFER-0005-prompt-plan.md
│   ├── KIP-INFER-0006-inference-receipt.md
│   ├── KIP-INFER-0007-replay-check.md
│   └── KIP-INFER-0008-native-api.md
│
├── crates/
│   ├── infer-contracts/
│   ├── infer-artifact/
│   ├── infer-prompt/
│   ├── infer-engine/
│   ├── infer-native/
│   ├── infer-kernels-cuda/
│   ├── infer-receipt/
│   ├── infer-worker/
│   ├── inferd/
│   ├── infer-cli/
│   ├── infer-backend-llamacpp/
│   └── infer-backend-ollama/
│
├── packages/
│   └── infer/
│       ├── src/
│       ├── test/
│       └── package.json
│
├── conformance/
│   ├── contracts/
│   ├── model-images/
│   ├── prompts/
│   ├── receipts/
│   ├── replay/
│   └── micro-model/
│
├── models/
│   ├── micro-transformer/
│   ├── llama-family/
│   ├── qwen-family/
│   ├── generic-moe/
│   ├── glm-4.7/
│   └── glm-5.3-research/
│
├── benchmarks/
│   ├── datasets/
│   ├── runners/
│   ├── reports/
│   └── schemas/
│
├── examples/
│   ├── openai-client/
│   ├── core-reflex/
│   ├── agents-host-effect/
│   └── native-rust/
│
└── docs/
    ├── ARCHITECTURE.md
    ├── MODEL_SUPPORT.md
    ├── SECURITY_MODEL.md
    ├── RECEIPTS.md
    ├── SCHEDULER.md
    ├── KV_CACHE.md
    ├── KERNELS.md
    └── OPERATIONS.md
```

Do not split every module into a crate on the first commit. Begin with the logical boundaries above, but merge low-churn components until compilation overhead justifies separation.

---

# 7. Dependency and ownership policy

## 7.1 Recommended Rust foundations

Use the following classes of dependencies:

| Purpose | Recommended approach |
|---|---|
| Async/server | Tokio, Axum, Tower |
| Serialization | Serde plus Knolo canonical-CBOR implementation |
| Hashing | SHA-256 |
| Signing | Ed25519 |
| Tensor substrate | `candle-core` and `candle-nn` behind a Knolo trait |
| Weight format | safetensors; Knolo GGUF parser or tightly scoped parser crate |
| Tokenizer | Hugging Face `tokenizers` crate |
| Template renderer | sandboxed `minijinja` subset |
| Model download | `hf-hub`, optional local/object-store adapters |
| Metrics | Prometheus-compatible metrics |
| Logging | `tracing` with a redacting layer |
| Hardware probe | sysinfo plus NVML/CUDA runtime adapters |
| Fuzzing | cargo-fuzz |
| Property tests | proptest |

## 7.2 Candle boundary

Use Candle as a tensor substrate, not as the public product architecture.

Rules:

1. No Candle type crosses Knolo public APIs.
2. `infer-engine` defines Knolo traits for tensors, devices, models, KV storage, and kernels.
3. `infer-native` implements those traits with Candle initially.
4. Architecture adapters, scheduler, cache, prompt compilation, sampling, placement, and receipts remain Knolo-owned.
5. Prefer `candle-core` and `candle-nn`; do not make `candle-transformers` the production architecture layer.
6. Custom performance kernels live in the small, isolated `infer-kernels-cuda` crate.
7. Upstream fixes should be contributed upstream where practical.

## 7.3 Third-party runtime policy

llama.cpp and Ollama may be explicit compatibility backends.

They must:

- retain their names;
- retain license and copyright notices;
- appear in `NOTICE`;
- have their exact binary/version identity recorded;
- never be selected silently;
- receive a lower or different verification class when Knolo cannot verify all runtime facts.

Do not create `knolo.cpp`.

---

# 8. Trust boundaries

```text
UNTRUSTED
- downloaded weight bytes
- GGUF metadata
- safetensors metadata
- tokenizer JSON
- chat templates
- model configs
- API requests
- Hub manifests before verification

TRUSTED AFTER VERIFICATION
- canonical .kmodel bytes
- artifact roots
- lockfile pins
- compiled prompt plans
- compiled placement plans
- signed manifests
- engine build descriptors

TRUSTED COMPUTING BASE
- Knolo contract verifier
- artifact parser
- prompt compiler
- placement planner
- native worker
- kernel bundle
- receipt builder
- signing key boundary

HOST-OWNED, OUTSIDE ENGINE
- credentials
- network transport policy
- tool implementations
- external side effects
- Knolo Agent authority
- application data
```

No model repository may cause arbitrary code execution. `trust_remote_code` is forbidden.

---

# 9. Canonical contracts and digest rules

Infer contracts must follow the same canonical culture as Knolo Core:

- definite-length canonical CBOR;
- text-keyed maps;
- shortest integer representation;
- duplicate keys rejected;
- indefinite forms rejected;
- map keys sorted canonically;
- decoded content must re-encode byte-for-byte;
- SHA-256 domain separation;
- lower-case hexadecimal digest strings.

## 9.1 Domain function

```text
H(domain, payload) =
SHA256(
  UTF8("knolo:" + domain + ":v1\0")
  ||
  canonical_cbor(payload)
)
```

Digest rendering:

```text
sha256-<64 lowercase hexadecimal characters>
```

## 9.2 Infer domains

Reserve at least:

```text
infer-model-image
infer-model-artifact
infer-tokenizer
infer-template
infer-config
infer-engine-build
infer-kernel-bundle
infer-hardware
infer-placement
infer-prompt-input
infer-prompt-tokens
infer-tools
infer-evidence
infer-sampler
infer-grammar
infer-request-intent
infer-execution-plan
infer-execution-event
infer-execution-trace
infer-output-tokens
infer-output-text
infer-receipt
infer-replay-check
infer-conformance
```

## 9.3 No floating-point values in rooted contracts

Core’s canonical CBOR subset rejects floats. Do not introduce JSON/CBOR floating-point ambiguity into Infer roots.

Represent generation values as fixed-point integers:

```text
temperature_micros     700000  = 0.7
top_p_millionths       950000  = 0.95
min_p_millionths        50000  = 0.05
repetition_penalty_micros
presence_penalty_micros
frequency_penalty_micros
```

The runtime converts fixed-point values to its internal floating representation only after the contract root is verified.

## 9.4 Versioning rules

- Every public object has a `kind` and `version`.
- A new incompatible shape requires a new major contract version.
- Unknown top-level fields are rejected.
- Extensions live only under a bounded, namespaced `extensions` map.
- Unknown extensions may be retained but cannot affect execution unless understood.
- Readers never silently substitute defaults for missing security-critical fields.
- Defaults are explicit in the normalized, rooted plan.

---

# 10. Knolo Model Image: `.kmodel`

Create a small, portable, content-addressed artifact called a **Knolo Model Image**.

File extension:

```text
.kmodel
```

It does **not** contain weight blobs.

It contains enough bounded metadata to make model interpretation portable and pinned:

- normalized model configuration;
- architecture adapter identifier;
- exact weight-file descriptors;
- tokenizer files or their embedded canonical representation;
- chat template;
- special token map;
- generation defaults;
- license metadata;
- source hints;
- supported capabilities;
- expected tensor inventory;
- supported precision/quantization combinations;
- optional placement recipe hints;
- optional publisher signature.

## 10.1 Why embed tokenizer and template data

A weight digest alone does not identify what prompt reaches the network.

Different tokenizers, templates, BOS/EOS handling, tool wrappers, or special-token maps can produce different token sequences from the same messages.

The `.kmodel` should therefore pin and preferably embed the small interpretation artifacts. Weight files remain external.

## 10.2 Example authoring manifest

YAML or JSON is authoring input only. The authoritative `.kmodel` output is canonical CBOR.

```yaml
kind: knolo.infer.model-image
version: 1

name: zai/glm-4.7-flash
variant: q5-k-m

architecture:
  family: glm-4.7
  adapter: knolo.glm47.v1

weights:
  format: gguf
  files:
    - path: model-q5-k-m.gguf
      size: 0
      sha256: sha256-...

tokenizer:
  embedded: tokenizer.json
  root: sha256-...

template:
  embedded: chat-template.jinja
  root: sha256-...

capabilities:
  - text-generation
  - tool-call-output

license:
  id: model-license-id
  acceptance_required: false

sources:
  - provider: huggingface
    repository: publisher/repository
    revision: immutable-revision

requirements:
  minimum_ram_bytes: 0
  minimum_vram_bytes: 0

signatures: []
```

`size: 0` is invalid in a built artifact; the compiler fills exact values and rejects unresolved files.

## 10.3 Artifact root

For sharded safetensors or any multi-file model:

```text
files = sort_by_utf8_path([
  {
    path,
    size_bytes,
    sha256
  }
])

artifact_root =
H("infer-model-artifact", { files })
```

Rules:

- normalized relative POSIX paths only;
- no `..`;
- no absolute paths;
- duplicate paths rejected;
- files sorted by normalized UTF-8 path;
- every size and digest checked before load;
- source URL is never identity;
- repository name, tag, or branch is never identity.

## 10.4 Model runtime root

```text
model_runtime_root =
H("infer-model-runtime", {
  model_image_root,
  artifact_root,
  tokenizer_root,
  template_root,
  config_root,
  architecture_adapter_root
})
```

This root is the effective model identity used by the scheduler, prefix cache, and receipt.

---

# 11. Lockfile design

Do not silently mutate the existing `knolo.lock.json` format.

Implement in two steps:

## Step A: Infer-local lockfile

Initial release:

```text
knolo.infer.lock.json
```

Example:

```json
{
  "kind": "knolo.infer.lock",
  "version": 1,
  "models": {
    "daily": {
      "modelImageRoot": "sha256-...",
      "artifactRoot": "sha256-...",
      "modelImagePath": ".knolo/models/daily.kmodel"
    }
  },
  "engine": {
    "channel": "native",
    "buildRoot": "sha256-..."
  },
  "profiles": {
    "workstation": {
      "placementRoot": "sha256-..."
    }
  }
}
```

## Step B: Shared Knolo lockfile

After Core and Infer agree on a cross-product schema, migrate to a namespaced `knolo.lock.json` revision that preserves all existing pack pins and unknown product sections.

No Infer agent should change Core’s current lockfile parser without:

- a written compatibility spec;
- migration tests;
- preservation of unknown sections;
- old CLI/new CLI interoperability tests;
- rollback tests.

---

# 12. Required versioned contracts

Implement and freeze these before optimizing the engine:

1. `ModelImageV1`
2. `ModelArtifactSetV1`
3. `EngineBuildDescriptorV1`
4. `KernelBundleDescriptorV1`
5. `HardwareProbeV1`
6. `PlacementPlanV1`
7. `PromptInputV1`
8. `PromptPlanV1`
9. `EvidenceBindingV1`
10. `SamplerPlanV1`
11. `GrammarPlanV1`
12. `InferenceIntentV1`
13. `ExecutionPlanV1`
14. `InferenceEventV1`
15. `InferenceReceiptV1`
16. `ReplayCheckReceiptV1`
17. `ModelConformanceReceiptV1`

Rust and TypeScript must verify the same golden vectors.

---

# 13. Engine build identity

Every native receipt must identify the actual binary and kernel bundle.

```rust
pub struct EngineBuildDescriptorV1 {
    pub kind: String,
    pub version: u32,
    pub binary_sha256: String,
    pub source_commit: String,
    pub cargo_lock_root: String,
    pub rustc_version: String,
    pub target_triple: String,
    pub build_profile: String,
    pub feature_set: Vec<String>,
    pub tensor_backend: String,
    pub tensor_backend_version: String,
    pub kernel_bundle_root: String,
}
```

The kernel bundle descriptor records:

- compiled CUDA architectures;
- CUDA toolkit version;
- source roots;
- compiler flags;
- build mode;
- embedded cubin/PTX roots;
- optional JIT compiler inputs and resulting code-object root.

If runtime JIT compilation is used, the receipt must bind:

```text
kernel source root
compiler version
flags
target architecture
resulting code-object root
```

Precompiled kernels are preferred for blessed profiles.

---

# 14. Hardware probe

Command:

```bash
knolo infer probe
```

The probe returns bounded, non-secret hardware capability data:

```text
CPU architecture and model class
physical/logical core counts
RAM total and available
NUMA topology when available
GPU vendor and model
VRAM total and available
compute capability
driver and runtime versions
supported precisions
storage class and available bytes
available native kernel bundles
```

Do not record device serial numbers by default.

A receipt may record the GPU model and compute capability because those facts can affect replay, but it should use a local opaque device slot rather than expose a globally unique hardware identifier.

---

# 15. Placement planner

Input:

```text
HardwareProbeV1
+ ModelImageV1
+ exact artifact inventory
+ runtime intent
+ memory policy
+ engine capabilities
```

Output:

```text
PlacementPlanV1
```

## 15.1 Placement plan contents

- model runtime root;
- selected devices;
- tensor groups assigned to each device;
- compute precision per group;
- weight storage precision;
- KV precision;
- CPU offload groups;
- MoE expert placement;
- context reservation;
- KV block size;
- workspace reservation;
- graph-capture mode;
- safety margin;
- expected memory totals;
- rejection reason if unsatisfiable.

## 15.2 Memory calculation

For ordinary MHA/GQA models, the adapter provides:

```text
kv_bytes_per_token =
2
× number_of_layers
× number_of_kv_heads
× head_dimension
× kv_element_bytes
```

The factor of two accounts for key and value.

The architecture adapter must override the formula for MLA, linear-attention state, hybrid attention, multimodal caches, or any model with a different state layout.

Total reservation:

```text
weight bytes
+ KV page-pool bytes
+ operator workspace
+ graph arenas
+ pinned host staging
+ tokenizer/template overhead
+ safety margin
```

Admission fails before execution when the declared limits cannot fit.

## 15.3 Reference workstation profile

Create a tested fixture named:

```text
workstation-ampere-dual
```

Reference class:

```text
Ryzen 7900X-class CPU
approximately 64 GB RAM
RTX 3090 24 GB primary compute GPU
RTX 3060 12 GB display/secondary GPU
```

Default policy:

- prefer the 3090 for the main model;
- reserve configurable VRAM on the display GPU;
- use the 3060 for a tiny model, embeddings, or explicit overflow;
- do not default to tensor parallel across asymmetric GPUs;
- do not assume that splitting across PCIe improves latency;
- benchmark every multi-GPU placement before blessing it;
- use CPU expert placement only when the exact model recipe and benchmark support it.

The planner must be generic. The profile is a fixture and recipe, not hard-coded application logic.

---

# 16. Prompt compiler

The prompt compiler is a security and provenance component, not a convenience formatter.

Pipeline:

```text
messages
+ system policy
+ tools
+ evidence context
+ chat template
+ special tokens
+ tokenizer
        ↓
normalized prompt input
        ↓
rendered text bytes
        ↓
exact token IDs
        ↓
PromptPlanV1
```

## 16.1 Required roots

Record separately:

- input messages root;
- tool-schema root;
- evidence binding root;
- chat-template root;
- tokenizer root;
- rendered-text root;
- token-ID root;
- special-token plan root;
- truncation plan root.

The exact token-ID root is authoritative for what enters the model.

## 16.2 Template execution

Use a sandboxed, bounded template renderer:

- no filesystem access;
- no network;
- no environment variables;
- no arbitrary functions;
- explicit allowlist of filters;
- loop and output limits;
- recursion disabled or tightly bounded;
- deterministic map iteration;
- deterministic whitespace;
- UTF-8 only.

Unsupported template behavior is a hard error.

## 16.3 Truncation

Truncation must never be implicit.

`PromptPlanV1` records:

- original token count;
- final token count;
- truncation strategy;
- removed message/evidence IDs;
- preserved system/tool sections;
- maximum context;
- reserved generation tokens.

For evidence-sensitive work, default to failure rather than silently dropping evidence.

## 16.4 Evidence binding

```rust
pub struct EvidenceBindingV1 {
    pub knowledge_image_root: Option<String>,
    pub knowledge_commit_root: Option<String>,
    pub query_receipt_ids: Vec<String>,
    pub reflex_receipt_ids: Vec<String>,
    pub context_root: String,
    pub ordered_evidence_ids: Vec<String>,
}
```

The engine does not query a `.knolo` image itself. The host integration queries Core and Reflex, then supplies bounded context plus this binding.

---

# 17. Native model runtime

## 17.1 Public engine traits

```rust
pub trait ExecutableModel: Send {
    fn identity(&self) -> &ModelRuntimeIdentity;
    fn capabilities(&self) -> &ModelCapabilities;
    fn kv_layout(&self) -> KvLayoutV1;

    fn prefill(
        &mut self,
        batch: &PrefillBatch,
        kv: &mut dyn KvStore,
    ) -> Result<PrefillOutput, EngineError>;

    fn decode(
        &mut self,
        batch: &DecodeBatch,
        kv: &mut dyn KvStore,
    ) -> Result<DecodeOutput, EngineError>;
}

pub trait ArchitectureAdapter: Send + Sync {
    fn id(&self) -> &'static str;
    fn validate_config(&self, config: &CanonicalModelConfig)
        -> Result<(), ModelError>;
    fn expected_tensors(&self, config: &CanonicalModelConfig)
        -> Result<TensorInventory, ModelError>;
    fn build(
        &self,
        source: &VerifiedWeightSource,
        placement: &PlacementPlanV1,
        backend: &dyn TensorBackend,
    ) -> Result<Box<dyn ExecutableModel>, ModelError>;
}
```

## 17.2 No dynamic model code

Architecture support is compiled into the binary.

A model config may select a known adapter, but may not provide executable code.

Unknown architecture:

```text
UNSUPPORTED_ARCHITECTURE
```

It must not trigger Python, shell commands, remote imports, or automatic compatibility fallback.

## 17.3 Initial architecture sequence

### Stage 0: Knolo micro-transformer

Create a tiny generated transformer with fixed synthetic weights and a tiny tokenizer.

Purpose:

- golden logits;
- deterministic CPU tests;
- KV tests;
- scheduler tests;
- receipt vectors;
- cross-language tests;
- no third-party model license.

### Stage 1: Dense Llama-family adapter

Implement:

- RMSNorm;
- RoPE;
- grouped-query attention;
- SwiGLU/MLP;
- causal mask;
- ordinary KV cache;
- safetensors F16/BF16.

### Stage 2: Quantized dense adapter

Add GGUF and selected quantizations in this order:

1. F16;
2. Q8_0;
3. Q6_K;
4. Q5_K_M;
5. Q4_K_M.

Each quantization requires:

- formal dequantization definition;
- CPU reference;
- GPU kernel;
- perplexity/accuracy delta report;
- artifact-specific conformance receipt.

### Stage 3: Generic MoE

Add:

- router logits;
- top-k expert selection;
- shared experts;
- deterministic expert tie-breaking;
- expert placement groups;
- grouped expert GEMM;
- CPU/GPU expert routing;
- expert load metrics.

### Stage 4: GLM-4.7-Flash

Implement as an explicit adapter after generic MoE.

No assumptions are inherited from model names. Support is based on verified official configuration and tensor inventory.

### Stage 5: GLM-5.3-Flash research

Separate adapter and milestone for:

- hybrid sparse/linear-attention schedule;
- KDA/linear state;
- mHC-related transformations;
- large MoE routing;
- MTP layer;
- multimodal projector/vision stack;
- million-token context policies.

Do not mark this adapter production-ready until each subsystem has independent conformance evidence.

---

# 18. Weight loading

## 18.1 Safetensors

Requirements:

- mmap where safe and supported;
- all headers bounded;
- shape multiplication overflow checked;
- dtype allowlist;
- tensor-name allowlist from adapter;
- all required tensors present;
- duplicate tensors rejected;
- unexpected tensors rejected unless adapter explicitly declares optional tensors;
- shard index verified;
- each file hash verified before mapping.

## 18.2 GGUF

Implement a bounded parser or use a narrowly scoped audited parser.

Requirements:

- file digest verified first;
- magic/version checked;
- metadata count bounded;
- tensor count bounded;
- dimensions bounded;
- offsets/alignment checked;
- overlapping ranges rejected;
- integer overflow checked;
- unsupported quantization rejected;
- template/tokenizer metadata treated as untrusted;
- `.kmodel` remains authoritative when metadata conflicts.

## 18.3 Runtime conversion

Do not silently quantize or transform weights during load.

A conversion produces a new content-addressed artifact plus a conversion receipt containing:

- source artifact root;
- converter build root;
- conversion configuration root;
- destination artifact root;
- validation result.

---

# 19. Tensor backend and kernels

## 19.1 Device sequence

1. CPU reference.
2. NVIDIA CUDA.
3. Metal after CUDA stability.
4. Other accelerators only through explicit backends.

## 19.2 Kernel categories

Initial native path may use substrate operations for correctness. Optimize in this order:

1. matrix multiplication via cuBLAS/cuBLASLt;
2. exact attention kernel;
3. RMSNorm;
4. RoPE;
5. fused residual operations;
6. quantized matrix multiplication;
7. grouped MoE GEMM;
8. top-k routing;
9. sampling and logit processors;
10. KV copy/scatter/gather.

## 19.3 Kernel registry

Each kernel implementation has:

- stable ID;
- supported dtype/device/shape constraints;
- source/build root;
- numerical tolerance class;
- deterministic capability flag;
- workspace requirements;
- benchmark evidence.

The planner selects only compatible kernels and writes the selection to `ExecutionPlanV1`.

No runtime auto-tuner may alter kernel choices in replay mode.

## 19.4 Numerical policy

Do not promise bitwise agreement between CPU and GPU.

Define model conformance tolerances for:

- per-layer hidden states;
- final logits;
- greedy next-token identity;
- generated token sequence under fixed conditions.

A model adapter can pass numerical tolerance while failing token parity; the conformance receipt must distinguish those outcomes.

---

# 20. Paged KV cache

## 20.1 Structure

```text
KV Page Pool
├── device-local pages
├── optional host-resident pages
├── free list
├── per-sequence block tables
├── prefix-cache index
└── eviction metadata
```

A page contains a fixed number of token slots for every layer/state component defined by `KvLayoutV1`.

Initial block sizes may include 16 and 32 tokens. The chosen size is part of the execution plan.

## 20.2 Transactional allocation

For each scheduling step:

1. calculate pages required;
2. reserve pages;
3. update a pending block table;
4. execute the model step;
5. commit the table on success;
6. release reservation on failure.

A failed kernel must not leave the allocator believing pages are valid.

## 20.3 Prefix cache

Cache entry identity:

```text
model_runtime_root
+ prompt_token_prefix_root
+ positional_plan_root
+ execution_precision_root
+ kv_layout_root
+ cache_namespace
```

Rules:

- exact token identity only;
- tenant/project namespace required;
- cross-tenant disabled by default;
- copy-on-write for branches;
- model unload invalidates associated pages;
- execution precision change invalidates the entry;
- cached and uncached logits must match the adapter’s tolerance;
- cache hit/miss and reused-token count are recorded.

## 20.4 Eviction

Default eviction policy may use recency plus recomputation cost, but must remain bounded and deterministic under the same event sequence.

High-priority active sequences cannot be evicted.

---

# 21. Scheduler

## 21.1 Request state machine

```text
ACCEPTED
→ NORMALIZING
→ TOKENIZING
→ ADMISSION_PENDING
→ QUEUED
→ PREFILL_READY
→ PREFILL_RUNNING
→ DECODE_READY
→ DECODE_RUNNING
→ STREAMING
→ FINALIZING
→ COMPLETE

Terminal alternatives:
CANCELLED
REJECTED
FAILED
WORKER_LOST
```

Every transition emits an `InferenceEventV1`.

## 21.2 Continuous batching

The scheduler builds a decode batch on every iteration from compatible sequences.

Compatibility includes:

- same model runtime root;
- same precision plan;
- same attention/KV layout;
- compatible grammar mode;
- compatible speculative mode;
- compatible replay policy;
- compatible kernel shape bucket.

Requests may enter or leave between decode iterations.

## 21.3 Chunked prefill

Long prompts must not monopolize the device.

Prefill is divided into bounded chunks. The scheduler can interleave:

- decode steps for interactive requests;
- chunks from long prompts;
- prefix-cache materialization.

The chunk size is part of the execution plan.

## 21.4 Fairness

Use weighted fair scheduling with explicit service classes:

```text
interactive
standard
batch
background
```

Each class has bounded queue time and token budget policy.

Tenant weights and concurrency limits come from host configuration, not model manifests.

## 21.5 No silent degradation

The scheduler must not silently:

- lower context length;
- switch model;
- switch backend;
- reduce precision;
- drop evidence;
- disable grammar constraints;
- move tensors to CPU.

If the plan cannot run, return a structured failure or require an explicitly approved replan.

---

# 22. Sampling and logit processing

## 22.1 Fixed processing order

Freeze an order such as:

1. model logits;
2. banned-token mask;
3. grammar mask;
4. repetition penalty;
5. presence/frequency penalties;
6. temperature transform;
7. top-k;
8. top-p;
9. min-p;
10. RNG sample;
11. stop-token evaluation.

Changing order changes behavior and therefore requires a new sampler contract version.

## 22.2 Greedy mode

For greedy generation:

- no RNG;
- ties resolved by lowest token ID;
- tie-breaking rule included in the sampler plan.

## 22.3 Random mode

Use a specified counter-based RNG such as Philox.

Record:

- algorithm and version;
- seed;
- request stream/subsequence;
- token-step counter.

Do not use an unspecified platform RNG.

## 22.4 Stop conditions

Record:

- EOS token IDs;
- stop-string roots;
- grammar termination;
- maximum tokens;
- cancellation;
- timeout;
- policy termination.

The output receipt includes the exact finish reason.

---

# 23. Structured generation and tools

Structured decoding is an engine feature; tool execution is not.

## 23.1 Grammar compiler

Later milestone:

```text
JSON Schema / regex / grammar
        ↓
bounded normalized grammar
        ↓
DFA/FSM
        ↓
token mask tables
```

The receipt records:

- source schema root;
- normalized grammar root;
- compiler build root;
- automaton root;
- validation outcome.

## 23.2 Tool calls

The engine may generate a structured tool-call object.

It must never execute the tool.

Knolo Agents or the application host:

- validates authority;
- checks budgets;
- executes the tool;
- records the effect;
- returns results for the next inference call.

---

# 24. Speculative decoding

Not in the initial native correctness release.

When added, support two explicit modes:

1. draft-model speculative decoding;
2. model-native MTP speculation.

The receipt binds:

- target model runtime root;
- draft model runtime root or MTP head root;
- speculation algorithm/version;
- proposal length;
- acceptance rule;
- accepted/rejected token counts;
- target-only verification steps;
- cache effects.

Disabling speculation must not change the target distribution for an algorithm claiming distribution preservation. Conformance tests must verify this statistically and with controlled fixtures.

---

# 25. Execution modes and replay assurance

Do not market “deterministic generation” as a blanket property.

Expose three execution modes:

## `throughput`

- continuous batching;
- prefix cache;
- dynamic shape buckets;
- approved auto-tuning;
- optional speculation;
- strongest throughput;
- provenance receipt only.

## `pinned`

- pinned model, engine, kernels, placement, prompt tokens, sampler, and seed;
- ordinary batching may be allowed if its trace is recorded;
- replay-checkable on a matching environment;
- no guarantee of bitwise equality.

## `isolated-replay`

- one request isolated from unrelated batching;
- fixed kernel plan;
- fixed allocation policy;
- auto-tuning disabled;
- prefix state either disabled or fully pinned;
- fixed seed and sampler;
- deterministic kernels where available;
- best setting for exact replay checking.

Receipt assurance classes:

```text
provenance
same_build_replayable
exact_replay_verified
```

`exact_replay_verified` is assigned only after a subsequent replay produces matching output token IDs and output bytes. It is not predicted in advance.

---

# 26. Inference intent and receipt structure

## 26.1 Semantic intent

Separate the user’s semantic execution request from dynamic runtime facts.

```rust
pub struct InferenceIntentV1 {
    pub kind: String,
    pub version: u32,
    pub model_runtime_root: String,
    pub prompt_plan_root: String,
    pub sampler_plan_root: String,
    pub grammar_plan_root: Option<String>,
    pub evidence_binding_root: Option<String>,
    pub requested_mode: String,
    pub limits_root: String,
}
```

`intent_root` is stable across repeated attempts with the same pinned request.

## 26.2 Receipt

```rust
pub struct InferenceReceiptV1 {
    pub kind: String,
    pub version: u32,

    pub receipt_id: String,
    pub intent_root: String,

    pub model: ModelReceiptBindingV1,
    pub engine: EngineReceiptBindingV1,
    pub hardware: HardwareReceiptBindingV1,
    pub placement: PlacementReceiptBindingV1,
    pub prompt: PromptReceiptBindingV1,
    pub knowledge: Option<EvidenceBindingV1>,
    pub sampler: SamplerReceiptBindingV1,
    pub execution: ExecutionReceiptBindingV1,
    pub output: OutputReceiptBindingV1,
    pub timing: TimingReceiptBindingV1,

    pub assurance: String,
    pub previous_receipt_root: Option<String>,
    pub signatures: Vec<ReceiptSignatureV1>,
}
```

## 26.3 Required model fields

- `.kmodel` root;
- weight artifact root;
- model runtime root;
- architecture adapter ID/root;
- config root;
- tokenizer root;
- template root;
- storage precision;
- compute precision.

## 26.4 Required engine fields

- engine build root;
- native or compatibility backend;
- backend build/version;
- binary hash;
- kernel bundle root;
- selected kernel plan root.

## 26.5 Required prompt fields

- messages/input root;
- tools root;
- evidence context root;
- rendered text root;
- token-ID root;
- prompt token count;
- truncation plan root.

## 26.6 Required execution fields

- execution plan root;
- placement root;
- scheduling mode;
- prefill chunk plan;
- KV block size;
- prefix-cache hit and reused token count;
- batch trace root;
- speculative plan root if used;
- event trace root;
- retry/attempt number.

## 26.7 Required output fields

- output token-ID root;
- output text root;
- token count;
- finish reason;
- optional structured-output validation root.

## 26.8 Privacy

Receipts omit raw prompt and output content by default.

They contain:

- roots;
- counts;
- IDs;
- bounded metadata;
- safe error codes.

Optional encrypted content sidecars are a later feature and must not change the base receipt contract.

---

# 27. Durable receipt journal

The receipt cannot be an afterthought generated only after successful output.

## 27.1 Request journal

Before execution:

1. validate and root the intent;
2. create an append-only request journal;
3. write `accepted`;
4. fsync according to receipt policy;
5. begin execution.

Each event commits to the previous event root:

```text
event_root_n =
H("infer-execution-event", {
  previous_event_root,
  event
})
```

The final receipt points to the terminal event root.

## 27.2 Receipt policies

### `durable-stream`

- intent is persisted before output;
- token bytes stream immediately;
- rolling output roots are maintained;
- final receipt arrives as the terminal event;
- consumers must treat prior chunks as pending verification until finalization.

### `atomic-verified`

- output is buffered;
- final receipt is persisted and signed;
- output and receipt are released together;
- intended for regulated/evidence-sensitive workflows.

### `compatibility`

- OpenAI-compatible stream;
- receipt ID delivered in headers/final metadata where possible;
- verification guarantees documented separately.

For Knolo Agents, external effects should not be committed based on streamed model output until the final receipt is available and policy accepts it.

## 27.3 Receipt store

Authoritative storage:

```text
~/.knolo/infer/
├── receipts/sha256/...
├── journals/<request-id>/...
├── models/sha256/...
├── model-images/sha256/...
└── cache/...
```

Use atomic temp-file creation, fsync where required, and rename.

An optional SQLite index may accelerate lookup, but canonical receipt files remain authoritative.

---

# 28. API design

## 28.1 Knolo-native API

Base:

```text
/knolo/infer/v1
```

Endpoints:

```text
POST /complete
POST /tokenize
POST /replay
GET  /models
GET  /models/{alias}
POST /models/load
POST /models/unload
GET  /receipts/{digest}
POST /receipts/verify
GET  /health
GET  /metrics
```

## 28.2 Native completion request

```json
{
  "model": "daily",
  "messages": [
    {
      "role": "user",
      "content": "Review the policy evidence."
    }
  ],
  "generation": {
    "temperatureMicros": 0,
    "topPMillionths": 1000000,
    "maxOutputTokens": 512,
    "seed": 42
  },
  "execution": {
    "mode": "pinned",
    "receiptPolicy": "durable-stream"
  },
  "knowledge": {
    "knowledgeImageRoot": "sha256-...",
    "queryReceiptIds": ["..."],
    "reflexReceiptIds": ["..."],
    "contextRoot": "sha256-..."
  }
}
```

JSON is normalized into the canonical CBOR contracts before execution.

## 28.3 Native response

Non-streaming:

```json
{
  "output": {
    "text": "...",
    "finishReason": "stop"
  },
  "receipt": {
    "receiptRoot": "sha256-...",
    "assurance": "same_build_replayable"
  }
}
```

Streaming SSE events:

```text
knolo.accepted
knolo.delta
knolo.usage
knolo.receipt
knolo.error
```

The final `knolo.receipt` event is authoritative.

## 28.4 OpenAI compatibility

Initial subset:

```text
GET  /v1/models
POST /v1/chat/completions
```

Rules:

- publish a compatibility matrix;
- reject unsupported fields;
- do not silently ignore generation parameters;
- return receipt identity in response metadata and headers;
- keep native receipts retrievable through the Knolo endpoint;
- do not claim complete compatibility until conformance tests prove it.

Later:

```text
POST /v1/responses
POST /v1/embeddings
```

---

# 29. CLI

```bash
knolo infer probe

knolo infer model build \
  ./model-image.yaml \
  --out ./dist/daily.kmodel

knolo infer model inspect ./dist/daily.kmodel
knolo infer model verify ./dist/daily.kmodel

knolo infer pin daily ./dist/daily.kmodel
knolo infer pull daily
knolo infer verify daily

knolo infer plan daily --intent interactive
knolo infer load daily
knolo infer serve --model daily

knolo infer run \
  --model daily \
  --prompt "Explain the evidence." \
  --mode pinned \
  --receipt ./receipt.cbor

knolo infer receipt show sha256-...
knolo infer receipt verify ./receipt.cbor
knolo infer replay ./receipt.cbor

knolo infer benchmark daily --suite smoke
knolo infer doctor
knolo infer gc
```

CLI output defaults to human-readable summaries. `--json` returns stable machine output. `--cbor` is available for canonical artifacts.

---

# 30. TypeScript SDK

Package:

```text
@knolo/infer
```

Responsibilities:

- typed native API client;
- OpenAI-compatible endpoint helper;
- receipt types and verification;
- model-image build/inspection helpers where appropriate;
- Core/Reflex composition adapter;
- Knolo Agents host-effect adapter;
- stream state that distinguishes pending output from finalized receipt;
- no model execution in Node.

Example:

```ts
import { InferClient } from '@knolo/infer';

const infer = new InferClient({
  baseUrl: 'http://127.0.0.1:6767',
});

const result = await infer.complete({
  model: 'daily',
  messages: [{ role: 'user', content: 'Review this claim.' }],
  generation: {
    temperatureMicros: 0,
    topPMillionths: 1_000_000,
    maxOutputTokens: 256,
    seed: 42,
  },
  execution: {
    mode: 'pinned',
    receiptPolicy: 'atomic-verified',
  },
});

console.log(result.output.text);
console.log(result.receipt.receiptRoot);
```

---

# 31. `@knolo/core` integration

The native engine has no mandatory runtime dependency on Core.

The integration package does:

```text
mount/query Knowledge Image
        ↓
receive deterministic query receipt
        ↓
build bounded evidence context
        ↓
pass context + EvidenceBindingV1 to Infer
```

Rules:

- lexical-first Core retrieval remains the default;
- no vector database is required;
- semantic reranking, if used, remains optional and separately receipted;
- Infer never mutates the Knowledge Image;
- Infer never invents evidence IDs;
- Knowledge Image and commit roots are passed unchanged.

---

# 32. Reflex integration

```text
Reflex request
        ↓
deterministic context selection
        ↓
selection.context
+ selection.receipt
        ↓
Infer prompt compiler
        ↓
generated output
        ↓
optional Reflex output validation
```

The Infer receipt binds:

- Reflex receipt ID/root;
- selected context root;
- tokenized context root;
- output root.

Reflex remains responsible for behavior selection and output-policy validation. Infer remains responsible for execution.

---

# 33. Knolo Agents integration

Implement a host effect, not a hard dependency:

```ts
import { createKnoloInferEffect } from '@knolo/infer/agents';
```

Flow:

```text
Agent node requests LLM effect
        ↓
Agent policy checks authority and budget
        ↓
Infer executes pinned request
        ↓
Infer returns output + receipt
        ↓
Agent records receipt reference in ordered event
        ↓
Agent transitions or suspends
```

The adapter must support policy conditions such as:

- required model alias/root;
- allowed execution modes;
- required Knowledge Image root;
- maximum prompt/output tokens;
- required receipt assurance;
- structured-output schema root;
- deny unverified compatibility backends.

Inference never executes the agent’s tools.

---

# 34. Hub integration

Hub publishes `.kmodel` artifacts and metadata, not weight blobs.

Hub record:

- publisher;
- `.kmodel` root;
- weight artifact root;
- source provider/revision hints;
- architecture;
- quantization;
- supported native-engine version range;
- license metadata;
- verified placement recipes;
- conformance receipts;
- benchmark receipts;
- signature;
- yanked status.

Weight sources remain:

- Hugging Face;
- local disk;
- enterprise object storage;
- customer-controlled storage.

Install flow:

```text
resolve .kmodel by digest
→ verify .kmodel
→ resolve exact weight descriptors
→ stream download to staging
→ hash while downloading
→ verify all files
→ atomically promote to local CAS
→ write lock pin
```

The Hub UI should call a model “native-supported” only when it has a valid `ModelConformanceReceiptV1` for the relevant engine build family.

---

# 35. Compatibility backends

## 35.1 Backend interface

```rust
pub trait InferenceBackend {
    fn backend_identity(&self) -> BackendIdentityV1;
    fn capabilities(&self) -> BackendCapabilitiesV1;
    fn load(&mut self, model: &ResolvedModel) -> Result<(), BackendError>;
    fn complete(
        &mut self,
        request: BackendRequestV1,
    ) -> Result<BackendResponseV1, BackendError>;
    fn unload(&mut self) -> Result<(), BackendError>;
}
```

## 35.2 Backend selection

```text
native
llama.cpp
ollama
```

Selection is explicit in config/CLI/request policy.

No silent fallback.

## 35.3 Verification classes

```text
native-verified
sidecar-artifact-verified
backend-reported
unverified
```

A compatibility backend cannot receive a native verification badge merely because it returned text.

---

# 36. Security design

## 36.1 Artifact security

- hash before parse where practical;
- bounded parsing;
- integer overflow checks;
- allocation limits;
- path traversal prevention;
- no symlink escape;
- no shell interpolation;
- no executable model files;
- no Python;
- no remote code;
- atomic download promotion;
- immutable CAS files after verification.

## 36.2 Worker sandbox

Linux production profile:

- separate unprivileged user;
- no external network;
- read-only model CAS;
- writable worker scratch only;
- seccomp profile after feature stabilization;
- resource limits;
- process group isolation;
- parent-death signal;
- bounded shared memory;
- direct argument arrays, never shell strings.

## 36.3 API security

- bind to localhost by default;
- remote bind requires explicit configuration;
- bearer or mTLS support;
- request body limits;
- rate limits;
- concurrency limits;
- tenant namespace;
- safe error responses;
- no prompt text in ordinary logs;
- metrics labels must not contain user content.

## 36.4 Cache side channels

- prefix cache scoped by tenant/project;
- cache-sharing policy explicit;
- no cross-tenant sharing by default;
- do not reveal whether another tenant’s prefix exists;
- cache metrics aggregated safely.

## 36.5 Receipt keys

- keys loaded through host-owned secret storage;
- key material never serialized;
- receipt contains key ID and signature only;
- rotation supported through explicit trusted metadata;
- unsigned local development receipts clearly marked.

## 36.6 Supply chain

Ship:

- `NOTICE`;
- third-party license inventory;
- SBOM;
- binary hashes;
- engine build descriptor;
- signed release manifest;
- reproducible-build instructions where feasible.

---

# 37. Failure model and error codes

Required stable errors include:

```text
MODEL_IMAGE_INVALID
MODEL_IMAGE_SIGNATURE_INVALID
MODEL_ARTIFACT_MISSING
MODEL_DIGEST_MISMATCH
TOKENIZER_INVALID
TEMPLATE_INVALID
UNSUPPORTED_ARCHITECTURE
UNSUPPORTED_QUANTIZATION
UNSUPPORTED_KERNEL
PLACEMENT_UNSATISFIABLE
INSUFFICIENT_MEMORY
CONTEXT_LIMIT_EXCEEDED
PROMPT_COMPILATION_FAILED
RECEIPT_REQUIRED
RECEIPT_PERSIST_FAILED
WORKER_START_FAILED
WORKER_LOST
CUDA_OOM
CUDA_FAULT
REQUEST_CANCELLED
REQUEST_TIMEOUT
BACKEND_NOT_ALLOWED
REPLAY_ENVIRONMENT_MISMATCH
REPLAY_OUTPUT_MISMATCH
```

Every failure returns:

- stable code;
- safe message;
- retryability;
- request/attempt ID;
- optional partial receipt root;
- no secrets or raw prompt unless explicitly enabled locally.

A failed or cancelled generation produces a terminal journal event. Partial output receives a partial output root and explicit non-complete status.

---

# 38. Observability

Metrics:

- queue depth by service class;
- admission rejection counts;
- model load time;
- verified bytes;
- time to first token;
- prefill tokens/sec;
- decode tokens/sec;
- active sequences;
- KV pages total/free/pinned;
- prefix hit rate and reused tokens;
- worker restarts;
- OOM count;
- receipt finalization latency;
- cancellation count;
- request duration;
- scheduler iteration duration.

Logs:

- structured;
- redacted;
- request IDs and roots;
- no prompt/output content by default.

Tracing:

```text
API
→ prompt compilation
→ admission
→ prefill chunks
→ decode iterations
→ finalization
```

The observability subsystem is not part of semantic execution identity unless a setting changes execution behavior.

---

# 39. Conformance strategy

## 39.1 Contract conformance

Rust and TypeScript share fixtures for:

- canonical CBOR;
- digest domains;
- fixed-point generation settings;
- `.kmodel`;
- artifact roots;
- prompt roots;
- placement roots;
- receipt roots;
- malformed input rejection.

## 39.2 Micro-model conformance

The synthetic Knolo micro-transformer provides:

- exact CPU logits;
- expected token sequence;
- KV page fixtures;
- prefix reuse fixtures;
- batching invariance fixtures;
- cancellation fixtures;
- replay fixtures.

## 39.3 Real-model correctness

For every supported architecture:

1. compare selected layer outputs to a frozen reference;
2. compare final logits within declared tolerances;
3. compare greedy next-token IDs;
4. compare a bounded greedy sequence;
5. compare cached vs uncached execution;
6. compare batch vs isolated execution;
7. compare chunked vs unchunked prefill;
8. validate quantized perplexity/accuracy delta;
9. issue `ModelConformanceReceiptV1`.

## 39.4 Corruption tests

At minimum:

- one-byte weight mutation;
- truncated shard;
- malformed safetensors header;
- malformed GGUF metadata;
- overlapping tensor ranges;
- wrong tensor shape;
- missing tensor;
- duplicated tensor;
- modified tokenizer;
- modified template;
- model-image root mismatch;
- artifact root mismatch;
- receipt mutation;
- event-chain mutation.

All must fail closed.

## 39.5 Runtime tests

- worker crash;
- CUDA OOM;
- cancellation during prefill;
- cancellation during decode;
- client disconnect;
- receipt-store failure;
- disk full;
- model unload under queued requests;
- daemon restart;
- stale lock;
- duplicate request ID;
- concurrent load request;
- prefix-cache eviction under load;
- long-running soak with bounded memory.

## 39.6 Fuzzing

Fuzz:

- canonical CBOR decoder;
- `.kmodel` parser;
- GGUF parser;
- safetensors inventory validation;
- template compiler;
- tokenizer boundaries;
- API normalization;
- IPC frames;
- receipt verifier.

---

# 40. Benchmark methodology

Do not publish unrooted “tokens per second” numbers.

Every benchmark report records:

- model image root;
- artifact root;
- engine build root;
- kernel bundle root;
- hardware probe root;
- placement root;
- prompt dataset root;
- prompt-length distribution;
- output-length distribution;
- concurrency;
- scheduler mode;
- cache policy;
- execution mode;
- warm/cold state;
- run count;
- statistics and raw report root.

Metrics:

- model verification time;
- model load time;
- time to first token;
- time per output token;
- prefill throughput;
- decode throughput;
- request throughput;
- p50/p95/p99 latency;
- peak RAM/VRAM;
- KV utilization;
- prefix reuse;
- cancellation latency;
- receipt overhead;
- model swap time.

Reference comparisons:

- current pinned llama.cpp build for workstation/GGUF;
- current pinned vLLM build for supported server-class safetensors models;
- optional pinned mistral.rs build for Rust-runtime comparison.

Comparisons must use the same:

- exact weight artifact;
- quantization;
- tokenizer/template;
- context;
- sampler;
- hardware;
- prompt/output distributions.

A model recipe is marked:

```text
experimental
conformant
blessed
```

A `blessed` recipe requires correctness, security, stability, and benchmark receipts. Fast but non-conformant is not blessed.

---

# 41. Delivery sequence and release gates

No phase advances because the code “seems to work.” Each phase has a gate.

## Phase 0 — Contracts and threat model

Deliver:

- KIP-INFER specifications;
- canonical encoder/decoder;
- Rust and TypeScript types;
- golden vectors;
- threat model;
- receipt schema;
- `.kmodel` schema.

Gate:

- cross-language roots match;
- malformed vectors fail;
- no unresolved contract ambiguity.

## Phase 1 — CPU reference engine

Deliver:

- micro-transformer;
- dense Llama-family adapter;
- safetensors loader;
- tokenizer/template compiler;
- greedy and seeded sampler;
- request journal;
- receipt generation;
- local CLI run.

Gate:

- exact micro-model fixtures;
- reference logits pass;
- receipt verification passes;
- no external model server.

## Phase 2 — CUDA single-request native execution

Deliver:

- Candle CUDA backend;
- GPU weight placement;
- ordinary KV cache;
- exact attention;
- kernel registry;
- hardware probe;
- placement plan;
- isolated execution.

Gate:

- CPU/GPU tolerance suite;
- greedy token parity on blessed fixture;
- stable memory under repeated runs;
- engine build and kernel roots in receipts.

## Phase 3 — Server, paging, and scheduling

Deliver:

- supervisor/worker;
- native API;
- OpenAI compatibility;
- paged KV;
- continuous batching;
- chunked prefill;
- cancellation;
- durable streaming;
- metrics.

Gate:

- batch/isolated invariance;
- bounded-memory soak;
- crash recovery;
- cancellation and disconnect tests;
- no unreceipted native completion.

## Phase 4 — GGUF and quantized kernels

Deliver:

- bounded GGUF parser;
- selected quant formats;
- quantized GEMM;
- conversion receipts;
- quantized conformance reports.

Gate:

- corruption fuzzing;
- accuracy/perplexity deltas recorded;
- memory estimates match observed bounds;
- no silent conversion.

## Phase 5 — Knolo product integration

Deliver:

- `@knolo/infer`;
- Core/Reflex adapter;
- Agents host effect;
- Hub `.kmodel` metadata;
- receipt UI data contract;
- model/knowledge receipt chain demo.

Gate:

- end-to-end evidence-to-output verification;
- Agents policy can deny the wrong model/root/backend;
- Hub installation verifies every artifact.

## Phase 6 — MoE and GLM-4.7

Deliver:

- generic MoE;
- expert placement;
- grouped kernels;
- GLM-4.7 adapter;
- workstation recipes.

Gate:

- router/expert conformance;
- single and mixed placement tests;
- no unsupported automatic fallback;
- recipe benchmark and conformance receipts.

## Phase 7 — Advanced serving

Deliver selectively:

- prefix cache;
- CUDA Graphs;
- structured decoding;
- speculative decoding;
- multi-model lifecycle;
- secondary GPU/tiny-model service.

Each feature has its own conformance and receipt fields.

## Phase 8 — GLM-5.3 research

Deliver only after a dedicated design review:

- hybrid attention state model;
- KDA/linear attention;
- mHC support;
- MTP speculation;
- multimodal input path;
- large MoE placement;
- extreme-context cache strategy.

Do not hold the production engine release for this phase.

---

# 42. Parallel agent workstreams

## Workstream A — Contracts

Own:

- KIP-INFER specs;
- canonical CBOR;
- digest domains;
- Rust/TS schemas;
- golden vectors;
- receipt verifier.

Must finish before other teams freeze public structures.

## Workstream B — Artifact system

Own:

- `.kmodel` compiler/verifier;
- CAS;
- safetensors inventory;
- GGUF parser;
- download staging;
- lockfile;
- signatures and license metadata.

## Workstream C — Prompt system

Own:

- tokenizer;
- template sandbox;
- prompt normalization;
- tool-schema normalization;
- truncation;
- evidence binding;
- token roots.

## Workstream D — Native model core

Own:

- tensor abstraction;
- CPU reference;
- architecture adapters;
- model graph;
- weight mapping;
- KV layout definitions;
- numerical conformance.

## Workstream E — CUDA and placement

Own:

- CUDA backend;
- kernel registry;
- kernel bundle identity;
- hardware probe;
- memory planner;
- placement;
- profiling.

## Workstream F — KV and scheduler

Own:

- paged KV;
- prefix cache;
- admission;
- continuous batching;
- chunked prefill;
- fairness;
- cancellation.

## Workstream G — Daemon and APIs

Own:

- supervisor/worker IPC;
- process lifecycle;
- native API;
- OpenAI API;
- streaming;
- metrics;
- safe logging.

## Workstream H — Knolo integrations

Own:

- TypeScript SDK;
- Core/Reflex adapter;
- Agents effect;
- Hub metadata;
- Studio data contract.

## Workstream I — Security and verification

Own:

- threat model;
- sandbox;
- fuzzing;
- corruption tests;
- SBOM/NOTICE;
- signing;
- release gates;
- benchmark receipts.

---

# 43. Initial issue backlog

Create these issues before implementation begins.

## Contracts

1. Specify canonical Infer CBOR subset.
2. Reserve digest domains.
3. Define fixed-point sampler representation.
4. Define `ModelImageV1`.
5. Define `ModelArtifactSetV1`.
6. Define `PromptPlanV1`.
7. Define `PlacementPlanV1`.
8. Define `InferenceReceiptV1`.
9. Define `ReplayCheckReceiptV1`.
10. Generate Rust/TypeScript golden vectors.

## Artifacts

11. Build `.kmodel` compiler.
12. Build `.kmodel` verifier.
13. Build model CAS.
14. Implement staged, resumable, hash-verifying pull.
15. Implement safetensors inventory verifier.
16. Design bounded GGUF parser.
17. Implement infer-local lockfile.
18. Add model signature verification.

## Native reference

19. Create synthetic micro-transformer.
20. Implement CPU tensor backend adapter.
21. Implement Llama-family model adapter.
22. Implement deterministic tokenizer/template fixture.
23. Implement sampler pipeline and Philox RNG.
24. Implement CPU inference receipt.

## CUDA

25. Add CUDA hardware probe.
26. Add Candle CUDA tensor backend.
27. Implement placement memory estimator.
28. Add exact attention path.
29. Add CUDA kernel identity registry.
30. Add CPU/GPU logit conformance.

## Serving

31. Implement worker IPC.
32. Implement worker supervisor.
33. Implement request journal.
34. Implement paged KV allocator.
35. Implement continuous decode scheduler.
36. Implement chunked prefill.
37. Implement cancellation.
38. Implement native streaming API.
39. Implement OpenAI chat completion subset.
40. Implement receipt store and verifier.

## Integration

41. Implement `@knolo/infer`.
42. Implement Core/Reflex composition.
43. Implement Agents host effect.
44. Implement Hub `.kmodel` record.
45. Implement end-to-end receipt chain demo.

## Hardening

46. Fuzz canonical CBOR.
47. Fuzz `.kmodel`.
48. Fuzz GGUF.
49. Add worker crash/OOM suite.
50. Add bounded-memory soak.
51. Generate SBOM and NOTICE checks.
52. Create signed release manifest.

---

# 44. Definition of done for the first native release

The first native release is done when a developer can execute:

```bash
knolo infer model verify ./daily.kmodel
knolo infer pin daily ./daily.kmodel
knolo infer pull daily
knolo infer plan daily --intent interactive
knolo infer serve --model daily
```

Then submit a request bound to a Knolo Knowledge Image and receive:

```json
{
  "output": {
    "text": "...",
    "finishReason": "stop"
  },
  "receipt": {
    "modelRuntimeRoot": "sha256-...",
    "artifactRoot": "sha256-...",
    "engineBuildRoot": "sha256-...",
    "kernelBundleRoot": "sha256-...",
    "placementRoot": "sha256-...",
    "promptTokenRoot": "sha256-...",
    "knowledgeImageRoot": "sha256-...",
    "queryReceiptIds": ["..."],
    "reflexReceiptIds": ["..."],
    "outputTokenRoot": "sha256-...",
    "outputTextRoot": "sha256-...",
    "assurance": "same_build_replayable"
  }
}
```

A second machine with the same pinned artifacts must be able to verify:

- `.kmodel` identity;
- weight identity;
- tokenizer/template identity;
- engine build identity;
- placement plan;
- prompt token identity;
- Knowledge Image and receipt references;
- sampler configuration;
- output identity;
- receipt signature.

The native path must run without llama.cpp, Ollama, mistral.rs, Python, or another model server.

---

# 45. Architecture decisions that agents must not reverse casually

## ADR-001 — Inference remains outside Core

`@knolo/core` stays model-neutral.

## ADR-002 — Native is not a sidecar wrapper

The native backend loads and executes weights directly.

## ADR-003 — Rust owns the runtime

TypeScript owns ergonomics and integration, not GPU process management.

## ADR-004 — Canonical contracts use integers, not floats

Sampler values are fixed-point in rooted objects.

## ADR-005 — Prompt token IDs are authoritative

Messages and rendered text are supporting identities; token IDs prove model input.

## ADR-006 — Model interpretation is pinned

Tokenizer, template, config, and special tokens belong in or are pinned by `.kmodel`.

## ADR-007 — Weight aliases are not identity

Exact artifact roots are identity.

## ADR-008 — No silent fallback

Backend, precision, placement, truncation, and degradation are explicit.

## ADR-009 — Tools remain host effects

Infer generates; Agents authorizes and acts.

## ADR-010 — Receipts are durable runtime output

A native completion is incomplete until its receipt policy is satisfied.

## ADR-011 — Performance features are receipted

Caching, batching, speculation, graph capture, and kernel selection are execution facts.

## ADR-012 — Third-party components retain their identity

Knolo does not rebadge llama.cpp, Ollama, Candle, or other dependencies.

---

# 46. Knolo-owned differentiation

The defensible product layer is the composition of:

```text
portable model interpretation
+ exact model artifact identity
+ hardware-aware placement
+ prompt/token identity
+ Knowledge Image binding
+ Reflex/query receipt binding
+ execution trace
+ output identity
+ replay verification
```

The most differentiated components are:

1. `.kmodel` as a portable, signed execution descriptor without redistributing weights;
2. the composed knowledge-model-prompt-execution receipt;
3. receipt-aware execution modes;
4. a placement plan that is content-addressed and independently inspectable;
5. prompt identity that commits both rendered bytes and exact token IDs;
6. an inference event chain that survives crashes and partial output;
7. policy integration that allows Knolo Agents to require specific model/knowledge/receipt properties;
8. Hub distribution of manifests, recipes, conformance evidence, and benchmark evidence rather than a weight CDN.

Do not dilute this differentiation by turning the product into a generic chat UI.

---

# 47. Research and implementation reference register

The agents should review the following primary materials while implementing. These references inform techniques; they are not source trees to rebadge.

## Knolo

- `HiveForensics-AI/knolo-core`
  - README and product boundary
  - V5 host deployment boundary
  - KIP-0003 canonical encoding and digest domains
  - Reflex package design
- `HiveForensics-AI/Knolo-Agents`
  - host-effect boundary
  - no-silent-fallback policy
  - durable events and checkpoints

## Serving and memory

- *Efficient Memory Management for Large Language Model Serving with PagedAttention* — arXiv:2309.06180
- *SGLang: Efficient Execution of Structured Language Model Programs* — arXiv:2312.07104
- TensorRT-LLM official serving documentation:
  - paged KV cache
  - in-flight batching
  - chunked prefill
  - disaggregated serving

## Kernels

- *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* — arXiv:2307.08691
- NVIDIA CUDA Graphs official documentation

## Decoding

- *Fast Inference from Transformers via Speculative Decoding* — arXiv:2211.17192
- *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — arXiv:2401.10774

## Rust and formats

- `huggingface/candle`
- `huggingface/safetensors`
- Hugging Face tokenizers and chat-template documentation
- `EricLBuehler/mistral.rs` as a feasibility and benchmark reference, not as Knolo’s codebase
- `ggml-org/llama.cpp` as the first compatibility backend and benchmark reference

## Target model metadata

- Official Z.AI GLM-4.7-Flash model card and configuration
- Official Z.AI GLM-5.3-Flash model card and configuration

---

# 48. Final directive to the build agents

Build the smallest correct native engine in this order:

```text
contracts
→ micro-model
→ CPU reference
→ receipts
→ CUDA single request
→ paged KV
→ scheduler
→ API
→ quantization
→ Knolo integration
→ MoE
→ GLM adapters
```

Do not begin by forking a model server.

Do not begin with a desktop UI.

Do not begin with GLM-5.3.

Do not optimize before the execution identities and receipts are correct.

Do not let the model artifact, tokenizer, template, placement, sampler, cache, or kernel plan exist outside the receipt chain.

The first release should be narrow, inspectable, and provably Knolo:

> **Pinned knowledge. Pinned model. Pinned execution. Verifiable output.**
