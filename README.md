# Knolo Infer

Knolo Infer is a first-party inference runtime. The native engine loads verified weights, compiles a pinned prompt and placement plan, runs prefill and decode itself, and emits a receipt that binds the model bytes, prompt tokens, engine build, placement, and output.

The architecture baseline is [KNOLO_INFER_NATIVE_ENGINE_FINAL_DESIGN.md](KNOLO_INFER_NATIVE_ENGINE_FINAL_DESIGN.md). This repository implements that design. It does not wrap llama.cpp, Ollama, or another model server.

## What is implemented

Milestone 1 has three layers so far.

The contract layer:

- definite-length canonical CBOR, the same subset as Knolo Core KIP-0003
- domain-separated SHA-256 digests (`sha256-` plus 64 lowercase hex digits)
- the one hundred eighteen versioned Infer objects, with unknown fields rejected
- fixed-point sampler settings (no floats in rooted contracts)
- a Rust crate, `infer-contracts`, and a TypeScript package, `@knolo/infer`, that agree on `conformance/contracts/vectors.json`
- a cross-check that the shared CBOR subset matches `@knolo/core`

The model-image layer:

- JSON or a bounded YAML subset compiles to a canonical `.kmodel`
- tokenizer and template bytes are embedded; weight bytes stay in external files
- safetensors headers are checked against the tensor inventory after the file hash matches
- `knolo.infer.lock.json` pins an alias to the model-image root, artifact root, and relative path
- `knolo-infer pull` is refused until download staging exists

The CPU micro-model:

- `models/micro-transformer/` is a checked-in f32 model with a 16-token vocabulary and two tiny layers
- `infer-engine` runs it with plain f32 loops and one KV block per sequence
- `infer-native` runs the same architecture through Candle on CPU, and with `--features cuda` on GPU `slot-0`
- greedy token ids must match exactly, and logits must agree within absolute tolerance `1e-4`
- an unknown architecture fails before weight files are opened

Prompt, sampler, and receipts:

- `infer-prompt` renders one allowlisted message loop and encodes `knolo.micro.tokens.v1`
- temperature 0 is lowest-id argmax; a non-zero temperature uses Philox-4x32-10
- `knolo-infer run` writes an append-only journal before the forward pass and a receipt of roots
- `knolo-infer receipt verify` checks that receipt; `knolo-infer replay` is what sets `exact_replay_verified`
- Rust and TypeScript must produce the same token-id root for `conformance/prompt/expected.json`

KV pages on the run path:

- `knolo-infer run` allocates from a pool of eight pages of 16 tokens
- a page joins the block table only when its token commits, and a released page is zeroed
- an active sequence is not evicted; prefix cache is not allocated
- the checked-in micro-model still fits in one page, and its greedy tokens match the single-block oracle

CPU scheduler, in-process:

- admission reserves pages for the prompt plus the token budget, and refuses a request that does not fit
- one iteration is a prefill chunk or one token for each selected decode sequence
- interactive, standard, batch, and background classes share the pool by virtual time
- cancelling a request releases its pages before the next forward
- tokens from a shared run match the same request run alone
- `knolo-infer run` remains one isolated sequence

CPU supervisor:

- `knolo-infer serve` listens on `127.0.0.1` and compiles the prompt
- `knolo-infer-worker` loads one pinned model, owns the page pool, and does not listen
- completions use `POST /knolo/infer/v1/complete`; cancellation and health are on the same prefix
- a worker exit returns `WORKER_LOST` for the request in flight and the listener stays up
- a shared completion matches the same request run alone
- the response has token ids and the runtime root, and it does not include a receipt

Native stream and OpenAI chat subset:

- `stream: true` on `POST /knolo/infer/v1/complete` sends `knolo.accepted`, one `knolo.delta` per output token, and `knolo.usage`
- `GET /v1/models` and `POST /v1/chat/completions` accept the served alias, fixed-point `temperature` and `top_p`, and `stream`
- unknown fields are refused, and a non-zero temperature still requires a seed
- `X-Knolo-Receipt` was `absent` until the serve-path journal below

Serve-path receipt journal:

- `knolo-infer serve` fsyncs `accepted` before it sends the request to the worker
- a `stop` or `length` completion stores a receipt of roots and returns its id in the body
- a native stream ends that completion with `knolo.receipt`; the stream header says `pending` until then
- assurance on this path is `compatibility`, because the worker does not rerun the sequence
- without the `cuda` feature the worker stays the reference oracle, so the receipt names `reference-f32` and the worker does not link Candle
- with `--features cuda` the worker places the model on `slot-0`, the receipt names that device, and the kernel bundle is the CUDA run bundle
- `GET /knolo/infer/v1/receipts/sha256-<hex>` reads that receipt back; a cancellation does not store one

Prometheus metrics:

- `GET /metrics` and `GET /knolo/infer/v1/metrics` return the same Prometheus text
- labels are service class, outcome, error code, page state, and OOM kind
- a request id, prompt, or model alias is not a label
- queue depth, KV pages, admission rejections, token totals, and request histograms come from the worker snapshot
- prefix-cache series stay at 0 because prefix cache is not allocated
- a worker exit leaves the listener up and counts one restart; live gauges drop to 0 until the next worker

Redacted tracing:

- each completion appends JSON lines under `traces/by-id/<request-id>.jsonl`
- the stages are api, prompt, admission, prefill, decode, and finalize
- a line may carry the request id, the prompt-plan root, and the receipt root
- the prompt, the output, and the token ids are not fields
- a failure before that file is opened appends one line to `traces/rejected.jsonl`
- the trace is not part of the receipt, and a trace write does not change token ids

Drain:

- `POST /knolo/infer/v1/drain` with `{}` stops new native and OpenAI completions
- a new completion is `SERVICE_DRAINING` and the body is not parsed
- a completion already admitted finishes with the same token ids and can still be cancelled
- the listener stays up, the worker stays loaded, and the call does not count a restart
- SIGINT and SIGTERM drain, wait up to 30 seconds, and then shut down

Unload:

- `POST /knolo/infer/v1/models/unload` with `{}` stops new completions, then drops the loaded model
- a new completion is `SERVICE_UNLOADED` and the body is not parsed
- a completion already admitted finishes with the same token ids and can still be cancelled
- when that work is done the worker zeros its KV pages and exits
- the listener stays up, a later completion does not start a worker, and the call does not count a restart

Restart:

- `POST /knolo/infer/v1/models/load` with `{}` loads that same pin again
- a bad body does not change the lifecycle, and a load during unload does not cancel admitted work
- after the worker has exited, load starts a fresh page pool and the same completion returns the same token ids
- load does not clear drain, does not replace a ready worker, and does not count a restart

Bounded-memory soak:

- the worker allocates its KV page pool once and does not add pages for later requests
- after a completion is written, the scheduler drops that request's prompt, tokens, and logits
- thirty-two repeats of the same prompt return the same token ids, and the page total stays at the pool size
- the worker's resident set after those repeats stays within 8 MiB of the sample taken once the model is loaded
- a second completion with the same request id is still refused, because that id already has a journal

Crash recovery:

- `knolo-infer serve` takes `{home}/daemon.lock` before it listens
- a second process fails with `CONTRACT_INVALID` while that owner is alive, and the first process keeps serving
- a lock whose pid is dead, or whose start time no longer matches, is replaced
- a journal left after `accepted` is sealed `failed` with the `WORKER_LOST` trace root, and that request id stays reserved
- a new request id returns the same token ids, and the new process does not count a worker restart for the recovery

GGUF:

- a file is parsed only after its size and SHA-256 match, and only up to 32 MiB
- the tensor allowlist is `F32`, `F16`, `Q8_0`, `Q4_K`, `Q5_K`, and `Q6_K`; other ggml types are `UNSUPPORTED_QUANTIZATION`
- `F16`, `Q8_0`, `Q4_K`, `Q5_K`, and `Q6_K` dequant to finite `f32` on the CPU, and that buffer is not written back as a new weight file
- `quant_gemm` multiplies those payloads, plus `F32`, by `f32` activations on the CPU and matches dequant-then-`gemv` when the expanded matrix fits in 32 MiB
- an explicit conversion writes a new little-endian `f32` artifact and a `knolo.infer.conversion-receipt`. It does not replace the source file. Dequant and `quant_gemm` do not call it
- `perplexity_delta` records the perplexity and accuracy delta of two `f32` logit matrices on a `knolo.infer.perplexity-report`. It does not run a model. Dequant, `quant_gemm`, and conversion do not call it
- `measure_placement_memory` records measured bytes that stay within a placement plan on a `knolo.infer.memory-estimate`. The micro plan's KV bytes are the eight-page pool. Run and serve do not call it
- `measure_micro_throughput` records prefill, decode, and request throughput for one cold run of the micro fixture on a `knolo.infer.throughput-report`. It does not run a model. The `throughput` execution mode stays refused. Run and serve do not call it
- `measure_micro_latency` records time to first token, time per output token, and the p50, p95, and p99 of that one request on a `knolo.infer.latency-report`. It does not run a model. Run and serve do not call it
- `measure_corruption_fuzz` mutates one seed for each of nine parsers and records a `knolo.infer.fuzz-report` when every mutation fails closed. It does not run a model. Run and serve do not call it
- `measure_cancellation_latency` records the time from a cancel request until that request is terminal on a `knolo.infer.cancellation-report`. It does not cancel a live request. Run and serve do not call it
- `measure_receipt_finalization` records the time from a terminal completion until the receipt is durable on a `knolo.infer.finalization-report`. A cancelled request stores no receipt. Run and serve do not call it
- `measure_receipt_overhead` records the accepted-event write plus the receipt-file write on a `knolo.infer.overhead-report`. It does not include the idle gap after the terminal event. Run and serve do not call it
- `measure_model_swap` records unload plus incoming verification plus incoming load on a `knolo.infer.swap-report`. The resident image is a different root and is not opened. Run and serve do not call it
- `measure_model_verification` records the byte count and the duration of one cold micro-image check on a `knolo.infer.verification-report`. A count above 32 MiB is `INSUFFICIENT_MEMORY`. Run and serve do not call it
- `measure_model_load` records the load after that check on a `knolo.infer.load-report`. Verification time stays on the verification report. Run and serve do not call it
- `measure_peak_memory` records peak host RAM and peak device VRAM for one cold micro run on a `knolo.infer.peak-report`. A `cpu` placement records zero VRAM. A count above 64 MiB is `INSUFFICIENT_MEMORY`. Run and serve do not call it
- `measure_kv_utilization` records occupied pages and tokens for the eight-page pool on a `knolo.infer.kv-report`. The micro fixture occupies one page. Run and serve do not call it
- `measure_prefix_reuse` records lookups, hits, misses, and reused tokens on a `knolo.infer.prefix-report`. The cache stays off, so every count is zero. Run and serve do not call it
- `measure_llama_comparison` records a pinned llama.cpp build on the same micro-fixture artifact on a `knolo.infer.llama-report`. The reference family is `gguf` and the verification class is `sidecar-artifact-verified`. It does not spawn llama.cpp. Run and serve do not call it
- `measure_vllm_comparison` records a pinned vLLM build on that same artifact on a `knolo.infer.vllm-report`. The reference family is `safetensors`. It does not spawn vLLM. Run and serve do not call it
- `measure_mistral_comparison` records a pinned mistral.rs build on that same artifact on a `knolo.infer.mistral-report`. The verification class is `backend-reported`. It does not spawn mistral.rs. Run and serve do not call it
- `measure_recipe_status` records `experimental`, `conformant`, or `blessed` on a `knolo.infer.recipe-report`. A blessed mark requires greedy token parity and four distinct receipt roots. Run and serve do not call it
- `measure_evidence_composition` records a host-supplied Knowledge Image, commit, query receipts, and Reflex receipts on a `knolo.infer.composition-report`. It does not open a `.knolo` image. Run and serve do not call it
- `measure_agent_effect` records allow or deny for one final receipt on a `knolo.infer.agent-effect-report`. A missing receipt and an unverified compatibility backend are denials. It does not call Agents or execute a tool. Run and serve do not call it
- `measure_hub_record` records `.kmodel` metadata, a license id, and native support on a `knolo.infer.hub-report`. Weight bytes stay off the record, and the source provider is `local`. Run and serve do not call it
- `measure_receipt_view` records receipt roots and counts on a `knolo.infer.studio-report`. Prompt text and output text are not fields. Run and serve do not call it
- `measure_receipt_chain` records the Knowledge Image, query receipt, Reflex receipt, inference receipt, and agent effect on a `knolo.infer.chain-report`. It does not open a `.knolo` image. Run and serve do not call it
- `measure_evidence_output` records that the output roots match that chain on a `knolo.infer.evidence-output-report`. A mismatched knowledge image issues no report. Run and serve do not call it
- `measure_hub_install` records that the `.kmodel`, weights, tokenizer, and template are present on a `knolo.infer.install-report`. A missing artifact is `MODEL_ARTIFACT_MISSING`. The source provider is `local`. Run and serve do not call it
- `measure_supply_notice` records NOTICE and SBOM roots on a `knolo.infer.notice-report`. A `cpu` inventory does not name Candle. It does not write an SBOM. Run and serve do not call it
- `measure_release_manifest` records the notice, SBOM, and binary-set roots on a `knolo.infer.release-report`. An unsigned release carries no signature. It does not verify an Ed25519 key. Run and serve do not call it
- `measure_binary_inventory` records the `knolo-infer` and `knolo-infer-worker` hashes on a `knolo.infer.binary-report`. A byte count above 64 MiB is `INSUFFICIENT_MEMORY`. It does not open either binary. Run and serve do not call it
- `measure_reproducible_build` records the lock root and the instruction source on a `knolo.infer.reproducible-report`. It does not run Cargo. Run and serve do not call it
- `measure_signature_gate` records an unsigned local release or one 64-byte Ed25519 block on a `knolo.infer.signature-report`. `keyVerified` stays false. Run and serve do not call it
- `measure_host_key` records a host-supplied match on a `knolo.infer.host-key-report`. `keyVerified` is true only for `matched`. Key bytes are not a field. Run and serve do not call it
- `measure_receipt_key` records host-store custody on a `knolo.infer.receipt-key-report`. A serialized key is rejected. Run and serve do not call it
- `measure_sandbox_profile` records an unprivileged worker, no external network, and deferred seccomp on a `knolo.infer.sandbox-report`. It does not apply the profile. Run and serve do not call it
- `measure_api_boundary` records localhost or an explicit remote bind, with authentication left as a hook, on a `knolo.infer.api-report`. It does not bind a socket. Run and serve do not call it
- `measure_cache_channel` records an isolated prefix policy on a `knolo.infer.cache-channel-report`. Another tenant's prefix stays hidden, and the index stays unallocated. Run and serve do not call it
- `measure_signature_equation` records a host-supplied Ed25519 equation status on a `knolo.infer.equation-report`. Key bytes are not a field. Run and serve do not call it
- `measure_safe_error` records one stable error code on a `knolo.infer.safe-error-report`. The message is that code, and the prompt stays off the report. Run and serve do not call it
- `measure_redacted_log` records one redacted completion line on a `knolo.infer.redaction-report`. Prompt text, output text, and paths stay omitted. Run and serve do not call it
- `measure_curve` records a host-supplied Ed25519 curve result on a `knolo.infer.curve-report`. An on-curve point is `verified`. The base point stays unmultiplied. Run and serve do not call it
- `measure_rollback` records the previous pin and the incoming pin on a `knolo.infer.rollback-report`. The lockfile stays unchanged. Run and serve do not call it
- `measure_disconnect` records a client disconnect on a `knolo.infer.disconnect-report`. The listener stays up. Run and serve do not call it
- `measure_receipt_store` records `RECEIPT_PERSIST_FAILED` on a `knolo.infer.receipt-store-report`. The receipt file stays unwritten. Run and serve do not call it
- `measure_base` records a host-supplied base-point multiplication on a `knolo.infer.base-report`. A multiplied point is `verified`. The public-key point stays unadded. Run and serve do not call it
- `measure_disk` records a full `journal`, `receipt`, `trace`, or `lock` store on a `knolo.infer.disk-report`. The target file stays unwritten. Run and serve do not call it
- `measure_queued_unload` records `SERVICE_UNLOADED` on a `knolo.infer.unload-report`. Queued work does not start. Run and serve do not call it
- `measure_daemon_restart` records a new owner on a `knolo.infer.restart-report`. A stale owner is replaced. Run and serve do not call it
- `measure_point` records a host-supplied point addition on a `knolo.infer.point-report`. An added point is `verified`. The points stay uncompared. Run and serve do not call it
- `measure_duplicate` records `CONTRACT_INVALID` on a `knolo.infer.duplicate-report`. The second copy does not start. Run and serve do not call it
- `measure_concurrent_load` records one worker on a `knolo.infer.concurrent-load-report`. A live worker is not replaced. Run and serve do not call it
- `measure_prefix_eviction` records zero evicted pages on a `knolo.infer.eviction-report`. The prefix index stays unallocated. Run and serve do not call it
- `measure_point_equality` records a host-supplied point comparison on a `knolo.infer.equality-report`. An equal comparison is `verified`. The challenge hash stays uncomputed. Run and serve do not call it
- `measure_cuda_oom` records `CUDA_OOM` on a `knolo.infer.oom-report`. The device is `slot-0`, and there is no fallback to CPU. Run and serve do not call it
- `measure_cuda_fault` records `CUDA_FAULT` on a `knolo.infer.fault-report`. The fault is not retryable, and CUDA graphs stay off. Run and serve do not call it
- `measure_timeout` records `REQUEST_TIMEOUT` on a `knolo.infer.timeout-report`. The listener stays up. Run and serve do not call it
- `measure_worker_start` records `WORKER_START_FAILED` on a `knolo.infer.worker-start-report`. A missing binary does not bind. Run and serve do not call it
- `measure_challenge` records a host-supplied challenge hash on a `knolo.infer.challenge-report`. A hashed challenge is `verified`. The scalar stays unreduced. Run and serve do not call it
- `measure_replay_environment` records `REPLAY_ENVIRONMENT_MISMATCH` on a `knolo.infer.replay-environment-report`. The forward does not run. Run and serve do not call it
- `measure_replay_output` records `REPLAY_OUTPUT_MISMATCH` on a `knolo.infer.replay-output-report`. The candidate output differs from the receipt. Run and serve do not call it
- `measure_worker_lost` records `WORKER_LOST` on a `knolo.infer.worker-lost-report`. The listener stays up and the exit counts a restart. Run and serve do not call it
- `measure_draining` records `SERVICE_DRAINING` on a `knolo.infer.draining-report`. The body is not parsed and the worker stays loaded. Run and serve do not call it
- `measure_scalar` records a host-supplied scalar reduction on a `knolo.infer.scalar-report`. A reduced scalar is `verified`. The public key stays unmultiplied. Run and serve do not call it
- `measure_digest_mismatch` records `MODEL_DIGEST_MISMATCH` on a `knolo.infer.digest-mismatch-report`. The header is not parsed. Run and serve do not call it
- `measure_tokenizer_invalid` records `TOKENIZER_INVALID` on a `knolo.infer.tokenizer-invalid-report`. The prompt is not compiled. Run and serve do not call it
- `measure_template_invalid` records `TEMPLATE_INVALID` on a `knolo.infer.template-invalid-report`. The template is not rendered. Run and serve do not call it
- `measure_architecture` records `UNSUPPORTED_ARCHITECTURE` on a `knolo.infer.architecture-report`. Weights are not opened. Run and serve do not call it
- `measure_public` records a host-supplied public-key multiplication on a `knolo.infer.public-report`. A multiplied public key is `verified`. The signature stays unchecked. Run and serve do not call it
- `measure_quantization` records `UNSUPPORTED_QUANTIZATION` on a `knolo.infer.quantization-report`. A precision refusal does not open weights. Run and serve do not call it
- `measure_kernel` records `UNSUPPORTED_KERNEL` on a `knolo.infer.kernel-report`. The kernel is not selected. Run and serve do not call it
- `measure_placement_refusal` records `PLACEMENT_UNSATISFIABLE` on a `knolo.infer.placement-refusal-report`. The device is not opened. Run and serve do not call it
- `measure_memory_refusal` records `INSUFFICIENT_MEMORY` on a `knolo.infer.memory-refusal-report`. No page is allocated. Run and serve do not call it
- `measure_signature_check` records a host-supplied signature check on a `knolo.infer.signature-check-report`. A checked signature is `verified`. The cofactor stays uncleared. Run and serve do not call it
- `measure_context_limit` records `CONTEXT_LIMIT_EXCEEDED` on a `knolo.infer.context-limit-report`. Tokens are not truncated. Run and serve do not call it
- `measure_prompt_compilation` records `PROMPT_COMPILATION_FAILED` on a `knolo.infer.prompt-compilation-report`. The prompt is not compiled. Run and serve do not call it
- `measure_image_invalid` records `MODEL_IMAGE_INVALID` on a `knolo.infer.image-invalid-report`. An empty image is not parsed. Run and serve do not call it
- `measure_image_signature` records `MODEL_IMAGE_SIGNATURE_INVALID` on a `knolo.infer.image-signature-report`. Key bytes are not a field. Run and serve do not call it
- `measure_cofactor` records a host-supplied cofactor clear on a `knolo.infer.cofactor-report`. A cleared cofactor is `verified`. The receipt stays unsigned. Run and serve do not call it
- `measure_artifact_missing` records `MODEL_ARTIFACT_MISSING` on a `knolo.infer.artifact-missing-report`. Pull does not download. Run and serve do not call it
- `measure_receipt_required` records `RECEIPT_REQUIRED` on a `knolo.infer.receipt-required-report`. A missing file is HTTP 404. Run and serve do not call it
- `measure_backend` records `BACKEND_NOT_ALLOWED` on a `knolo.infer.backend-report`. The refused mode is throughput. Run and serve do not call it
- `measure_digest_invalid` records `DIGEST_INVALID` on a `knolo.infer.digest-invalid-report`. The payload is not hashed. Run and serve do not call it
- `measure_receipt_sign` records a host-supplied receipt signature on a `knolo.infer.receipt-sign-report`. A signed receipt is `verified`. The receipt stays unverified. Run and serve do not call it
- `measure_canonical_cbor` records `CANONICAL_CBOR_INVALID` on a `knolo.infer.canonical-cbor-report`. The document is not decoded. Run and serve do not call it
- `measure_contract_invalid` records `CONTRACT_INVALID` on a `knolo.infer.contract-invalid-report`. The field name is not a field. Run and serve do not call it
- `measure_grammar_refusal` records a grammar the compiler did not build on a `knolo.infer.grammar-refusal-report`. The grammar is not compiled. Run and serve do not call it
- `measure_tool_refusal` records a tool call the engine did not execute on a `knolo.infer.tool-refusal-report`. Authority stays unchecked. Run and serve do not call it
- `measure_receipt_verify` records a host-supplied receipt verification on a `knolo.infer.receipt-verify-report`. A verified receipt is `verified`. The domain stays unseparated. Run and serve do not call it
- `measure_speculative` records a speculative request the engine did not run on a `knolo.infer.speculative-report`. Accepted tokens stay zero. Run and serve do not call it
- `measure_cuda_graph` records an uncaptured CUDA graph on a `knolo.infer.graph-report`. The device is `slot-0`. Run and serve do not call it
- `measure_multi_model` records a second model the worker did not load on a `knolo.infer.multi-model-report`. One resident model stays. Run and serve do not call it
- `measure_secondary` records a secondary service the worker did not start on a `knolo.infer.secondary-report`. The slot is `slot-1`. Run and serve do not call it
- `measure_domain` records a host-supplied domain prefix on a `knolo.infer.domain-report`. A separated domain is `verified`. The payload stays unhashed. Run and serve do not call it
- `measure_attention` records an attention modification that was not applied on a `knolo.infer.attention-report`. Attention stays exact. Run and serve do not call it
- `measure_moe` records a mixture-of-experts request the router did not run on a `knolo.infer.moe-report`. Selected experts stay zero. Run and serve do not call it
- `measure_expert_placement` records an expert the planner did not place on a `knolo.infer.expert-placement-report`. Placed experts stay zero. Run and serve do not call it
- `measure_grouped_kernel` records a grouped kernel that was not selected on a `knolo.infer.grouped-kernel-report`. The code is `UNSUPPORTED_KERNEL`. Run and serve do not call it
- `measure_router` records router outputs that were not compared on a `knolo.infer.router-report`. Parity stays false. Run and serve do not call it
- `measure_glm` records `knolo.glm.v1` on a `knolo.infer.glm-report`. The micro adapter issues no report. Run and serve do not call it
- `measure_workstation` records a workstation recipe that is not blessed on a `knolo.infer.workstation-report`. The benchmark does not run. Run and serve do not call it
- `measure_mixed_placement` records a mixed placement that was not selected on a `knolo.infer.mixed-placement-report`. Automatic fallback stays off. Run and serve do not call it
- `measure_capacity` records expert capacity that was not applied on a `knolo.infer.capacity-report`. Capacity tokens stay zero. Run and serve do not call it
- a manifest with `format: gguf` is still `MODEL_IMAGE_INVALID`

This machine can prove the CPU path. `cargo test --workspace` does not enable CUDA. `knolo-infer run` and `knolo-infer serve` then place the model on `cpu`. With `--features cuda`, both place it on `slot-0` and the receipt names that device. The specs are `spec/KIP-INFER-0022-cuda-run.md` and `spec/KIP-INFER-0023-cuda-serve.md`.

## Develop

```bash
cargo test
npm install
npm test
```

`cargo test -p infer-engine` rewrites `models/micro-transformer/` and `conformance/micro-model/expected.json`. `cargo test -p infer-native` checks Candle CPU against the live oracle. Neither test uses the network.

Build and check a model image:

```bash
cargo run -p infer-cli -- model build conformance/model-image/manifest.json --out /tmp/micro.kmodel
cargo run -p infer-cli -- model verify /tmp/micro.kmodel --weights conformance/model-image
cargo run -p infer-cli -- pin micro conformance/model-image/micro.kmodel --weights conformance/model-image
```

Rust 1.85 or newer and Node.js 22 are enough for the default build. CUDA is not required for `cargo test --workspace`. A CUDA run needs `nvcc` on `PATH`. `CUDA_ROOT` and `CUDA_HOME` are the toolkit prefix. `LD_LIBRARY_PATH` includes that prefix's `lib` directory, which is where `libcublas` is loaded. `cudarc` reads `CUDA_ROOT`.

```bash
cargo test -p infer-native --features cuda --offline
cargo test -p infer-receipt --features cuda --lib cuda_run --offline
cargo test -p infer-cli --features cuda --test cli cli_runs_verifies --offline
cargo test -p infer-serve --features cuda --offline
```

`cargo test` rewrites `conformance/contracts/vectors.json` from the Rust fixtures, and rewrites `conformance/model-image/micro.kmodel`, `weights.safetensors`, and `expected.json` from the checked-in manifests. `npm test` checks that `@knolo/infer` accepts the contract bytes, reproduces the digests, and rejects the malformed vectors.

## Layout

```text
spec/                         KIP-INFER contracts, including serve metrics, traces, drain, unload, restart, and the memory soak
crates/infer-contracts/       Rust canonical contracts
crates/infer-artifact/        .kmodel compiler, safetensors inventory, lockfile
crates/infer-prompt/          bounded template and micro tokenizer
crates/infer-engine/          f32 oracle, micro-transformer, sampler, paged KV, scheduler
crates/infer-native/          Candle CPU micro-transformer
crates/infer-receipt/         journal, engine identity, receipt verification
crates/infer-serve/           supervisor, worker process, loopback HTTP
crates/infer-cli/             knolo-infer binary
packages/infer/               @knolo/infer verifier
conformance/contracts/        shared golden vectors
conformance/model-image/      authoring fixture and generated .kmodel
conformance/micro-model/      oracle logits and greedy tokens
conformance/prompt/           shared prompt render and token-id roots
models/micro-transformer/     synthetic weights and .kmodel
docs/SECURITY_MODEL.md        parser and trust constraints
docs/CPU_REFERENCE.md         CPU micro-model limits
docs/MILESTONE_1_CONFORMANCE.md  Milestone 1 gate review
```

License: Apache-2.0. See [NOTICE](NOTICE).
