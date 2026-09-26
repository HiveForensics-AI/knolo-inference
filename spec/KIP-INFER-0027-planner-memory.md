# KIP-INFER-0027 — Planner memory estimate

Status: `measure_placement_memory` compares one `PlacementPlanV1` with measured byte counts for weight, KV, workspace, staging, and overhead. It returns a `knolo.infer.memory-estimate` when every measured count is within that plan's declared bound. It does not allocate a KV pool, does not run a model, does not read a GGUF payload, and does not write a weight file. `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`, and `perplexity_delta` do not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The micro placement's `expectedKvBytes` is the eight-page host pool those commands allocate. A measurement above a declared bound issues no report. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Page pool

For an f32 key and value cache the reservation is:

```text
kv_bytes_per_token = 2 × layers × kv_heads × head_dim × 4
pool_bytes = kv_bytes_per_token × block_size × page_count
```

`planned_kv_bytes` is that pool. `PagedKv::new` allocates it and no other size. One page is `layers × block_size × kv_heads × head_dim × 8` bytes. A pool above 64 MiB is `INSUFFICIENT_MEMORY`, and the message says the kv page pool exceeds 64 MiB. A layout that is not a paged f32 pool is `CONTRACT_INVALID`, and the message says the kv layout is not a paged f32 pool.

The micro-model placement reserves eight pages (`CPU_KV_PAGE_POOL`). `expectedKvBytes` is `planned_kv_bytes` of the micro layout and that page count. It equals `PagedKv::resident_bytes` for the same pool. Admitting or releasing a sequence does not change that count. One page is `micro_kv_bytes`. The context reservation stays 16 tokens, and the block size stays 16. Graph capture stays `off`.

The other micro reservations are unchanged: workspace `4096`, safety margin `1024`, staging `0`, and overhead `0`. `expectedTotalBytes` is the sum of weight, KV, workspace, staging, overhead, and the safety margin.

## Report

The report is a versioned contract. Its identity root is `H(infer-memory, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `placementRoot` | root of the `PlacementPlanV1` that declared the bounds |
| `estimatorBuildRoot` | root of the `EngineBuildDescriptorV1` that measured |
| `declaredWeightBytes` | `expectedWeightBytes` |
| `declaredKvBytes` | `expectedKvBytes` |
| `declaredWorkspaceBytes` | `expectedWorkspaceBytes` |
| `declaredStagingBytes` | `expectedStagingBytes` |
| `declaredOverheadBytes` | `expectedOverheadBytes` |
| `declaredMarginBytes` | `safetyMarginBytes` |
| `declaredTotalBytes` | `expectedTotalBytes` |
| `measuredWeightBytes` | measured weight bytes |
| `measuredKvBytes` | measured KV bytes |
| `measuredWorkspaceBytes` | measured workspace bytes |
| `measuredStagingBytes` | measured staging bytes |
| `measuredOverheadBytes` | measured overhead bytes |
| `measuredTotalBytes` | the sum of the five measured counts |
| `headroomBytes` | declared total minus measured total |
| `validationResult` | `within-bounds` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.memory-estimate` and `version` is `1`. The contract count is twenty. `infer-memory` is the report domain.

Any other validation result is `CONTRACT_INVALID`, and the message says the field has an unsupported value. A measured total that is not the sum of the five measured counts is `CONTRACT_INVALID`, and the message says the measured bytes do not match. A declared total that is not the sum of the six declared counts is `CONTRACT_INVALID`, and the message says the declared bytes do not match. A sum that overflows is `CONTRACT_INVALID`, and the message says the measured bytes overflow or the declared bytes overflow. A measured count above its declared count is `CONTRACT_INVALID`, and the message says that measured count exceeds the placement bound. `headroomBytes` must equal the declared total minus the measured total. Any other headroom is `CONTRACT_INVALID`, and the message says the memory headroom does not match. A report is not issued when the measurement fails.

## Measurement

`measure_placement_memory` takes the plan, the five measured counts, and the estimator build root. It returns the report. It does not create a file and it does not allocate the pool. The caller supplies the counts, including a `PagedKv::resident_bytes` taken from a pool the caller already built.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the memory estimate.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. Measured weight bytes exceed `expectedWeightBytes`: `INSUFFICIENT_MEMORY`, and the message says the measured weight bytes exceed the placement bound.
5. Measured KV bytes exceed `expectedKvBytes`: `INSUFFICIENT_MEMORY`, and the message says the measured kv bytes exceed the placement bound.
6. Measured workspace bytes exceed `expectedWorkspaceBytes`: `INSUFFICIENT_MEMORY`, and the message says the measured workspace bytes exceed the placement bound.
7. Measured staging bytes exceed `expectedStagingBytes`: `INSUFFICIENT_MEMORY`, and the message says the measured staging bytes exceed the placement bound.
8. Measured overhead bytes exceed `expectedOverheadBytes`: `INSUFFICIENT_MEMORY`, and the message says the measured overhead bytes exceed the placement bound.
9. The measured sum overflows: `CONTRACT_INVALID`, and the message says the measured bytes overflow.

A count equal to its bound is within bounds. The safety margin is not a measured count. It remains inside the declared total, so the headroom is at least the margin when every measured count equals its bound. An unused reservation increases the headroom.

`verify_memory_estimate` checks the stored report and recomputes the measurement. Checks run in this order:

1. The report fails `validate`, and that failure is returned.
2. The estimator build root differs: `CONTRACT_INVALID`, and the message says the estimator build root does not match.
3. The placement root differs: `CONTRACT_INVALID`, and the message says the placement root does not match.
4. A declared count differs from the plan. The checks are weight, KV, workspace, staging, overhead, margin, then total. The message says that declared count does not match.
5. A measured count differs from the observation. The checks are weight, KV, workspace, staging, overhead, then total. The message says that measured count does not match.
6. Recomputing the measurement fails, and that failure is returned.
7. The recomputed report bytes differ: `CONTRACT_INVALID`, and the message says the memory estimate validation did not match.

`measure_placement_memory` runs that verify before it returns. A verify failure returns no report.

## Files

`write_memory_estimate` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the memory estimate directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the memory estimate path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the memory estimate output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The plan and the measured buffers are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_placement_memory`. Their placement root now includes the eight-page KV reservation. Token ids are unchanged. The throughput report is KIP-INFER-0028. Latency and corruption fuzzing stay later. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter.
