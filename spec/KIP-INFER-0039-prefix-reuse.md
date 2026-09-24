# KIP-INFER-0039 — Prefix reuse

Status: `measure_prefix_reuse` records lookups, hits, misses, and reused tokens for one cold `knolo.micro.v1` run. It returns a `knolo.infer.prefix-report`. It does not allocate a prefix index, does not open a model file, does not run a model, does not write a serve journal, does not read a GGUF payload, does not allocate a KV pool, and does not write a weight file. `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`, `measure_micro_throughput`, `measure_micro_latency`, `measure_corruption_fuzz`, `measure_cancellation_latency`, `measure_receipt_finalization`, `measure_receipt_overhead`, `measure_model_swap`, `measure_model_verification`, `measure_model_load`, `measure_peak_memory`, and `measure_kv_utilization` do not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Counts

Cache policy is `off`. Contract version 1 rejects an enabled prefix cache, so this report records that no lookup ran. Lookups, hits, misses, and reused tokens are each zero. A non-zero count issues no report. There is no floating-point value in the report. Queue time is not a field. A hit is not inferred from text.

One cold run records lookup count `0`, hit count `0`, miss count `0`, and reused tokens `0`.

## Report

The report is a versioned contract. Its identity root is `H(infer-prefix, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `modelImageRoot` | root of the micro model image |
| `artifactRoot` | root of the weight artifact |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` that ran |
| `placementRoot` | root of the `PlacementPlanV1` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `lookupCount` | `0` |
| `hitCount` | `0` |
| `missCount` | `0` |
| `reusedTokens` | `0` |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.prefix-report` and `version` is `1`. The contract count is thirty-two. `infer-prefix` is the report domain.

Any other validation result is `CONTRACT_INVALID`, and the message says the field has an unsupported value. A report is not issued when the measurement fails.

## Measurement

`measure_prefix_reuse` takes the placement plan and one observation. The observation carries the model image root, the artifact root, the engine build root, the execution mode, the cache policy, the concurrency, the run count, the warm state, the request count, and the four counts. It returns the report. It does not create a file and it does not call the model. The caller supplies the counts from a completed cold run of the micro fixture.

The plan is the micro fixture plan: context reservation 16, KV block size 16, and KV precision `f32`. Graph capture stays `off`. Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the prefix report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the prefix report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the prefix report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says prefix concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says prefix run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says prefix warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says prefix request count is one.
12. The lookup count is not zero: `CONTRACT_INVALID`, and the message says prefix lookups are zero while the cache is off.
13. The hit count is not zero: `CONTRACT_INVALID`, and the message says prefix hits are zero while the cache is off.
14. The miss count is not zero: `CONTRACT_INVALID`, and the message says prefix misses are zero while the cache is off.
15. The reused token count is not zero: `CONTRACT_INVALID`, and the message says reused tokens are zero while the cache is off.

`verify_prefix_reuse` checks the stored report and recomputes the measurement. Checks run in this order:

1. The report fails `validate`, and that failure is returned.
2. The model image root, artifact root, or engine build root differs. The message says that root does not match.
3. The placement root differs: `CONTRACT_INVALID`, and the message says the placement root does not match.
4. The execution mode, cache policy, concurrency, run count, warm state, or request count differs. The message says that field does not match.
5. A count differs. The message says that count does not match.
6. Recomputing the measurement fails, and that failure is returned.
7. The recomputed report bytes differ: `CONTRACT_INVALID`, and the message says the prefix validation did not match.

`measure_prefix_reuse` runs that verify before it returns. A verify failure returns no report.

## Files

`write_prefix_report` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the prefix directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the prefix path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the prefix output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The plan and the token ids are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_prefix_reuse`. Token ids are unchanged. The prefix index is not allocated. Reference comparisons are KIP-INFER-0040, KIP-INFER-0041, and KIP-INFER-0042. The recipe mark is KIP-INFER-0043. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter. The dense Llama-family adapter stays deferred.
