# KIP-INFER-0038 — KV utilization

Status: `measure_kv_utilization` records the peak pages and peak tokens occupied by one cold `knolo.micro.v1` run on the eight-page pool. It returns a `knolo.infer.kv-report`. It does not allocate a KV pool, does not open a model file, does not run a model, does not write a serve journal, does not read a GGUF payload, and does not write a weight file. `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`, `measure_micro_throughput`, `measure_micro_latency`, `measure_corruption_fuzz`, `measure_cancellation_latency`, `measure_receipt_finalization`, `measure_receipt_overhead`, `measure_model_swap`, `measure_model_verification`, `measure_model_load`, and `measure_peak_memory` do not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Occupancy

The pool is 8 pages of 16 tokens, 128 token slots. The micro fixture fits in one page, so the peak page count is 1. The peak token count is from 1 through 16. Utilization is `peakTokens * 1000000 / 128`, integer division, remainder discarded. There is no floating-point value in the report. The idle free count after reap is not a field.

Sixteen peak tokens record utilization `125000`. One peak token records utilization `7812`. Four peak tokens record utilization `31250`. A peak of zero tokens issues no report. A peak above 16 tokens is `CONTEXT_LIMIT_EXCEEDED` and issues no report. A stored report above 16 tokens is `CONTRACT_INVALID`.

## Report

The report is a versioned contract. Its identity root is `H(infer-kv, document)`. Version 1 accepts only these fields:

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
| `pageTotal` | `8` |
| `pageSizeTokens` | `16` |
| `peakPages` | `1` |
| `peakTokens` | tokens occupied at the peak |
| `utilizationMillionths` | token slots occupied, in millionths of the pool |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.kv-report` and `version` is `1`. The contract count is thirty-one. `infer-kv` is the report domain.

Any other validation result is `CONTRACT_INVALID`, and the message says the field has an unsupported value. A report is not issued when the measurement fails.

## Measurement

`measure_kv_utilization` takes the placement plan and one observation. The observation carries the model image root, the artifact root, the engine build root, the execution mode, the cache policy, the concurrency, the run count, the warm state, the request count, the peak page count, and the peak token count. Page total and page size are the micro pool. It returns the report. It does not create a file and it does not call the model. The caller supplies the occupancy from a completed cold run of the micro fixture.

The plan is the micro fixture plan: context reservation 16, KV block size 16, and KV precision `f32`. Graph capture stays `off`. Device `cpu` and device `slot-0` both record a report. The host page pool is the same on both. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the kv report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the kv report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the kv report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says kv concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says kv run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says kv warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says kv request count is one.
12. The peak token count is zero: `CONTRACT_INVALID`, and the message says kv peak tokens are zero.
13. The peak token count is above 16: `CONTEXT_LIMIT_EXCEEDED`, and the message says kv peak tokens exceed the micro context.
14. The peak page count is not 1: `CONTRACT_INVALID`, and the message says the micro fixture occupies one page.

`verify_kv_utilization` checks the stored report and recomputes the measurement. Checks run in this order:

1. The report fails `validate`, and that failure is returned. A page total other than 8, or a page size other than 16, says the kv report is the micro fixture. A utilization that is not the formula says kv utilization does not match.
2. The model image root, artifact root, or engine build root differs. The message says that root does not match.
3. The placement root differs: `CONTRACT_INVALID`, and the message says the placement root does not match.
4. The execution mode, cache policy, concurrency, run count, warm state, or request count differs. The message says that field does not match.
5. The peak pages or peak tokens differ. The message says that count does not match.
6. Recomputing the measurement fails, and that failure is returned.
7. The recomputed report bytes differ: `CONTRACT_INVALID`, and the message says the kv validation did not match.

`measure_kv_utilization` runs that verify before it returns. A verify failure returns no report.

## Files

`write_kv_report` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the kv directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the kv path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the kv output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The plan and the page pool are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_kv_utilization`. Token ids are unchanged. Prefix reuse stays later. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter. The dense Llama-family adapter stays deferred.
