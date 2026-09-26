# KIP-INFER-0031 — Cancellation latency

Status: `measure_cancellation_latency` records the time from a cancel request until that request is terminal for one cold run of the `knolo.micro.v1` fixture. It returns a `knolo.infer.cancellation-report`. It does not run a model, does not cancel a live request, does not read a GGUF payload, does not allocate a KV pool, and does not write a weight file. `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`, `measure_micro_throughput`, `measure_micro_latency`, and `measure_corruption_fuzz` do not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Duration

A latency figure is an integer number of nanoseconds. There is no floating-point value in the report. Queue time is not a field.

Both times are measured from accept. The cancel request time is greater than zero. The terminal time is greater than the request time. Cancellation latency is the terminal time minus the request time.

A prefill cancel has no output tokens. A decode cancel has at least one output token, and those ids are the tokens produced before the cancel became terminal. One prefill cancel requested at 1_000 nanoseconds and terminal at 1_400 nanoseconds has cancellation latency `400`. One decode cancel requested at 2_000 nanoseconds and terminal at 2_750 nanoseconds has cancellation latency `750`.

## Report

The report is a versioned contract. Its identity root is `H(infer-cancellation, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `modelImageRoot` | root of the micro model image |
| `artifactRoot` | root of the weight artifact |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` that ran |
| `placementRoot` | root of the `PlacementPlanV1` |
| `promptTokenRoot` | `H(infer-prompt-tokens, prompt ids)` |
| `outputTokenRoot` | `H(infer-output-tokens, output ids)` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `cancelStage` | `prefill` or `decode` |
| `prefillTokens` | prompt length |
| `decodeTokens` | output length, zero for prefill |
| `requestedNanos` | cancel request time |
| `terminalNanos` | terminal time |
| `cancellationLatencyNanos` | terminal minus request |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.cancellation-report` and `version` is `1`. The contract count is twenty-four. `infer-cancellation` is the report domain.

Any other validation result is `CONTRACT_INVALID`, and the message says the field has an unsupported value. A stored latency that is not the difference above is `CONTRACT_INVALID`, and the message says cancellation latency does not match. A prefill plus decode token count above 16 is `CONTRACT_INVALID`, and the message says the micro prompt does not fit in the reserved context. A report is not issued when the measurement fails.

## Measurement

`measure_cancellation_latency` takes the placement plan and one observation. The observation carries the model image root, the artifact root, the engine build root, the execution mode, the cache policy, the concurrency, the run count, the warm state, the request count, the cancel stage, the prompt ids, the output ids, and the two times. It returns the report. It does not create a file and it does not call the model. The caller supplies ids and times from a completed micro-fixture cancel.

The plan is the micro fixture plan: context reservation 16, KV block size 16, and KV precision `f32`. Graph capture stays `off`. Device `cpu` and device `slot-0` both record a report. The token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the cancellation report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the cancellation report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the cancellation report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says cancellation concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says cancellation run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says cancellation warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says cancellation request count is one.
12. `cancelStage` is neither `prefill` nor `decode`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
13. The prefill token count is zero: `CONTRACT_INVALID`, and the message says the prefill token count is zero.
14. The stage is `prefill` and the decode token count is not zero: `CONTRACT_INVALID`, and the message says the prefill cancel has output tokens.
15. The stage is `decode` and the decode token count is zero: `CONTRACT_INVALID`, and the message says the decode cancel has no output tokens.
16. The prompt length differs from the prefill token count: `CONTRACT_INVALID`, and the message says the prefill token count does not match.
17. The output length differs from the decode token count: `CONTRACT_INVALID`, and the message says the decode token count does not match.
18. The prefill count plus the decode count overflows or exceeds 16: `CONTEXT_LIMIT_EXCEEDED`. The overflow message says the micro context length overflows. The limit message says the micro prompt does not fit in the reserved context.
19. A prompt id, then an output id, is outside the micro vocabulary: `CONTRACT_INVALID`, and the message says the cancellation token is outside the micro vocabulary.
20. The cancel request time is zero: `CONTRACT_INVALID`, and the message says the cancel request time is zero.
21. The terminal time is not after the request time: `CONTRACT_INVALID`, and the message says the cancel terminal is not after the request.

`verify_cancellation_latency` checks the stored report and recomputes the measurement. Checks run in this order:

1. The report fails `validate`, and that failure is returned.
2. The model image root, artifact root, or engine build root differs. The message says that root does not match.
3. The placement root differs: `CONTRACT_INVALID`, and the message says the placement root does not match.
4. The prompt token root or the output token root differs. The message says that root does not match.
5. The execution mode, cache policy, concurrency, run count, warm state, request count, or cancel stage differs. The message says that field does not match.
6. A token count or a time differs. The message says that count or time does not match.
7. Recomputing the measurement fails, and that failure is returned.
8. The recomputed report bytes differ: `CONTRACT_INVALID`, and the message says the cancellation validation did not match.

`measure_cancellation_latency` runs that verify before it returns. A verify failure returns no report.

## Files

`write_cancellation_report` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the cancellation directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the cancellation path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the cancellation output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The prompt ids, the output ids, and the plan are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_cancellation_latency`. Token ids are unchanged. Receipt finalization latency is KIP-INFER-0032. Receipt overhead stays later. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter.
