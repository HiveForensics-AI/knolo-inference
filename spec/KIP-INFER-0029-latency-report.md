# KIP-INFER-0029 — Micro fixture latency

Status: `measure_micro_latency` records time to first token, time per output token, and the p50, p95, and p99 request latency for one cold run of the `knolo.micro.v1` fixture. It returns a `knolo.infer.latency-report`. It does not run a model, does not read a GGUF payload, does not allocate a KV pool, and does not write a weight file. `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`, and `measure_micro_throughput` do not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Durations

A latency figure is an integer number of nanoseconds. There is no floating-point value in the report. Queue time is not a field.

Time to first token is the prefill duration. The first token of this isolated run is available when prefill finishes.

Time per output token is the decode duration divided by the decode token count. The division is integer division. The remainder is discarded. A quotient of zero is recorded when the decode duration is shorter than one nanosecond per output token.

Request latency is the sum of the prefill duration and the decode duration. The sample count is one, because `requestCount` is `1` and `runCount` is `1`. The nearest-rank percentile of that one sample, at 50, at 95, and at 99, is the sample itself. `latencyP50Nanos`, `latencyP95Nanos`, and `latencyP99Nanos` are therefore equal to the request latency.

One request whose prefill took one second and whose four output tokens took one second has time to first token `1_000_000_000`, time per output token `250_000_000`, and request latency `2_000_000_000`. The three percentiles are `2_000_000_000`. Two output tokens in five nanoseconds have time per output token `2`. Two output tokens in one nanosecond have time per output token `0`.

## Report

The report is a versioned contract. Its identity root is `H(infer-latency, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `modelImageRoot` | root of the micro model image |
| `artifactRoot` | root of the weight artifact |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` that ran |
| `placementRoot` | root of the `PlacementPlanV1` |
| `promptTokenRoot` | `H(infer-prompt-tokens, prompt ids)` |
| `outputTokenRoot` | `H(infer-output-tokens, greedy ids)` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `prefillTokens` | prompt length |
| `decodeTokens` | greedy output length |
| `requestCount` | `1` |
| `prefillNanos` | prefill duration |
| `decodeNanos` | decode duration |
| `timeToFirstTokenNanos` | the prefill duration |
| `timePerOutputTokenNanos` | the integer decode duration per output token |
| `requestLatencyNanos` | prefill plus decode |
| `latencyP50Nanos` | the request latency |
| `latencyP95Nanos` | the request latency |
| `latencyP99Nanos` | the request latency |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.latency-report` and `version` is `1`. The contract count is twenty-two. `infer-latency` is the report domain.

Any other validation result is `CONTRACT_INVALID`, and the message says the field has an unsupported value. A stored duration that is not the value above is `CONTRACT_INVALID`, and the message says that latency does not match. A prefill plus decode token count above 16 is `CONTRACT_INVALID`, and the message says the micro prompt does not fit in the reserved context. A report is not issued when the measurement fails.

## Measurement

`measure_micro_latency` takes the placement plan and one observation. The observation carries the model image root, the artifact root, the engine build root, the execution mode, the cache policy, the concurrency, the run count, the warm state, the request count, the prompt ids, the greedy output ids, and the two durations. It returns the report. It does not create a file and it does not call the model. The caller supplies ids from a completed micro-fixture run.

The plan is the micro fixture plan: context reservation 16, KV block size 16, and KV precision `f32`. Graph capture stays `off`. Device `cpu` and device `slot-0` both record a report. The token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the latency report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the latency report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the latency report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says latency concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says latency run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says latency warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says latency request count is one.
12. The prefill token count is zero: `CONTRACT_INVALID`, and the message says the prefill token count is zero.
13. The decode token count is zero: `CONTRACT_INVALID`, and the message says the decode token count is zero.
14. The prompt length differs from the prefill token count: `CONTRACT_INVALID`, and the message says the prefill token count does not match.
15. The output length differs from the decode token count: `CONTRACT_INVALID`, and the message says the decode token count does not match.
16. The prefill count plus the decode count overflows or exceeds 16: `CONTEXT_LIMIT_EXCEEDED`. The overflow message says the micro context length overflows. The limit message says the micro prompt does not fit in the reserved context.
17. A prompt id, then an output id, is outside the micro vocabulary: `CONTRACT_INVALID`, and the message says the latency token is outside the micro vocabulary.
18. The prefill duration is zero: `CONTRACT_INVALID`, and the message says the prefill duration is zero.
19. The decode duration is zero: `CONTRACT_INVALID`, and the message says the decode duration is zero.
20. The duration sum overflows: `CONTRACT_INVALID`, and the message says the latency duration overflows.

`verify_micro_latency` checks the stored report and recomputes the measurement. Checks run in this order:

1. The report fails `validate`, and that failure is returned.
2. The model image root, artifact root, or engine build root differs. The message says that root does not match.
3. The placement root differs: `CONTRACT_INVALID`, and the message says the placement root does not match.
4. The prompt token root or the output token root differs. The message says that root does not match.
5. The execution mode, cache policy, concurrency, run count, warm state, or request count differs. The message says that field does not match.
6. A token count or a duration differs. The message says that count or duration does not match.
7. Recomputing the measurement fails, and that failure is returned.
8. The recomputed report bytes differ: `CONTRACT_INVALID`, and the message says the latency validation did not match.

`measure_micro_latency` runs that verify before it returns. A verify failure returns no report.

## Files

`write_latency_report` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the latency directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the latency path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the latency output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The prompt ids, the output ids, and the plan are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_micro_latency`. Token ids are unchanged. Corruption fuzzing is KIP-INFER-0030. Cancellation latency is KIP-INFER-0031. Receipt finalization latency stays later. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter.
