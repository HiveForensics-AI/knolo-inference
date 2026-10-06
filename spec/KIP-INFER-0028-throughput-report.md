# KIP-INFER-0028 — Micro fixture throughput

Status: `measure_micro_throughput` records prefill throughput, decode throughput, and request throughput for one cold run of the `knolo.micro.v1` fixture. It returns a `knolo.infer.throughput-report`. It does not run a model, does not read a GGUF payload, does not allocate a KV pool, and does not write a weight file. `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`, `perplexity_delta`, and `measure_placement_memory` do not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Rate

A throughput figure is an integer count of tokens, or of requests, per second, in micros. There is no floating-point value in the report.

```text
tokens_per_second_micros = count × 1_000_000 × 1_000_000_000 / nanos
```

The division is integer division. The remainder is discarded. `count` and `nanos` are both greater than zero. A product or a quotient that does not fit in `u64` is `CONTRACT_INVALID`, and the message says the throughput rate overflows.

Prefill throughput uses the prompt length and the prefill nanoseconds. Decode throughput uses the output length and the decode nanoseconds. Request throughput uses the request count and the sum of those two durations. Queue time is not a field of this report.

One token in three nanoseconds is `333_333_333_333_333` micros. Three prompt tokens in one second are `3_000_000` micros. The request rate of one request whose prefill and decode each took one second is `500_000` micros.

## Report

The report is a versioned contract. Its identity root is `H(infer-throughput, document)`. Version 1 accepts only these fields:

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
| `prefillTokensPerSecondMicros` | the prefill rate |
| `decodeTokensPerSecondMicros` | the decode rate |
| `requestsPerSecondMicros` | the request rate |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.throughput-report` and `version` is `1`. The contract count is twenty-one. `infer-throughput` is the report domain.

Any other validation result is `CONTRACT_INVALID`, and the message says the field has an unsupported value. A stored rate that is not the division above is `CONTRACT_INVALID`, and the message says that throughput does not match. A prefill plus decode token count above 16 is `CONTRACT_INVALID`, and the message says the micro prompt does not fit in the reserved context. A report is not issued when the measurement fails.

## Measurement

`measure_micro_throughput` takes the placement plan and one observation. The observation carries the model image root, the artifact root, the engine build root, the execution mode, the cache policy, the concurrency, the run count, the warm state, the request count, the prompt ids, the greedy output ids, and the two durations. It returns the report. It does not create a file and it does not call the model. The caller supplies ids from a completed micro-fixture run.

The plan is the micro fixture plan: context reservation 16, KV block size 16, and KV precision `f32`. Graph capture stays `off`. Device `cpu` and device `slot-0` both record a report. The token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the throughput report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the throughput report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the throughput report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says throughput concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says throughput run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says throughput warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says throughput request count is one.
12. The prefill token count is zero: `CONTRACT_INVALID`, and the message says the prefill token count is zero.
13. The decode token count is zero: `CONTRACT_INVALID`, and the message says the decode token count is zero.
14. The prompt length differs from the prefill token count: `CONTRACT_INVALID`, and the message says the prefill token count does not match.
15. The output length differs from the decode token count: `CONTRACT_INVALID`, and the message says the decode token count does not match.
16. The prefill count plus the decode count overflows or exceeds 16: `CONTEXT_LIMIT_EXCEEDED`. The overflow message says the micro context length overflows. The limit message says the micro prompt does not fit in the reserved context.
17. A prompt id, then an output id, is outside the micro vocabulary: `CONTRACT_INVALID`, and the message says the throughput token is outside the micro vocabulary.
18. The prefill duration is zero: `CONTRACT_INVALID`, and the message says the prefill duration is zero.
19. The decode duration is zero: `CONTRACT_INVALID`, and the message says the decode duration is zero.
20. The duration sum overflows: `CONTRACT_INVALID`, and the message says the throughput duration overflows.
21. A rate overflows: `CONTRACT_INVALID`, and the message says the throughput rate overflows.

`verify_micro_throughput` checks the stored report and recomputes the measurement. Checks run in this order:

1. The report fails `validate`, and that failure is returned.
2. The model image root, artifact root, or engine build root differs. The message says that root does not match.
3. The placement root differs: `CONTRACT_INVALID`, and the message says the placement root does not match.
4. The prompt token root or the output token root differs. The message says that root does not match.
5. The execution mode, cache policy, concurrency, run count, warm state, or request count differs. The message says that field does not match.
6. A token count or a duration differs. The message says that count or duration does not match.
7. Recomputing the measurement fails, and that failure is returned.
8. The recomputed report bytes differ: `CONTRACT_INVALID`, and the message says the throughput validation did not match.

`measure_micro_throughput` runs that verify before it returns. A verify failure returns no report.

## Files

`write_throughput_report` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the throughput directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the throughput path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the throughput output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The prompt ids, the output ids, and the plan are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_micro_throughput`. Token ids are unchanged. The latency report is KIP-INFER-0029. Cancellation latency, receipt finalization latency, and corruption fuzzing stay later. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter.
