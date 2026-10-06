# KIP-INFER-0033 — Receipt overhead

Status: `measure_receipt_overhead` records the nanoseconds spent writing the accepted event and the receipt file for one cold run of the `knolo.micro.v1` fixture that stores a receipt. It returns a `knolo.infer.overhead-report`. It does not run a model, does not write a serve journal, does not read a GGUF payload, does not allocate a KV pool, and does not write a weight file. `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`, `measure_micro_throughput`, `measure_micro_latency`, `measure_corruption_fuzz`, `measure_cancellation_latency`, and `measure_receipt_finalization` do not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Duration

A duration is an integer number of nanoseconds. There is no floating-point value in the report. Queue time is not a field. These figures are write durations, not timestamps measured from accept.

The accepted write is the fsync of the accepted event before submit. The receipt write is the fsync of the receipt file after the terminal event. Both are greater than zero. Receipt overhead is their sum. The idle gap between the terminal event and the start of the receipt write is not included. That gap is part of the finalization latency in KIP-INFER-0032, and this report does not store it.

A `stop` receipt has at least one output token. A `length` receipt may have none. One `stop` whose accepted write is 120 nanoseconds and whose receipt write is 40 nanoseconds has receipt overhead `160`. One `length` with no output tokens, an accepted write of 80 nanoseconds, and a receipt write of 15 nanoseconds has receipt overhead `95`.

A `cancelled` request stores no receipt. A finish reason of `error` or `timeout` is a failed journal and stores no receipt. Any other finish reason is unsupported in this slice. A sum that overflows issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-overhead, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `modelImageRoot` | root of the micro model image |
| `artifactRoot` | root of the weight artifact |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` that ran |
| `placementRoot` | root of the `PlacementPlanV1` |
| `promptTokenRoot` | `H(infer-prompt-tokens, prompt ids)` |
| `outputTokenRoot` | `H(infer-output-tokens, output ids)` |
| `receiptRoot` | root of the stored inference receipt |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `finishReason` | `stop` or `length` |
| `prefillTokens` | prompt length |
| `decodeTokens` | output length |
| `acceptedWriteNanos` | accepted-event fsync |
| `receiptWriteNanos` | receipt-file fsync |
| `receiptOverheadNanos` | accepted write plus receipt write |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.overhead-report` and `version` is `1`. The contract count is twenty-six. `infer-overhead` is the report domain.

Any other validation result is `CONTRACT_INVALID`, and the message says the field has an unsupported value. A stored overhead that is not the sum above is `CONTRACT_INVALID`, and the message says receipt overhead does not match. A prefill plus decode token count above 16 is `CONTRACT_INVALID`, and the message says the micro prompt does not fit in the reserved context. A report is not issued when the measurement fails.

## Measurement

`measure_receipt_overhead` takes the placement plan and one observation. The observation carries the model image root, the artifact root, the engine build root, the receipt root, the execution mode, the cache policy, the concurrency, the run count, the warm state, the request count, the finish reason, the prompt ids, the output ids, and the two write durations. It returns the report. It does not create a file and it does not call the model. The caller supplies ids, the receipt root, and durations from a completed micro-fixture receipt.

The plan is the micro fixture plan: context reservation 16, KV block size 16, and KV precision `f32`. Graph capture stays `off`. Device `cpu` and device `slot-0` both record a report. The token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the overhead report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the overhead report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the overhead report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says overhead concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says overhead run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says overhead warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says overhead request count is one.
12. `finishReason` is `cancelled`: `CONTRACT_INVALID`, and the message says a cancelled request stores no receipt.
13. `finishReason` is `error` or `timeout`: `CONTRACT_INVALID`, and the message says a failed request stores no receipt.
14. `finishReason` is neither `stop` nor `length`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
15. The prefill token count is zero: `CONTRACT_INVALID`, and the message says the prefill token count is zero.
16. The finish reason is `stop` and the decode token count is zero: `CONTRACT_INVALID`, and the message says a stop receipt has no output tokens.
17. The prompt length differs from the prefill token count: `CONTRACT_INVALID`, and the message says the prefill token count does not match.
18. The output length differs from the decode token count: `CONTRACT_INVALID`, and the message says the decode token count does not match.
19. The prefill count plus the decode count overflows or exceeds 16: `CONTEXT_LIMIT_EXCEEDED`. The overflow message says the micro context length overflows. The limit message says the micro prompt does not fit in the reserved context.
20. A prompt id, then an output id, is outside the micro vocabulary: `CONTRACT_INVALID`, and the message says the overhead token is outside the micro vocabulary.
21. The accepted write is zero: `CONTRACT_INVALID`, and the message says the accepted write time is zero.
22. The receipt write is zero: `CONTRACT_INVALID`, and the message says the receipt write time is zero.
23. The sum overflows: `CONTRACT_INVALID`, and the message says receipt overhead overflows.

`verify_receipt_overhead` checks the stored report and recomputes the measurement. Checks run in this order:

1. The report fails `validate`, and that failure is returned.
2. The model image root, artifact root, or engine build root differs. The message says that root does not match.
3. The placement root differs: `CONTRACT_INVALID`, and the message says the placement root does not match.
4. The prompt token root or the output token root differs. The message says that root does not match.
5. The receipt root differs: `CONTRACT_INVALID`, and the message says the receipt root does not match.
6. The execution mode, cache policy, concurrency, run count, warm state, request count, or finish reason differs. The message says that field does not match.
7. A token count or a write duration differs. The message says that count or time does not match.
8. Recomputing the measurement fails, and that failure is returned.
9. The recomputed report bytes differ: `CONTRACT_INVALID`, and the message says the overhead validation did not match.

`measure_receipt_overhead` runs that verify before it returns. A verify failure returns no report.

## Files

`write_overhead_report` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the overhead directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the overhead path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the overhead output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The prompt ids, the output ids, the plan, and the inference receipt are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_receipt_overhead`. Token ids are unchanged. Model swap time is KIP-INFER-0034. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter.
