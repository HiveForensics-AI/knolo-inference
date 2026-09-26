# KIP-INFER-0097 — Context limit

Status: `measure_context_limit` records one cold micro fixture whose prompt does not fit the reserved context. It returns a `knolo.infer.context-limit-report`. The reason is `prompt`, `budget`, or `overflow`. The context is 16 tokens. A prompt reason records 17 through 64 prompt tokens. A budget reason records a prompt of 1 through 16 plus a reservation of 1 through 16 whose sum exceeds 16. An overflow reason records a prompt plus a reservation that does not fit in `u32`. Tokens are not truncated. The code is `CONTEXT_LIMIT_EXCEEDED` and it is not retryable. The forward does not run and no receipt is stored. It does not drop tokens. `measure_signature_check` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

A prompt reason above 64 tokens is `CONTEXT_LIMIT_EXCEEDED` and issues no report. A stored report above that cap is `CONTRACT_INVALID`.

## Report

The report is a versioned contract. Its identity root is `H(infer-context-limit, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `prompt`, `budget`, or `overflow` |
| `promptTokens` | at least `1`; `17` through `64` for `prompt`; `1` through `16` for `budget` |
| `reservedTokens` | `0` through `16` for `prompt`; `1` through `16` for `budget` |
| `contextTokens` | `16` |
| `truncated` | `false` |
| `code` | `CONTEXT_LIMIT_EXCEEDED` |
| `retryable` | `false` |
| `forwardRan` | `false` |
| `receiptStored` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.context-limit-report` and `version` is `1`. The contract count is ninety-three. `infer-context-limit` is the report domain. Token ids are not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the context-limit extensions are empty. A prompt count of zero says a context limit has a prompt. A context other than 16 says the context-limit report is the micro fixture. `truncated` true says tokens are not truncated. A code other than `CONTEXT_LIMIT_EXCEEDED` says a context limit is CONTEXT_LIMIT_EXCEEDED. `retryable` true says a context limit is not retryable. `forwardRan` true says a context limit does not run the forward. `receiptStored` true says a context limit stores no receipt. A prompt reason at or below 16 says a prompt that fits is not a prompt limit. A stored prompt reason above 64 says prompt tokens exceed the record cap. A prompt reason that reserves more than 16 says a prompt limit does not reserve past the context. A budget reason whose prompt exceeds 16 says a budget limit prompt fits in the context. A budget reservation of zero or above 16 says a budget limit reserves at most the context. A budget whose sum fits says a budget that fits is not a context limit. An overflow whose sum fits in `u32` says an overflowing context does not fit in u32.

## Measurement

`measure_context_limit` takes the placement plan and one observation. It returns the report. It does not truncate the prompt and it does not run the forward.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the context-limit report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the context-limit report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the context-limit report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says context-limit concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says context-limit run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says context-limit warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says context-limit request count is one.
12. A `prompt` reason above 64 tokens: `CONTEXT_LIMIT_EXCEEDED`, and the message says prompt tokens exceed the record cap.
13. The report checks in the Report section, in the order written there.

`verify_context_limit` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the context-limit validation did not match.

## Files

`write_context_limit_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the context-limit directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the context-limit path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the context-limit output already exists.

The report is created with `create_new`, written, and `fsync`ed. Tokens are not truncated.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_context_limit`. The prompt-compilation record is KIP-INFER-0098. Cofactor clearing has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
