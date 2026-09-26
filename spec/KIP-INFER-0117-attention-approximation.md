# KIP-INFER-0117 — Attention approximation

Status: `measure_attention` records one attention modification the engine did not apply for a cold micro fixture. It returns a `knolo.infer.attention-report`. The reason is `approximation`, `window`, `sparsity`, or `quantized-kv`. The code is `CONTRACT_INVALID` and it is not retryable. Attention stays exact. `approximated`, `windowApplied`, `sparsityApplied`, and `kvQuantized` stay false. A window refusal names 1 through 16 tokens. Exactly 16 is recorded. A span above 16 is `CONTEXT_LIMIT_EXCEEDED` and issues no report. The other reasons carry no window. The forward does not run and no receipt is stored. It does not approximate attention. `measure_domain` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-attention, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `approximation`, `window`, `sparsity`, or `quantized-kv` |
| `windowTokens` | `1` through `16` for `window`; `0` otherwise |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `attentionExact` | `true` |
| `approximated` | `false` |
| `windowApplied` | `false` |
| `sparsityApplied` | `false` |
| `kvQuantized` | `false` |
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

`kind` is `knolo.infer.attention-report` and `version` is `1`. The contract count is one hundred eighteen. `infer-attention` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the attention extensions are empty. A code other than `CONTRACT_INVALID` says an attention modification is CONTRACT_INVALID. `retryable` true says an attention modification is not retryable. `attentionExact` false says attention stays exact. `approximated` true says an approximation is not applied. `windowApplied` true says a sliding window is not applied. `sparsityApplied` true says a sparsity rule is not applied. `kvQuantized` true says quantized KV is not applied. `forwardRan` true says an attention modification does not run the forward. `receiptStored` true says an attention modification stores no receipt. A window span above 16 on a stored report says the window span exceeds the record cap. A window of zero says a window refusal names a span. An approximation, sparsity, or quantized-kv reason with a window says that reason carries no window.

## Measurement

`measure_attention` takes the placement plan and one observation. It returns the report. Attention is not modified.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the attention report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the attention report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the attention report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says attention concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says attention run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says attention warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says attention request count is one.
12. The report checks in the Report section, in the order written there.

`verify_attention` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the attention validation did not match.

## Files

`write_attention_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the attention directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the attention path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the attention output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_attention`. Payload hashing has not started. Hybrid attention has not started. Linear attention has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
