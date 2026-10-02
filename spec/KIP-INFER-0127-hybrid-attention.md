# KIP-INFER-0127 — Hybrid attention

Status: `measure_hybrid` records one hybrid-attention state the engine did not allocate for a cold micro fixture. It returns a `knolo.infer.hybrid-report`. The reason is `state`, `sliding`, or `full`. The code is `CONTRACT_INVALID` and it is not retryable. Attention stays exact. `hybridAllocated`, `slidingApplied`, and `fullMixed` stay false. A state refusal names 1 through 16 tokens. Exactly 16 is recorded. A span above 16 is `CONTEXT_LIMIT_EXCEEDED` and issues no report. A sliding refusal and a full refusal carry no state. The forward does not run and no receipt is stored. It does not allocate hybrid state. `measure_payload` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-hybrid, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `state`, `sliding`, or `full` |
| `stateTokens` | `1` through `16` for `state`; `0` otherwise |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `attentionExact` | `true` |
| `hybridAllocated` | `false` |
| `slidingApplied` | `false` |
| `fullMixed` | `false` |
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

`kind` is `knolo.infer.hybrid-report` and `version` is `1`. The contract count is one hundred twenty-three. `infer-hybrid` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the hybrid extensions are empty. A code other than `CONTRACT_INVALID` says hybrid attention is CONTRACT_INVALID. `retryable` true says hybrid attention is not retryable. `attentionExact` false says attention stays exact. `hybridAllocated` true says hybrid state is not allocated. `slidingApplied` true says a sliding layer is not applied. `fullMixed` true says full attention is not mixed. `forwardRan` true says hybrid attention does not run the forward. `receiptStored` true says hybrid attention stores no receipt. A state span above 16 on a stored report says hybrid state exceeds the record cap. A state of zero says a state refusal names a span. A sliding or full reason with a state says that reason carries no state.

## Measurement

`measure_hybrid` takes the placement plan and one observation. It returns the report. Hybrid attention state is not allocated.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the hybrid report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the hybrid report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the hybrid report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says hybrid concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says hybrid run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says hybrid warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says hybrid request count is one.
12. The report checks in the Report section, in the order written there.

`verify_hybrid` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the hybrid validation did not match.

## Files

`write_hybrid_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the hybrid directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the hybrid path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the hybrid output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_hybrid`. Digest binding has not started. Multimodal input has not started. Large expert placement has not started. Extreme-context cache has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
