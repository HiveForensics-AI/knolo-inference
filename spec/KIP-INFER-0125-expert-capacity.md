# KIP-INFER-0125 — Expert capacity

Status: `measure_capacity` records one expert-capacity request the router did not apply for a cold micro fixture. It returns a `knolo.infer.capacity-report`. The reason is `overflow`, `drop`, or `balance`. The code is `CONTRACT_INVALID` and it is not retryable. Each reason names 1 through 16 requested tokens. Exactly 16 is recorded. A count above 16 is `CONTEXT_LIMIT_EXCEEDED` and issues no report. Capacity tokens stay 0. `overflowed`, `dropped`, and `balanced` stay false. The forward does not run and no receipt is stored. It does not drop a token. `measure_mixed_placement` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-capacity, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `overflow`, `drop`, or `balance` |
| `requestedTokens` | `1` through `16` |
| `capacityTokens` | `0` |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `overflowed` | `false` |
| `dropped` | `false` |
| `balanced` | `false` |
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

`kind` is `knolo.infer.capacity-report` and `version` is `1`. The contract count is one hundred eighteen. `infer-capacity` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the capacity extensions are empty. A code other than `CONTRACT_INVALID` says expert capacity is CONTRACT_INVALID. `retryable` true says expert capacity is not retryable. `overflowed` true says expert overflow is not placed. `dropped` true says expert tokens are not dropped. `balanced` true says expert load is not balanced. Capacity tokens other than 0 say expert capacity stays zero. A stored report above 16 says capacity tokens exceed the record cap. A requested count of 0 says that reason names a token count. `forwardRan` true says expert capacity does not run the forward. `receiptStored` true says expert capacity stores no receipt.

## Measurement

`measure_capacity` takes the placement plan and one observation. It returns the report. Expert capacity is not applied.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the capacity report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the capacity report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the capacity report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says capacity concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says capacity run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says capacity warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says capacity request count is one.
12. The report checks in the Report section, in the order written there.

`verify_capacity` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the capacity validation did not match.

## Files

`write_capacity_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the capacity directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the capacity path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the capacity output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_capacity`. Payload hashing has not started. Hybrid attention has not started. Linear attention has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
