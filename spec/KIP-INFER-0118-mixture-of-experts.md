# KIP-INFER-0118 — Mixture of experts

Status: `measure_moe` records one mixture-of-experts request the router did not run for a cold micro fixture. It returns a `knolo.infer.moe-report`. The reason is `router`, `expert`, or `shared`. The code is `CONTRACT_INVALID` and it is not retryable. Each reason names 1 through 16 requested experts. Exactly 16 is recorded. A count above 16 is `CONTRACT_INVALID` and issues no report. Selected experts stay 0. `routed`, `sharedUsed`, `tieBroken`, `grouped`, and `expertPlaced` stay false. The forward does not run and no receipt is stored. It does not run a router. `measure_attention` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-moe, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `router`, `expert`, or `shared` |
| `requestedExperts` | `1` through `16` |
| `selectedExperts` | `0` |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `routed` | `false` |
| `sharedUsed` | `false` |
| `tieBroken` | `false` |
| `grouped` | `false` |
| `expertPlaced` | `false` |
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

`kind` is `knolo.infer.moe-report` and `version` is `1`. The contract count is one hundred eighteen. `infer-moe` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the moe extensions are empty. A code other than `CONTRACT_INVALID` says a mixture of experts is CONTRACT_INVALID. `retryable` true says a mixture of experts is not retryable. `routed` true says a router does not run. `sharedUsed` true says a shared expert is not used. `tieBroken` true says expert ties are not broken. `grouped` true says grouped kernels stay off. `expertPlaced` true says an expert is not placed. Selected experts other than 0 say selected experts stay zero. A requested count above 16 says the expert count exceeds the record cap. A requested count of 0 says that reason names an expert. `forwardRan` true says a mixture of experts does not run the forward. `receiptStored` true says a mixture of experts stores no receipt.

## Measurement

`measure_moe` takes the placement plan and one observation. It returns the report. The router does not run.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the moe report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the moe report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the moe report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says moe concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says moe run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says moe warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says moe request count is one.
12. The report checks in the Report section, in the order written there.

`verify_moe` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the moe validation did not match.

## Files

`write_moe_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the moe directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the moe path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the moe output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_moe`. Payload hashing has not started. Hybrid attention has not started. Linear attention has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
