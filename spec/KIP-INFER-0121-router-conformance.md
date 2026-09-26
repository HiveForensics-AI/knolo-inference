# KIP-INFER-0121 — Router conformance

Status: `measure_router` records one router comparison the engine did not run for a cold micro fixture. It returns a `knolo.infer.router-report`. The reason is `logits`, `selection`, or `load`. Each reason names 1 through 16 samples. Exactly 16 is recorded. A count above 16 is `CONTRACT_INVALID` and issues no report. The reference root and the candidate root differ from each other and from the engine build and the placement. `compared`, `parity`, and `loadRecorded` stay false. The forward does not run and no receipt is stored. It does not run a router. `measure_grouped_kernel` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-router, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `referenceRoot` | root distinct from the engine build and the placement |
| `candidateRoot` | root distinct from the reference, the engine build, and the placement |
| `reason` | `logits`, `selection`, or `load` |
| `sampleCount` | `1` through `16` |
| `compared` | `false` |
| `parity` | `false` |
| `loadRecorded` | `false` |
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

`kind` is `knolo.infer.router-report` and `version` is `1`. The contract count is one hundred eighteen. `infer-router` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the router extensions are empty. A reference root equal to the engine build says the reference repeats the engine build. A reference root equal to the placement says the reference repeats the placement. A candidate root equal to the engine build says the candidate repeats the engine build. A candidate root equal to the placement says the candidate repeats the placement. A candidate root equal to the reference says the candidate repeats the reference. `compared` true says router outputs are not compared. `parity` true says router parity is not claimed. `loadRecorded` true says expert load is not recorded. A sample count above 16 says the router sample exceeds the record cap. A sample count of 0 says that reason names a sample. `forwardRan` true says a router comparison does not run the forward. `receiptStored` true says a router comparison stores no receipt.

## Measurement

`measure_router` takes the placement plan and one observation. It returns the report. Router outputs are not compared.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the router report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the router report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the router report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says router concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says router run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says router warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says router request count is one.
12. The report checks in the Report section, in the order written there.

`verify_router` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the router validation did not match.

## Files

`write_router_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the router directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the router path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the router output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_router`. Payload hashing has not started. Hybrid attention has not started. Linear attention has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
