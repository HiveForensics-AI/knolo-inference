# KIP-INFER-0119 — Expert placement

Status: `measure_expert_placement` records one expert placement the planner did not apply for a cold micro fixture. It returns a `knolo.infer.expert-placement-report`. The reason is `residency`, `offload`, or `asymmetric`. The code is `CONTRACT_INVALID` and it is not retryable. A residency refusal and an offload refusal name one device. An asymmetric refusal names two. A count above 2 is `CONTRACT_INVALID` and issues no report. Placed experts stay 0. `cpuOffload`, `residentMoved`, `tensorParallel`, and `automaticFallback` stay false. The forward does not run and no receipt is stored. It does not place an expert. `measure_moe` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-expert-placement, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `residency`, `offload`, or `asymmetric` |
| `deviceCount` | `1` for `residency` and `offload`; `2` for `asymmetric` |
| `expertsPlaced` | `0` |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `cpuOffload` | `false` |
| `residentMoved` | `false` |
| `tensorParallel` | `false` |
| `automaticFallback` | `false` |
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

`kind` is `knolo.infer.expert-placement-report` and `version` is `1`. The contract count is one hundred eighteen. `infer-expert-placement` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the expert-placement extensions are empty. A code other than `CONTRACT_INVALID` says an expert placement is CONTRACT_INVALID. `retryable` true says an expert placement is not retryable. `cpuOffload` true says experts are not offloaded. `residentMoved` true says resident experts are not moved. `tensorParallel` true says tensor parallel stays off. `automaticFallback` true says placement does not fall back. Placed experts other than 0 say placed experts stay zero. A device count above 2 says the device count exceeds the record cap. `forwardRan` true says an expert placement does not run the forward. `receiptStored` true says an expert placement stores no receipt.

## Measurement

`measure_expert_placement` takes the placement plan and one observation. It returns the report. The expert is not placed.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the expert-placement report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the expert-placement report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the expert-placement report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says expert-placement concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says expert-placement run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says expert-placement warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says expert-placement request count is one.
12. The report checks in the Report section, in the order written there.

`verify_expert_placement` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the expert-placement validation did not match.

## Files

`write_expert_placement_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the expert-placement directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the expert-placement path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the expert-placement output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_expert_placement`. Payload hashing has not started. Hybrid attention has not started. Linear attention has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
