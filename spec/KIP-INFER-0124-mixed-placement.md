# KIP-INFER-0124 — Mixed placement

Status: `measure_mixed_placement` records one placement mix the planner did not select for a cold micro fixture. It returns a `knolo.infer.mixed-placement-report`. The reason is `single`, `mixed`, or `fallback`. A single record names one device, uses code `none`, and sets `singleRecorded`. A mixed record names two devices and is `CONTRACT_INVALID`. A fallback record names one device and is `CONTRACT_INVALID`. A count above 2 is `CONTRACT_INVALID` and issues no report. `mixedSelected` and `automaticFallback` stay false. The forward does not run and no receipt is stored. It does not place a model. `measure_workstation` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-mixed-placement, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `single`, `mixed`, or `fallback` |
| `deviceCount` | `2` for `mixed`; `1` otherwise |
| `code` | `none` for `single`; `CONTRACT_INVALID` otherwise |
| `retryable` | `false` |
| `singleRecorded` | `true` only for `single` |
| `mixedSelected` | `false` |
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

`kind` is `knolo.infer.mixed-placement-report` and `version` is `1`. The contract count is one hundred eighteen. `infer-mixed-placement` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the mixed-placement extensions are empty. `retryable` true says a placement record is not retryable. `mixedSelected` true says mixed placement is not selected. `automaticFallback` true says placement does not fall back. A device count above 2 says the device count exceeds the record cap. A single record that is not code `none` says a single placement is not a refusal. A single record that does not set `singleRecorded` says a single placement records one device. A mixed record that is not `CONTRACT_INVALID` says a mixed placement is CONTRACT_INVALID. A mixed record that sets `singleRecorded` says only a single placement records one device. A fallback record that is not `CONTRACT_INVALID` says an automatic fallback is CONTRACT_INVALID. `forwardRan` true says a placement record does not run the forward. `receiptStored` true says a placement record stores no receipt.

## Measurement

`measure_mixed_placement` takes the placement plan and one observation. It returns the report. Mixed placement is not selected.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the mixed-placement report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the mixed-placement report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the mixed-placement report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says mixed-placement concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says mixed-placement run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says mixed-placement warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says mixed-placement request count is one.
12. The report checks in the Report section, in the order written there.

`verify_mixed_placement` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the mixed-placement validation did not match.

## Files

`write_mixed_placement_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the mixed-placement directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the mixed-placement path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the mixed-placement output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_mixed_placement`. Payload hashing has not started. Hybrid attention has not started. Linear attention has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
