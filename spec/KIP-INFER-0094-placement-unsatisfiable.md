# KIP-INFER-0094 — Placement unsatisfiable

Status: `measure_placement_refusal` records one placement the micro fixture cannot satisfy for a cold micro fixture. It returns a `knolo.infer.placement-refusal-report`. The reason is `device`, `toolkit`, `capability`, or `bytes`. A toolkit refusal does not probe the device. A device refusal probes slot-0 and does not see it. A capability refusal and a byte refusal probe slot-0 and see it. The code is `PLACEMENT_UNSATISFIABLE` and it is not retryable. The device is not opened and there is no fallback. The forward does not run and no receipt is stored. It does not open a device. `measure_kernel` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-placement-refusal, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `device`, `toolkit`, `capability`, or `bytes` |
| `code` | `PLACEMENT_UNSATISFIABLE` |
| `retryable` | `false` |
| `probeReached` | `false` for `toolkit`; `true` otherwise |
| `slotVisible` | `true` for `capability` and `bytes`; `false` otherwise |
| `deviceOpened` | `false` |
| `cpuFallback` | `false` |
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

`kind` is `knolo.infer.placement-refusal-report` and `version` is `1`. The contract count is eighty-eight. `infer-placement-refusal` is the report domain. A device path is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the placement-refusal extensions are empty. A code other than `PLACEMENT_UNSATISFIABLE` says an unsatisfiable placement is PLACEMENT_UNSATISFIABLE. `retryable` true says an unsatisfiable placement is not retryable. `forwardRan` true says an unsatisfiable placement does not run the forward. `receiptStored` true says an unsatisfiable placement stores no receipt. `deviceOpened` true says an unsatisfiable placement does not open a device. `cpuFallback` true says an unsatisfiable placement does not fall back. `probeReached` true on `toolkit` says a toolkit refusal does not probe the device. `slotVisible` true on `toolkit` says a toolkit refusal does not claim a visible device. `probeReached` false on `device` says a device refusal probes slot-0. `slotVisible` true on `device` says a missing device is not visible. `probeReached` false on `capability` says a capability refusal probes slot-0. `slotVisible` false on `capability` says a capability refusal sees slot-0. `probeReached` false on `bytes` says a byte refusal probes slot-0. `slotVisible` false on `bytes` says a byte refusal sees slot-0.

## Measurement

`measure_placement_refusal` takes the placement plan and one observation. It returns the report. It does not open a device and it does not fall back.

Device `cpu` and device `slot-0` both record a report. The recorded plan is a micro plan that already validates. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the placement-refusal report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the placement-refusal report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the placement-refusal report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says placement-refusal concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says placement-refusal run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says placement-refusal warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says placement-refusal request count is one.
12. The report checks in the Report section, in the order written there.

`verify_placement_refusal` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the placement-refusal validation did not match.

## Files

`write_placement_refusal_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the placement-refusal directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the placement-refusal path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the placement-refusal output already exists.

The report is created with `create_new`, written, and `fsync`ed. No device is opened.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_placement_refusal`. The insufficient-memory record is KIP-INFER-0095. The signature check has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
