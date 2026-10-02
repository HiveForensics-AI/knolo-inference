# KIP-INFER-0123 — Workstation recipe

Status: `measure_workstation` records one workstation recipe the engine did not bless for a cold micro fixture. It returns a `knolo.infer.workstation-report`. The reason is `profile`, `benchmark`, or `fallback`. The profile root and the benchmark root differ from each other and from the engine build and the placement. A benchmark record names 1 through 32 suite entries. Exactly 32 is recorded. A profile record and a fallback record carry no suite. A count above 32 is `CONTRACT_INVALID` and issues no report. `blessed`, `benchmarkRecorded`, and `fallbackSelected` stay false. The forward does not run and no receipt is stored. It does not run a benchmark. `measure_glm` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-workstation, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `profileRoot` | root distinct from the engine build and the placement |
| `benchmarkRoot` | root distinct from the profile, the engine build, and the placement |
| `reason` | `profile`, `benchmark`, or `fallback` |
| `benchmarkCount` | `1` through `32` for `benchmark`; `0` otherwise |
| `blessed` | `false` |
| `benchmarkRecorded` | `false` |
| `fallbackSelected` | `false` |
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

`kind` is `knolo.infer.workstation-report` and `version` is `1`. The contract count is one hundred eighteen. `infer-workstation` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the workstation extensions are empty. A profile root equal to the engine build says the profile repeats the engine build. A profile root equal to the placement says the profile repeats the placement. A benchmark root equal to the engine build says the benchmark repeats the engine build. A benchmark root equal to the placement says the benchmark repeats the placement. A benchmark root equal to the profile says the benchmark repeats the profile. `blessed` true says a workstation recipe is not blessed. `benchmarkRecorded` true says a workstation benchmark does not run. `fallbackSelected` true says a workstation recipe does not fall back. A suite above 32 says the benchmark count exceeds the record cap. A benchmark record with no suite says a benchmark record names a suite. A profile record with a suite says a profile record carries no benchmark. A fallback record with a suite says a fallback record carries no benchmark. `forwardRan` true says a workstation recipe does not run the forward. `receiptStored` true says a workstation recipe stores no receipt.

## Measurement

`measure_workstation` takes the placement plan and one observation. It returns the report. The workstation recipe is not blessed.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the workstation report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the workstation report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the workstation report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says workstation concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says workstation run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says workstation warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says workstation request count is one.
12. The report checks in the Report section, in the order written there.

`verify_workstation` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the workstation validation did not match.

## Files

`write_workstation_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the workstation directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the workstation path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the workstation output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_workstation`. Payload hashing has not started. Hybrid attention has not started. Linear attention has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
