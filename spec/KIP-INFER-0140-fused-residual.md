# KIP-INFER-0140 — Fused residual

Status: `measure_residual` records one fused residual the engine did not apply for a cold micro fixture. It returns a `knolo.infer.residual-report`. The reason is `fused`, `add`, or `norm`. The code is `UNSUPPORTED_KERNEL` and it is not retryable. Each reason names 1 through 16 layers. Exactly 16 is recorded. A count above 16 is `CONTRACT_INVALID` and issues no report. Applied layers stay 0. `fusedApplied`, `added`, and `normFolded` stay false. The forward does not run and no receipt is stored. It does not fuse a residual. `measure_rope` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-residual, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `fused`, `add`, or `norm` |
| `requestedLayers` | `1` through `16` |
| `appliedLayers` | `0` |
| `code` | `UNSUPPORTED_KERNEL` |
| `retryable` | `false` |
| `fusedApplied` | `false` |
| `added` | `false` |
| `normFolded` | `false` |
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

`kind` is `knolo.infer.residual-report` and `version` is `1`. The contract count is one hundred thirty-three. `infer-residual` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the residual extensions are empty. A code other than `UNSUPPORTED_KERNEL` says a fused residual is UNSUPPORTED_KERNEL. `retryable` true says a fused residual is not retryable. `fusedApplied` true says a fused residual is not applied. `added` true says a residual add is not applied. `normFolded` true says a folded norm is not applied. `appliedLayers` other than 0 says applied residual layers stay zero. A stored report above 16 says residual layers exceed the record cap. A requested count of 0 says that reason names a layer count. `forwardRan` true says a fused residual does not run the forward. `receiptStored` true says a fused residual stores no receipt.

## Measurement

`measure_residual` takes the placement plan and one observation. It returns the report. A residual is not fused.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the residual report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the residual report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the residual report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says residual concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says residual run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says residual warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says residual request count is one.
12. The report checks in the Report section, in the order written there.

A count above 16 on the observation is `CONTRACT_INVALID` and issues no report.

`verify_residual` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the residual validation did not match.

## Files

`write_residual_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the residual directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the residual path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the residual output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_residual`. Metal backend has not started. KV scatter has not started. Disaggregated serving has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
