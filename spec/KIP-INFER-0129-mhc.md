# KIP-INFER-0129 — mHC support

Status: `measure_mhc` records one manifold-constrained hyper-connection the engine did not apply for a cold micro fixture. It returns a `knolo.infer.mhc-report`. The reason is `connection`, `manifold`, or `residual`. The code is `CONTRACT_INVALID` and it is not retryable. Each reason names 1 through 16 requested streams. Exactly 16 is recorded. A count above 16 is `CONTRACT_INVALID` and issues no report. Applied streams stay 0. `connected`, `manifoldApplied`, and `residualMapped` stay false. The forward does not run and no receipt is stored. It does not apply an mHC connection. `measure_linear` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-mhc, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `connection`, `manifold`, or `residual` |
| `requestedStreams` | `1` through `16` |
| `appliedStreams` | `0` |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `connected` | `false` |
| `manifoldApplied` | `false` |
| `residualMapped` | `false` |
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

`kind` is `knolo.infer.mhc-report` and `version` is `1`. The contract count is one hundred twenty-three. `infer-mhc` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the mhc extensions are empty. A code other than `CONTRACT_INVALID` says an mHC connection is CONTRACT_INVALID. `retryable` true says an mHC connection is not retryable. `connected` true says an mHC connection is not applied. `manifoldApplied` true says a manifold constraint is not applied. `residualMapped` true says a residual map is not applied. Applied streams other than 0 say mHC streams stay zero. A stored report above 16 says mHC streams exceed the record cap. A requested count of 0 says that reason names a stream count. `forwardRan` true says an mHC connection does not run the forward. `receiptStored` true says an mHC connection stores no receipt.

## Measurement

`measure_mhc` takes the placement plan and one observation. It returns the report. An mHC connection is not applied.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the mhc report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the mhc report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the mhc report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says mhc concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says mhc run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says mhc warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says mhc request count is one.
12. The report checks in the Report section, in the order written there.

`verify_mhc` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the mhc validation did not match.

## Files

`write_mhc_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the mhc directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the mhc path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the mhc output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_mhc`. Digest binding has not started. Multimodal input has not started. Large expert placement has not started. Extreme-context cache has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
