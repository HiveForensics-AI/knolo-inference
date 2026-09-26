# KIP-INFER-0104 — Backend not allowed

Status: `measure_backend` records one throughput mode the reference refused for a cold micro fixture. It returns a `knolo.infer.backend-report`. The surface is `run` or `measure`. The requested mode is `throughput`. The code is `BACKEND_NOT_ALLOWED` and it is not retryable. The backend is not selected. The forward does not run and no receipt is stored. The report's own execution mode stays `isolated-replay` or `pinned`. It does not select a backend. `measure_receipt_required` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-backend, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `surface` | `run` or `measure` |
| `requestedMode` | `throughput` |
| `code` | `BACKEND_NOT_ALLOWED` |
| `retryable` | `false` |
| `backendSelected` | `false` |
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

`kind` is `knolo.infer.backend-report` and `version` is `1`. The contract count is ninety-eight. `infer-backend` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the backend extensions are empty. A surface other than `run` or `measure` says the field has an unsupported value. A requested mode other than `throughput` says the refused mode is throughput. A code other than `BACKEND_NOT_ALLOWED` says a refused backend is BACKEND_NOT_ALLOWED. `retryable` true says a refused backend is not retryable. `backendSelected` true says the backend is not selected. `forwardRan` true says a refused backend does not run the forward. `receiptStored` true says a refused backend stores no receipt.

## Measurement

`measure_backend` takes the placement plan and one observation. It returns the report. It does not select the backend and it does not run the forward.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the backend report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the backend report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the backend report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says backend concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says backend run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says backend warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says backend request count is one.
12. The report checks in the Report section, in the order written there.

`verify_backend` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the backend validation did not match.

## Files

`write_backend_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the backend directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the backend path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the backend output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_backend`. Receipt signing has not started. A canonical-CBOR record has not started. A contract-invalid record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
