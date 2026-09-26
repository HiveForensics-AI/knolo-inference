# KIP-INFER-0070 — Queued unload

Status: `measure_queued_unload` records one unload while requests are still queued for a cold micro fixture. It returns a `knolo.infer.unload-report`. The queue holds 1 through 16 requests. Admitted work is 0 through 16. Queued work does not start. Admitted work finishes. The code is `SERVICE_UNLOADED` and it is not retryable. The listener stays up and the model is not loaded again. An empty admission is `unloaded` and the worker has exited. Admitted work keeps the lifecycle `unloading` and the worker stays until that work finishes. It does not unload a worker. `measure_disk` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Queue

The request id is 1 through 64 characters of ASCII letters, digits, and hyphens. A queue of zero issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-unload, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `requestId` | a request token |
| `queuedRequests` | `1` through `16` |
| `admittedRequests` | `0` through `16` |
| `queuedStarted` | `false` |
| `admittedFinished` | `true` |
| `lifecycle` | `unloaded` when nothing is admitted, otherwise `unloading` |
| `workerExited` | `true` only when nothing is admitted |
| `code` | `SERVICE_UNLOADED` |
| `retryable` | `false` |
| `listenerUp` | `true` |
| `modelReloaded` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.unload-report` and `version` is `1`. The contract count is sixty-four. `infer-unload` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the unload extensions are empty. An empty or oversized request id says the field is empty or outside its bounds. A request id with any other character says the request id is a token. A queue of zero says an unload under queue has a queued request. A queue above 16 says the unload queue is at most 16. Admitted work above 16 says admitted work is at most 16. `queuedStarted` true says queued work does not start. `admittedFinished` false says admitted work finishes. A code other than `SERVICE_UNLOADED` says a queued unload is SERVICE_UNLOADED. `retryable` true says a queued unload is not retryable. `listenerUp` false says the listener stays up. `modelReloaded` true says a queued unload does not load the model. An empty admission whose lifecycle is not `unloaded` says an empty admission is unloaded. An empty admission whose worker has not exited says an empty admission exits the worker. Admitted work whose lifecycle is not `unloading` says admitted work keeps the lifecycle unloading. Admitted work whose worker has exited says the worker stays until admitted work finishes.

## Measurement

`measure_queued_unload` takes the placement plan and one observation. It returns the report. It does not unload a worker and it does not start a second worker.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the unload report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the unload report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the unload report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says unload concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says unload run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says unload warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says unload request count is one.
12. The report checks in the Report section, in the order written there.

`verify_queued_unload` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the unload validation did not match.

## Files

`write_unload_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the unload directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the unload path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the unload output already exists.

The report is created with `create_new`, written, and `fsync`ed. No worker is stopped.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_queued_unload`. The daemon restart record is KIP-INFER-0071. Point addition has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
