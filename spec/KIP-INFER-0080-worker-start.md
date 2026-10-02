# KIP-INFER-0080 — Worker start

Status: `measure_worker_start` records one worker that did not become ready for a cold micro fixture. It returns a `knolo.infer.worker-start-report`. The failure is `missing-binary`, `not-ready`, or `socket`. The code is `WORKER_START_FAILED` and it is retryable. No receipt is stored. The worker stays down, and the restart counter stays at zero. A missing binary does not bind and does not spawn. A worker that is not ready was spawned, and the listener stays up. A socket failure does not spawn the worker, and the listener stays up. It does not spawn a process. `measure_timeout` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-worker-start, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `startFailure` | `missing-binary`, `not-ready`, or `socket` |
| `code` | `WORKER_START_FAILED` |
| `retryable` | `true` |
| `receiptStored` | `false` |
| `listenerUp` | `false` only for `missing-binary` |
| `processSpawned` | `true` only for `not-ready` |
| `workerReady` | `false` |
| `restartCount` | `0` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.worker-start-report` and `version` is `1`. The contract count is seventy-three. `infer-worker-start` is the report domain. A path is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the worker-start extensions are empty. A failure other than the three named failures says the field has an unsupported value. A code other than `WORKER_START_FAILED` says a start failure is WORKER_START_FAILED. `retryable` false says a start failure is retryable. `receiptStored` true says a start failure stores no receipt. `workerReady` true says a start failure leaves the worker down. A restart count other than zero says a start failure does not count a restart. A missing binary with `listenerUp` true says a missing worker binary does not bind. A missing binary with `processSpawned` true says a missing worker binary does not spawn. A worker that is not ready with `listenerUp` false says the listener stays up. A worker that is not ready with `processSpawned` false says a worker that is not ready was spawned. A socket failure with `listenerUp` false says the listener stays up. A socket failure with `processSpawned` true says a socket failure does not spawn the worker.

## Measurement

`measure_worker_start` takes the placement plan and one observation. It returns the report. It does not spawn a process and it does not bind a socket.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the worker-start report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the worker-start report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the worker-start report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says worker-start concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says worker-start run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says worker-start warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says worker-start request count is one.
12. The report checks in the Report section, in the order written there.

`verify_worker_start` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the worker-start validation did not match.

## Files

`write_worker_start_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the worker-start directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the worker-start path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the worker-start output already exists.

The report is created with `create_new`, written, and `fsync`ed. No process is spawned.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_worker_start`. The challenge hash has not started. A replay-mismatch record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
