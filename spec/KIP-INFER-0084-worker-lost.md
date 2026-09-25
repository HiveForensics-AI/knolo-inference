# KIP-INFER-0084 — Worker lost

Status: `measure_worker_lost` records one ready worker that exited during a request for a cold micro fixture. It returns a `knolo.infer.worker-lost-report`. The code is `WORKER_LOST` and it is retryable. No receipt is stored. The listener stays up and the supervisor stays up. The HTTP status is 503. The open journal is sealed, and the exit counts a restart. It does not kill a process. `measure_replay_output` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-worker-lost, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `requestId` | the in-flight request token |
| `code` | `WORKER_LOST` |
| `retryable` | `true` |
| `receiptStored` | `false` |
| `listenerUp` | `true` |
| `supervisorExited` | `false` |
| `httpStatus` | `503` |
| `journalSealed` | `true` |
| `restartCounted` | `true` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.worker-lost-report` and `version` is `1`. The contract count is seventy-eight. `infer-worker-lost` is the report domain. A path is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the worker-lost extensions are empty. A request id that is empty, longer than 64 bytes, or outside ASCII letters, digits, and `-` says the request id is a token. A code other than `WORKER_LOST` says a lost worker is WORKER_LOST. `retryable` false says a lost worker is retryable. `receiptStored` true says a lost worker stores no receipt. `listenerUp` false says the listener stays up. `supervisorExited` true says a lost worker does not exit the supervisor. An HTTP status other than 503 says a lost worker is HTTP 503. `journalSealed` false says a lost worker seals the open journal. `restartCounted` false says a lost worker counts a restart.

## Measurement

`measure_worker_lost` takes the placement plan and one observation. It returns the report. It does not kill a process.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the worker-lost report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the worker-lost report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the worker-lost report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says worker-lost concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says worker-lost run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says worker-lost warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says worker-lost request count is one.
12. The report checks in the Report section, in the order written there.

`verify_worker_lost` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the worker-lost validation did not match.

## Files

`write_worker_lost_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the worker-lost directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the worker-lost path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the worker-lost output already exists.

The report is created with `create_new`, written, and `fsync`ed. No process is killed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_worker_lost`. The draining record is KIP-INFER-0085. The scalar reduction has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
