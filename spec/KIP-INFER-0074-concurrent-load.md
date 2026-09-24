# KIP-INFER-0074 — Concurrent load

Status: `measure_concurrent_load` records one concurrent load for a cold micro fixture. It returns a `knolo.infer.concurrent-load-report`. The phase is `ready`, `unloading`, or `down`. A ready worker is left running. An unloading worker still has admitted work and is not replaced. A down worker is started once. A second worker is not started. A live worker is not replaced. The restart counter stays zero. The listener stays up. Inflight is 0 through 16. It does not start a worker. `measure_duplicate` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Phase

`ready` records lifecycle `serving`, `workerStarted` false, and inflight 0 through 16. `unloading` records lifecycle `unloading`, `workerStarted` false, and inflight 1 through 16. `down` records lifecycle `serving`, `workerStarted` true, and inflight 0.

## Report

The report is a versioned contract. Its identity root is `H(infer-concurrent-load, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `phase` | `ready`, `unloading`, or `down` |
| `lifecycle` | `serving` for `ready` and `down`, otherwise `unloading` |
| `inflight` | `0` through `16`, and `0` for `down` |
| `workerStarted` | `true` only for `down` |
| `secondWorker` | `false` |
| `workerReplaced` | `false` |
| `restartCount` | `0` |
| `listenerUp` | `true` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.concurrent-load-report` and `version` is `1`. The contract count is sixty-eight. `infer-concurrent-load` is the report domain. A pid is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the concurrent-load extensions are empty. A phase outside the three names says the field has an unsupported value. `secondWorker` true says a concurrent load starts one worker. `workerReplaced` true says a concurrent load does not replace a live worker. A restart count other than zero says a concurrent load does not count a restart. `listenerUp` false says the listener stays up. Inflight above 16 says a concurrent load admits at most 16 requests. A ready lifecycle other than `serving` says a ready load stays serving. A ready load with `workerStarted` true says a ready worker is left running. An unloading lifecycle other than `unloading` says an unloading load stays unloading. An unloading load with `workerStarted` true says an unloading load does not start a worker. An unloading load with inflight zero says an unloading load still has admitted work. A down lifecycle other than `serving` says a down load returns to serving. A down load with `workerStarted` false says a down worker is started. A down load with inflight other than zero says a down load has no admitted work.

## Measurement

`measure_concurrent_load` takes the placement plan and one observation. It returns the report. It does not start a worker and it does not bind a socket.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the concurrent-load report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the concurrent-load report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the concurrent-load report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says concurrent-load concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says concurrent-load run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says concurrent-load warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says concurrent-load request count is one.
12. The report checks in the Report section, in the order written there.

`verify_concurrent_load` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the concurrent-load validation did not match.

## Files

`write_concurrent_load_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the concurrent-load directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the concurrent-load path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the concurrent-load output already exists.

The report is created with `create_new`, written, and `fsync`ed. No worker is started.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_concurrent_load`. The prefix-eviction record is KIP-INFER-0075. Point comparison has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
