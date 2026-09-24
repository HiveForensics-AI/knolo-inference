# KIP-INFER-0071 — Daemon restart

Status: `measure_daemon_restart` records one daemon restart for a cold micro fixture. It returns a `knolo.infer.restart-report`. The reason is `stale-lock`, `clean-exit`, or `killed`. The previous owner root and the incoming owner root differ, and neither repeats the engine build or the placement. `stale-lock` and `killed` replace the lock and seal open journals. `clean-exit` does not replace a held lock and has no open journal. The listener comes back. The worker restart counter on the new process is zero. The record does not spawn a process and it does not start a second worker. `measure_queued_unload` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Owner

An owner root is a digest. A pid and a kernel start time are not fields.

## Report

The report is a versioned contract. Its identity root is `H(infer-restart, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `previousOwnerRoot` | root of the previous owner record |
| `incomingOwnerRoot` | root of the incoming owner record |
| `reason` | `stale-lock`, `clean-exit`, or `killed` |
| `lockReplaced` | `true` for `stale-lock` and `killed` |
| `journalsSealed` | `true` for `stale-lock` and `killed` |
| `listenerUp` | `true` |
| `workerRestartCount` | `0` |
| `processSpawned` | `false` |
| `secondWorker` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.restart-report` and `version` is `1`. The contract count is sixty-four. `infer-restart` is the report domain. A pid is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the restart extensions are empty. A previous owner equal to the engine build says the previous owner repeats the engine build. An incoming owner equal to the engine build says the incoming owner repeats the engine build. An incoming owner equal to the previous owner says the incoming owner repeats the previous owner. A previous owner equal to the placement says the previous owner repeats the placement. An incoming owner equal to the placement says the incoming owner repeats the placement. `processSpawned` true says a restart record does not spawn a process. `secondWorker` true says a restart record starts one worker. `listenerUp` false says the listener comes back. A restart count other than zero says a new process starts the restart counter at zero. A stale or killed owner with `lockReplaced` false says a stale owner is replaced. A stale or killed owner with `journalsSealed` false says a stale owner seals open journals. A clean exit with `lockReplaced` true says a clean exit does not replace a held lock. A clean exit with `journalsSealed` true says a clean exit has no open journal.

## Measurement

`measure_daemon_restart` takes the placement plan and one observation. It returns the report. It does not spawn a process and it does not read `/proc`.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the restart report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the restart report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the restart report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says restart concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says restart run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says restart warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says restart request count is one.
12. The report checks in the Report section, in the order written there.

`verify_daemon_restart` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the restart validation did not match.

## Files

`write_restart_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the restart directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the restart path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the restart output already exists.

The report is created with `create_new`, written, and `fsync`ed. No process is spawned.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_daemon_restart`. A duplicate request id has not started. Point addition has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
