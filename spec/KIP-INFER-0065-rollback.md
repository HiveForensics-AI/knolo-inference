# KIP-INFER-0065 — Rollback record

Status: `measure_rollback` records one rollback of the infer-local pin for a cold micro fixture. It returns a `knolo.infer.rollback-report`. The previous model image differs from the incoming model image, and the previous artifact differs from the incoming artifact. The reason is `pin-mismatch`, `failed-load`, or `operator`. `lockfileMutated` and `coreLockTouched` stay false. It does not rewrite `knolo.infer.lock.json` and it does not open a Core lockfile. `measure_curve` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Pins

Each image root and each artifact root differs from the engine build root. The incoming image differs from the previous image. The incoming artifact differs from the previous artifact.

## Report

The report is a versioned contract. Its identity root is `H(infer-rollback, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `previousImageRoot` | root of the pin being replaced |
| `incomingImageRoot` | root of the pin being restored |
| `previousArtifactRoot` | root of the artifact being replaced |
| `incomingArtifactRoot` | root of the artifact being restored |
| `reason` | `pin-mismatch`, `failed-load`, or `operator` |
| `lockfileMutated` | `false` |
| `coreLockTouched` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.rollback-report` and `version` is `1`. The contract count is sixty. `infer-rollback` is the report domain. A path is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the rollback extensions are empty. A previous image equal to the engine build says the previous image repeats the engine build. An incoming image equal to the engine build says the incoming image repeats the engine build. An incoming image equal to the previous image says the incoming image repeats the previous image. A previous artifact equal to the engine build says the previous artifact repeats the engine build. An incoming artifact equal to the engine build says the incoming artifact repeats the engine build. An incoming artifact equal to the previous artifact says the incoming artifact repeats the previous artifact. `lockfileMutated` true says a rollback record leaves the lockfile unchanged. `coreLockTouched` true says a rollback record leaves the core lockfile unchanged. Any other reason says the field has an unsupported value.

## Measurement

`measure_rollback` takes the placement plan and one observation. It returns the report. It does not open a lockfile.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the rollback report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the rollback report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the rollback report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says rollback concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says rollback run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says rollback warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says rollback request count is one.
12. The report checks in the Report section, in the order written there.

`verify_rollback` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the rollback validation did not match.

## Files

`write_rollback_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the rollback directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the rollback path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the rollback output already exists.

The report is created with `create_new`, written, and `fsync`ed. The lockfile is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_rollback`. The client disconnect record is KIP-INFER-0066. The base-point record is KIP-INFER-0068. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
