# KIP-INFER-0082 — Replay environment

Status: `measure_replay_environment` records one replay whose environment did not match the original receipt for a cold micro fixture. It returns a `knolo.infer.replay-environment-report`. The mismatched field is `prompt`, `sampler`, `model`, `engine`, or `placement`. The code is `REPLAY_ENVIRONMENT_MISMATCH` and it is not retryable. The replay check is not stored. The forward does not run, and output tokens are not compared. Assurance is `incomplete`. It does not open a receipt and does not run a model. `measure_challenge` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-replay-environment, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `receiptRoot` | a root distinct from the engine build and the placement |
| `mismatchedField` | `prompt`, `sampler`, `model`, `engine`, or `placement` |
| `code` | `REPLAY_ENVIRONMENT_MISMATCH` |
| `retryable` | `false` |
| `checkStored` | `false` |
| `forwardRan` | `false` |
| `outputCompared` | `false` |
| `assurance` | `incomplete` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.replay-environment-report` and `version` is `1`. The contract count is seventy-eight. `infer-replay-environment` is the report domain. Token ids are not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the replay-environment extensions are empty. A receipt root equal to the engine build says the receipt repeats the engine build. A receipt root equal to the placement says the receipt repeats the placement. A field other than the five named fields says the field has an unsupported value. A code other than `REPLAY_ENVIRONMENT_MISMATCH` says an environment mismatch is REPLAY_ENVIRONMENT_MISMATCH. `retryable` true says an environment mismatch is not retryable. `checkStored` true says an environment mismatch stores no replay check. `forwardRan` true says an environment mismatch does not run the forward. `outputCompared` true says an environment mismatch does not compare output tokens. An assurance other than `incomplete` says an environment mismatch is incomplete.

## Measurement

`measure_replay_environment` takes the placement plan and one observation. It returns the report. It does not open a receipt and it does not run the forward.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the replay-environment report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the replay-environment report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the replay-environment report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says replay-environment concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says replay-environment run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says replay-environment warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says replay-environment request count is one.
12. The report checks in the Report section, in the order written there.

`verify_replay_environment` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the replay-environment validation did not match.

## Files

`write_replay_environment_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the replay-environment directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the replay-environment path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the replay-environment output already exists.

The report is created with `create_new`, written, and `fsync`ed. The receipt is not opened.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_replay_environment`. The replay output record is KIP-INFER-0083. The scalar reduction has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
