# KIP-INFER-0083 — Replay output

Status: `measure_replay_output` records one replay whose output tokens did not match the original receipt for a cold micro fixture. It returns a `knolo.infer.replay-output-report`. The code is `REPLAY_OUTPUT_MISMATCH` and it is not retryable. The environment matched, so the forward is recorded as having run. The replay check is not stored. Assurance is `incomplete`. The candidate output root differs from the receipt root. Token ids are not a field. It does not run a model. `measure_replay_environment` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-replay-output, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `receiptRoot` | a root distinct from the engine build |
| `candidateOutputRoot` | a root distinct from the receipt |
| `code` | `REPLAY_OUTPUT_MISMATCH` |
| `retryable` | `false` |
| `checkStored` | `false` |
| `forwardRan` | `true` |
| `environmentMatched` | `true` |
| `assurance` | `incomplete` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.replay-output-report` and `version` is `1`. The contract count is seventy-eight. `infer-replay-output` is the report domain. Token ids are not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the replay-output extensions are empty. A receipt root equal to the engine build says the receipt repeats the engine build. A candidate output root equal to the receipt says the candidate output repeats the receipt. A code other than `REPLAY_OUTPUT_MISMATCH` says an output mismatch is REPLAY_OUTPUT_MISMATCH. `retryable` true says an output mismatch is not retryable. `checkStored` true says an output mismatch stores no replay check. `forwardRan` false says an output mismatch ran the forward. `environmentMatched` false says an output mismatch follows a matching environment. An assurance other than `incomplete` says an output mismatch is incomplete.

## Measurement

`measure_replay_output` takes the placement plan and one observation. It returns the report. It does not run the forward and it does not write token ids.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the replay-output report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the replay-output report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the replay-output report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says replay-output concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says replay-output run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says replay-output warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says replay-output request count is one.
12. The report checks in the Report section, in the order written there.

`verify_replay_output` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the replay-output validation did not match.

## Files

`write_replay_output_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the replay-output directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the replay-output path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the replay-output output already exists.

The report is created with `create_new`, written, and `fsync`ed. Token ids are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_replay_output`. The worker-lost record is KIP-INFER-0084. The scalar reduction has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
