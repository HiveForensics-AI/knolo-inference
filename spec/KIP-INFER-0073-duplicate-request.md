# KIP-INFER-0073 — Duplicate request

Status: `measure_duplicate` records one duplicate request id for a cold micro fixture. It returns a `knolo.infer.duplicate-report`. The code is `CONTRACT_INVALID` and it is not retryable. The second copy does not start, opens no journal, and stores no receipt. The admitted request stays. The listener stays up. The occupant root differs from the engine build and the placement. It does not admit the second copy. `measure_point` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Request

The request id is 1 through 64 characters of ASCII letters, digits, and hyphens. Prompt text is not a field.

## Report

The report is a versioned contract. Its identity root is `H(infer-duplicate, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `requestId` | a request token |
| `occupantRoot` | root of the admitted request |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `duplicateStarted` | `false` |
| `occupantKept` | `true` |
| `secondJournal` | `false` |
| `listenerUp` | `true` |
| `receiptStored` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.duplicate-report` and `version` is `1`. The contract count is sixty-eight. `infer-duplicate` is the report domain. Prompt text is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the duplicate extensions are empty. An empty or oversized request id says the field is empty or outside its bounds. A request id with any other character says the request id is a token. An occupant root equal to the engine build says the occupant repeats the engine build. An occupant root equal to the placement says the occupant repeats the placement. A code other than `CONTRACT_INVALID` says a duplicate request id is CONTRACT_INVALID. `retryable` true says a duplicate request id is not retryable. `duplicateStarted` true says a duplicate request does not start. `occupantKept` false says the admitted request stays. `secondJournal` true says a duplicate request opens no journal. `listenerUp` false says the listener stays up. `receiptStored` true says a duplicate request stores no receipt.

## Measurement

`measure_duplicate` takes the placement plan and one observation. It returns the report. It does not admit the second copy and it does not open a journal.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the duplicate report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the duplicate report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the duplicate report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says duplicate concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says duplicate run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says duplicate warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says duplicate request count is one.
12. The report checks in the Report section, in the order written there.

`verify_duplicate` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the duplicate validation did not match.

## Files

`write_duplicate_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the duplicate directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the duplicate path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the duplicate output already exists.

The report is created with `create_new`, written, and `fsync`ed. No journal is opened.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_duplicate`. The concurrent load record is KIP-INFER-0074. Point comparison has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
