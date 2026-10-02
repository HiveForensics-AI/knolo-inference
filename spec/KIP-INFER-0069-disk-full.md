# KIP-INFER-0069 — Disk full

Status: `measure_disk` records one full disk for a cold micro fixture. It returns a `knolo.infer.disk-report`. The store is `journal`, `receipt`, `trace`, or `lock`. Free bytes are zero. Needed bytes are 1 through 1 MiB. The target file stays unwritten and no other file is removed to make space. A `journal`, `receipt`, or `lock` failure is `RECEIPT_PERSIST_FAILED` and retryable, and it stores no receipt. A `trace` failure leaves the code `none`, is not retryable, and leaves the receipt stored. The listener stays up except for `lock`, which does not bind. It does not fill a disk and does not write the target file. `measure_base` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Space

The request id is 1 through 64 characters of ASCII letters, digits, and hyphens. A needed count above 1 MiB is `INSUFFICIENT_MEMORY` and issues no report. A stored report above that cap is `CONTRACT_INVALID`. Exactly 1 MiB is recorded. A needed count of zero is `CONTRACT_INVALID`.

## Report

The report is a versioned contract. Its identity root is `H(infer-disk, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `requestId` | a request token |
| `store` | `journal`, `receipt`, `trace`, or `lock` |
| `freeBytes` | `0` |
| `neededBytes` | `1` through `1048576` |
| `fileWritten` | `false` |
| `spaceReclaimed` | `false` |
| `listenerUp` | `false` only for `lock` |
| `code` | `RECEIPT_PERSIST_FAILED`, or `none` for `trace` |
| `retryable` | `true` for a durable store, otherwise `false` |
| `receiptStored` | `true` only for `trace` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.disk-report` and `version` is `1`. The contract count is sixty-four. `infer-disk` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the disk extensions are empty. An empty or oversized request id says the field is empty or outside its bounds. A request id with any other character says the request id is a token. A free byte count other than zero says a disk-full record has no free bytes. A needed count of zero says a disk-full record needs space. A stored needed count above 1 MiB says needed bytes exceed 1 MiB. `fileWritten` true says a disk-full record writes no file. `spaceReclaimed` true says a disk-full record reclaims no space. A durable store whose code is not `RECEIPT_PERSIST_FAILED` says a durable store failure is RECEIPT_PERSIST_FAILED. A durable store that is not retryable says a durable store failure is retryable. A durable store with `receiptStored` true says a full durable store stores no receipt. A trace whose code is not `none` says a trace write does not fail the completion. A retryable trace says a trace write is not a retryable failure. A trace with `receiptStored` false says a full trace leaves the receipt stored. A lock with `listenerUp` true says a lock store that is full does not bind. Any other store with `listenerUp` false says the listener stays up.

## Measurement

`measure_disk` takes the placement plan and one observation. It returns the report. It does not write the target file and it does not delete another file.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the disk report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the disk report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the disk report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says disk concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says disk run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says disk warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says disk request count is one.
12. `neededBytes` is above 1 MiB: `INSUFFICIENT_MEMORY`, and the message says needed bytes exceed 1 MiB.
13. The report checks in the Report section, in the order written there.

`verify_disk` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the disk validation did not match.

## Files

`write_disk_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the disk directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the disk path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the disk output already exists.

The report is created with `create_new`, written, and `fsync`ed. The target file of the full store is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_disk`. The queued unload record is KIP-INFER-0070. Point addition has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
