# KIP-INFER-0095 — Insufficient memory

Status: `measure_memory_refusal` records one cold micro fixture that needed more host memory than the bound allows. It returns a `knolo.infer.memory-refusal-report`. The reason is `pool`, `output`, or `admission`. Free bytes are zero. Needed bytes are 1 through 64 MiB. Exactly 64 MiB is recorded. The code is `INSUFFICIENT_MEMORY` and it is retryable. A full pool has no free page and does not hold the queue. An output cap is not a full pool and does not hold the queue. An admission cap holds the queue and is not a full pool. The listener stays up. No page is allocated, the forward does not run, and no receipt is stored. It does not allocate a page. `measure_placement_refusal` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

A needed count above 64 MiB is `INSUFFICIENT_MEMORY` and issues no report. A stored report above that cap is `CONTRACT_INVALID`. A needed count of zero is `CONTRACT_INVALID`.

## Report

The report is a versioned contract. Its identity root is `H(infer-memory-refusal, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `pool`, `output`, or `admission` |
| `freeBytes` | `0` |
| `neededBytes` | `1` through 64 MiB |
| `code` | `INSUFFICIENT_MEMORY` |
| `retryable` | `true` |
| `residentFull` | `true` for `pool`; `false` otherwise |
| `queueHeld` | `true` for `admission`; `false` otherwise |
| `allocated` | `false` |
| `forwardRan` | `false` |
| `receiptStored` | `false` |
| `listenerUp` | `true` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.memory-refusal-report` and `version` is `1`. The contract count is eighty-eight. `infer-memory-refusal` is the report domain. A path is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the memory-refusal extensions are empty. A free byte count other than zero says an insufficient memory record has no free bytes. A needed count of zero says an insufficient memory record needs bytes. A stored needed count above 64 MiB says needed bytes exceed 64 MiB. A code other than `INSUFFICIENT_MEMORY` says an insufficient memory record is INSUFFICIENT_MEMORY. `retryable` false says an insufficient memory record is retryable. `forwardRan` true says an insufficient memory record does not run the forward. `receiptStored` true says an insufficient memory record stores no receipt. `listenerUp` false says the listener stays up. `allocated` true says an insufficient memory record does not allocate. `residentFull` false on `pool` says a full pool has no free page. `queueHeld` true on `pool` says a full pool does not hold the queue. `residentFull` true on `output` says an output cap is not a full pool. `queueHeld` true on `output` says an output cap does not hold the queue. `residentFull` true on `admission` says an admission cap is not a full pool. `queueHeld` false on `admission` says an admission cap holds the queue.

## Measurement

`measure_memory_refusal` takes the placement plan and one observation. It returns the report. It does not allocate a page and it does not read a weight file.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the memory-refusal report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the memory-refusal report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the memory-refusal report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says memory-refusal concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says memory-refusal run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says memory-refusal warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says memory-refusal request count is one.
12. `neededBytes` is above 64 MiB: `INSUFFICIENT_MEMORY`, and the message says needed bytes exceed 64 MiB.
13. The report checks in the Report section, in the order written there.

`verify_memory_refusal` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the memory-refusal validation did not match.

## Files

`write_memory_refusal_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the memory-refusal directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the memory-refusal path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the memory-refusal output already exists.

The report is created with `create_new`, written, and `fsync`ed. No page is allocated.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_memory_refusal`. The signature check has not started. A context-limit record has not started. A prompt-compilation record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
