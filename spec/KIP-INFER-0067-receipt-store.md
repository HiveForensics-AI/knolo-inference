# KIP-INFER-0067 — Receipt-store failure

Status: `measure_receipt_store` records one receipt-store failure for a cold micro fixture. It returns a `knolo.infer.receipt-store-report`. The code is `RECEIPT_PERSIST_FAILED` and it is retryable. `receiptStored` stays false. The journal event is `failed`. A partial receipt is `absent` or a root that differs from the engine build and the placement. It does not write a receipt file. `measure_disconnect` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Failure

The request id is 1 through 64 characters of ASCII letters, digits, and hyphens. Prompt text is not a field.

## Report

The report is a versioned contract. Its identity root is `H(infer-receipt-store, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `requestId` | a request token |
| `code` | `RECEIPT_PERSIST_FAILED` |
| `retryable` | `true` |
| `receiptStored` | `false` |
| `journalEvent` | `failed` |
| `partialReceipt` | `absent` or a `sha256-` root |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.receipt-store-report` and `version` is `1`. The contract count is sixty. `infer-receipt-store` is the report domain. Prompt text is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the receipt store extensions are empty. An empty or oversized request id says the field is empty or outside its bounds. A request id with any other character says the request id is a token. A code other than `RECEIPT_PERSIST_FAILED` says a receipt-store failure is RECEIPT_PERSIST_FAILED. `retryable` false says a receipt-store failure is retryable. `receiptStored` true says a receipt-store failure stores no receipt. A journal event other than `failed` says the journal seals failed. A partial receipt that is neither `absent` nor a root says a partial receipt is absent or a root. A partial receipt equal to the engine build says the partial receipt repeats the engine build. A partial receipt equal to the placement says the partial receipt repeats the placement.

## Measurement

`measure_receipt_store` takes the placement plan and one observation. It returns the report. It does not write a receipt file.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the receipt store report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the receipt store report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the receipt store report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says receipt store concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says receipt store run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says receipt store warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says receipt store request count is one.
12. The report checks in the Report section, in the order written there.

`verify_receipt_store` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the receipt store validation did not match.

## Files

`write_receipt_store_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the receipt store directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the receipt store path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the receipt store output already exists.

The report is created with `create_new`, written, and `fsync`ed. The inference receipt is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_receipt_store`. The base-point record is KIP-INFER-0068. The disk-full record is KIP-INFER-0069. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
