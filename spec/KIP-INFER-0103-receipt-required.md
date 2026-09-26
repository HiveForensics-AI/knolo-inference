# KIP-INFER-0103 — Receipt required

Status: `measure_receipt_required` records one receipt the verifier could not use for a cold micro fixture. It returns a `knolo.infer.receipt-required-report`. The reason is `file`, `request`, or `journal`. The code is `RECEIPT_REQUIRED` and it is not retryable. A missing file is not read, has no request id, does not open the journal, and is HTTP 404. A missing request id was read from the receipt and does not open the journal. An empty journal was opened and names a request id. Those two verify-path refusals have HTTP status 0. The event count is 0. The listener stays up. The forward does not run and no receipt is stored. It does not read a receipt file and it does not open a journal. `measure_artifact_missing` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-receipt-required, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `receiptRoot` | root distinct from the engine build and the placement |
| `reason` | `file`, `request`, or `journal` |
| `code` | `RECEIPT_REQUIRED` |
| `retryable` | `false` |
| `httpStatus` | `404` for `file`; `0` for `request` and `journal` |
| `receiptRead` | `false` for `file`; `true` otherwise |
| `requestBound` | `true` for `journal`; `false` otherwise |
| `journalOpened` | `true` for `journal`; `false` otherwise |
| `eventCount` | `0` |
| `listenerUp` | `true` |
| `forwardRan` | `false` |
| `receiptStored` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.receipt-required-report` and `version` is `1`. The contract count is ninety-eight. `infer-receipt-required` is the report domain. Receipt bytes are not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the receipt-required extensions are empty. A receipt root equal to the engine build says the receipt repeats the engine build. A receipt root equal to the placement says the receipt repeats the placement. A code other than `RECEIPT_REQUIRED` says a missing receipt is RECEIPT_REQUIRED. `retryable` true says a missing receipt is not retryable. A non-zero event count says a required receipt has no journal events. `listenerUp` false says the listener stays up. `forwardRan` true says a missing receipt does not run the forward. `receiptStored` true says a missing receipt stores no receipt. A file reason that was read says a missing receipt file is not read. A file reason with an HTTP status other than 404 says a missing receipt file is HTTP 404. A request or journal reason with a non-zero HTTP status says a verify-path refusal has no HTTP status. A request reason that was not read says a missing request id was read. A journal reason that was not opened says an empty journal was opened.

## Measurement

`measure_receipt_required` takes the placement plan and one observation. It returns the report. It does not read a receipt file and it does not open a journal.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the receipt-required report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the receipt-required report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the receipt-required report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says receipt-required concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says receipt-required run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says receipt-required warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says receipt-required request count is one.
12. The report checks in the Report section, in the order written there.

`verify_receipt_required` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the receipt-required validation did not match.

## Files

`write_receipt_required_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the receipt-required directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the receipt-required path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the receipt-required output already exists.

The report is created with `create_new`, written, and `fsync`ed. No receipt bytes are read.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_receipt_required`. Receipt signing has not started. A canonical-CBOR record has not started. A contract-invalid record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
