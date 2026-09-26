# KIP-INFER-0107 — Canonical CBOR

Status: `measure_canonical_cbor` records one canonical header the decoder refused for a cold micro fixture. It returns a `knolo.infer.canonical-cbor-report`. The reason is `indefinite`, `tag`, `order`, or `shortest`. The code is `CANONICAL_CBOR_INVALID` and it is not retryable. An indefinite refusal is additional information 31 on major 0 through 5 or 7, and it does not read the argument. A tag refusal is major 6, or major 7 with an additional information other than 20, 21, 22, or 31, and it does not read the argument. An order refusal is major 5 with additional information 0 through 23, and it read the map length. A shortest refusal is major 0 through 5 with additional information 24 through 27, and it read the argument. `valueAccepted`, `keysOrdered`, and `reencoded` stay false. The forward does not run and no receipt is stored. It does not decode the document. `measure_receipt_sign` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

A major above 7 is `CANONICAL_CBOR_INVALID` and issues no report. Additional information above 31 is `CANONICAL_CBOR_INVALID` and issues no report. An order reason above 23 is `CANONICAL_CBOR_INVALID` and issues no report. A stored report above any of those caps is `CONTRACT_INVALID`.

## Report

The report is a versioned contract. Its identity root is `H(infer-canonical-cbor, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `indefinite`, `tag`, `order`, or `shortest` |
| `major` | `0` through `7` |
| `additionalInfo` | `0` through `31` |
| `code` | `CANONICAL_CBOR_INVALID` |
| `retryable` | `false` |
| `argumentRead` | `false` for `indefinite` and `tag`; `true` for `order` and `shortest` |
| `valueAccepted` | `false` |
| `keysOrdered` | `false` |
| `reencoded` | `false` |
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

`kind` is `knolo.infer.canonical-cbor-report` and `version` is `1`. The contract count is one hundred three. `infer-canonical-cbor` is the report domain. The document bytes are not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the canonical-cbor extensions are empty. A code other than `CANONICAL_CBOR_INVALID` says an invalid canonical document is CANONICAL_CBOR_INVALID. `retryable` true says an invalid canonical document is not retryable. `valueAccepted` true says a canonical refusal does not accept the value. `keysOrdered` true says a canonical refusal does not order the keys. `reencoded` true says a canonical refusal does not re-encode the document. `forwardRan` true says a canonical refusal does not run the forward. `receiptStored` true says a canonical refusal stores no receipt. A stored major above 7 says CBOR major exceeds the record cap. A stored additional information above 31 says additional information exceeds the record cap. An indefinite refusal whose additional information is not 31 says an indefinite refusal is additional information 31. An indefinite refusal on major 6 says an indefinite refusal is not a tag. An indefinite or tag refusal with `argumentRead` true says that refusal does not read the argument. A tag refusal on any other header says a tag refusal is a tag or a rejected simple value. An order refusal whose major is not 5 says an order refusal is a map. A stored order length above 23 says map length exceeds the record cap. An order refusal with `argumentRead` false says an order refusal read the map length. A shortest refusal on major 6 or 7 says a shortest refusal is an integer or a length. A shortest refusal outside additional information 24 through 27 says a shortest refusal is a wide argument. A shortest refusal with `argumentRead` false says a shortest refusal read the argument.

## Measurement

`measure_canonical_cbor` takes the placement plan and one observation. It returns the report. It does not decode the document.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the canonical-cbor report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the canonical-cbor report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the canonical-cbor report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says canonical-cbor concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says canonical-cbor run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says canonical-cbor warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says canonical-cbor request count is one.
12. `major` is above 7: `CANONICAL_CBOR_INVALID`, and the message says CBOR major exceeds the record cap.
13. `additionalInfo` is above 31: `CANONICAL_CBOR_INVALID`, and the message says additional information exceeds the record cap.
14. An `order` reason above 23: `CANONICAL_CBOR_INVALID`, and the message says map length exceeds the record cap.
15. The report checks in the Report section, in the order written there.

`verify_canonical_cbor` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the canonical-cbor validation did not match.

## Files

`write_canonical_cbor_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the canonical-cbor directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the canonical-cbor path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the canonical-cbor output already exists.

The report is created with `create_new`, written, and `fsync`ed. The refused document is not a field.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_canonical_cbor`. Receipt verification has not started. A speculative-decoding record has not started. A CUDA-graph record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
