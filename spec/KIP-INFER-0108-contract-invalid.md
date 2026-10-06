# KIP-INFER-0108 — Contract invalid

Status: `measure_contract_invalid` records one contract field the decoder refused for a cold micro fixture. It returns a `knolo.infer.contract-invalid-report`. The reason is `missing`, `type`, `unknown`, or `value`. The code is `CONTRACT_INVALID` and it is not retryable. A missing field carries no bytes, is not present, and does not accept the type. A type refusal, an unknown field, and a value refusal name 1 through 80 bytes. Exactly 80 is recorded. A type refusal and an unknown field are present and do not accept the type. A value refusal is present and accepted the type. `valueAccepted` and `decoded` stay false. The forward does not run and no receipt is stored. It does not accept the document. The field name is not a field. `measure_canonical_cbor` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

A type, unknown, or value reason above 80 bytes is `CONTRACT_INVALID` and issues no report. A stored report above that cap is `CONTRACT_INVALID`.

## Report

The report is a versioned contract. Its identity root is `H(infer-contract-invalid, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `missing`, `type`, `unknown`, or `value` |
| `fieldBytes` | `0` for `missing`; `1` through `80` otherwise |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `fieldPresent` | `false` for `missing`; `true` otherwise |
| `typeAccepted` | `true` only for `value` |
| `valueAccepted` | `false` |
| `decoded` | `false` |
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

`kind` is `knolo.infer.contract-invalid-report` and `version` is `1`. The contract count is one hundred three. `infer-contract-invalid` is the report domain. The field name is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the contract-invalid extensions are empty. A code other than `CONTRACT_INVALID` says an invalid contract is CONTRACT_INVALID. `retryable` true says an invalid contract is not retryable. `decoded` true says a contract refusal does not decode the document. `valueAccepted` true says a contract refusal does not accept the value. `forwardRan` true says a contract refusal does not run the forward. `receiptStored` true says a contract refusal stores no receipt. A missing field with bytes says a missing field carries no bytes. A missing field that is present says a missing field is not present. A missing field that accepts the type says a missing field does not accept the type. A type, unknown, or value refusal with no bytes says that refusal names the field. A stored length above 80 says field exceeds the record cap. A type refusal that is absent says a type refusal is present. A type refusal that accepts the type says a type refusal does not accept the type. An unknown field that is absent says an unknown field is present. An unknown field that accepts the type says an unknown field does not accept the type. A value refusal that is absent says a value refusal is present. A value refusal that does not accept the type says a value refusal accepted the type.

## Measurement

`measure_contract_invalid` takes the placement plan and one observation. It returns the report. It does not accept the document.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the contract-invalid report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the contract-invalid report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the contract-invalid report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says contract-invalid concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says contract-invalid run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says contract-invalid warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says contract-invalid request count is one.
12. A `type`, `unknown`, or `value` reason above 80 bytes: `CONTRACT_INVALID`, and the message says field exceeds the record cap.
13. The report checks in the Report section, in the order written there.

`verify_contract_invalid` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the contract-invalid validation did not match.

## Files

`write_contract_invalid_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the contract-invalid directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the contract-invalid path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the contract-invalid output already exists.

The report is created with `create_new`, written, and `fsync`ed. The field name is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_contract_invalid`. Receipt verification has not started. A speculative-decoding record has not started. A CUDA-graph record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
