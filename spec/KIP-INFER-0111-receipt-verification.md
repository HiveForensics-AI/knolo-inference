# KIP-INFER-0111 — Receipt verification

Status: `measure_receipt_verify` records one host-supplied Ed25519 receipt verification for a cold micro fixture. It returns a `knolo.infer.receipt-verify-report`. `verified` and `rejected` say the host checked the receipt. `unsigned-local` does not. A public key is 32 bytes, a signature is 64 bytes, and a scalar is 32 bytes when the host checked them. Key bytes are not a field. `domainSeparated` stays false. It does not verify the receipt and does not separate the domain. `measure_receipt_sign` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-receipt-verify, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `releaseRoot` | root distinct from the engine build |
| `messageRoot` | root distinct from the engine build and the release |
| `verifyStatus` | `verified`, `rejected`, or `unsigned-local` |
| `receiptVerified` | `true` for `verified` and `rejected`; `false` for `unsigned-local` |
| `domainSeparated` | `false` |
| `publicKeyBytes` | `32` when checked; `0` when unsigned |
| `signatureBytes` | `64` when checked; `0` when unsigned |
| `scalarBytes` | `32` when checked; `0` when unsigned |
| `keyMaterialPresent` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `verified` for `verified`; `recorded` otherwise |
| `extensions` | an empty map |

`kind` is `knolo.infer.receipt-verify-report` and `version` is `1`. The contract count is one hundred eight. `infer-receipt-verify` is the report domain. Key bytes and signature bytes are not fields.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the receipt-verify extensions are empty. A release root equal to the engine build says the release repeats the engine build. A message root equal to the engine build says the verified message repeats the engine build. A message root equal to the release says the verified message repeats the release. `keyMaterialPresent` true says key material stays in host storage. `domainSeparated` true says the domain stays unseparated. An unsigned release with `receiptVerified` true says an unsigned release verifies the receipt. A verified status with `receiptVerified` false says a verified receipt records the check. A rejected status with `receiptVerified` false says a rejected receipt records the check. A verified status stores `validationResult` `verified`, and any other result says a verified receipt is verified. An unsigned or rejected status stores `recorded`, and any other result says only a verified receipt is verified.

## Measurement

`measure_receipt_verify` takes the placement plan and one observation. It returns the report. It does not verify a receipt.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the receipt-verify report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the receipt-verify report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the receipt-verify report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says receipt-verify concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says receipt-verify run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says receipt-verify warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says receipt-verify request count is one.
12. The report checks in the Report section, in the order written there.

`verify_receipt_verify` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the receipt-verify validation did not match.

## Files

`write_receipt_verify_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the receipt-verify directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the receipt-verify path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the receipt-verify output already exists.

The report is created with `create_new`, written, and `fsync`ed. Key bytes are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_receipt_verify`. Domain separation has not started. An attention-approximation record has not started. A mixture-of-experts record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
