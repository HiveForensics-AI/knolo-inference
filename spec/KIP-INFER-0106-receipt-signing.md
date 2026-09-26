# KIP-INFER-0106 — Receipt signing

Status: `measure_receipt_sign` records one host-supplied Ed25519 receipt signature for a cold micro fixture. It returns a `knolo.infer.receipt-sign-report`. `signed` and `rejected` say the host signed the receipt. `unsigned-local` does not. A public key is 32 bytes, a signature is 64 bytes, and a scalar is 32 bytes when the host signed them. Key bytes are not a field. `receiptVerified` stays false. It does not sign the receipt and does not verify it. `measure_cofactor` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-receipt-sign, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `releaseRoot` | root distinct from the engine build |
| `messageRoot` | root distinct from the engine build and the release |
| `signStatus` | `signed`, `rejected`, or `unsigned-local` |
| `receiptSigned` | `true` for `signed` and `rejected`; `false` for `unsigned-local` |
| `receiptVerified` | `false` |
| `publicKeyBytes` | `32` when signed; `0` when unsigned |
| `signatureBytes` | `64` when signed; `0` when unsigned |
| `scalarBytes` | `32` when signed; `0` when unsigned |
| `keyMaterialPresent` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `verified` for `signed`; `recorded` otherwise |
| `extensions` | an empty map |

`kind` is `knolo.infer.receipt-sign-report` and `version` is `1`. The contract count is one hundred three. `infer-receipt-sign` is the report domain. Key bytes and signature bytes are not fields.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the receipt-sign extensions are empty. A release root equal to the engine build says the release repeats the engine build. A message root equal to the engine build says the signed message repeats the engine build. A message root equal to the release says the signed message repeats the release. `keyMaterialPresent` true says key material stays in host storage. `receiptVerified` true says the receipt stays unverified. An unsigned release that carries bytes says an unsigned release carries a public key, signature bytes, or a scalar. An unsigned release with `receiptSigned` true says an unsigned release signs the receipt. A signed or rejected status whose counts are not 32, 64, and 32 says ed25519 public keys are 32 bytes, ed25519 signatures are 64 bytes, or ed25519 scalars are 32 bytes. A signed status with `receiptSigned` false says a signed receipt records the signature. A rejected status with `receiptSigned` false says a rejected receipt records the signature. `verified` on any status other than `signed` says only a signed receipt is verified. A signed status that is not `verified` says a signed receipt is verified.

## Measurement

`measure_receipt_sign` takes the placement plan and one observation. It returns the report. It does not read key bytes and it does not verify a receipt.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the receipt-sign report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the receipt-sign report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the receipt-sign report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says receipt-sign concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says receipt-sign run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says receipt-sign warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says receipt-sign request count is one.
12. The report checks in the Report section, in the order written there.

`verify_receipt_sign` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the receipt-sign validation did not match.

## Files

`write_receipt_sign_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the receipt-sign directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the receipt-sign path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the receipt-sign output already exists.

The report is created with `create_new`, written, and `fsync`ed. No key bytes are read.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_receipt_sign`. Receipt verification has not started. A speculative-decoding record has not started. A CUDA-graph record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
