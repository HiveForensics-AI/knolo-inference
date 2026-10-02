# KIP-INFER-0096 — Signature check

Status: `measure_signature_check` records one host-supplied Ed25519 signature check for a cold micro fixture. It returns a `knolo.infer.signature-check-report`. `checked` and `rejected` say the host checked the signature. `unsigned-local` does not. A public key is 32 bytes, a signature is 64 bytes, and a scalar is 32 bytes when the host checked them. Key bytes are not a field. `cofactorCleared` stays false. It does not check the signature and does not clear the cofactor. `measure_public` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-signature-check, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `releaseRoot` | root distinct from the engine build and the signed message |
| `messageRoot` | root distinct from the engine build and the release |
| `checkStatus` | `checked`, `rejected`, or `unsigned-local` |
| `signatureChecked` | `true` for `checked` and `rejected`; `false` for `unsigned-local` |
| `cofactorCleared` | `false` |
| `publicKeyBytes` | `32` when checked, otherwise `0` |
| `signatureBytes` | `64` when checked, otherwise `0` |
| `scalarBytes` | `32` when checked, otherwise `0` |
| `keyMaterialPresent` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `verified` for `checked`; `recorded` otherwise |
| `extensions` | an empty map |

`kind` is `knolo.infer.signature-check-report` and `version` is `1`. The contract count is ninety-three. `infer-signature-check` is the report domain. Key bytes and signature bytes are not fields.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the signature-check extensions are empty. A release root equal to the engine build says the release repeats the engine build. A message root equal to the engine build says the signed message repeats the engine build. A message root equal to the release says the signed message repeats the release. `keyMaterialPresent` true says key material stays in host storage. `cofactorCleared` true says the cofactor stays uncleared. An unsigned public-key count other than zero says an unsigned release carries a public key. An unsigned signature count other than zero says an unsigned release carries signature bytes. An unsigned scalar count other than zero says an unsigned release carries a scalar. An unsigned release with `signatureChecked` true says an unsigned release checks a signature. A checked public-key count other than 32 says ed25519 public keys are 32 bytes. A checked signature count other than 64 says ed25519 signatures are 64 bytes. A checked scalar count other than 32 says ed25519 scalars are 32 bytes. A checked release with `signatureChecked` false says a checked signature records the check. A rejected release with `signatureChecked` false says a rejected signature records the check. `validationResult` `verified` on any status other than `checked` says only a checked signature is verified. A checked release with `validationResult` `recorded` says a checked signature is verified.

## Measurement

`measure_signature_check` takes the placement plan and one observation. It returns the report. It does not read the public key, the signature, or the scalar, and it does not clear the cofactor.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the signature-check report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the signature-check report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the signature-check report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says signature-check concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says signature-check run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says signature-check warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says signature-check request count is one.
12. The report checks in the Report section, in the order written there.

`verify_signature_check` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the signature-check validation did not match.

## Files

`write_signature_check_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the signature-check directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the signature-check path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the signature-check output already exists.

The report is created with `create_new`, written, and `fsync`ed. No key bytes are read.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_signature_check`. The context-limit record is KIP-INFER-0097. Cofactor clearing has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
