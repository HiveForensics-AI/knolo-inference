# KIP-INFER-0091 — Public-key multiplication

Status: `measure_public` records one host-supplied Ed25519 public-key multiplication for a cold micro fixture. It returns a `knolo.infer.public-report`. `multiplied` and `rejected` say the host multiplied the public key. `unsigned-local` does not. A public key is 32 bytes, a signature is 64 bytes, and a scalar is 32 bytes when the host multiplied them. Key bytes are not a field. `signatureChecked` stays false. It does not multiply the public key and does not check the signature. `measure_scalar` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-public, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `releaseRoot` | root distinct from the engine build and the signed message |
| `messageRoot` | root distinct from the engine build and the release |
| `publicStatus` | `multiplied`, `rejected`, or `unsigned-local` |
| `publicMultiplied` | `true` for `multiplied` and `rejected`; `false` for `unsigned-local` |
| `signatureChecked` | `false` |
| `publicKeyBytes` | `32` when multiplied, otherwise `0` |
| `signatureBytes` | `64` when multiplied, otherwise `0` |
| `scalarBytes` | `32` when multiplied, otherwise `0` |
| `keyMaterialPresent` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `verified` for `multiplied`; `recorded` otherwise |
| `extensions` | an empty map |

`kind` is `knolo.infer.public-report` and `version` is `1`. The contract count is eighty-eight. `infer-public` is the report domain. Key bytes and signature bytes are not fields.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the public extensions are empty. A release root equal to the engine build says the release repeats the engine build. A message root equal to the engine build says the signed message repeats the engine build. A message root equal to the release says the signed message repeats the release. `keyMaterialPresent` true says key material stays in host storage. `signatureChecked` true says the signature check stays uncomputed. An unsigned public-key count other than zero says an unsigned release carries a public key. An unsigned signature count other than zero says an unsigned release carries signature bytes. An unsigned scalar count other than zero says an unsigned release carries a scalar. An unsigned release with `publicMultiplied` true says an unsigned release multiplies the public key. A multiplied public-key count other than 32 says ed25519 public keys are 32 bytes. A multiplied signature count other than 64 says ed25519 signatures are 64 bytes. A multiplied scalar count other than 32 says ed25519 scalars are 32 bytes. A multiplied release with `publicMultiplied` false says a multiplied public key records the multiplication. A rejected release with `publicMultiplied` false says a rejected public key records the multiplication. `validationResult` `verified` on any status other than `multiplied` says only a multiplied public key is verified. A multiplied release with `validationResult` `recorded` says a multiplied public key is verified.

## Measurement

`measure_public` takes the placement plan and one observation. It returns the report. It does not read the public key, the signature, or the scalar, and it does not check the signature.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the public report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the public report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the public report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says public concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says public run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says public warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says public request count is one.
12. The report checks in the Report section, in the order written there.

`verify_public` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the public validation did not match.

## Files

`write_public_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the public directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the public path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the public output already exists.

The report is created with `create_new`, written, and `fsync`ed. No key bytes are read.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_public`. The unsupported-quantization record is KIP-INFER-0092. The signature check has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
