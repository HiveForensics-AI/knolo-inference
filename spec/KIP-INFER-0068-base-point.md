# KIP-INFER-0068 — Base point

Status: `measure_base` records one host-supplied Ed25519 base-point multiplication for a cold micro fixture. It returns a `knolo.infer.base-report`. `multiplied` and `rejected` say the host multiplied the base point by the scalar. `unsigned-local` does not. A public key is 32 bytes, a signature is 64 bytes, and a scalar is 32 bytes when the host multiplied it. Key bytes are not a field. `pointAdded` stays false. It does not add the public-key point and does not load a secret. `measure_curve` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Result

The release root differs from the engine build root. The signed message root differs from both. `keyMaterialPresent` is false. `multiplied` carries validation result `verified`. `rejected` and `unsigned-local` carry `recorded`.

## Report

The report is a versioned contract. Its identity root is `H(infer-base, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `releaseRoot` | root of the release manifest |
| `messageRoot` | root of the signed bytes |
| `baseStatus` | `multiplied`, `rejected`, or `unsigned-local` |
| `baseMultiplied` | `true` only for `multiplied` and `rejected` |
| `publicKeyBytes` | `32` when multiplied, otherwise `0` |
| `signatureBytes` | `64` when multiplied, otherwise `0` |
| `scalarBytes` | `32` when multiplied, otherwise `0` |
| `pointAdded` | `false` |
| `keyMaterialPresent` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `verified` for `multiplied`, otherwise `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.base-report` and `version` is `1`. The contract count is sixty-four. `infer-base` is the report domain. Key bytes and signature bytes are not fields.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the base extensions are empty. A release root equal to the engine build says the release repeats the engine build. A message root equal to the engine build says the signed message repeats the engine build. A message root equal to the release says the signed message repeats the release. `keyMaterialPresent` true says key material stays in host storage. `pointAdded` true says the public-key point stays unadded. An unsigned public-key count other than zero says an unsigned release carries a public key. An unsigned signature count other than zero says an unsigned release carries signature bytes. An unsigned scalar count other than zero says an unsigned release carries a scalar. An unsigned release with `baseMultiplied` true says an unsigned release multiplies the base point. A multiplied public-key count other than 32 says ed25519 public keys are 32 bytes. A multiplied signature count other than 64 says ed25519 signatures are 64 bytes. A multiplied scalar count other than 32 says ed25519 scalars are 32 bytes. A multiplied release with `baseMultiplied` false says a multiplied release multiplies the base point. A multiplied result whose validation result is not `verified` says a multiplied base point is verified. A rejected or unsigned result whose validation result is `verified` says only a multiplied base point is verified.

## Measurement

`measure_base` takes the placement plan and one observation. It returns the report. It does not read the public key, the signature, or the scalar, and it does not add the public-key point.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement. The stored validation result is `verified` when the status is `multiplied` and `recorded` otherwise.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the base report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the base report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the base report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says base concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says base run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says base warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says base request count is one.
12. The report checks in the Report section, in the order written there.

`verify_base` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the base validation did not match.

## Files

`write_base_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the base directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the base path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the base output already exists.

The report is created with `create_new`, written, and `fsync`ed. The public key, the signature, and the scalar are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_base`. The disk-full record is KIP-INFER-0069. Point addition has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
