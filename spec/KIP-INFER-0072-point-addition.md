# KIP-INFER-0072 — Point addition

Status: `measure_point` records one host-supplied Ed25519 point addition for a cold micro fixture. It returns a `knolo.infer.point-report`. `added` and `rejected` say the host added the public-key point. `unsigned-local` does not. A public key is 32 bytes, a signature is 64 bytes, and a scalar is 32 bytes when the host added it. Key bytes are not a field. `pointsEqual` stays false. It does not compare the points and does not load a secret. `measure_base` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Result

The release root differs from the engine build root. The signed message root differs from both. `keyMaterialPresent` is false. `added` carries validation result `verified`. `rejected` and `unsigned-local` carry `recorded`.

## Report

The report is a versioned contract. Its identity root is `H(infer-point, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `releaseRoot` | root of the release manifest |
| `messageRoot` | root of the signed bytes |
| `pointStatus` | `added`, `rejected`, or `unsigned-local` |
| `pointAdded` | `true` only for `added` and `rejected` |
| `publicKeyBytes` | `32` when added, otherwise `0` |
| `signatureBytes` | `64` when added, otherwise `0` |
| `scalarBytes` | `32` when added, otherwise `0` |
| `pointsEqual` | `false` |
| `keyMaterialPresent` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `verified` for `added`, otherwise `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.point-report` and `version` is `1`. The contract count is sixty-eight. `infer-point` is the report domain. Key bytes and signature bytes are not fields.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the point extensions are empty. A release root equal to the engine build says the release repeats the engine build. A message root equal to the engine build says the signed message repeats the engine build. A message root equal to the release says the signed message repeats the release. `keyMaterialPresent` true says key material stays in host storage. `pointsEqual` true says the points stay uncompared. An unsigned public-key count other than zero says an unsigned release carries a public key. An unsigned signature count other than zero says an unsigned release carries signature bytes. An unsigned scalar count other than zero says an unsigned release carries a scalar. An unsigned release with `pointAdded` true says an unsigned release adds the public-key point. An added public-key count other than 32 says ed25519 public keys are 32 bytes. An added signature count other than 64 says ed25519 signatures are 64 bytes. An added scalar count other than 32 says ed25519 scalars are 32 bytes. An added release with `pointAdded` false says an added release adds the public-key point. An added result whose validation result is not `verified` says an added public-key point is verified. A rejected or unsigned result whose validation result is `verified` says only an added public-key point is verified.

## Measurement

`measure_point` takes the placement plan and one observation. It returns the report. It does not read the public key, the signature, or the scalar, and it does not compare the points.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement. The stored validation result is `verified` when the status is `added` and `recorded` otherwise.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the point report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the point report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the point report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says point concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says point run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says point warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says point request count is one.
12. The report checks in the Report section, in the order written there.

`verify_point` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the point validation did not match.

## Files

`write_point_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the point directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the point path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the point output already exists.

The report is created with `create_new`, written, and `fsync`ed. The public key, the signature, and the scalar are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_point`. The duplicate request id is KIP-INFER-0073. Point comparison has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
