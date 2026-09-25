# KIP-INFER-0076 — Point comparison

Status: `measure_point_equality` records one host-supplied Ed25519 point comparison for a cold micro fixture. It returns a `knolo.infer.equality-report`. `equal` and `unequal` say the host compared the two points. `unsigned-local` does not. A public key is 32 bytes, a signature is 64 bytes, and a scalar is 32 bytes when the host compared them. Key bytes are not a field. `challengeHashed` stays false. It does not hash the challenge and does not load a secret. `measure_point` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Result

The release root differs from the engine build root. The signed message root differs from both. `keyMaterialPresent` is false. `equal` carries validation result `verified`. `unequal` and `unsigned-local` carry `recorded`.

## Report

The report is a versioned contract. Its identity root is `H(infer-equality, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `releaseRoot` | root of the release manifest |
| `messageRoot` | root of the signed bytes |
| `equalityStatus` | `equal`, `unequal`, or `unsigned-local` |
| `pointsCompared` | `true` only for `equal` and `unequal` |
| `pointsEqual` | `true` only for `equal` |
| `challengeHashed` | `false` |
| `publicKeyBytes` | `32` when compared, otherwise `0` |
| `signatureBytes` | `64` when compared, otherwise `0` |
| `scalarBytes` | `32` when compared, otherwise `0` |
| `keyMaterialPresent` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `verified` for `equal`, otherwise `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.equality-report` and `version` is `1`. The contract count is seventy-three. `infer-equality` is the report domain. Key bytes and signature bytes are not fields.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the equality extensions are empty. A release root equal to the engine build says the release repeats the engine build. A message root equal to the engine build says the signed message repeats the engine build. A message root equal to the release says the signed message repeats the release. `keyMaterialPresent` true says key material stays in host storage. `challengeHashed` true says the challenge hash stays uncomputed. An unsigned public-key count other than zero says an unsigned release carries a public key. An unsigned signature count other than zero says an unsigned release carries signature bytes. An unsigned scalar count other than zero says an unsigned release carries a scalar. An unsigned release with `pointsCompared` true says an unsigned release compares the points. An unsigned release with `pointsEqual` true says an unsigned release says the points are equal. An unsigned result whose validation result is `verified` says only an equal point comparison is verified. A compared public-key count other than 32 says ed25519 public keys are 32 bytes. A compared signature count other than 64 says ed25519 signatures are 64 bytes. A compared scalar count other than 32 says ed25519 scalars are 32 bytes. A compared release with `pointsCompared` false says a compared release compares the points. An equal result with `pointsEqual` false says an equal comparison says the points are equal. An equal result whose validation result is not `verified` says an equal point comparison is verified. An unequal result with `pointsEqual` true says an unequal comparison says the points differ. An unequal result whose validation result is `verified` says only an equal point comparison is verified.

## Measurement

`measure_point_equality` takes the placement plan and one observation. It returns the report. It does not read the public key, the signature, or the scalar, and it does not hash the challenge.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement. The stored validation result is `verified` when the status is `equal` and `recorded` otherwise.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the equality report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the equality report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the equality report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says equality concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says equality run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says equality warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says equality request count is one.
12. The report checks in the Report section, in the order written there.

`verify_point_equality` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the equality validation did not match.

## Files

`write_equality_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the equality directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the equality path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the equality output already exists.

The report is created with `create_new`, written, and `fsync`ed. The public key, the signature, and the scalar are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_point_equality`. The CUDA out-of-memory record is KIP-INFER-0077. The challenge hash has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
