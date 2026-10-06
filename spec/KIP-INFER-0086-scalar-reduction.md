# KIP-INFER-0086 — Scalar reduction

Status: `measure_scalar` records one host-supplied Ed25519 scalar reduction for a cold micro fixture. It returns a `knolo.infer.scalar-report`. `reduced` and `rejected` say the host reduced the scalar. `unsigned-local` does not. A public key is 32 bytes, a signature is 64 bytes, and a scalar is 32 bytes when the host reduced them. Key bytes are not a field. `publicMultiplied` stays false. It does not reduce the scalar and does not multiply the public key. `measure_challenge` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-scalar, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `releaseRoot` | distinct from the engine build |
| `messageRoot` | distinct from the engine build and the release |
| `scalarStatus` | `reduced`, `rejected`, or `unsigned-local` |
| `scalarReduced` | `true` only for `reduced` and `rejected` |
| `publicMultiplied` | `false` |
| `publicKeyBytes` | `32` when reduced or rejected, otherwise `0` |
| `signatureBytes` | `64` when reduced or rejected, otherwise `0` |
| `scalarBytes` | `32` when reduced or rejected, otherwise `0` |
| `keyMaterialPresent` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `verified` for `reduced`, otherwise `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.scalar-report` and `version` is `1`. The contract count is eighty-three. `infer-scalar` is the report domain. Key bytes and signature bytes are not fields.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the scalar extensions are empty. A release root equal to the engine build says the release repeats the engine build. A message root equal to the engine build says the signed message repeats the engine build. A message root equal to the release says the signed message repeats the release. `keyMaterialPresent` true says key material stays in host storage. `publicMultiplied` true says the public-key multiplication stays uncomputed. An unsigned public-key count other than zero says an unsigned release carries a public key. An unsigned signature count other than zero says an unsigned release carries signature bytes. An unsigned scalar count other than zero says an unsigned release carries a scalar. An unsigned release with `scalarReduced` true says an unsigned release reduces the scalar. A reduced release with `scalarReduced` false says a reduced scalar records the reduction. A rejected release with `scalarReduced` false says a rejected scalar records the reduction. A reduced release whose validation result is not `verified` says a reduced scalar is verified. Any other status whose validation result is `verified` says only a reduced scalar is verified.

## Measurement

`measure_scalar` takes the placement plan and one observation. It returns the report. It does not reduce the scalar and it does not multiply the public key.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the scalar report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the scalar report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the scalar report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says scalar concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says scalar run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says scalar warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says scalar request count is one.
12. The report checks in the Report section, in the order written there.

`verify_scalar` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the scalar validation did not match.

## Files

`write_scalar_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the scalar directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the scalar path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the scalar output already exists.

The report is created with `create_new`, written, and `fsync`ed. The scalar is not reduced.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_scalar`. The digest-mismatch record is KIP-INFER-0087. The public-key multiplication has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
