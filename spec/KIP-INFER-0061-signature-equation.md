# KIP-INFER-0061 — Signature equation

Status: `measure_signature_equation` records one host-supplied Ed25519 equation status for a cold micro fixture. It returns a `knolo.infer.equation-report`. `accepted` and `rejected` say the host evaluated the equation. `unsigned-local` does not. A public key is 32 bytes and a signature is 64 bytes when the host evaluated it. Key bytes are not a field. It does not compute the Ed25519 curve and does not load a secret. `measure_cache_channel` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Status

The release root differs from the engine build root. The signed message root differs from both. `keyMaterialPresent` is false. `accepted` carries validation result `verified`. `rejected` and `unsigned-local` carry `recorded`.

## Report

The report is a versioned contract. Its identity root is `H(infer-equation, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `releaseRoot` | root of the release manifest |
| `messageRoot` | root of the signed bytes |
| `equationStatus` | `accepted`, `rejected`, or `unsigned-local` |
| `equationEvaluated` | `true` only for `accepted` and `rejected` |
| `publicKeyBytes` | `32` when evaluated, otherwise `0` |
| `signatureBytes` | `64` when evaluated, otherwise `0` |
| `keyMaterialPresent` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `verified` for `accepted`, otherwise `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.equation-report` and `version` is `1`. The contract count is fifty-six. `infer-equation` is the report domain. Key bytes and signature bytes are not fields.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the equation extensions are empty. A release root equal to the engine build says the release repeats the engine build. A message root equal to the engine build says the signed message repeats the engine build. A message root equal to the release says the signed message repeats the release. `keyMaterialPresent` true says key material stays in host storage. An unsigned public-key count other than zero says an unsigned release carries a public key. An unsigned signature count other than zero says an unsigned release carries signature bytes. An unsigned release with `equationEvaluated` true says an unsigned release evaluates an equation. A checked public-key count other than 32 says ed25519 public keys are 32 bytes. A checked signature count other than 64 says ed25519 signatures are 64 bytes. A checked release with `equationEvaluated` false says a checked release evaluates the equation. An accepted release whose validation result is not `verified` says an accepted equation is verified. A rejected or unsigned release whose validation result is `verified` says only an accepted equation is verified.

## Measurement

`measure_signature_equation` takes the placement plan and one observation. It returns the report. It does not read the public key or the signature.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement. The stored validation result is `verified` when the status is `accepted` and `recorded` otherwise.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the equation report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the equation report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the equation report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says equation concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says equation run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says equation warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says equation request count is one.
12. The report checks in the Report section, in the order written there.

`verify_signature_equation` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the equation validation did not match.

## Files

`write_equation_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the equation directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the equation path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the equation output already exists.

The report is created with `create_new`, written, and `fsync`ed. The public key and the signature are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_signature_equation`. The safe error envelope is KIP-INFER-0062. Computing the Ed25519 curve has not started. The signature shape gate still rejects `keyVerified` true. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
