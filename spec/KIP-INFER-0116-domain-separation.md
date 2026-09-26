# KIP-INFER-0116 — Domain separation

Status: `measure_domain` records one host-supplied Ed25519 domain prefix for a cold micro fixture. It returns a `knolo.infer.domain-report`. `separated` and `rejected` say the host applied the prefix. `unsigned-local` does not. A public key is 32 bytes, a signature is 64 bytes, and a scalar is 32 bytes when the host separated them. Key bytes are not a field. `payloadHashed` stays false. It does not separate the domain and does not hash the payload. `measure_receipt_verify` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-domain, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `domainRoot` | root distinct from the engine build and the placement |
| `messageRoot` | root distinct from the domain, the engine build, and the placement |
| `separateStatus` | `separated`, `rejected`, or `unsigned-local` |
| `domainSeparated` | `true` for `separated` and `rejected`; `false` for `unsigned-local` |
| `payloadHashed` | `false` |
| `publicKeyBytes` | `32` when separated; `0` when unsigned |
| `signatureBytes` | `64` when separated; `0` when unsigned |
| `scalarBytes` | `32` when separated; `0` when unsigned |
| `keyMaterialPresent` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `verified` for `separated`; `recorded` otherwise |
| `extensions` | an empty map |

`kind` is `knolo.infer.domain-report` and `version` is `1`. The contract count is one hundred eighteen. `infer-domain` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the domain extensions are empty. A domain root equal to the engine build says the domain repeats the engine build. A domain root equal to the placement says the domain repeats the placement. A message root equal to the engine build says the separated message repeats the engine build. A message root equal to the placement says the separated message repeats the placement. A message root equal to the domain says the separated message repeats the domain. `keyMaterialPresent` true says key material stays in host storage. `payloadHashed` true says the payload stays unhashed. An unsigned release with `domainSeparated` true says an unsigned release separates the domain. A separated status with `domainSeparated` false says a separated domain records the prefix. A rejected status with `domainSeparated` false says a rejected domain records the prefix. A separated status stores `validationResult` `verified`, and any other result says a separated domain is verified. An unsigned or rejected status stores `recorded`, and any other result says only a separated domain is verified.

## Measurement

`measure_domain` takes the placement plan and one observation. It returns the report. The domain is not separated.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the domain report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the domain report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the domain report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says domain concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says domain run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says domain warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says domain request count is one.
12. The report checks in the Report section, in the order written there.

`verify_domain` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the domain validation did not match.

## Files

`write_domain_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the domain directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the domain path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the domain output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_domain`. Payload hashing has not started. Hybrid attention has not started. Linear attention has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
