# KIP-INFER-0126 — Payload hashing

Status: `measure_payload` records one host-supplied payload hash for a cold micro fixture. It returns a `knolo.infer.payload-report`. `hashed` and `rejected` say the host hashed the payload. `unsigned-local` does not. A public key is 32 bytes, a signature is 64 bytes, and a scalar is 32 bytes when the host hashed them. Key bytes are not a field. `digestBound` stays false. It does not hash the payload and does not bind the digest. `measure_capacity` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-payload, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `payloadRoot` | root distinct from the engine build and the placement |
| `domainRoot` | root distinct from the payload, the engine build, and the placement |
| `hashStatus` | `hashed`, `rejected`, or `unsigned-local` |
| `payloadHashed` | `true` for `hashed` and `rejected`; `false` for `unsigned-local` |
| `digestBound` | `false` |
| `publicKeyBytes` | `32` when hashed; `0` when unsigned |
| `signatureBytes` | `64` when hashed; `0` when unsigned |
| `scalarBytes` | `32` when hashed; `0` when unsigned |
| `keyMaterialPresent` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `verified` for `hashed`; `recorded` otherwise |
| `extensions` | an empty map |

`kind` is `knolo.infer.payload-report` and `version` is `1`. The contract count is one hundred twenty-three. `infer-payload` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the payload extensions are empty. A payload root equal to the engine build says the hashed payload repeats the engine build. A payload root equal to the placement says the hashed payload repeats the placement. A domain root equal to the engine build says the payload domain repeats the engine build. A domain root equal to the placement says the payload domain repeats the placement. A domain root equal to the payload says the payload domain repeats the payload. `keyMaterialPresent` true says key material stays in host storage. `digestBound` true says the digest stays unbound. An unsigned release with `payloadHashed` true says an unsigned release hashes the payload. A hashed status with `payloadHashed` false says a hashed payload records the digest. A rejected status with `payloadHashed` false says a rejected payload records the digest. A hashed status stores `validationResult` `verified`, and any other result says a hashed payload is verified. An unsigned or rejected status stores `recorded`, and any other result says only a hashed payload is verified.

## Measurement

`measure_payload` takes the placement plan and one observation. It returns the report. The payload is not hashed.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the payload report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the payload report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the payload report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says payload concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says payload run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says payload warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says payload request count is one.
12. The report checks in the Report section, in the order written there.

`verify_payload` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the payload validation did not match.

## Files

`write_payload_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the payload directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the payload path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the payload output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_payload`. Digest binding has not started. Multimodal input has not started. Large expert placement has not started. Extreme-context cache has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
