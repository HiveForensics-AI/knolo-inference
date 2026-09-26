# KIP-INFER-0056 — Host-key check

Status: `measure_host_key` records one host-supplied signature match for a cold micro fixture. It returns a `knolo.infer.host-key-report`. `matched` verifies the host key. `rejected` and `unsigned-local` do not. Key bytes are not a field. It does not evaluate the Ed25519 equation and does not load a secret. `measure_signature_gate` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Match

The release root differs from the engine build root. The host-key root differs from both. `keyMaterialPresent` is false. `matched` carries one key id of at most 128 characters, 64 signature bytes, `keyVerified` true, and validation result `verified`. `rejected` carries the same shape with `keyVerified` false and validation result `recorded`. `unsigned-local` carries an empty key id, a zero byte count, `keyVerified` false, and validation result `recorded`.

## Report

The report is a versioned contract. Its identity root is `H(infer-host-key, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `releaseRoot` | root of the release manifest |
| `hostKeyRoot` | root of the host key id |
| `signatureStatus` | `matched`, `rejected`, or `unsigned-local` |
| `keyId` | empty, or the host key id |
| `signatureBytes` | `0` or `64`, matching the status |
| `keyVerified` | `true` only when the status is `matched` |
| `keyMaterialPresent` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `verified` for `matched`, otherwise `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.host-key-report` and `version` is `1`. The contract count is fifty-two. `infer-host-key` is the report domain. Key bytes are not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the host key extensions are empty. A release root equal to the engine build says the release repeats the engine build. A host-key root equal to the engine build says the host key repeats the engine build. A host-key root equal to the release says the host key repeats the release. `keyMaterialPresent` true says key material stays in host storage. An unsigned release with a key id says an unsigned release names a key. An unsigned release with a non-zero byte count says an unsigned release carries signature bytes. An unsigned release with `keyVerified` true says an unsigned release verifies a key. An unsigned release whose validation result is not `recorded` says an unsigned release is recorded. An empty or oversized key id says the field is empty or outside its bounds. A matched or rejected byte count other than 64 says ed25519 signatures are 64 bytes. A matched release with `keyVerified` false says a matched release verifies the host key. A matched release whose validation result is not `verified` says a matched release is verified. A rejected release with `keyVerified` true says a rejected release verifies a key. A rejected release whose validation result is not `recorded` says a rejected release is recorded.

## Measurement

`measure_host_key` takes the placement plan and one observation. It returns the report. It does not create a key file.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the host key report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the host key report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the host key report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says host key concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says host key run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says host key warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says host key request count is one.
12. The report checks in the Report section, in the order written there.

`verify_host_key` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the host key validation did not match.

## Files

`write_host_key_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the host key directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the host key path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the host key output already exists.

The report is created with `create_new`, written, and `fsync`ed. The key bytes are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_host_key`. Evaluating the Ed25519 equation has not started. The receipt-key custody record is KIP-INFER-0057. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred. The signature shape gate still rejects `keyVerified` true.
