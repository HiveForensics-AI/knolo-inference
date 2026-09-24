# KIP-INFER-0055 — Signature shape gate

Status: `measure_signature_gate` records the signature shape for one cold micro fixture. It returns a `knolo.infer.signature-report`. `unsigned-local` has an empty key id, a zero byte count, and a zero signature count. `shape-checked` has one block: a key id of at most 128 characters and a byte count of 64. `keyVerified` is false. It does not read the signature bytes and does not verify an Ed25519 key. `measure_reproducible_build` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Shape

The release root differs from the engine build root. A claim that the key was verified is `CONTRACT_INVALID`, and the message says signature keys stay unverified. An empty signature list remains the unsigned local form.

## Report

The report is a versioned contract. Its identity root is `H(infer-signature, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `releaseRoot` | root of the release manifest |
| `signatureStatus` | `unsigned-local` or `shape-checked` |
| `signatureCount` | `0` or `1`, matching the status |
| `keyId` | empty, or the shape-checked key id |
| `signatureBytes` | `0` or `64`, matching the status |
| `keyVerified` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.signature-report` and `version` is `1`. The contract count is forty-eight. `infer-signature` is the report domain. Signature bytes are not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the signature extensions are empty. A release root equal to the engine build says the release repeats the engine build. An unsigned release with a non-zero count says an unsigned release carries a signature. An unsigned release with a key id says an unsigned release names a key. An unsigned release with a non-zero byte count says an unsigned release carries signature bytes. A shape-checked release whose count is not one says a shape-checked release carries one signature. A shape-checked byte count other than 64 says ed25519 signatures are 64 bytes. An empty or oversized key id says the field is empty or outside its bounds. `keyVerified` true says signature keys stay unverified.

## Measurement

`measure_signature_gate` takes the placement plan and one observation. It returns the report. It does not create a signature file.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the signature report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the signature report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the signature report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says signature concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says signature run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says signature warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says signature request count is one.
12. The report checks in the Report section, in the order written there.

`verify_signature_gate` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the signature validation did not match.

## Files

`write_signature_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the signature directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the signature path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the signature output already exists.

The report is created with `create_new`, written, and `fsync`ed. The signature bytes are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_signature_gate`. The host-key check is KIP-INFER-0056. A worker sandbox profile is KIP-INFER-0058. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
