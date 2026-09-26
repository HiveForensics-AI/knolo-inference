# KIP-INFER-0057 — Receipt-key custody

Status: `measure_receipt_key` records where one receipt signing key lives for a cold micro fixture. It returns a `knolo.infer.receipt-key-report`. Custody is `host-store`. Key material is not serialized. `unsigned-local` names no key and does not rotate. `shape-checked` names one key id. Rotation to a different key id requires trusted metadata. It does not load a secret and does not evaluate Ed25519. `measure_host_key` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Custody

The receipt root differs from the engine build root. A current key has an empty previous key id and trusted metadata false. A rotated key has a different previous key id of at most 128 characters and trusted metadata true.

## Report

The report is a versioned contract. Its identity root is `H(infer-receipt-key, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `receiptRoot` | root of the inference receipt |
| `custody` | `host-store` |
| `keyMaterialSerialized` | `false` |
| `keyId` | empty, or the current key id |
| `signatureStatus` | `unsigned-local` or `shape-checked` |
| `signatureBytes` | `0` or `64`, matching the status |
| `rotation` | `current` or `rotated` |
| `previousKeyId` | empty, or the previous key id |
| `trustedMetadata` | `true` only when rotation is `rotated` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.receipt-key-report` and `version` is `1`. The contract count is fifty-two. `infer-receipt-key` is the report domain. Key bytes are not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the receipt key extensions are empty. A receipt root equal to the engine build says the receipt repeats the engine build. A custody other than `host-store` says receipt keys stay in host storage. `keyMaterialSerialized` true says key material stays out of the receipt. An unsigned receipt with a key id says an unsigned receipt names a key. An unsigned receipt with a non-zero byte count says an unsigned receipt carries signature bytes. An unsigned receipt whose rotation is not `current` says an unsigned receipt rotates a key. An unsigned receipt with a previous key id says an unsigned receipt names a previous key. An unsigned receipt with trusted metadata says an unsigned receipt names rotation. An empty or oversized key id says the field is empty or outside its bounds. A shape-checked byte count other than 64 says ed25519 signatures are 64 bytes. A current key with a previous key id says a current key names a previous key. A current key with trusted metadata says a current key names rotation. A rotated key whose previous id repeats the current id says rotation repeats the current key. A rotated key without trusted metadata says rotation uses trusted metadata.

## Measurement

`measure_receipt_key` takes the placement plan and one observation. It returns the report. It does not open a secret store.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the receipt key report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the receipt key report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the receipt key report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says receipt key concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says receipt key run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says receipt key warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says receipt key request count is one.
12. The report checks in the Report section, in the order written there.

`verify_receipt_key` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the receipt key validation did not match.

## Files

`write_receipt_key_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the receipt key directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the receipt key path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the receipt key output already exists.

The report is created with `create_new`, written, and `fsync`ed. The secret is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_receipt_key`. The worker sandbox profile is KIP-INFER-0058. Evaluating the Ed25519 equation has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred. The signature shape gate still rejects `keyVerified` true.
