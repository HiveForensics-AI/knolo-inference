# KIP-INFER-0105 — Digest invalid

Status: `measure_digest_invalid` records one digest text or domain the parser refused for a cold micro fixture. It returns a `knolo.infer.digest-invalid-report`. The reason is `prefix`, `length`, `alphabet`, or `domain`. The code is `DIGEST_INVALID` and it is not retryable. A prefix refusal carries no hex and accepts neither the prefix, the length, nor the alphabet. A length refusal accepted the prefix and is 1 through 128 hex characters, excluding 64. An alphabet refusal is 64 characters, accepted the prefix and the length, and does not accept the alphabet. A domain refusal is 64 hex characters, accepted the prefix, the length, and the alphabet, and names a label of 1 through 64 lowercase letters, digits, or hyphens that is outside the allowlist. Earlier reasons name the domain `none`. `domainAccepted`, `hashed`, and `fileOpened` stay false. The forward does not run and no receipt is stored. It does not hash a payload and it does not open a receipt. `measure_backend` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

A length above 128 hex characters is `DIGEST_INVALID` and issues no report. A domain above 64 bytes is `DIGEST_INVALID` and issues no report. A stored report above either cap is `CONTRACT_INVALID`.

## Report

The report is a versioned contract. Its identity root is `H(infer-digest-invalid, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `prefix`, `length`, `alphabet`, or `domain` |
| `hexLength` | `0` for `prefix`; `1` through `128` excluding `64` for `length`; `64` for `alphabet` and `domain` |
| `domain` | `none` except for `domain`, which names an unknown label |
| `code` | `DIGEST_INVALID` |
| `retryable` | `false` |
| `prefixAccepted` | `false` for `prefix`; `true` otherwise |
| `lengthAccepted` | `true` for `alphabet` and `domain`; `false` otherwise |
| `alphabetAccepted` | `true` for `domain`; `false` otherwise |
| `domainAccepted` | `false` |
| `hashed` | `false` |
| `fileOpened` | `false` |
| `forwardRan` | `false` |
| `receiptStored` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.digest-invalid-report` and `version` is `1`. The contract count is ninety-eight. `infer-digest-invalid` is the report domain. The digest text is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the digest-invalid extensions are empty. A code other than `DIGEST_INVALID` says an invalid digest is DIGEST_INVALID. `retryable` true says an invalid digest is not retryable. `domainAccepted` true says the domain stays unaccepted. `hashed` true says a digest refusal does not hash the payload. `fileOpened` true says a digest refusal does not open a receipt. `forwardRan` true says a digest refusal does not run the forward. `receiptStored` true says a digest refusal stores no receipt. A prefix refusal with hex says a prefix refusal carries no hex. A length of zero or 64 says a length refusal is not 64 lowercase hex characters. A stored length above 128 says digest hex exceeds the record cap. A domain of `none` or empty on a domain refusal says a domain refusal names the domain. A stored domain above 64 bytes says domain exceeds the record cap. A domain label with any other character says a domain refusal names a domain label. A domain that is on the allowlist says a domain refusal names a domain outside the allowlist.

## Measurement

`measure_digest_invalid` takes the placement plan and one observation. It returns the report. It does not hash a payload and it does not open a receipt.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the digest-invalid report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the digest-invalid report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the digest-invalid report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says digest-invalid concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says digest-invalid run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says digest-invalid warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says digest-invalid request count is one.
12. A `length` reason above 128 hex characters: `DIGEST_INVALID`, and the message says digest hex exceeds the record cap.
13. A `domain` reason above 64 bytes: `DIGEST_INVALID`, and the message says domain exceeds the record cap.
14. The report checks in the Report section, in the order written there.

`verify_digest_invalid` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the digest-invalid validation did not match.

## Files

`write_digest_invalid_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the digest-invalid directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the digest-invalid path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the digest-invalid output already exists.

The report is created with `create_new`, written, and `fsync`ed. No digest text is hashed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_digest_invalid`. Receipt signing has not started. A canonical-CBOR record has not started. A contract-invalid record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
