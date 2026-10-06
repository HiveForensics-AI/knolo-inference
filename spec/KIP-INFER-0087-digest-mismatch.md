# KIP-INFER-0087 — Digest mismatch

Status: `measure_digest_mismatch` records one weight file whose size or digest did not match for a cold micro fixture. It returns a `knolo.infer.digest-mismatch-report`. The mismatch is `size` or `digest`. The code is `MODEL_DIGEST_MISMATCH` and it is not retryable. A size mismatch does not read the body. A digest mismatch reads the body and does not parse the header. The forward does not run and no receipt is stored. It does not open a weight file. `measure_scalar` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-digest-mismatch, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `artifactRoot` | distinct from the engine build and the placement |
| `mismatch` | `size` or `digest` |
| `code` | `MODEL_DIGEST_MISMATCH` |
| `retryable` | `false` |
| `headerParsed` | `false` |
| `bodyRead` | `true` only for `digest` |
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

`kind` is `knolo.infer.digest-mismatch-report` and `version` is `1`. The contract count is eighty-three. `infer-digest-mismatch` is the report domain. A path is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the digest-mismatch extensions are empty. An artifact root equal to the engine build says the artifact repeats the engine build. An artifact root equal to the placement says the artifact repeats the placement. A mismatch other than `size` or `digest` says the field has an unsupported value. A code other than `MODEL_DIGEST_MISMATCH` says a digest mismatch is MODEL_DIGEST_MISMATCH. `retryable` true says a digest mismatch is not retryable. `headerParsed` true says a digest mismatch does not parse the header. A size mismatch with `bodyRead` true says a size mismatch does not read the body. A digest mismatch with `bodyRead` false says a digest mismatch reads the body. `forwardRan` true says a digest mismatch does not run the forward. `receiptStored` true says a digest mismatch stores no receipt.

## Measurement

`measure_digest_mismatch` takes the placement plan and one observation. It returns the report. It does not open a weight file and it does not parse a header.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the digest-mismatch report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the digest-mismatch report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the digest-mismatch report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says digest-mismatch concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says digest-mismatch run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says digest-mismatch warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says digest-mismatch request count is one.
12. The report checks in the Report section, in the order written there.

`verify_digest_mismatch` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the digest-mismatch validation did not match.

## Files

`write_digest_mismatch_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the digest-mismatch directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the digest-mismatch path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the digest-mismatch output already exists.

The report is created with `create_new`, written, and `fsync`ed. No weight file is opened.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_digest_mismatch`. The tokenizer record is KIP-INFER-0088. The public-key multiplication has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
