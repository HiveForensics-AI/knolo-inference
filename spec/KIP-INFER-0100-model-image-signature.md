# KIP-INFER-0100 — Model image signature

Status: `measure_image_signature` records one signature block the image compiler refused for a cold micro fixture. It returns a `knolo.infer.image-signature-report`. The reason is `algorithm`, `length`, or `count`. The code is `MODEL_IMAGE_SIGNATURE_INVALID` and it is not retryable. An algorithm refusal carries one signature, does not read the signature bytes, and does not accept the algorithm. A length refusal carries one signature, accepted the algorithm, and read a length other than 0 or 64, at most 256 bytes. A count refusal carries 9 through 16 signatures, does not read the signature bytes, and does not accept the algorithm. Weights are not opened. Key material stays in host storage. The forward does not run and no receipt is stored. It does not read a key. `measure_image_invalid` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

A length above 256 bytes is `MODEL_IMAGE_SIGNATURE_INVALID` and issues no report. A count above 16 is `MODEL_IMAGE_SIGNATURE_INVALID` and issues no report. A stored report above either cap is `CONTRACT_INVALID`.

## Report

The report is a versioned contract. Its identity root is `H(infer-image-signature, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `imageRoot` | root distinct from the engine build and the placement |
| `reason` | `algorithm`, `length`, or `count` |
| `signatureCount` | `1` for `algorithm` and `length`; `9` through `16` for `count` |
| `signatureBytes` | `0` for `algorithm` and `count`; `1` through `256`, excluding `64`, for `length` |
| `code` | `MODEL_IMAGE_SIGNATURE_INVALID` |
| `retryable` | `false` |
| `algorithmAccepted` | `true` for `length`; `false` otherwise |
| `weightsOpened` | `false` |
| `forwardRan` | `false` |
| `receiptStored` | `false` |
| `keyMaterialPresent` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.image-signature-report` and `version` is `1`. The contract count is ninety-three. `infer-image-signature` is the report domain. Key bytes are not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the image-signature extensions are empty. An image root equal to the engine build says the image repeats the engine build. An image root equal to the placement says the image repeats the placement. A code other than `MODEL_IMAGE_SIGNATURE_INVALID` says an invalid signature block is MODEL_IMAGE_SIGNATURE_INVALID. `retryable` true says an invalid signature block is not retryable. `keyMaterialPresent` true says key material stays in host storage. `weightsOpened` true says a signature refusal does not open weights. `forwardRan` true says a signature refusal does not run the forward. `receiptStored` true says a signature refusal stores no receipt. An algorithm refusal whose count is not one says an algorithm refusal carries one signature. An algorithm refusal that read bytes says an algorithm refusal does not read the signature. An algorithm refusal that accepted the algorithm says an algorithm refusal does not accept the algorithm. A length refusal whose count is not one says a length refusal carries one signature. A length of zero or 64 says a length refusal read a signature that is not 64 bytes. A stored length above 256 says signature bytes exceed the record cap. A length refusal that did not accept the algorithm says a length refusal accepted the algorithm. A count of eight or fewer says eight signatures are still a shape check. A stored count above 16 says signature count exceeds the record cap. A count refusal that read bytes says a count refusal does not read the signature. A count refusal that accepted the algorithm says a count refusal does not accept the algorithm.

## Measurement

`measure_image_signature` takes the placement plan and one observation. It returns the report. It does not read the signature bytes and it does not open a weight file.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the image-signature report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the image-signature report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the image-signature report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says image-signature concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says image-signature run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says image-signature warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says image-signature request count is one.
12. A `length` reason above 256 bytes: `MODEL_IMAGE_SIGNATURE_INVALID`, and the message says signature bytes exceed the record cap.
13. A `count` reason above 16 signatures: `MODEL_IMAGE_SIGNATURE_INVALID`, and the message says signature count exceeds the record cap.
14. The report checks in the Report section, in the order written there.

`verify_image_signature` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the image-signature validation did not match.

## Files

`write_image_signature_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the image-signature directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the image-signature path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the image-signature output already exists.

The report is created with `create_new`, written, and `fsync`ed. No key bytes are read.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_image_signature`. Cofactor clearing has not started. A missing-artifact record has not started. A receipt-required record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
