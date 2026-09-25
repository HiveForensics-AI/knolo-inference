# KIP-INFER-0099 — Model image invalid

Status: `measure_image_invalid` records one model image the compiler refused for a cold micro fixture. It returns a `knolo.infer.image-invalid-report`. The reason is `empty`, `canonical`, `format`, or `inventory`. The code is `MODEL_IMAGE_INVALID` and it is not retryable. An empty image and a non-canonical image are not parsed and do not open weights. A format refusal parsed the image and does not open weights. An inventory refusal parsed the image and opened the weights. The forward does not run and no receipt is stored. It does not open a weight file. `measure_prompt_compilation` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-image-invalid, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `imageRoot` | root distinct from the engine build and the placement |
| `reason` | `empty`, `canonical`, `format`, or `inventory` |
| `code` | `MODEL_IMAGE_INVALID` |
| `retryable` | `false` |
| `imageParsed` | `true` for `format` and `inventory`; `false` otherwise |
| `weightsOpened` | `true` for `inventory`; `false` otherwise |
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

`kind` is `knolo.infer.image-invalid-report` and `version` is `1`. The contract count is ninety-three. `infer-image-invalid` is the report domain. A path is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the image-invalid extensions are empty. An image root equal to the engine build says the image repeats the engine build. An image root equal to the placement says the image repeats the placement. A code other than `MODEL_IMAGE_INVALID` says an invalid image is MODEL_IMAGE_INVALID. `retryable` true says an invalid image is not retryable. `forwardRan` true says an invalid image does not run the forward. `receiptStored` true says an invalid image stores no receipt. An empty image that parsed says an empty image is not parsed. An empty image that opened weights says an empty image does not open weights. A non-canonical image that parsed says a non-canonical image is not parsed. A non-canonical image that opened weights says a non-canonical image does not open weights. A format refusal that did not parse says a format refusal parsed the image. A format refusal that opened weights says a format refusal does not open weights. An inventory refusal that did not parse says an inventory refusal parsed the image. An inventory refusal that did not open weights says an inventory refusal opens the weights.

## Measurement

`measure_image_invalid` takes the placement plan and one observation. It returns the report. It does not open a weight file.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the image-invalid report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the image-invalid report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the image-invalid report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says image-invalid concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says image-invalid run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says image-invalid warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says image-invalid request count is one.
12. The report checks in the Report section, in the order written there.

`verify_image_invalid` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the image-invalid validation did not match.

## Files

`write_image_invalid_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the image-invalid directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the image-invalid path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the image-invalid output already exists.

The report is created with `create_new`, written, and `fsync`ed. No weight file is opened.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_image_invalid`. The invalid signature block is KIP-INFER-0100. Cofactor clearing has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
