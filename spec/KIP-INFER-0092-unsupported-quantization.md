# KIP-INFER-0092 — Unsupported quantization

Status: `measure_quantization` records one weight whose precision, dtype, ggml type, or quantization version is outside the allowlist for a cold micro fixture. It returns a `knolo.infer.quantization-report`. The reason is `precision`, `dtype`, `ggml`, or `version`. The code is `UNSUPPORTED_QUANTIZATION` and it is not retryable. A precision refusal does not open weights and does not read the payload. A dtype refusal opens the weights and reads the payload. A ggml type and a quantization version open the weights and do not read the payload. The forward does not run and no receipt is stored. It does not open a weight file and does not dequant a payload. `measure_public` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-quantization, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `artifactRoot` | root distinct from the engine build and the placement |
| `reason` | `precision`, `dtype`, `ggml`, or `version` |
| `code` | `UNSUPPORTED_QUANTIZATION` |
| `retryable` | `false` |
| `weightsOpened` | `false` for `precision`; `true` otherwise |
| `payloadRead` | `true` for `dtype`; `false` otherwise |
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

`kind` is `knolo.infer.quantization-report` and `version` is `1`. The contract count is eighty-eight. `infer-quantization` is the report domain. Weight bytes are not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the quantization extensions are empty. An artifact root equal to the engine build says the artifact repeats the engine build. An artifact root equal to the placement says the artifact repeats the placement. A code other than `UNSUPPORTED_QUANTIZATION` says an unsupported quantization is UNSUPPORTED_QUANTIZATION. `retryable` true says an unsupported quantization is not retryable. `forwardRan` true says an unsupported quantization does not run the forward. `receiptStored` true says an unsupported quantization stores no receipt. `weightsOpened` true on `precision` says a precision refusal does not open weights. `payloadRead` true on `precision` says a precision refusal does not read the payload. `weightsOpened` false on `dtype` says a dtype refusal opens the weights. `payloadRead` false on `dtype` says a dtype refusal reads the payload. `weightsOpened` false on `ggml` says a ggml type opens the weights. `payloadRead` true on `ggml` says a ggml type does not read the payload. `weightsOpened` false on `version` says a quantization version opens the weights. `payloadRead` true on `version` says a quantization version does not read the payload.

## Measurement

`measure_quantization` takes the placement plan and one observation. It returns the report. It does not open a weight file and it does not dequant a payload.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the quantization report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the quantization report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the quantization report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says quantization concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says quantization run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says quantization warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says quantization request count is one.
12. The report checks in the Report section, in the order written there.

`verify_quantization` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the quantization validation did not match.

## Files

`write_quantization_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the quantization directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the quantization path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the quantization output already exists.

The report is created with `create_new`, written, and `fsync`ed. No weight file is opened.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_quantization`. The unsupported-kernel record is KIP-INFER-0093. The signature check has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
