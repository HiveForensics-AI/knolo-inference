# KIP-INFER-0090 — Unsupported architecture

Status: `measure_architecture` records one architecture adapter that is not compiled in for a cold micro fixture. It returns a `knolo.infer.architecture-report`. The rejected adapter is `knolo.llama.v1`. The code is `UNSUPPORTED_ARCHITECTURE` and it is not retryable. Weights are not opened, the forward does not run, and no receipt is stored. `knolo.micro.v1` says the micro adapter is compiled in and issues no report. It does not open a weight file. `measure_template_invalid` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-architecture, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `rejectedAdapter` | `knolo.llama.v1` |
| `code` | `UNSUPPORTED_ARCHITECTURE` |
| `retryable` | `false` |
| `weightsOpened` | `false` |
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

`kind` is `knolo.infer.architecture-report` and `version` is `1`. The contract count is eighty-three. `infer-architecture` is the report domain. A path is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the architecture extensions are empty. `rejectedAdapter` `knolo.micro.v1` says the micro adapter is compiled in. Any other adapter says the field has an unsupported value. A code other than `UNSUPPORTED_ARCHITECTURE` says an unsupported architecture is UNSUPPORTED_ARCHITECTURE. `retryable` true says an unsupported architecture is not retryable. `weightsOpened` true says an unsupported architecture does not open weights. `forwardRan` true says an unsupported architecture does not run the forward. `receiptStored` true says an unsupported architecture stores no receipt.

## Measurement

`measure_architecture` takes the placement plan and one observation. It returns the report. It does not open a weight file and it does not select an adapter.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the architecture report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the architecture report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the architecture report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says architecture concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says architecture run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says architecture warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says architecture request count is one.
12. The report checks in the Report section, in the order written there.

`verify_architecture` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the architecture validation did not match.

## Files

`write_architecture_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the architecture directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the architecture path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the architecture output already exists.

The report is created with `create_new`, written, and `fsync`ed. No weight file is opened.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_architecture`. The public-key multiplication has not started. An unsupported-quantization record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
