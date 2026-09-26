# KIP-INFER-0114 — Multi-model lifecycle

Status: `measure_multi_model` records one second model the worker did not load for a cold micro fixture. It returns a `knolo.infer.multi-model-report`. The reason is `second`, `replace`, or `parallel`. The code is `CONTRACT_INVALID` and it is not retryable. A second-model refusal and a replace refusal name one device. A parallel refusal names two devices. Exactly 2 is recorded. A count above 2 is `CONTRACT_INVALID` and issues no report. A stored report above that cap is `CONTRACT_INVALID`. `modelsLoaded` stays 1. `secondLoaded`, `residentReplaced`, and `tensorParallel` stay false. The resident root and the incoming root differ from each other and from the engine build and the placement. The forward does not run and no receipt is stored. It does not load a model. `measure_concurrent_load` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-multi-model, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `residentRoot` | root distinct from the engine build and the placement |
| `incomingRoot` | root distinct from the resident model, the engine build, and the placement |
| `reason` | `second`, `replace`, or `parallel` |
| `deviceCount` | `1` for `second` and `replace`; `2` for `parallel` |
| `modelsLoaded` | `1` |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `secondLoaded` | `false` |
| `residentReplaced` | `false` |
| `tensorParallel` | `false` |
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

`kind` is `knolo.infer.multi-model-report` and `version` is `1`. The contract count is one hundred eight. `infer-multi-model` is the report domain. A model alias is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the multi-model extensions are empty. A resident root equal to the engine build says the resident model repeats the engine build. A resident root equal to the placement says the resident model repeats the placement. An incoming root equal to the engine build says the incoming model repeats the engine build. An incoming root equal to the placement says the incoming model repeats the placement. An incoming root equal to the resident model says the incoming model repeats the resident model. A code other than `CONTRACT_INVALID` says a multi-model record is CONTRACT_INVALID. `retryable` true says a multi-model record is not retryable. A loaded count other than 1 says a multi-model record keeps one resident model. `secondLoaded` true says a multi-model record does not load a second model. `residentReplaced` true says a multi-model record does not replace the resident model. `tensorParallel` true says tensor parallel stays off. `forwardRan` true says a multi-model record does not run the forward. `receiptStored` true says a multi-model record stores no receipt. A stored count above 2 says device count exceeds the record cap. A second-model refusal whose count is not 1 says a second-model refusal names one device. A replace refusal whose count is not 1 says a replace refusal names one device. A parallel refusal whose count is not 2 says a parallel refusal names two devices.

## Measurement

`measure_multi_model` takes the placement plan and one observation. It returns the report. It does not load a model.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the multi-model report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the multi-model report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the multi-model report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says multi-model concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says multi-model run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says multi-model warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says multi-model request count is one.
12. A device count above 2: `CONTRACT_INVALID`, and the message says device count exceeds the record cap.
13. The report checks in the Report section, in the order written there.

`verify_multi_model` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the multi-model validation did not match.

## Files

`write_multi_model_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the multi-model directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the multi-model path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the multi-model output already exists.

The report is created with `create_new`, written, and `fsync`ed. A second model is not loaded.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_multi_model`. Domain separation has not started. An attention-approximation record has not started. A mixture-of-experts record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
