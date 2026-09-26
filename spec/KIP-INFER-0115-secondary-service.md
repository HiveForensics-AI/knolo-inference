# KIP-INFER-0115 — Secondary service

Status: `measure_secondary` records one secondary-slot service the worker did not start for a cold micro fixture. It returns a `knolo.infer.secondary-report`. The reason is `tiny`, `embeddings`, or `overflow`. The code is `CONTRACT_INVALID` and it is not retryable. The secondary slot is `slot-1`. A tiny-model request names the model and does not ask for embeddings or overflow. An embeddings request asks for embeddings and does not name a generation model or ask for overflow. An overflow request names the model and asks for overflow, and it does not ask for embeddings. `serviceStarted`, `embeddingsRan`, `overflowPlaced`, `deviceOpened`, and `tensorParallel` stay false. The primary root and the secondary root differ from each other and from the engine build and the placement. The forward does not run and no receipt is stored. It does not open a device and it does not run embeddings. The slot name is not a product name. `measure_multi_model` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-secondary, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `primaryRoot` | root distinct from the engine build and the placement |
| `secondaryRoot` | root distinct from the primary model, the engine build, and the placement |
| `reason` | `tiny`, `embeddings`, or `overflow` |
| `secondarySlot` | `slot-1` |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `modelNamed` | `true` for `tiny` and `overflow`; `false` for `embeddings` |
| `embeddingsRequested` | `true` only for `embeddings` |
| `overflowRequested` | `true` only for `overflow` |
| `serviceStarted` | `false` |
| `embeddingsRan` | `false` |
| `overflowPlaced` | `false` |
| `deviceOpened` | `false` |
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

`kind` is `knolo.infer.secondary-report` and `version` is `1`. The contract count is one hundred eight. `infer-secondary` is the report domain. A product name is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the secondary extensions are empty. A primary root equal to the engine build says the primary model repeats the engine build. A primary root equal to the placement says the primary model repeats the placement. A secondary root equal to the engine build says the secondary model repeats the engine build. A secondary root equal to the placement says the secondary model repeats the placement. A secondary root equal to the primary model says the secondary model repeats the primary model. A slot other than `slot-1` says a secondary service names slot-1. A code other than `CONTRACT_INVALID` says a secondary service is CONTRACT_INVALID. `retryable` true says a secondary service is not retryable. `serviceStarted` true says a secondary service does not start. `deviceOpened` true says a secondary service does not open a device. `tensorParallel` true says tensor parallel stays off. `embeddingsRan` true says a secondary service does not run embeddings. `overflowPlaced` true says a secondary service does not place overflow. `forwardRan` true says a secondary service does not run the forward. `receiptStored` true says a secondary service stores no receipt. A tiny-model request that does not name the model says a tiny-model request names the model. A tiny-model request that asks for embeddings says a tiny-model request does not ask for embeddings. A tiny-model request that asks for overflow says a tiny-model request does not ask for overflow. An embeddings request that names a model says an embeddings request does not name a model. An embeddings request that does not ask for embeddings says an embeddings request asks for embeddings. An embeddings request that asks for overflow says an embeddings request does not ask for overflow. An overflow request that does not name the model says an overflow request names the model. An overflow request that asks for embeddings says an overflow request does not ask for embeddings. An overflow request that does not ask for overflow says an overflow request asks for overflow.

## Measurement

`measure_secondary` takes the placement plan and one observation. It returns the report. It does not start a service.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the secondary report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the secondary report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the secondary report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says secondary concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says secondary run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says secondary warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says secondary request count is one.
12. The report checks in the Report section, in the order written there.

`verify_secondary` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the secondary validation did not match.

## Files

`write_secondary_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the secondary directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the secondary path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the secondary output already exists.

The report is created with `create_new`, written, and `fsync`ed. A device is not opened.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_secondary`. Domain separation has not started. An attention-approximation record has not started. A mixture-of-experts record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
