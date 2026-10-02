# KIP-INFER-0122 — GLM-4.7 adapter

Status: `measure_glm` records one GLM-4.7 adapter the binary does not compile for a cold micro fixture. It returns a `knolo.infer.glm-report`. The adapter is `knolo.glm.v1`. `knolo.micro.v1` issues no report. The reason is `adapter`, `inventory`, or `recipe`. The code is `UNSUPPORTED_ARCHITECTURE` and it is not retryable. An adapter refusal does not open weights and does not read an inventory. An inventory refusal says the weights were opened and the inventory was not read. A recipe refusal says the weights were opened and the inventory was read. `recipeBlessed` stays false. The forward does not run and no receipt is stored. It does not open a weight file. `measure_router` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-glm, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `adapter`, `inventory`, or `recipe` |
| `rejectedAdapter` | `knolo.glm.v1` |
| `code` | `UNSUPPORTED_ARCHITECTURE` |
| `retryable` | `false` |
| `weightsOpened` | `false` for `adapter`; `true` for `inventory` and `recipe` |
| `inventoryRead` | `true` only for `recipe` |
| `recipeBlessed` | `false` |
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

`kind` is `knolo.infer.glm-report` and `version` is `1`. The contract count is one hundred eighteen. `infer-glm` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the glm extensions are empty. `knolo.micro.v1` says the micro adapter is compiled in. Any other adapter says a glm record names knolo.glm.v1. A code other than `UNSUPPORTED_ARCHITECTURE` says an unsupported glm adapter is UNSUPPORTED_ARCHITECTURE. `retryable` true says a glm adapter is not retryable. `recipeBlessed` true says a glm recipe is not blessed. An adapter refusal that opens weights says an adapter refusal does not open weights. An adapter refusal that reads the inventory says an adapter refusal does not read the inventory. An inventory refusal that does not open weights says an inventory refusal opened the weights. An inventory refusal that reads the inventory says an inventory refusal does not read the inventory. A recipe refusal that does not open weights says a recipe refusal opened the weights. A recipe refusal that does not read the inventory says a recipe refusal read the inventory. `forwardRan` true says a glm adapter does not run the forward. `receiptStored` true says a glm adapter stores no receipt.

## Measurement

`measure_glm` takes the placement plan and one observation. It returns the report. The GLM adapter is not compiled in.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the glm report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the glm report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the glm report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says glm concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says glm run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says glm warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says glm request count is one.
12. The report checks in the Report section, in the order written there.

`verify_glm` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the glm validation did not match.

## Files

`write_glm_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the glm directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the glm path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the glm output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_glm`. Payload hashing has not started. Hybrid attention has not started. Linear attention has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
