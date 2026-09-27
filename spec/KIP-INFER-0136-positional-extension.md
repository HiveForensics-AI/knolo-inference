# KIP-INFER-0136 — Positional extension

Status: `measure_positional` records one positional extension the engine did not apply for a cold micro fixture. It returns a `knolo.infer.positional-report`. The reason is `yarn`, `ntk`, or `rope`. The code is `CONTRACT_INVALID` and it is not retryable. Each reason names a scale of 1 through 16. Exactly 16 is recorded. A scale above 16 is `CONTRACT_INVALID` and issues no report. Applied scale stays 0. `yarnApplied`, `ntkApplied`, and `ropeScaled` stay false. The forward does not run and no receipt is stored. It does not rescale RoPE. `measure_vision` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-positional, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `yarn`, `ntk`, or `rope` |
| `requestedScale` | `1` through `16` |
| `appliedScale` | `0` |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `yarnApplied` | `false` |
| `ntkApplied` | `false` |
| `ropeScaled` | `false` |
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

`kind` is `knolo.infer.positional-report` and `version` is `1`. The contract count is one hundred thirty-three. `infer-positional` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the positional extensions are empty. A code other than `CONTRACT_INVALID` says positional extension is CONTRACT_INVALID. `retryable` true says positional extension is not retryable. `yarnApplied` true says a YaRN scale is not applied. `ntkApplied` true says an NTK scale is not applied. `ropeScaled` true says RoPE is not rescaled. `appliedScale` other than 0 says applied positional scale stays zero. A stored report above 16 says positional scale exceeds the record cap. A requested count of 0 says that reason names a scale. `forwardRan` true says positional extension does not run the forward. `receiptStored` true says positional extension stores no receipt.

## Measurement

`measure_positional` takes the placement plan and one observation. It returns the report. RoPE is not rescaled.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the positional report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the positional report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the positional report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says positional concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says positional run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says positional warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says positional request count is one.
12. The report checks in the Report section, in the order written there.

A count above 16 on the observation is `CONTRACT_INVALID` and issues no report.

`verify_positional` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the positional validation did not match.

## Files

`write_positional_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the positional directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the positional path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the positional output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_positional`. Exact attention has not started. RMSNorm has not started. RoPE has not started. Fused residual has not started. Metal backend has not started. KV scatter has not started. Disaggregated serving has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
