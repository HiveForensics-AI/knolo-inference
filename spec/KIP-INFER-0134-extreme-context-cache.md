# KIP-INFER-0134 — Extreme-context cache

Status: `measure_extreme_cache` records one million-token cache policy the engine did not apply for a cold micro fixture. It returns a `knolo.infer.extreme-cache-report`. The reason is `window`, `yarn`, or `page`. The code is `CONTRACT_INVALID` and it is not retryable. Each reason names 1 through 16 tokens. Exactly 16 is recorded. A count above 16 is `CONTEXT_LIMIT_EXCEEDED` and issues no report. Cached tokens stay 0. `windowed`, `yarnApplied`, and `paged` stay false. The forward does not run and no receipt is stored. It does not allocate an extreme-context page. `measure_large_placement` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-extreme-cache, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `window`, `yarn`, or `page` |
| `requestedTokens` | `1` through `16` |
| `cachedTokens` | `0` |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `windowed` | `false` |
| `yarnApplied` | `false` |
| `paged` | `false` |
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

`kind` is `knolo.infer.extreme-cache-report` and `version` is `1`. The contract count is one hundred thirty-three. `infer-extreme-cache` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the extreme-cache extensions are empty. A code other than `CONTRACT_INVALID` says extreme-context cache is CONTRACT_INVALID. `retryable` true says extreme-context cache is not retryable. `windowed` true says an extreme window is not applied. `yarnApplied` true says a YaRN scale is not applied. `paged` true says an extreme page is not allocated. `cachedTokens` other than 0 says cached extreme-context tokens stay zero. A stored report above 16 says extreme-context tokens exceed the record cap. A requested count of 0 says that reason names a token count. `forwardRan` true says extreme-context cache does not run the forward. `receiptStored` true says extreme-context cache stores no receipt.

## Measurement

`measure_extreme_cache` takes the placement plan and one observation. It returns the report. An extreme-context page is not allocated.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the extreme-cache report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the extreme-cache report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the extreme-cache report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says extreme-cache concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says extreme-cache run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says extreme-cache warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says extreme-cache request count is one.
12. The report checks in the Report section, in the order written there.

A count above 16 on the observation is `CONTEXT_LIMIT_EXCEEDED` and issues no report.

`verify_extreme_cache` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the extreme-cache validation did not match.

## Files

`write_extreme_cache_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the extreme-cache directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the extreme-cache path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the extreme-cache output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_extreme_cache`. Vision projector has not started. Positional extension has not started. Exact attention has not started. RMSNorm has not started. RoPE has not started. Fused residual has not started. Metal backend has not started. KV scatter has not started. Disaggregated serving has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
