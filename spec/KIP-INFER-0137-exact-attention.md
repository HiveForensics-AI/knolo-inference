# KIP-INFER-0137 — Exact attention

Status: `measure_exact_attention` records one exact attention kernel the engine did not select for a cold micro fixture. It returns a `knolo.infer.exact-attention-report`. The reason is `flash`, `tile`, or `split`. The code is `UNSUPPORTED_KERNEL` and it is not retryable. Each reason names 1 through 16 tiles. Exactly 16 is recorded. A count above 16 is `CONTRACT_INVALID` and issues no report. Selected tiles stay 0. `flashSelected`, `tiled`, and `split` stay false. The forward does not run and no receipt is stored. It does not select an exact attention kernel. `measure_positional` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-exact-attention, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `flash`, `tile`, or `split` |
| `requestedTiles` | `1` through `16` |
| `selectedTiles` | `0` |
| `code` | `UNSUPPORTED_KERNEL` |
| `retryable` | `false` |
| `flashSelected` | `false` |
| `tiled` | `false` |
| `split` | `false` |
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

`kind` is `knolo.infer.exact-attention-report` and `version` is `1`. The contract count is one hundred thirty-three. `infer-exact-attention` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the exact-attention extensions are empty. A code other than `UNSUPPORTED_KERNEL` says an exact attention kernel is UNSUPPORTED_KERNEL. `retryable` true says an exact attention kernel is not retryable. `flashSelected` true says a flash kernel is not selected. `tiled` true says an attention tile is not selected. `split` true says a split kernel is not selected. `selectedTiles` other than 0 says selected attention tiles stay zero. A stored report above 16 says exact attention tiles exceed the record cap. A requested count of 0 says that reason names a tile count. `forwardRan` true says an exact attention kernel does not run the forward. `receiptStored` true says an exact attention kernel stores no receipt.

## Measurement

`measure_exact_attention` takes the placement plan and one observation. It returns the report. An exact attention kernel is not selected.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the exact-attention report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the exact-attention report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the exact-attention report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says exact-attention concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says exact-attention run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says exact-attention warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says exact-attention request count is one.
12. The report checks in the Report section, in the order written there.

A count above 16 on the observation is `CONTRACT_INVALID` and issues no report.

`verify_exact_attention` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the exact-attention validation did not match.

## Files

`write_exact_attention_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the exact-attention directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the exact-attention path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the exact-attention output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_exact_attention`. RMSNorm has not started. RoPE has not started. Fused residual has not started. Metal backend has not started. KV scatter has not started. Disaggregated serving has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
