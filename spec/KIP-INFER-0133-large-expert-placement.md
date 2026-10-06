# KIP-INFER-0133 — Large expert placement

Status: `measure_large_placement` records one large expert placement the planner did not apply for a cold micro fixture. It returns a `knolo.infer.large-placement-report`. The reason is `span`, `shard`, or `replicate`. The code is `CONTRACT_INVALID` and it is not retryable. Each reason names 1 through 16 experts. Exactly 16 is recorded. A count above 16 is `CONTRACT_INVALID` and issues no report. Placed experts stay 0. `spanned`, `sharded`, and `replicated` stay false. The forward does not run and no receipt is stored. It does not place an expert. `measure_multimodal` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-large-placement, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `span`, `shard`, or `replicate` |
| `requestedExperts` | `1` through `16` |
| `placedExperts` | `0` |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `spanned` | `false` |
| `sharded` | `false` |
| `replicated` | `false` |
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

`kind` is `knolo.infer.large-placement-report` and `version` is `1`. The contract count is one hundred thirty-three. `infer-large-placement` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the large-placement extensions are empty. A code other than `CONTRACT_INVALID` says large placement is CONTRACT_INVALID. `retryable` true says large placement is not retryable. `spanned` true says a span is not applied. `sharded` true says a shard is not applied. `replicated` true says a replica is not applied. `placedExperts` other than 0 says placed experts stay zero. A stored report above 16 says large placement exceeds the record cap. A requested count of 0 says that reason names an expert count. `forwardRan` true says large placement does not run the forward. `receiptStored` true says large placement stores no receipt.

## Measurement

`measure_large_placement` takes the placement plan and one observation. It returns the report. An expert is not placed.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the large-placement report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the large-placement report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the large-placement report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says large-placement concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says large-placement run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says large-placement warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says large-placement request count is one.
12. The report checks in the Report section, in the order written there.

A count above 16 on the observation is `CONTRACT_INVALID` and issues no report.

`verify_large_placement` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the large-placement validation did not match.

## Files

`write_large_placement_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the large-placement directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the large-placement path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the large-placement output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_large_placement`. Extreme-context cache has not started. Vision projector has not started. Positional extension has not started. Exact attention has not started. RMSNorm has not started. RoPE has not started. Fused residual has not started. Metal backend has not started. KV scatter has not started. Disaggregated serving has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
