# KIP-INFER-0113 — CUDA graph

Status: `measure_cuda_graph` records one CUDA graph constraint that was not captured for a cold micro fixture. It returns a `knolo.infer.graph-report`. The reason is `identity`, `kernel`, `workspace`, or `shape`. The device is `slot-0`. A `cpu` placement issues no report. The code is `CONTRACT_INVALID` and it is not retryable. `graphCaptured` stays false. Captured nodes, workspace bytes, and shape buckets stay 0. `kernelRepeated` stays false. The forward does not run and no receipt is stored. It does not capture a graph. `measure_cuda_fault` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-graph, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `identity`, `kernel`, `workspace`, or `shape` |
| `device` | `slot-0` |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `graphCaptured` | `false` |
| `capturedNodes` | `0` |
| `workspaceBytes` | `0` |
| `shapeBuckets` | `0` |
| `kernelRepeated` | `false` |
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

`kind` is `knolo.infer.graph-report` and `version` is `1`. The contract count is one hundred eight. `infer-graph` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the graph extensions are empty. A device other than `slot-0` says a cuda graph record names slot-0. A code other than `CONTRACT_INVALID` says a cuda graph record is CONTRACT_INVALID. `retryable` true says a cuda graph record is not retryable. `graphCaptured` true says cuda graphs stay off. A non-zero node count says captured nodes stay zero while graphs are off. A non-zero workspace says graph workspace stays zero while graphs are off. A non-zero bucket count says shape buckets stay zero while graphs are off. `kernelRepeated` true says a cuda graph record does not repeat a kernel. `forwardRan` true says a cuda graph record does not run the forward. `receiptStored` true says a cuda graph record stores no receipt.

## Measurement

`measure_cuda_graph` takes the placement plan and one observation. It returns the report. It does not capture a graph.

A `cpu` placement issues no report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the graph report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the graph report is the micro fixture.
5. The device is not `slot-0`: `CONTRACT_INVALID`, and the message says a cuda graph record names slot-0.
6. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
7. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
8. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the graph report.
9. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says graph concurrency is one.
10. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says graph run count is one.
11. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says graph warm state is cold.
12. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says graph request count is one.
13. The report checks in the Report section, in the order written there.

`verify_cuda_graph` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the graph validation did not match.

## Files

`write_cuda_graph_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the graph directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the graph path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the graph output already exists.

The report is created with `create_new`, written, and `fsync`ed. A graph is not captured.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_cuda_graph`. Domain separation has not started. An attention-approximation record has not started. A mixture-of-experts record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
