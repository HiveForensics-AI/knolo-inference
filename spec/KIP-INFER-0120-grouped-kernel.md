# KIP-INFER-0120 — Grouped kernel

Status: `measure_grouped_kernel` records one grouped expert kernel the planner did not select for a cold micro fixture. It returns a `knolo.infer.grouped-kernel-report`. The reason is `gemm` or `routing`. The code is `UNSUPPORTED_KERNEL` and it is not retryable. A gemm refusal names 1 through 16 groups. Exactly 16 is recorded. A routing refusal carries no group. A count above 16 is `CONTRACT_INVALID` and issues no report. `kernelSelected`, `grouped`, and `routingRan` stay false. The forward does not run and no receipt is stored. It does not select a kernel. `measure_expert_placement` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-grouped-kernel, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `gemm` or `routing` |
| `groupCount` | `1` through `16` for `gemm`; `0` for `routing` |
| `code` | `UNSUPPORTED_KERNEL` |
| `retryable` | `false` |
| `kernelSelected` | `false` |
| `grouped` | `false` |
| `routingRan` | `false` |
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

`kind` is `knolo.infer.grouped-kernel-report` and `version` is `1`. The contract count is one hundred eighteen. `infer-grouped-kernel` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the grouped-kernel extensions are empty. A code other than `UNSUPPORTED_KERNEL` says a grouped kernel is UNSUPPORTED_KERNEL. `retryable` true says a grouped kernel is not retryable. `kernelSelected` true says a grouped kernel is not selected. `grouped` true says a grouped gemm is not selected. `routingRan` true says top-k routing does not run. A group count above 16 says the group count exceeds the record cap. A gemm refusal with no group says a grouped gemm names a group. A routing refusal with a group says a routing refusal carries no group. `forwardRan` true says a grouped kernel does not run the forward. `receiptStored` true says a grouped kernel stores no receipt.

## Measurement

`measure_grouped_kernel` takes the placement plan and one observation. It returns the report. The grouped kernel is not selected.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the grouped-kernel report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the grouped-kernel report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the grouped-kernel report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says grouped-kernel concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says grouped-kernel run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says grouped-kernel warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says grouped-kernel request count is one.
12. The report checks in the Report section, in the order written there.

`verify_grouped_kernel` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the grouped-kernel validation did not match.

## Files

`write_grouped_kernel_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the grouped-kernel directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the grouped-kernel path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the grouped-kernel output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_grouped_kernel`. Payload hashing has not started. Hybrid attention has not started. Linear attention has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
