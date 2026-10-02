# KIP-INFER-0078 — CUDA fault

Status: `measure_cuda_fault` records one CUDA fault for a cold micro fixture. It returns a `knolo.infer.fault-report`. The device is `slot-0`. The class is `kernel` or `device`. The code is `CUDA_FAULT` and it is not retryable. The supervisor stays up, the listener stays up, and no receipt is stored. There is no fallback to CPU. CUDA graphs stay off. It does not query a device. `measure_cuda_oom` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-fault, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `device` | `slot-0` |
| `faultClass` | `kernel` or `device` |
| `code` | `CUDA_FAULT` |
| `retryable` | `false` |
| `receiptStored` | `false` |
| `listenerUp` | `true` |
| `supervisorExited` | `false` |
| `cpuFallback` | `false` |
| `graphCaptured` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.fault-report` and `version` is `1`. The contract count is seventy-three. `infer-fault` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the fault extensions are empty. A device other than `slot-0` says a cuda fault record names slot-0. A class other than `kernel` or `device` says the field has an unsupported value. A code other than `CUDA_FAULT` says a cuda fault record is CUDA_FAULT. `retryable` true says a cuda fault record is not retryable. `receiptStored` true says a cuda fault record stores no receipt. `listenerUp` false says the listener stays up. `supervisorExited` true says a cuda fault does not exit the supervisor. `cpuFallback` true says a cuda fault does not fall back to cpu. `graphCaptured` true says cuda graphs stay off.

## Measurement

`measure_cuda_fault` takes the placement plan and one observation. It returns the report. It does not query the device and it does not capture a graph.

Device `slot-0` records a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the fault report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the fault report is the micro fixture.
5. The placement device is not `slot-0`: `CONTRACT_INVALID`, and the message says a cuda fault record names slot-0.
6. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
7. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
8. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the fault report.
9. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says fault concurrency is one.
10. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says fault run count is one.
11. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says fault warm state is cold.
12. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says fault request count is one.
13. The report checks in the Report section, in the order written there.

`verify_cuda_fault` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the fault validation did not match.

## Files

`write_fault_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the fault directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the fault path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the fault output already exists.

The report is created with `create_new`, written, and `fsync`ed. The device is not queried.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_cuda_fault`. The request-timeout record is KIP-INFER-0079. The challenge hash has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
