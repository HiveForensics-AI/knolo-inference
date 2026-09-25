# KIP-INFER-0077 — CUDA out of memory

Status: `measure_cuda_oom` records one CUDA out-of-memory result for a cold micro fixture. It returns a `knolo.infer.oom-report`. The device is `slot-0`. Free device bytes are zero. Needed bytes are 1 through 64 MiB. The code is `CUDA_OOM` and it is retryable. The supervisor stays up, the listener stays up, and no receipt is stored. There is no fallback to CPU. It does not allocate device memory. `measure_point_equality` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Space

A needed count above 64 MiB is `INSUFFICIENT_MEMORY` and issues no report. A stored report above that cap is `CONTRACT_INVALID`. Exactly 64 MiB is recorded. A needed count of zero is `CONTRACT_INVALID`. A `cpu` placement is `CONTRACT_INVALID`.

## Report

The report is a versioned contract. Its identity root is `H(infer-oom, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `device` | `slot-0` |
| `freeBytes` | `0` |
| `neededBytes` | `1` through `67108864` |
| `code` | `CUDA_OOM` |
| `retryable` | `true` |
| `receiptStored` | `false` |
| `listenerUp` | `true` |
| `supervisorExited` | `false` |
| `cpuFallback` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.oom-report` and `version` is `1`. The contract count is seventy-three. `infer-oom` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the oom extensions are empty. A device other than `slot-0` says a cuda oom record names slot-0. A free byte count other than zero says a cuda oom record has no free device bytes. A needed count of zero says a cuda oom record needs device bytes. A stored needed count above 64 MiB says needed bytes exceed 64 MiB. A code other than `CUDA_OOM` says a cuda oom record is CUDA_OOM. `retryable` false says a cuda oom record is retryable. `receiptStored` true says a cuda oom record stores no receipt. `listenerUp` false says the listener stays up. `supervisorExited` true says a cuda oom does not exit the supervisor. `cpuFallback` true says a cuda oom does not fall back to cpu.

## Measurement

`measure_cuda_oom` takes the placement plan and one observation. It returns the report. It does not allocate device memory and it does not move the request to CPU.

Device `slot-0` records a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the oom report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the oom report is the micro fixture.
5. The placement device is not `slot-0`: `CONTRACT_INVALID`, and the message says a cuda oom record names slot-0.
6. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
7. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
8. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the oom report.
9. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says oom concurrency is one.
10. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says oom run count is one.
11. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says oom warm state is cold.
12. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says oom request count is one.
13. `neededBytes` is above 64 MiB: `INSUFFICIENT_MEMORY`, and the message says needed bytes exceed 64 MiB.
14. The report checks in the Report section, in the order written there.

`verify_cuda_oom` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the oom validation did not match.

## Files

`write_oom_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the oom directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the oom path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the oom output already exists.

The report is created with `create_new`, written, and `fsync`ed. No device allocation is performed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_cuda_oom`. The CUDA fault record is KIP-INFER-0078. The challenge hash has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
