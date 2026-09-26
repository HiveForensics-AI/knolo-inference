# KIP-INFER-0093 — Unsupported kernel

Status: `measure_kernel` records one backend the micro adapter does not select for a cold micro fixture. It returns a `knolo.infer.kernel-report`. The reason is `backend` or `feature`. A foreign backend does not request CUDA. A missing feature requests `candle-cuda`. The code is `UNSUPPORTED_KERNEL` and it is not retryable. The kernel is not selected and the device is not opened. The forward does not run and no receipt is stored. It does not open a device. `measure_quantization` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-kernel, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `backend` or `feature` |
| `code` | `UNSUPPORTED_KERNEL` |
| `retryable` | `false` |
| `cudaRequested` | `false` for `backend`; `true` for `feature` |
| `kernelSelected` | `false` |
| `deviceOpened` | `false` |
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

`kind` is `knolo.infer.kernel-report` and `version` is `1`. The contract count is eighty-eight. `infer-kernel` is the report domain. A backend name is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the kernel extensions are empty. A code other than `UNSUPPORTED_KERNEL` says an unsupported kernel is UNSUPPORTED_KERNEL. `retryable` true says an unsupported kernel is not retryable. `forwardRan` true says an unsupported kernel does not run the forward. `receiptStored` true says an unsupported kernel stores no receipt. `kernelSelected` true says an unsupported kernel is not selected. `deviceOpened` true says an unsupported kernel does not open a device. `cudaRequested` true on `backend` says a foreign backend does not request cuda. `cudaRequested` false on `feature` says a missing feature requests candle-cuda.

## Measurement

`measure_kernel` takes the placement plan and one observation. It returns the report. It does not select a kernel and it does not open a device.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the kernel report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the kernel report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the kernel report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says kernel concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says kernel run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says kernel warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says kernel request count is one.
12. The report checks in the Report section, in the order written there.

`verify_kernel` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the kernel validation did not match.

## Files

`write_kernel_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the kernel directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the kernel path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the kernel output already exists.

The report is created with `create_new`, written, and `fsync`ed. No device is opened.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_kernel`. The placement-refusal record is KIP-INFER-0094. The signature check has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
