# KIP-INFER-0037 — Peak RAM and VRAM

Status: `measure_peak_memory` records the peak host RAM and the peak device VRAM of one cold `knolo.micro.v1` run. It returns a `knolo.infer.peak-report`. It does not sample a process, does not query a device, does not open a model file, does not run a model, does not write a serve journal, does not read a GGUF payload, does not allocate a KV pool, and does not write a weight file. `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`, `measure_micro_throughput`, `measure_micro_latency`, `measure_corruption_fuzz`, `measure_cancellation_latency`, `measure_receipt_finalization`, `measure_receipt_overhead`, `measure_model_swap`, `measure_model_verification`, and `measure_model_load` do not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Bytes

A byte count is an integer. There is no floating-point value in the report. Queue time is not a field. The planner's declared totals stay on the KIP-INFER-0027 estimate.

Peak RAM is the host high-water mark. It is greater than zero on both `cpu` and `slot-0`. Peak VRAM is the device high-water mark. A `cpu` placement records zero. A `slot-0` placement records a count greater than zero. A count above 64 MiB is `INSUFFICIENT_MEMORY` and issues no report. A stored report above that cap is `CONTRACT_INVALID`. Exactly 64 MiB is recorded.

One `cpu` run with 4096 host bytes and no device bytes records peak RAM `4096` and peak VRAM `0`. One `slot-0` run with 4096 host bytes and 2048 device bytes records those two counts.

## Report

The report is a versioned contract. Its identity root is `H(infer-peak, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `modelImageRoot` | root of the micro model image |
| `artifactRoot` | root of the weight artifact |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` that ran |
| `placementRoot` | root of the `PlacementPlanV1` |
| `device` | `cpu` or `slot-0` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `peakRamBytes` | host high-water mark |
| `peakVramBytes` | device high-water mark |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.peak-report` and `version` is `1`. The contract count is thirty. `infer-peak` is the report domain.

Any other validation result is `CONTRACT_INVALID`, and the message says the field has an unsupported value. A report is not issued when the measurement fails.

## Measurement

`measure_peak_memory` takes the placement plan and one observation. The observation carries the model image root, the artifact root, the engine build root, the execution mode, the cache policy, the concurrency, the run count, the warm state, the request count, and the two peaks. The device is taken from the plan. It returns the report. It does not create a file and it does not call the model. The caller supplies the peaks from a completed cold run of the micro fixture.

The plan is the micro fixture plan: context reservation 16, KV block size 16, and KV precision `f32`. Graph capture stays `off`. The device list is exactly `cpu` or exactly `slot-0`. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the peak report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the peak report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the peak report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says peak concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says peak run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says peak warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says peak request count is one.
12. The device list is not exactly `cpu` or exactly `slot-0`: `CONTRACT_INVALID`, and the message says the peak report is the micro fixture.
13. The peak RAM is zero: `CONTRACT_INVALID`, and the message says peak ram is zero.
14. The peak RAM is above 64 MiB: `INSUFFICIENT_MEMORY`, and the message says peak ram exceeds 64 MiB.
15. The device is `cpu` and the peak VRAM is not zero: `CONTRACT_INVALID`, and the message says cpu placement has no device memory.
16. The device is `slot-0` and the peak VRAM is zero: `CONTRACT_INVALID`, and the message says slot-0 placement records device memory.
17. The peak VRAM is above 64 MiB: `INSUFFICIENT_MEMORY`, and the message says peak vram exceeds 64 MiB.

`verify_peak_memory` checks the stored report and recomputes the measurement. Checks run in this order:

1. The report fails `validate`, and that failure is returned.
2. The model image root, artifact root, or engine build root differs. The message says that root does not match.
3. The placement root differs: `CONTRACT_INVALID`, and the message says the placement root does not match.
4. The device differs from the plan. The message says the device does not match.
5. The execution mode, cache policy, concurrency, run count, warm state, or request count differs. The message says that field does not match.
6. The peak RAM or peak VRAM differs. The message says that count does not match.
7. Recomputing the measurement fails, and that failure is returned.
8. The recomputed report bytes differ: `CONTRACT_INVALID`, and the message says the peak validation did not match.

`measure_peak_memory` runs that verify before it returns. A verify failure returns no report.

## Files

`write_peak_report` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the peak directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the peak path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the peak output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The plan is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_peak_memory`. Token ids are unchanged. KV utilization and prefix reuse stay later. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter. The dense Llama-family adapter stays deferred.
