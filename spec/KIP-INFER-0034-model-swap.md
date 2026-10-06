# KIP-INFER-0034 — Model swap time

Status: `measure_model_swap` records the nanoseconds to replace one resident model image with the incoming `knolo.micro.v1` fixture for one cold swap. It returns a `knolo.infer.swap-report`. It does not unload a worker, does not open a model file, does not run a model, does not write a serve journal, does not read a GGUF payload, does not allocate a KV pool, and does not write a weight file. `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`, `measure_micro_throughput`, `measure_micro_latency`, `measure_corruption_fuzz`, `measure_cancellation_latency`, `measure_receipt_finalization`, and `measure_receipt_overhead` do not call it. `measure_model_verification` and `measure_model_load` do not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Duration

A duration is an integer number of nanoseconds. There is no floating-point value in the report. Queue time is not a field. The drain of an admitted request is not a field. Unload time starts after inflight work is already zero, which is the same point KIP-INFER-0013 uses before the worker drops the model.

Unload time is the drop of the resident image. Incoming verification time is the check of the incoming image. Incoming load time is the load of that image after verification, and it does not include the verification time. Each of the three is greater than zero. Model swap time is their sum.

One swap whose unload is 50 nanoseconds, whose incoming verification is 30 nanoseconds, and whose incoming load is 80 nanoseconds has model swap time `160`. One swap of 10, 20, and 5 nanoseconds has model swap time `35`.

The incoming model image root and the incoming artifact root are the report's `modelImageRoot` and `artifactRoot`. The resident roots are separate fields. A swap whose resident model image root equals the incoming model image root is not a replacement. A swap whose resident artifact root equals the incoming artifact root is not a replacement. A sum that overflows issues no report. The resident image is not opened. This slice does not serve two models.

## Report

The report is a versioned contract. Its identity root is `H(infer-swap, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `modelImageRoot` | root of the incoming micro model image |
| `artifactRoot` | root of the incoming weight artifact |
| `residentModelImageRoot` | root of the image that was unloaded |
| `residentArtifactRoot` | root of the artifact that was unloaded |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` that ran |
| `placementRoot` | root of the incoming `PlacementPlanV1` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `unloadNanos` | resident drop |
| `incomingVerificationNanos` | incoming image check |
| `incomingLoadNanos` | incoming load after that check |
| `modelSwapNanos` | unload plus incoming verification plus incoming load |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.swap-report` and `version` is `1`. The contract count is twenty-seven. `infer-swap` is the report domain.

Any other validation result is `CONTRACT_INVALID`, and the message says the field has an unsupported value. A stored swap time that is not the sum above is `CONTRACT_INVALID`, and the message says model swap time does not match. A report is not issued when the measurement fails.

## Measurement

`measure_model_swap` takes the incoming placement plan and one observation. The observation carries the incoming model image root, the incoming artifact root, the resident model image root, the resident artifact root, the engine build root, the execution mode, the cache policy, the concurrency, the run count, the warm state, the request count, and the three durations. It returns the report. It does not create a file and it does not call the model. The caller supplies roots and durations from a completed cold swap onto the micro fixture.

The plan is the micro fixture plan: context reservation 16, KV block size 16, and KV precision `f32`. Graph capture stays `off`. Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the swap report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the swap report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the swap report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says swap concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says swap run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says swap warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says swap request count is one.
12. The resident model image root equals the incoming model image root: `CONTRACT_INVALID`, and the message says a swap replaces a different model image.
13. The resident artifact root equals the incoming artifact root: `CONTRACT_INVALID`, and the message says a swap replaces a different artifact.
14. The unload time is zero: `CONTRACT_INVALID`, and the message says the unload time is zero.
15. The incoming verification time is zero: `CONTRACT_INVALID`, and the message says the incoming verification time is zero.
16. The incoming load time is zero: `CONTRACT_INVALID`, and the message says the incoming load time is zero.
17. The sum overflows: `CONTRACT_INVALID`, and the message says model swap time overflows.

`verify_model_swap` checks the stored report and recomputes the measurement. Checks run in this order:

1. The report fails `validate`, and that failure is returned.
2. The model image root, artifact root, or engine build root differs. The message says that root does not match.
3. The placement root differs: `CONTRACT_INVALID`, and the message says the placement root does not match.
4. The resident model image root or the resident artifact root differs. The message says that root does not match.
5. The execution mode, cache policy, concurrency, run count, warm state, or request count differs. The message says that field does not match.
6. A duration differs. The message says that time does not match.
7. Recomputing the measurement fails, and that failure is returned.
8. The recomputed report bytes differ: `CONTRACT_INVALID`, and the message says the swap validation did not match.

`measure_model_swap` runs that verify before it returns. A verify failure returns no report.

## Files

`write_swap_report` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the swap directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the swap path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the swap output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The plan and the resident bytes are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_model_swap`. Token ids are unchanged. Model verification time is KIP-INFER-0035. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter. The dense Llama-family adapter stays deferred.
