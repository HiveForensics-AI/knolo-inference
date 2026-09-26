# KIP-INFER-0036 — Model load time

Status: `measure_model_load` records the nanoseconds to load one cold `knolo.micro.v1` fixture after verification has already finished. It returns a `knolo.infer.load-report`. It does not open a model file, does not run a model, does not write a serve journal, does not read a GGUF payload, does not allocate a KV pool, and does not write a weight file. `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`, `measure_micro_throughput`, `measure_micro_latency`, `measure_corruption_fuzz`, `measure_cancellation_latency`, `measure_receipt_finalization`, `measure_receipt_overhead`, `measure_model_swap`, and `measure_model_verification` do not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Duration

A duration is an integer number of nanoseconds. There is no floating-point value in the report. Queue time is not a field. Verification time is not a field. That time stays on the KIP-INFER-0035 report. Model load time is the load after that check, and it is greater than zero.

One load of 80 nanoseconds records model load time `80`. A load time of zero issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-load, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `modelImageRoot` | root of the micro model image |
| `artifactRoot` | root of the weight artifact |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` that ran |
| `placementRoot` | root of the `PlacementPlanV1` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `modelLoadNanos` | load after verification |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.load-report` and `version` is `1`. The contract count is twenty-nine. `infer-load` is the report domain.

Any other validation result is `CONTRACT_INVALID`, and the message says the field has an unsupported value. A report is not issued when the measurement fails.

## Measurement

`measure_model_load` takes the placement plan and one observation. The observation carries the model image root, the artifact root, the engine build root, the execution mode, the cache policy, the concurrency, the run count, the warm state, the request count, and the load duration. It returns the report. It does not create a file and it does not call the model. The caller supplies the duration from a completed cold load of the micro fixture.

The plan is the micro fixture plan: context reservation 16, KV block size 16, and KV precision `f32`. Graph capture stays `off`. Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the load report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the load report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the load report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says load concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says load run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says load warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says load request count is one.
12. The load time is zero: `CONTRACT_INVALID`, and the message says the model load time is zero.

`verify_model_load` checks the stored report and recomputes the measurement. Checks run in this order:

1. The report fails `validate`, and that failure is returned.
2. The model image root, artifact root, or engine build root differs. The message says that root does not match.
3. The placement root differs: `CONTRACT_INVALID`, and the message says the placement root does not match.
4. The execution mode, cache policy, concurrency, run count, warm state, or request count differs. The message says that field does not match.
5. The load time differs. The message says the model load time does not match.
6. Recomputing the measurement fails, and that failure is returned.
7. The recomputed report bytes differ: `CONTRACT_INVALID`, and the message says the load validation did not match.

`measure_model_load` runs that verify before it returns. A verify failure returns no report.

## Files

`write_load_report` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the load directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the load path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the load output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The plan and the model bytes are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_model_load`. Token ids are unchanged. Peak RAM and VRAM stay later. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter. The dense Llama-family adapter stays deferred.
