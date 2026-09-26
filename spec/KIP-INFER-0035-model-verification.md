# KIP-INFER-0035 — Model verification time

Status: `measure_model_verification` records the nanoseconds to verify one cold `knolo.micro.v1` image and the byte count that check covered. It returns a `knolo.infer.verification-report`. It does not open a model file, does not run a model, does not write a serve journal, does not read a GGUF payload, does not allocate a KV pool, and does not write a weight file. `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`, `measure_micro_throughput`, `measure_micro_latency`, `measure_corruption_fuzz`, `measure_cancellation_latency`, `measure_receipt_finalization`, `measure_receipt_overhead`, and `measure_model_swap` do not call it. `measure_model_load` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Duration

A duration is an integer number of nanoseconds. There is no floating-point value in the report. Queue time is not a field. Load time is not a field. Model verification time is the check of the image and its weight bytes, and it is greater than zero.

Verified bytes are the count of bytes that check covered. The count is greater than zero and at most 32 MiB, the same parser cap KIP-INFER-0017 uses. One verification of 4096 bytes in 40 nanoseconds records those two figures. The measurement does not stat a file. The caller supplies the count and the duration.

## Report

The report is a versioned contract. Its identity root is `H(infer-verification, document)`. Version 1 accepts only these fields:

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
| `verifiedBytes` | bytes the check covered |
| `modelVerificationNanos` | that check |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.verification-report` and `version` is `1`. The contract count is twenty-eight. `infer-verification` is the report domain.

Any other validation result is `CONTRACT_INVALID`, and the message says the field has an unsupported value. A stored report whose verified byte count is zero, or whose count is above 32 MiB, is `CONTRACT_INVALID`. The zero message says the verified byte count is zero. The cap message says verified bytes exceed the parser cap. A report is not issued when the measurement fails.

## Measurement

`measure_model_verification` takes the placement plan and one observation. The observation carries the model image root, the artifact root, the engine build root, the execution mode, the cache policy, the concurrency, the run count, the warm state, the request count, the verified byte count, and the duration. It returns the report. It does not create a file and it does not call the model. The caller supplies the count and the duration from a completed cold verification of the micro fixture.

The plan is the micro fixture plan: context reservation 16, KV block size 16, and KV precision `f32`. Graph capture stays `off`. Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the verification report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the verification report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the verification report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says verification concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says verification run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says verification warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says verification request count is one.
12. The verified byte count is zero: `CONTRACT_INVALID`, and the message says the verified byte count is zero.
13. The verified byte count is above 32 MiB: `INSUFFICIENT_MEMORY`, and the message says verified bytes exceed the parser cap.
14. The verification time is zero: `CONTRACT_INVALID`, and the message says the model verification time is zero.

`verify_model_verification` checks the stored report and recomputes the measurement. Checks run in this order:

1. The report fails `validate`, and that failure is returned.
2. The model image root, artifact root, or engine build root differs. The message says that root does not match.
3. The placement root differs: `CONTRACT_INVALID`, and the message says the placement root does not match.
4. The execution mode, cache policy, concurrency, run count, warm state, or request count differs. The message says that field does not match.
5. The verified byte count or the verification time differs. The message says that count or time does not match.
6. Recomputing the measurement fails, and that failure is returned.
7. The recomputed report bytes differ: `CONTRACT_INVALID`, and the message says the verification validation did not match.

`measure_model_verification` runs that verify before it returns. A verify failure returns no report.

## Files

`write_verification_report` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the verification directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the verification path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the verification output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The plan and the model bytes are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_model_verification`. Token ids are unchanged. Model load time is KIP-INFER-0036. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter. The dense Llama-family adapter stays deferred.
