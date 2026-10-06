# KIP-INFER-0040 — llama.cpp reference comparison

Status: `measure_llama_comparison` records one cold comparison of the micro fixture with a pinned llama.cpp build. It returns a `knolo.infer.llama-report`. It does not spawn llama.cpp, does not open a model file, does not run a model, does not write a serve journal, does not read a GGUF payload, does not allocate a KV pool, and does not write a weight file. `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`, `measure_micro_throughput`, `measure_micro_latency`, `measure_corruption_fuzz`, `measure_cancellation_latency`, `measure_receipt_finalization`, `measure_receipt_overhead`, `measure_model_swap`, `measure_model_verification`, `measure_model_load`, `measure_peak_memory`, `measure_kv_utilization`, and `measure_prefix_reuse` do not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Same pins

The comparison uses one weight artifact. The reference artifact root equals the native artifact root. Quantization is `f32` on both sides. The tokenizer, template, sampler, hardware probe, prompt distribution, and output distribution roots are the same on both sides. Context is 16 tokens. The reference build root differs from the engine build root. The reference runtime is `llama.cpp`. The reference family is `gguf`, which names the compatibility loader of the pin. The native artifact stays safetensors. The verification class is `sidecar-artifact-verified`. `native-verified` is rejected.

## Report

The report is a versioned contract. Its identity root is `H(infer-llama, document)`. Version 1 accepts only these fields:

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
| `quantization` | `f32` |
| `contextTokens` | `16` |
| `tokenizerRoot` | root shared with the reference |
| `templateRoot` | root shared with the reference |
| `samplerRoot` | root shared with the reference |
| `hardwareProbeRoot` | root shared with the reference |
| `promptDistributionRoot` | root shared with the reference |
| `outputDistributionRoot` | root shared with the reference |
| `referenceRuntime` | `llama.cpp` |
| `referenceFamily` | `gguf` |
| `referenceBuildRoot` | root of the pinned llama.cpp build |
| `referenceArtifactRoot` | the same artifact root |
| `referenceQuantization` | `f32` |
| `referenceTokenizerRoot` | the same tokenizer root |
| `referenceTemplateRoot` | the same template root |
| `referenceSamplerRoot` | the same sampler root |
| `referenceHardwareRoot` | the same hardware probe root |
| `referencePromptDistributionRoot` | the same prompt distribution root |
| `referenceOutputDistributionRoot` | the same output distribution root |
| `referenceVerificationClass` | `sidecar-artifact-verified` |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.llama-report` and `version` is `1`. The contract count is thirty-three. `infer-llama` is the report domain.

Any other validation result is `CONTRACT_INVALID`, and the message says the field has an unsupported value. A report is not issued when the measurement fails.

## Measurement

`measure_llama_comparison` takes the placement plan and one observation. The observation carries the roots and the reference pin. It returns the report. It does not create a file and it does not call the model. The caller supplies the pin from a completed cold run of the micro fixture.

The plan is the micro fixture plan: context reservation 16, KV block size 16, and KV precision `f32`. Graph capture stays `off`. Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the llama report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the llama report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the llama report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says llama concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says llama run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says llama warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says llama request count is one.
12. Quantization is not `f32`, or the context is not 16: `CONTRACT_INVALID`, and the message says the llama comparison is the micro fixture.
13. `referenceRuntime` is not `llama.cpp`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
14. `referenceFamily` is not `gguf`: `CONTRACT_INVALID`, and the message says llama.cpp comparison is a gguf pin.
15. `referenceVerificationClass` is `native-verified`: `CONTRACT_INVALID`, and the message says a compatibility backend is not native-verified.
16. `referenceVerificationClass` is not `sidecar-artifact-verified`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
17. The reference build root equals the engine build root: `CONTRACT_INVALID`, and the message says the reference build is the engine build.
18. The reference artifact root differs: `CONTRACT_INVALID`, and the message says the artifact root does not match.
19. The reference quantization differs: `CONTRACT_INVALID`, and the message says quantization does not match.
20. The tokenizer, template, sampler, hardware probe, prompt distribution, or output distribution root differs. The message says that root does not match.

`verify_llama_comparison` checks the stored report and recomputes the measurement. Checks run in this order:

1. The report fails `validate`, and that failure is returned.
2. A stored root or count differs from the observation or the plan. The message says that field does not match.
3. Recomputing the measurement fails, and that failure is returned.
4. The recomputed report bytes differ: `CONTRACT_INVALID`, and the message says the llama validation did not match.

`measure_llama_comparison` runs that verify before it returns. A verify failure returns no report.

## Files

`write_llama_report` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the llama directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the llama path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the llama output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The plan is not written. llama.cpp is not executed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_llama_comparison`. Token ids are unchanged. The vLLM comparison is KIP-INFER-0041. The mistral.rs comparison is KIP-INFER-0042. The recipe mark is KIP-INFER-0043. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter. The dense Llama-family adapter stays deferred.
