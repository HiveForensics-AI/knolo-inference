# KIP-INFER-0041 — vLLM reference comparison

Status: `measure_vllm_comparison` records one cold comparison of the micro fixture with a pinned vLLM build. It returns a `knolo.infer.vllm-report`. It does not spawn vLLM, does not open a model file, does not run a model, does not write a serve journal, does not read a GGUF payload, does not allocate a KV pool, and does not write a weight file. `measure_llama_comparison` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Same pins

The comparison uses one weight artifact. The reference artifact root equals the native artifact root. Quantization is `f32` on both sides. The tokenizer, template, sampler, hardware probe, prompt distribution, and output distribution roots are the same on both sides. Context is 16 tokens. The reference build root differs from the engine build root. The reference runtime is `vllm`. The reference family is `safetensors`, which names the compatibility loader of the pin. The verification class is `sidecar-artifact-verified`. `native-verified` is rejected. A `gguf` family is rejected.

## Report

The report is a versioned contract. Its identity root is `H(infer-vllm, document)`. Version 1 accepts the same fields as the llama.cpp report, with these fixed values:

| Field | Value |
| --- | --- |
| `referenceRuntime` | `vllm` |
| `referenceFamily` | `safetensors` |
| `referenceVerificationClass` | `sidecar-artifact-verified` |
| `validationResult` | `recorded` |

`kind` is `knolo.infer.vllm-report` and `version` is `1`. The contract count is thirty-four. `infer-vllm` is the report domain.

## Measurement

`measure_vllm_comparison` takes the placement plan and one observation. Checks run in the same order as KIP-INFER-0040, with these vLLM messages:

1. The plan and cold-run checks name the vllm report.
2. Quantization or context outside the micro fixture: the message says the vllm comparison is the micro fixture.
3. `referenceRuntime` is not `vllm`: the message says the field has an unsupported value.
4. `referenceFamily` is not `safetensors`: the message says vllm comparison is a safetensors pin.
5. `referenceVerificationClass` is `native-verified`: the message says a compatibility backend is not native-verified.
6. Any other class: the message says the field has an unsupported value.
7. A repeated build, or a root that is not shared, uses the same messages as the llama.cpp comparison.

`verify_vllm_comparison` recomputes the report. A byte mismatch says the vllm validation did not match.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged.

## Files

`write_vllm_report` uses `create_new` inside a caller-supplied directory. A missing directory says the vllm directory does not exist. A symlink escape says the vllm path leaves the directory. An existing output says the vllm output already exists. The plan is not written. vLLM is not executed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_vllm_comparison`. Token ids are unchanged. The mistral.rs comparison is KIP-INFER-0042. The recipe mark is KIP-INFER-0043. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
