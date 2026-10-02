# KIP-INFER-0026 — Perplexity report

Status: `perplexity_delta` compares one reference `f32` logit matrix with one candidate `f32` logit matrix on the same target token ids. It returns a `knolo.infer.perplexity-report`. It does not run a model, does not read a GGUF payload, and does not write a weight file. `dequant_gguf`, `quant_gemm`, and `convert_gguf_tensor` do not call it. `knolo-infer run` and `knolo-infer serve` do not call it. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Report

The report is a versioned contract. Its identity root is `H(infer-perplexity, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `referenceLogitRoot` | `H(infer-logits, little-endian f32 bytes)` of the reference matrix |
| `candidateLogitRoot` | the same digest of the candidate matrix |
| `targetTokenRoot` | `output_token_root` of the target ids |
| `reporterBuildRoot` | root of the `EngineBuildDescriptorV1` that measured |
| `tokenCount` | the number of scored positions |
| `referencePerplexityMicros` | reference perplexity, in micros |
| `candidatePerplexityMicros` | candidate perplexity, in micros |
| `perplexityDeltaMicros` | candidate minus reference, in micros |
| `greedyParity` | true when every lowest-index argmax matches |
| `targetAccuracyMillionths` | candidate argmax hits on the targets, in millionths |
| `maxAbsLogitDeltaMillionths` | the largest absolute logit difference, in millionths |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.perplexity-report` and `version` is `1`. Any other validation result is `CONTRACT_INVALID`, and the message says the field has an unsupported value. A report is not issued when the measurement fails. `perplexityDeltaMicros` must equal the candidate perplexity minus the reference perplexity. Any other difference is `CONTRACT_INVALID`, and the message says the perplexity delta does not match. `tokenCount` of 0 is `CONTRACT_INVALID`, and the message says the perplexity token count is zero. `targetAccuracyMillionths` above `1000000` is `CONTRACT_INVALID`, and the message says the target accuracy is outside its bounds.

`infer-logits` hashes the raw little-endian `f32` bytes in row-major order. It is not a contract kind. The target token root reuses `infer-output-tokens`.

## Measurement

`perplexity_delta` takes the vocabulary, the target ids, the reference logits, the candidate logits, and the reporter build root. Both matrices are row-major. Row `t` has `vocab` logits and scores target `t`. The function returns the report. It does not create a file. The logit slices are not written back.

Checks run in this order. The first failure is the one returned, and it returns no report. A failure before the value checks does not read a logit.

1. `vocab` is 0: `CONTRACT_INVALID`, and the message says the perplexity vocab is zero.
2. The token count is 0: `CONTRACT_INVALID`, and the message says the perplexity token count is zero.
3. The element count or its byte length overflows: `CONTRACT_INVALID`, and the message says the perplexity logit length overflows.
4. The matrix is above 32 MiB: `INSUFFICIENT_MEMORY`, and the message says the perplexity logit matrix exceeds 32 MiB.
5. A target id is outside the vocabulary: `CONTRACT_INVALID`, and the message says the perplexity target is outside the vocab. Targets are scanned from the first position. The token count is the target length.
6. The reference length is not `tokenCount * vocab`: `CONTRACT_INVALID`, and the message says the perplexity reference length does not match.
7. The candidate length fails that same check, and the message says the perplexity candidate length does not match.
8. A logit is not finite. Reference rows are scanned first, then candidate rows, each in element order. The failure is `CONTRACT_INVALID`, and the message says the perplexity logit is not finite.

Each row's negative log likelihood is the natural log of the softmax probability of its target. The maximum logit is subtracted before the exponential. A log probability in `(0, 1e-6]` nats is treated as 0. A larger positive log probability, a non-finite sum, or a perplexity that does not fit in `u64` micros is `CONTRACT_INVALID`, and the message says perplexity is outside its bounds. No report is issued.

The perplexity is `exp(mean nll)` over the rows. Micros are that value times `1000000`, rounded half away from zero. The same rounding converts the largest absolute logit difference into millionths. A difference that does not fit in `u64`, or a perplexity delta that does not fit in `i64`, is the same out-of-bounds failure.

`greedyParity` is true only when every row's lowest-index argmax agrees. `targetAccuracyMillionths` is `(matches * 1000000 + tokenCount / 2) / tokenCount` in integer arithmetic. `matches` counts rows whose candidate argmax equals the target.

`verify_perplexity_report` checks the stored roots and recomputes the measurement. Checks run in this order:

1. `validationResult` is not `recorded`, the delta does not match the two perplexities, or another receipt validation fails.
2. The reporter build root differs: `CONTRACT_INVALID`, and the message says the reporter build root does not match.
3. `tokenCount` differs from the target length: the message says the perplexity token count does not match.
4. The reference logit root differs: the message says the reference logit root does not match.
5. The candidate logit root differs: the message says the candidate logit root does not match.
6. The target token root differs: the message says the target token root does not match.
7. Recomputing the measurement fails, and that failure is returned.
8. The recomputed report bytes differ: `CONTRACT_INVALID`, and the message says the perplexity validation did not match.

`perplexity_delta` runs that verify before it returns. A verify failure returns no report.

## Files

`write_perplexity_report` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the perplexity directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the perplexity path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the perplexity output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The logit matrices are not written.

## Out of this slice

This slice does not issue `ModelConformanceReceiptV1`. A CUDA quantized kernel stays off. The planner memory estimate is KIP-INFER-0027. Throughput, latency, and corruption fuzzing stay later. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter.
