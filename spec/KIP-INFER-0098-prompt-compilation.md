# KIP-INFER-0098 — Prompt compilation

Status: `measure_prompt_compilation` records one prompt the compiler refused for a cold micro fixture. It returns a `knolo.infer.prompt-compilation-report`. The failure is `empty`, `vocab`, or `size`. The code is `PROMPT_COMPILATION_FAILED` and it is not retryable. The template rendered and the tokenizer parsed. The prompt is not compiled, the forward does not run, and no receipt is stored. An empty prompt has no tokens and no rejected token. A vocabulary failure has 1 through 16 tokens and one rejected token from 16 through 1024. A prompt that is too large records no token count and no rejected token. It does not compile a prompt. `measure_context_limit` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

A vocabulary failure whose token count is above 16 is `CONTEXT_LIMIT_EXCEEDED` and issues no report. A stored report above 16 is `CONTRACT_INVALID`. A rejected token above 1024 is `PROMPT_COMPILATION_FAILED` and issues no report. A stored report above that cap is `CONTRACT_INVALID`.

## Report

The report is a versioned contract. Its identity root is `H(infer-prompt-compilation, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `failure` | `empty`, `vocab`, or `size` |
| `tokenCount` | `0` for `empty` and `size`; `1` through `16` for `vocab` |
| `rejectedToken` | `0` for `empty` and `size`; `16` through `1024` for `vocab` |
| `code` | `PROMPT_COMPILATION_FAILED` |
| `retryable` | `false` |
| `templateRendered` | `true` |
| `tokenizerParsed` | `true` |
| `promptCompiled` | `false` |
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

`kind` is `knolo.infer.prompt-compilation-report` and `version` is `1`. The contract count is ninety-three. `infer-prompt-compilation` is the report domain. Prompt text is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the prompt-compilation extensions are empty. A code other than `PROMPT_COMPILATION_FAILED` says a prompt failure is PROMPT_COMPILATION_FAILED. `retryable` true says a prompt failure is not retryable. `promptCompiled` true says a prompt failure does not compile the prompt. `forwardRan` true says a prompt failure does not run the forward. `receiptStored` true says a prompt failure stores no receipt. `templateRendered` false says a prompt failure rendered the template. `tokenizerParsed` false says a prompt failure parsed the tokenizer. An empty prompt with any token says an empty prompt has no tokens. An empty prompt with a rejected token says an empty prompt has no rejected token. A vocabulary failure with no tokens says a vocabulary failure has a prompt. A stored vocabulary failure above 16 tokens says the prompt compilation is the micro fixture. A rejected token below 16 says a vocabulary failure is outside the micro vocabulary. A stored rejected token above 1024 says a rejected token exceeds the record cap. A size failure with a token count says a prompt that is too large has no token count. A size failure with a rejected token says a prompt that is too large has no rejected token.

## Measurement

`measure_prompt_compilation` takes the placement plan and one observation. It returns the report. It does not compile a prompt and it does not render a template.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the prompt-compilation report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the prompt-compilation report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the prompt-compilation report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says prompt-compilation concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says prompt-compilation run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says prompt-compilation warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says prompt-compilation request count is one.
12. A `vocab` failure above 16 tokens: `CONTEXT_LIMIT_EXCEEDED`, and the message says the prompt compilation is the micro fixture.
13. A `vocab` failure whose rejected token is above 1024: `PROMPT_COMPILATION_FAILED`, and the message says a rejected token exceeds the record cap.
14. The report checks in the Report section, in the order written there.

`verify_prompt_compilation` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the prompt-compilation validation did not match.

## Files

`write_prompt_compilation_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the prompt-compilation directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the prompt-compilation path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the prompt-compilation output already exists.

The report is created with `create_new`, written, and `fsync`ed. The prompt is not compiled.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_prompt_compilation`. The invalid-image record is KIP-INFER-0099. Cofactor clearing has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
