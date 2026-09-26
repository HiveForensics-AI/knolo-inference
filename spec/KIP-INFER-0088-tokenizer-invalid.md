# KIP-INFER-0088 — Tokenizer invalid

Status: `measure_tokenizer_invalid` records one tokenizer the prompt compiler refused for a cold micro fixture. It returns a `knolo.infer.tokenizer-invalid-report`. The failure is `root`, `file`, `special`, or `encode`. The code is `TOKENIZER_INVALID` and it is not retryable. A root mismatch does not render the template and does not parse the tokenizer. A file rejection is checked after the template renders and does not accept the tokenizer. A special-token check and an encode failure both happen after the template rendered and the tokenizer parsed. The prompt is not compiled, the forward does not run, and no receipt is stored. It does not parse a tokenizer. `measure_digest_mismatch` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-tokenizer-invalid, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `tokenizerRoot` | distinct from the engine build and the placement |
| `failure` | `root`, `file`, `special`, or `encode` |
| `code` | `TOKENIZER_INVALID` |
| `retryable` | `false` |
| `tokenizerParsed` | `true` only for `special` and `encode` |
| `templateRendered` | `true` only for `file`, `special`, and `encode` |
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

`kind` is `knolo.infer.tokenizer-invalid-report` and `version` is `1`. The contract count is eighty-three. `infer-tokenizer-invalid` is the report domain. Tokenizer bytes are not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the tokenizer extensions are empty. A tokenizer root equal to the engine build says the tokenizer repeats the engine build. A tokenizer root equal to the placement says the tokenizer repeats the placement. A failure other than the four named failures says the field has an unsupported value. A code other than `TOKENIZER_INVALID` says a tokenizer failure is TOKENIZER_INVALID. `retryable` true says a tokenizer failure is not retryable. `promptCompiled` true says a tokenizer failure does not compile the prompt. `forwardRan` true says a tokenizer failure does not run the forward. `receiptStored` true says a tokenizer failure stores no receipt. A root mismatch that parsed the tokenizer says a root mismatch does not parse the tokenizer. A root mismatch that rendered the template says a root mismatch does not render the template. A file rejection that parsed the tokenizer says a tokenizer file is not accepted. A file rejection that did not render the template says a tokenizer file is checked after the template renders. A special-token check that did not parse the tokenizer says a special token is checked after the tokenizer parses. A special-token check that did not render the template says a special token is checked after the template renders. An encode failure that did not parse the tokenizer says an encode failure parsed the tokenizer. An encode failure that did not render the template says an encode failure rendered the template.

## Measurement

`measure_tokenizer_invalid` takes the placement plan and one observation. It returns the report. It does not parse a tokenizer and it does not render a template.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the tokenizer report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the tokenizer report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the tokenizer report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says tokenizer concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says tokenizer run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says tokenizer warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says tokenizer request count is one.
12. The report checks in the Report section, in the order written there.

`verify_tokenizer_invalid` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the tokenizer validation did not match.

## Files

`write_tokenizer_invalid_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the tokenizer directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the tokenizer path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the tokenizer output already exists.

The report is created with `create_new`, written, and `fsync`ed. The tokenizer is not parsed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_tokenizer_invalid`. The template record is KIP-INFER-0089. The public-key multiplication has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
