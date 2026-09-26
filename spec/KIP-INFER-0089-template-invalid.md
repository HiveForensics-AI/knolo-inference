# KIP-INFER-0089 — Template invalid

Status: `measure_template_invalid` records one template the prompt compiler refused for a cold micro fixture. It returns a `knolo.infer.template-invalid-report`. The failure is `root`, `encoding`, `grammar`, or `cap`. The code is `TEMPLATE_INVALID` and it is not retryable. The template is not rendered, the tokenizer is not parsed, and the prompt is not compiled. The forward does not run and no receipt is stored. It does not render a template. `measure_tokenizer_invalid` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-template-invalid, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `templateRoot` | distinct from the engine build and the placement |
| `failure` | `root`, `encoding`, `grammar`, or `cap` |
| `code` | `TEMPLATE_INVALID` |
| `retryable` | `false` |
| `templateRendered` | `false` |
| `tokenizerParsed` | `false` |
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

`kind` is `knolo.infer.template-invalid-report` and `version` is `1`. The contract count is eighty-three. `infer-template-invalid` is the report domain. Template text is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the template extensions are empty. A template root equal to the engine build says the template repeats the engine build. A template root equal to the placement says the template repeats the placement. A failure other than the four named failures says the field has an unsupported value. A code other than `TEMPLATE_INVALID` says a template failure is TEMPLATE_INVALID. `retryable` true says a template failure is not retryable. `templateRendered` true says a template failure does not render the prompt. `tokenizerParsed` true says a template failure does not parse the tokenizer. `promptCompiled` true says a template failure does not compile the prompt. `forwardRan` true says a template failure does not run the forward. `receiptStored` true says a template failure stores no receipt.

## Measurement

`measure_template_invalid` takes the placement plan and one observation. It returns the report. It does not render a template and it does not parse a tokenizer.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the template report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the template report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the template report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says template concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says template run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says template warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says template request count is one.
12. The report checks in the Report section, in the order written there.

`verify_template_invalid` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the template validation did not match.

## Files

`write_template_invalid_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the template directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the template path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the template output already exists.

The report is created with `create_new`, written, and `fsync`ed. The template is not rendered.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_template_invalid`. The unsupported-architecture record is KIP-INFER-0090. The public-key multiplication has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
