# KIP-INFER-0047 — Receipt view

Status: `measure_receipt_view` records the fields a Studio panel may show for one cold micro-fixture receipt. It returns a `knolo.infer.studio-report`. It does not render a panel, does not open a receipt file, does not run a model, and does not write prompt text or output text. `measure_hub_record` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## View

The view stores roots and counts. `outputTextRoot` is a digest. Prompt text and output text are not fields. `extensions` is empty, so those strings are not carried there either. A `stop` receipt has at least one output token. A `length` receipt may have none. `cancelled` and `error` are not views, because those finishes store no receipt.

## Report

The report is a versioned contract. Its identity root is `H(infer-studio, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `modelImageRoot` | root of the micro model image |
| `artifactRoot` | root of the weight artifact |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` that ran |
| `placementRoot` | root of the `PlacementPlanV1` |
| `receiptRoot` | root of the inference receipt |
| `promptTokenRoot` | root of the prompt token ids |
| `outputTokenRoot` | root of the output token ids |
| `outputTextRoot` | root of the output text |
| `knowledgeImageRoot` | root of the Knowledge Image |
| `evidenceRoot` | root of the evidence binding |
| `promptTokenCount` | prompt token count, at least `1` |
| `outputTokenCount` | output token count |
| `finishReason` | `stop` or `length` |
| `assurance` | `same_build_replayable` or `compatibility` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.studio-report` and `version` is `1`. The contract count is forty. `infer-studio` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the studio view extensions are empty. A prompt count of zero is `CONTRACT_INVALID`, and the message says the studio view has no prompt. A prompt count plus an output count above 16 is `CONTEXT_LIMIT_EXCEEDED`, and the message says the studio view is the micro fixture. A `stop` finish with no output tokens is `CONTRACT_INVALID`, and the message says a stop receipt has no output tokens.

## Measurement

`measure_receipt_view` takes the placement plan and one observation. It returns the report. It does not create a file.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement, and they are not fields of the report.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the studio report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the studio report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the studio report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says studio concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says studio run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says studio warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says studio request count is one.
12. `extensions` is not empty: `CONTRACT_INVALID`, and the message says the studio view extensions are empty.
13. `assurance` or `finishReason` is outside the lists above: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
14. `promptTokenCount` is zero: `CONTRACT_INVALID`, and the message says the studio view has no prompt.
15. The token counts exceed the micro context: `CONTEXT_LIMIT_EXCEEDED`, and the message says the studio view is the micro fixture.
16. `finishReason` is `stop` and `outputTokenCount` is zero: `CONTRACT_INVALID`, and the message says a stop receipt has no output tokens.

`verify_receipt_view` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the studio validation did not match.

## Files

`write_studio_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the studio directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the studio path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the studio output already exists.

The receipt is created with `create_new`, written, and `fsync`ed. The inference receipt is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_receipt_view`. The receipt chain demo is KIP-INFER-0048. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
