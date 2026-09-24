# KIP-INFER-0049 — Evidence to output

Status: `measure_evidence_output` records that one cold micro-fixture output is bound to the receipt chain's evidence. It returns a `knolo.infer.evidence-output-report`. The evidence root, knowledge image, query receipt, Reflex receipt, output token root, and output text root on the output must equal the chain side. It does not open a `.knolo` image, does not run a model, and does not write prompt text or output text. `measure_receipt_chain` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Binding

The report stores each matched root once. A mismatch issues no report. The chain root differs from the inference receipt root. A `stop` output has at least one output token. A `length` output may have none.

## Report

The report is a versioned contract. Its identity root is `H(infer-evidence-output, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `evidenceRoot` | root of the evidence binding |
| `knowledgeImageRoot` | root of the Knowledge Image |
| `queryReceiptRoot` | root of the query receipt |
| `reflexReceiptRoot` | root of the Reflex receipt |
| `receiptRoot` | root of the inference receipt |
| `chainRoot` | root of the receipt chain |
| `outputTokenRoot` | root of the output token ids |
| `outputTextRoot` | root of the output text |
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
| `validationResult` | `verified` |
| `extensions` | an empty map |

`kind` is `knolo.infer.evidence-output-report` and `version` is `1`. The contract count is forty-four. `infer-evidence-output` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the evidence output extensions are empty. A prompt count of zero is `CONTRACT_INVALID`, and the message says the evidence output has no prompt. A prompt count plus an output count above 16 is `CONTEXT_LIMIT_EXCEEDED`, and the message says the evidence output is the micro fixture. A `stop` finish with no output tokens is `CONTRACT_INVALID`, and the message says a stop output has no output tokens. A chain root that equals the receipt root says the chain repeats the receipt.

## Measurement

`measure_evidence_output` takes the placement plan and one observation. The observation carries the chain side and the output side of each bound root. It returns the report. It does not create a file.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement, and they are not fields of the report.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the evidence report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the evidence report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the evidence report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says evidence concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says evidence run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says evidence warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says evidence request count is one.
12. The evidence roots differ: `CONTRACT_INVALID`, and the message says the evidence root does not bind the output.
13. The knowledge image roots differ: `CONTRACT_INVALID`, and the message says the knowledge image does not bind the output.
14. The query receipt roots differ: `CONTRACT_INVALID`, and the message says the query receipt does not bind the output.
15. The Reflex receipt roots differ: `CONTRACT_INVALID`, and the message says the reflex receipt does not bind the output.
16. The output token roots differ: `CONTRACT_INVALID`, and the message says the output token root does not bind the receipt.
17. The output text roots differ: `CONTRACT_INVALID`, and the message says the output text root does not bind the receipt.
18. The report checks in the Report section, in the order written there.

`verify_evidence_output` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the evidence output validation did not match.

## Files

`write_evidence_output_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the evidence directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the evidence path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the evidence output already exists.

The report is created with `create_new`, written, and `fsync`ed. The inference receipt is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_evidence_output`. The Hub installation check is KIP-INFER-0050. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
