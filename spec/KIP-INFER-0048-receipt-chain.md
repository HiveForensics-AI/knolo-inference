# KIP-INFER-0048 — Receipt chain

Status: `measure_receipt_chain` records one ordered chain for a cold micro fixture. It returns a `knolo.infer.chain-report`. The links are the Knowledge Image, one query receipt, one Reflex receipt, the inference receipt, and the agent effect. It does not open a `.knolo` image, does not run a model, does not call Agents, and does not write prompt text or output text. `measure_receipt_view` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Chain

The five links are distinct. The knowledge commit differs from the knowledge image. The weight artifact differs from the model runtime. A `stop` chain has at least one output token. A `length` chain may have none. `cancelled` and `error` are not chains, because those finishes store no receipt. A second check of the same plan and observation verifies the stored roots without opening the model.

## Report

The report is a versioned contract. Its identity root is `H(infer-chain, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `modelRuntimeRoot` | root of the model runtime |
| `artifactRoot` | root of the weight artifact |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` that ran |
| `kernelBundleRoot` | root of the kernel bundle |
| `placementRoot` | root of the `PlacementPlanV1` |
| `promptTokenRoot` | root of the prompt token ids |
| `knowledgeImageRoot` | root of the Knowledge Image |
| `knowledgeCommitRoot` | root of the knowledge commit |
| `queryReceiptRoot` | root of the query receipt |
| `queryReceiptCount` | query receipts, at least `1` and at most `256` |
| `reflexReceiptRoot` | root of the Reflex receipt |
| `reflexReceiptCount` | Reflex receipts, at least `1` and at most `256` |
| `receiptRoot` | root of the inference receipt |
| `effectRoot` | root of the agent-effect report |
| `outputTokenRoot` | root of the output token ids |
| `outputTextRoot` | root of the output text |
| `promptTokenCount` | prompt token count, at least `1` |
| `outputTokenCount` | output token count |
| `finishReason` | `stop` or `length` |
| `assurance` | `same_build_replayable` or `compatibility` |
| `linkCount` | `5` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `verified` |
| `extensions` | an empty map |

`kind` is `knolo.infer.chain-report` and `version` is `1`. The contract count is forty-four. `infer-chain` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the receipt chain extensions are empty. A prompt count of zero is `CONTRACT_INVALID`, and the message says the receipt chain has no prompt. A prompt count plus an output count above 16 is `CONTEXT_LIMIT_EXCEEDED`, and the message says the receipt chain is the micro fixture. A `stop` finish with no output tokens is `CONTRACT_INVALID`, and the message says a stop chain has no output tokens. A repeated knowledge commit says the knowledge commit repeats the image. A repeated artifact says the artifact repeats the model runtime. A query count of zero says the receipt chain has no query receipt. A Reflex count of zero says the receipt chain has no reflex receipt. A count above 256 says the receipt chain evidence list is too large. A link count other than five says the receipt chain has five links. Any two of the five links that are equal says the receipt chain repeats a link.

## Measurement

`measure_receipt_chain` takes the placement plan and one observation. It returns the report. It does not create a file.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement, and they are not fields of the report.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the chain report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the chain report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the chain report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says chain concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says chain run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says chain warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says chain request count is one.
12. The report checks in the Report section, in the order written there.

`verify_receipt_chain` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the chain validation did not match.

## Files

`write_chain_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the chain directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the chain path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the chain output already exists.

The report is created with `create_new`, written, and `fsync`ed. The inference receipt is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_receipt_chain`. The evidence-to-output check is KIP-INFER-0049. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred. Ed25519 signatures stay shape-checked.
