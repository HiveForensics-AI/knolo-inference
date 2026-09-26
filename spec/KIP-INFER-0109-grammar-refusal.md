# KIP-INFER-0109 — Grammar refusal

Status: `measure_grammar_refusal` records one grammar the compiler did not build for a cold micro fixture. It returns a `knolo.infer.grammar-refusal-report`. The reason is `schema`, `automaton`, or `mask`. The code is `CONTRACT_INVALID` and it is not retryable. A schema refusal carries no source, does not open the source, and does not root the grammar. An automaton refusal and a mask refusal name 1 through 4096 source bytes. Exactly 4096 is recorded. An automaton refusal opened the source and does not root the grammar. A mask refusal opened the source and rooted the grammar. `maskApplied` and `grammarCompiled` stay false. The forward does not run and no receipt is stored. It does not compile a grammar and it does not apply a mask. The source bytes are not a field. `measure_contract_invalid` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

An automaton or mask reason above 4096 bytes is `CONTRACT_INVALID` and issues no report. A stored report above that cap is `CONTRACT_INVALID`.

## Report

The report is a versioned contract. Its identity root is `H(infer-grammar-refusal, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `schema`, `automaton`, or `mask` |
| `sourceBytes` | `0` for `schema`; `1` through `4096` otherwise |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `sourceOpened` | `false` for `schema`; `true` otherwise |
| `grammarRooted` | `true` only for `mask` |
| `maskApplied` | `false` |
| `grammarCompiled` | `false` |
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

`kind` is `knolo.infer.grammar-refusal-report` and `version` is `1`. The contract count is one hundred three. `infer-grammar-refusal` is the report domain. `infer-grammar` stays the grammar-plan domain. Source bytes are not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the grammar-refusal extensions are empty. A code other than `CONTRACT_INVALID` says a grammar refusal is CONTRACT_INVALID. `retryable` true says a grammar refusal is not retryable. `maskApplied` true says a grammar refusal does not apply a mask. `grammarCompiled` true says a grammar refusal does not compile the grammar. `forwardRan` true says a grammar refusal does not run the forward. `receiptStored` true says a grammar refusal stores no receipt. A schema refusal with bytes says a schema refusal carries no source. A schema refusal that opens the source says a schema refusal does not open the source. A schema refusal that roots the grammar says a schema refusal does not root the grammar. An automaton or mask refusal with no bytes says that refusal names the source. A stored length above 4096 says grammar source exceeds the record cap. An automaton refusal that does not open the source says an automaton refusal opened the source. An automaton refusal that roots the grammar says an automaton refusal does not root the grammar. A mask refusal that does not open the source says a mask refusal opened the source. A mask refusal that does not root the grammar says a mask refusal rooted the grammar.

## Measurement

`measure_grammar_refusal` takes the placement plan and one observation. It returns the report. It does not compile a grammar.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the grammar-refusal report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the grammar-refusal report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the grammar-refusal report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says grammar-refusal concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says grammar-refusal run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says grammar-refusal warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says grammar-refusal request count is one.
12. An `automaton` or `mask` reason above 4096 bytes: `CONTRACT_INVALID`, and the message says grammar source exceeds the record cap.
13. The report checks in the Report section, in the order written there.

`verify_grammar_refusal` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the grammar-refusal validation did not match.

## Files

`write_grammar_refusal_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the grammar-refusal directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the grammar-refusal path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the grammar-refusal output already exists.

The report is created with `create_new`, written, and `fsync`ed. The grammar source is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_grammar_refusal`. Receipt verification has not started. A speculative-decoding record has not started. A CUDA-graph record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
