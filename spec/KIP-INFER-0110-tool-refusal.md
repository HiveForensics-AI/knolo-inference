# KIP-INFER-0110 — Tool refusal

Status: `measure_tool_refusal` records one tool call the engine did not execute for a cold micro fixture. It returns a `knolo.infer.tool-refusal-report`. The reason is `name`, `object`, or `execute`. The code is `CONTRACT_INVALID` and it is not retryable. A name refusal carries no name, does not accept the name, and does not generate an object. An object refusal and an execute refusal name 1 through 64 bytes. Exactly 64 is recorded. An object refusal accepted the name and does not generate an object. An execute refusal accepted the name and generated the object. The object root differs from the engine build and the placement. `toolExecuted`, `authorityChecked`, and `budgetChecked` stay false. The forward does not run and no receipt is stored. It does not call Agents and it does not execute a tool. The tool name is not a field. `measure_grammar_refusal` does not call it. `measure_agent_effect` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

An object or execute reason above 64 bytes is `CONTRACT_INVALID` and issues no report. A stored report above that cap is `CONTRACT_INVALID`.

## Report

The report is a versioned contract. Its identity root is `H(infer-tool-refusal, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `objectRoot` | root distinct from the engine build and the placement |
| `reason` | `name`, `object`, or `execute` |
| `nameBytes` | `0` for `name`; `1` through `64` otherwise |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `nameAccepted` | `false` for `name`; `true` otherwise |
| `objectGenerated` | `true` only for `execute` |
| `toolExecuted` | `false` |
| `authorityChecked` | `false` |
| `budgetChecked` | `false` |
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

`kind` is `knolo.infer.tool-refusal-report` and `version` is `1`. The contract count is one hundred three. `infer-tool-refusal` is the report domain. `infer-tools` stays reserved for a tool list. The tool name is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the tool-refusal extensions are empty. An object root equal to the engine build says the tool object repeats the engine build. An object root equal to the placement says the tool object repeats the placement. A code other than `CONTRACT_INVALID` says a tool refusal is CONTRACT_INVALID. `retryable` true says a tool refusal is not retryable. `toolExecuted` true says a tool refusal does not execute the tool. `authorityChecked` true says a tool refusal does not check authority. `budgetChecked` true says a tool refusal does not check a budget. `forwardRan` true says a tool refusal does not run the forward. `receiptStored` true says a tool refusal stores no receipt. A name refusal with bytes says a name refusal carries no name. A name refusal that accepts the name says a name refusal does not accept the name. A name refusal that generates an object says a name refusal does not generate an object. An object or execute refusal with no bytes says that refusal names the tool. A stored length above 64 says tool name exceeds the record cap. An object refusal that does not accept the name says an object refusal accepted the name. An object refusal that generates an object says an object refusal does not generate an object. An execute refusal that does not accept the name says an execute refusal accepted the name. An execute refusal that does not generate an object says an execute refusal generated the object.

## Measurement

`measure_tool_refusal` takes the placement plan and one observation. It returns the report. It does not execute a tool.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the tool-refusal report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the tool-refusal report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the tool-refusal report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says tool-refusal concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says tool-refusal run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says tool-refusal warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says tool-refusal request count is one.
12. An `object` or `execute` reason above 64 bytes: `CONTRACT_INVALID`, and the message says tool name exceeds the record cap.
13. The report checks in the Report section, in the order written there.

`verify_tool_refusal` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the tool-refusal validation did not match.

## Files

`write_tool_refusal_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the tool-refusal directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the tool-refusal path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the tool-refusal output already exists.

The report is created with `create_new`, written, and `fsync`ed. The tool name is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_tool_refusal`. Receipt verification has not started. A speculative-decoding record has not started. A CUDA-graph record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
