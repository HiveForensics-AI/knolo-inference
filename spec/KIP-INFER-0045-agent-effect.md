# KIP-INFER-0045 — Agents host effect

Status: `measure_agent_effect` records the policy decision for one cold micro-fixture request. It returns a `knolo.infer.agent-effect-report`. It does not call Agents, does not execute a tool, does not run a model, and does not write a receipt journal. `measure_evidence_composition` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A malformed measurement issues no report. A policy denial is a recorded decision.

## Decision

The decision is computed. `allow` with reason `accepted` is the only pair that lets the effect run. A denial still returns a report with `validationResult` `recorded`. The first matching reason wins:

| Order | Condition | Reason |
| --- | --- | --- |
| 1 | the final receipt is absent | `stream-not-authorization` |
| 2 | the backend is not `reference-f32` or `candle-cuda` | `unverified-backend` |
| 3 | the model runtime root differs from the required root | `model-runtime-rejected` |
| 4 | the artifact root differs from the required root | `artifact-rejected` |
| 5 | the knowledge image root differs from the required root | `knowledge-image-rejected` |
| 6 | the execution mode is outside the allowed list | `execution-mode-rejected` |
| 7 | prompt tokens exceed the prompt budget | `prompt-budget-exceeded` |
| 8 | output tokens exceed the output budget | `output-budget-exceeded` |
| 9 | the assurance does not meet the required assurance | `assurance-rejected` |
| 10 | every check passed | `accepted` |

`llama.cpp`, `ollama`, `vllm`, and `mistral.rs` are unverified compatibility backends. Required assurance `compatibility` also accepts `same_build_replayable`. Required assurance `same_build_replayable` accepts only that value. Streamed tokens are not authorization.

## Report

The report is a versioned contract. Its identity root is `H(infer-agent-effect, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `modelImageRoot` | root of the micro model image |
| `artifactRoot` | root of the weight artifact on the request |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` that ran |
| `placementRoot` | root of the `PlacementPlanV1` |
| `modelRuntimeRoot` | runtime root on the request |
| `requiredModelRuntimeRoot` | runtime root the policy requires |
| `requiredArtifactRoot` | artifact root the policy requires |
| `knowledgeImageRoot` | knowledge image root on the request |
| `requiredKnowledgeImageRoot` | knowledge image root the policy requires |
| `backend` | `reference-f32`, `candle-cuda`, `llama.cpp`, `ollama`, `vllm`, or `mistral.rs` |
| `assurance` | `same_build_replayable` or `compatibility` |
| `requiredAssurance` | `same_build_replayable` or `compatibility` |
| `executionMode` | `isolated-replay` or `pinned` |
| `allowedExecutionModes` | a sorted unique list of those two modes, one or both |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `promptTokens` | prompt token count |
| `outputTokens` | output token count |
| `maxPromptTokens` | prompt budget, `1` through `16` |
| `maxOutputTokens` | output budget, `1` through `16` |
| `receiptPresent` | whether a final receipt exists |
| `decision` | `allow` or `deny` |
| `reason` | the reason from the table |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.agent-effect-report` and `version` is `1`. The contract count is forty. `infer-agent-effect` is the report domain.

A stored decision that is not the computed pair is `CONTRACT_INVALID`, and the message says the agent effect decision does not match. A prompt count plus an output count above 16 is `CONTEXT_LIMIT_EXCEEDED`, and the message says the agent effect is the micro fixture. That failure issues no report.

## Measurement

`measure_agent_effect` takes the placement plan and one observation. It returns the report. It does not create a file.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged. Opening `candle-cuda` as a backend name does not open a device.

Checks run in this order before the decision table. The first failure returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the agent effect report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the agent effect report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the agent effect report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says agent effect concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says agent effect run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says agent effect warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says agent effect request count is one.
12. `backend`, `assurance`, or `requiredAssurance` is outside the lists above: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
13. The allowed mode list is empty: `CONTRACT_INVALID`, and the message says the agent effect allows no execution mode.
14. The allowed mode list is longer than two, or a budget is outside `1` through `16`: `CONTRACT_INVALID`, and the message says the agent effect is the micro fixture.
15. The allowed mode list is not strictly sorted and unique: `CONTRACT_INVALID`, and the message says the field must be strictly sorted and unique.
16. Prompt tokens plus output tokens exceed 16: `CONTEXT_LIMIT_EXCEEDED`, and the message says the agent effect is the micro fixture.

`verify_agent_effect` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the agent effect validation did not match.

## Files

`write_agent_effect_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the agent effect directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the agent effect path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the agent effect output already exists.

The receipt is created with `create_new`, written, and `fsync`ed. Token ids are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_agent_effect`. Inference does not execute the agent's tools. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred. The Hub record is KIP-INFER-0046.
