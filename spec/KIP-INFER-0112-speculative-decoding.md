# KIP-INFER-0112 — Speculative decoding

Status: `measure_speculative` records one speculative request the engine did not run for a cold micro fixture. It returns a `knolo.infer.speculative-report`. The reason is `draft`, `mtp`, or `plan`. The code is `CONTRACT_INVALID` and it is not retryable. A draft refusal and an MTP refusal name 1 through 16 proposal tokens. Exactly 16 is recorded. A plan refusal carries no proposal. Accepted tokens and rejected tokens stay 0. `speculated`, `distributionChanged`, and `cacheAffected` stay false. The target root and the proposal root differ from each other and from the engine build and the placement. The forward does not run and no receipt is stored. It does not run a draft model and it does not run an MTP head. `measure_receipt_verify` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

A draft or MTP proposal above 16 tokens is `CONTEXT_LIMIT_EXCEEDED` and issues no report. A stored report above that cap is `CONTRACT_INVALID`.

## Report

The report is a versioned contract. Its identity root is `H(infer-speculative, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `targetRoot` | root distinct from the engine build and the placement |
| `proposalRoot` | root distinct from the target, the engine build, and the placement |
| `reason` | `draft`, `mtp`, or `plan` |
| `proposalTokens` | `0` for `plan`; `1` through `16` otherwise |
| `acceptedTokens` | `0` |
| `rejectedTokens` | `0` |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `speculated` | `false` |
| `distributionChanged` | `false` |
| `cacheAffected` | `false` |
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

`kind` is `knolo.infer.speculative-report` and `version` is `1`. The contract count is one hundred eight. `infer-speculative` is the report domain. Token ids are not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the speculative extensions are empty. A target root equal to the engine build says the target repeats the engine build. A target root equal to the placement says the target repeats the placement. A proposal root equal to the engine build says the proposal repeats the engine build. A proposal root equal to the placement says the proposal repeats the placement. A proposal root equal to the target says the proposal repeats the target. A code other than `CONTRACT_INVALID` says a speculative refusal is CONTRACT_INVALID. `retryable` true says a speculative refusal is not retryable. `speculated` true says a speculative refusal does not speculate. A non-zero accepted count says accepted tokens stay zero while speculation is off. A non-zero rejected count says rejected tokens stay zero while speculation is off. `distributionChanged` true says speculation does not change the target distribution. `cacheAffected` true says speculation does not affect the cache. `forwardRan` true says a speculative refusal does not run the forward. `receiptStored` true says a speculative refusal stores no receipt. A stored length above 16 says speculative proposal exceeds the record cap. A plan refusal with proposal tokens says a plan refusal carries no proposal. A draft refusal with no tokens says a draft refusal names a proposal. An MTP refusal with no tokens says an mtp refusal names a proposal.

## Measurement

`measure_speculative` takes the placement plan and one observation. It returns the report. It does not speculate.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the speculative report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the speculative report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the speculative report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says speculative concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says speculative run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says speculative warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says speculative request count is one.
12. A `draft` or `mtp` reason above 16 tokens: `CONTEXT_LIMIT_EXCEEDED`, and the message says a speculative proposal exceeds the context.
13. The report checks in the Report section, in the order written there.

`verify_speculative` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the speculative validation did not match.

## Files

`write_speculative_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the speculative directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the speculative path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the speculative output already exists.

The report is created with `create_new`, written, and `fsync`ed. Token ids are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_speculative`. Domain separation has not started. An attention-approximation record has not started. A mixture-of-experts record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
