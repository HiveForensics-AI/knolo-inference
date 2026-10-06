# KIP-INFER-0130 — MTP speculation

Status: `measure_mtp` records one multi-token prediction head the engine did not run for a cold micro fixture. It returns a `knolo.infer.mtp-report`. The reason is `head`, `draft`, or `accept`. The code is `CONTRACT_INVALID` and it is not retryable. Each reason names 1 through 16 proposal tokens. Exactly 16 is recorded. A count above 16 is `CONTEXT_LIMIT_EXCEEDED` and issues no report. Accepted tokens stay 0. `headRan`, `drafted`, `accepted`, and `speculated` stay false. The forward does not run and no receipt is stored. It does not run an MTP head. `measure_mhc` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it.

## Report

The report is a versioned contract. Its identity root is `H(infer-mtp, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `reason` | `head`, `draft`, or `accept` |
| `proposalTokens` | `1` through `16` |
| `acceptedTokens` | `0` |
| `code` | `CONTRACT_INVALID` |
| `retryable` | `false` |
| `headRan` | `false` |
| `drafted` | `false` |
| `accepted` | `false` |
| `speculated` | `false` |
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

`kind` is `knolo.infer.mtp-report` and `version` is `1`. The contract count is one hundred twenty-three. `infer-mtp` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the mtp extensions are empty. A code other than `CONTRACT_INVALID` says an MTP head is CONTRACT_INVALID. `retryable` true says an MTP head is not retryable. `headRan` true says an MTP head does not run. `drafted` true says an MTP draft is not produced. `accepted` true says an MTP proposal is not accepted. `speculated` true says speculation stays off. Accepted tokens other than 0 say accepted MTP tokens stay zero. A stored report above 16 says MTP tokens exceed the record cap. A proposal count of 0 says that reason names a proposal. `forwardRan` true says an MTP head does not run the forward. `receiptStored` true says an MTP head stores no receipt.

## Measurement

`measure_mtp` takes the placement plan and one observation. It returns the report. An MTP head is not run.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the mtp report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the mtp report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the mtp report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says mtp concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says mtp run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says mtp warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says mtp request count is one.
12. The report checks in the Report section, in the order written there.

`verify_mtp` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the mtp validation did not match.

## Files

`write_mtp_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the mtp directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the mtp path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the mtp output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_mtp`. Digest binding has not started. Multimodal input has not started. Large expert placement has not started. Extreme-context cache has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
