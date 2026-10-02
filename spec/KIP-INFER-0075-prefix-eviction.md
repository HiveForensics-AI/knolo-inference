# KIP-INFER-0075 — Prefix eviction

Status: `measure_prefix_eviction` records prefix-cache eviction under load for one cold micro fixture. It returns a `knolo.infer.eviction-report`. The load is 1 through 16 requests. Evicted pages and evicted tokens are zero. An active sequence is not evicted. The prefix index is not allocated. The listener stays up. Cache policy stays `off`. It does not allocate a prefix index. `measure_concurrent_load` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Load

A load of zero issues no report. A load above 16 issues no report. A non-zero eviction count issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-eviction, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `loadRequests` | `1` through `16` |
| `evictedPages` | `0` |
| `evictedTokens` | `0` |
| `activeEvicted` | `false` |
| `prefixAllocated` | `false` |
| `listenerUp` | `true` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.eviction-report` and `version` is `1`. The contract count is sixty-eight. `infer-eviction` is the report domain. A prefix token is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the eviction extensions are empty. A load of zero says eviction under load names a request. A load above 16 says eviction under load admits at most 16 requests. An evicted page count other than zero says prefix eviction is zero while the cache is off. An evicted token count other than zero says evicted tokens are zero while the cache is off. `activeEvicted` true says an active sequence is not evicted. `prefixAllocated` true says prefix cache is not allocated. `listenerUp` false says the listener stays up.

## Measurement

`measure_prefix_eviction` takes the placement plan and one observation. It returns the report. It does not allocate a prefix index and it does not evict a page.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the eviction report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the eviction report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the eviction report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says eviction concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says eviction run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says eviction warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says eviction request count is one.
12. The report checks in the Report section, in the order written there.

`verify_prefix_eviction` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the eviction validation did not match.

## Files

`write_eviction_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the eviction directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the eviction path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the eviction output already exists.

The report is created with `create_new`, written, and `fsync`ed. No prefix index is allocated.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_prefix_eviction`. Point comparison has not started. A CUDA OOM record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
