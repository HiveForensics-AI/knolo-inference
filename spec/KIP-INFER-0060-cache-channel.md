# KIP-INFER-0060 — Cache side channel

Status: `measure_cache_channel` records the prefix-cache disclosure policy for one cold micro fixture. It returns a `knolo.infer.cache-channel-report`. Sharing is `isolated`. Cross-tenant sharing is off. Whether another tenant's prefix exists stays `hidden`. Cache metrics are `aggregate`. The prefix index stays unallocated. It does not look up a prefix. `measure_api_boundary` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Scope

The tenant root differs from the engine build root. The project root differs from both. Cache policy stays `off`.

## Report

The report is a versioned contract. Its identity root is `H(infer-cache-channel, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `tenantRoot` | root of the tenant namespace |
| `projectRoot` | root of the project namespace |
| `sharingPolicy` | `isolated` |
| `crossTenant` | `false` |
| `existenceDisclosure` | `hidden` |
| `metricsScope` | `aggregate` |
| `prefixAllocated` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.cache-channel-report` and `version` is `1`. The contract count is fifty-six. `infer-cache-channel` is the report domain. A prefix token is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the cache channel extensions are empty. A tenant root equal to the engine build says the tenant repeats the engine build. A project root equal to the engine build says the project repeats the engine build. A project root equal to the tenant says the project repeats the tenant. A sharing policy other than `isolated` says cache sharing is isolated. `crossTenant` true says cross-tenant sharing is off. An existence disclosure other than `hidden` says another tenant prefix stays hidden. A metrics scope other than `aggregate` says cache metrics stay aggregated. `prefixAllocated` true says the prefix index stays unallocated.

## Measurement

`measure_cache_channel` takes the placement plan and one observation. It returns the report. It does not allocate a prefix index.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the cache channel report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the cache channel report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the cache channel report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says cache channel concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says cache channel run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says cache channel warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says cache channel request count is one.
12. The report checks in the Report section, in the order written there.

`verify_cache_channel` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the cache channel validation did not match.

## Files

`write_cache_channel_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the cache channel directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the cache channel path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the cache channel output already exists.

The report is created with `create_new`, written, and `fsync`ed. A prefix entry is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_cache_channel`. The signature equation record is KIP-INFER-0061. Computing the Ed25519 curve has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
