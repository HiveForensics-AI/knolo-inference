# KIP-INFER-0059 — API boundary

Status: `measure_api_boundary` records the API boundary for one cold micro fixture. It returns a `knolo.infer.api-report`. `localhost` does not set a remote bind. `remote` requires that bind to be explicit. Authentication is `hook`. The body limit is 1 byte through 1 MiB. The rate is 1 through 256 requests per minute. The concurrency limit is 1. Ordinary logs omit prompt text. Metrics labels are counts. It does not bind a socket and does not parse a request. `measure_sandbox_profile` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Boundary

The tenant root differs from the engine build root. Exactly 1 MiB is recorded. Exactly 256 requests per minute is recorded. Bearer authentication and mTLS stay outside this report.

## Report

The report is a versioned contract. Its identity root is `H(infer-api, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `tenantRoot` | root of the tenant namespace |
| `bind` | `localhost` or `remote` |
| `remoteExplicit` | `true` only when `bind` is `remote` |
| `auth` | `hook` |
| `bodyLimitBytes` | `1` through 1 MiB |
| `ratePerMinute` | `1` through `256` |
| `concurrencyLimit` | `1` |
| `promptLog` | `omitted` |
| `metricsLabels` | `counts` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.api-report` and `version` is `1`. The contract count is fifty-two. `infer-api` is the report domain. Prompt text is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the api extensions are empty. A tenant root equal to the engine build says the tenant repeats the engine build. `localhost` with `remoteExplicit` true says localhost does not set a remote bind. `remote` with `remoteExplicit` false says a remote bind is explicit. An authentication value other than `hook` says authentication stays a hook. A body limit of zero says the request body limit is zero. A stored body limit above 1 MiB says the request body exceeds 1 MiB. A rate of zero says the api rate is zero. A rate above 256 says the api rate exceeds 256. A concurrency limit other than 1 says the api concurrency limit is one. A prompt log other than `omitted` says ordinary logs omit prompt text. Metrics labels other than `counts` say metrics labels omit user content.

## Measurement

`measure_api_boundary` takes the placement plan and one observation. It returns the report. It does not bind a socket.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the api report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the api report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the api report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says api concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says api run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says api warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says api request count is one.
12. `bodyLimitBytes` is above 1 MiB: `INSUFFICIENT_MEMORY`, and the message says the request body exceeds 1 MiB.
13. The report checks in the Report section, in the order written there.

`verify_api_boundary` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the api validation did not match.

## Files

`write_api_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the api directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the api path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the api output already exists.

The report is created with `create_new`, written, and `fsync`ed. The request body is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_api_boundary`. The cache side-channel report is KIP-INFER-0060. Evaluating the Ed25519 equation has not started. Authentication stays a hook. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
