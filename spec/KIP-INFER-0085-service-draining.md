# KIP-INFER-0085 — Service draining

Status: `measure_draining` records one new completion refused because the service is draining for a cold micro fixture. It returns a `knolo.infer.draining-report`. The lifecycle is `draining` or `drained`. The code is `SERVICE_DRAINING` and it is retryable. No receipt is stored. The body is not parsed. The listener stays up and the worker stays loaded. The refusal does not count a restart. The HTTP status is 503. It does not drain the listener and does not parse a body. `measure_worker_lost` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-draining, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `lifecycle` | `draining` or `drained` |
| `code` | `SERVICE_DRAINING` |
| `retryable` | `true` |
| `receiptStored` | `false` |
| `bodyParsed` | `false` |
| `listenerUp` | `true` |
| `workerLoaded` | `true` |
| `restartCounted` | `false` |
| `httpStatus` | `503` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.draining-report` and `version` is `1`. The contract count is seventy-eight. `infer-draining` is the report domain. A request id is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the draining extensions are empty. A lifecycle other than `draining` or `drained` says the field has an unsupported value. A code other than `SERVICE_DRAINING` says a draining refusal is SERVICE_DRAINING. `retryable` false says a draining refusal is retryable. `receiptStored` true says a draining refusal stores no receipt. `bodyParsed` true says a draining refusal does not parse the body. `listenerUp` false says the listener stays up. `workerLoaded` false says a draining refusal leaves the worker loaded. `restartCounted` true says a draining refusal does not count a restart. An HTTP status other than 503 says a draining refusal is HTTP 503.

## Measurement

`measure_draining` takes the placement plan and one observation. It returns the report. It does not parse a body and it does not drain the listener.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the draining report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the draining report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the draining report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says draining concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says draining run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says draining warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says draining request count is one.
12. The report checks in the Report section, in the order written there.

`verify_draining` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the draining validation did not match.

## Files

`write_draining_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the draining directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the draining path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the draining output already exists.

The report is created with `create_new`, written, and `fsync`ed. The body is not parsed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_draining`. The scalar reduction has not started. A digest-mismatch record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
