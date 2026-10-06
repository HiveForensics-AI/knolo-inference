# KIP-INFER-0079 — Request timeout

Status: `measure_timeout` records one request timeout for a cold micro fixture. It returns a `knolo.infer.timeout-report`. The stage is `admission`, `completion`, or `cancellation`. The wait is greater than zero. The code is `REQUEST_TIMEOUT` and it is retryable. The HTTP status is 504. The listener stays up, the worker is not lost, and no receipt is stored. It does not arm a timer. `measure_cuda_fault` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Wait

The request id is 1 through 64 characters of ASCII letters, digits, and hyphens. A wait of zero is `CONTRACT_INVALID`.

## Report

The report is a versioned contract. Its identity root is `H(infer-timeout, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `requestId` | a request token |
| `timeoutStage` | `admission`, `completion`, or `cancellation` |
| `waitedNanos` | greater than `0` |
| `code` | `REQUEST_TIMEOUT` |
| `retryable` | `true` |
| `receiptStored` | `false` |
| `listenerUp` | `true` |
| `workerLost` | `false` |
| `httpStatus` | `504` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.timeout-report` and `version` is `1`. The contract count is seventy-three. `infer-timeout` is the report domain. Prompt text is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the timeout extensions are empty. An empty or oversized request id says the field is empty or outside its bounds. A request id with any other character says the request id is a token. A stage other than the three named stages says the field has an unsupported value. A wait of zero says a timeout record waits. A code other than `REQUEST_TIMEOUT` says a timeout record is REQUEST_TIMEOUT. `retryable` false says a timeout record is retryable. `receiptStored` true says a timeout record stores no receipt. `listenerUp` false says the listener stays up. `workerLost` true says a timeout is not a lost worker. An HTTP status other than 504 says a timeout record is HTTP 504.

## Measurement

`measure_timeout` takes the placement plan and one observation. It returns the report. It does not arm a timer and it does not cancel a live request.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the timeout report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the timeout report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the timeout report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says timeout concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says timeout run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says timeout warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says timeout request count is one.
12. The report checks in the Report section, in the order written there.

`verify_timeout` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the timeout validation did not match.

## Files

`write_timeout_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the timeout directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the timeout path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the timeout output already exists.

The report is created with `create_new`, written, and `fsync`ed. No timer is armed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_timeout`. The worker-start record is KIP-INFER-0080. The challenge hash has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
