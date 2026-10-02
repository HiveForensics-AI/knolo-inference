# KIP-INFER-0062 — Safe error

Status: `measure_safe_error` records one stable failure for a cold micro fixture. It returns a `knolo.infer.safe-error-report`. The message is the stable code. Retryability follows that code. The attempt is 1. The prompt and any secret stay absent. A partial receipt is `absent` or a root. It does not return the prompt. `measure_signature_equation` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Envelope

The request id is 1 through 64 characters of ASCII letters, digits, and hyphens. A partial receipt root differs from the engine build and from the placement.

## Report

The report is a versioned contract. Its identity root is `H(infer-safe-error, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `code` | one stable error code |
| `message` | the same stable code |
| `retryable` | the code's retryability |
| `requestId` | a request token |
| `attempt` | `1` |
| `promptPresent` | `false` |
| `secretPresent` | `false` |
| `partialReceipt` | `absent` or a `sha256-` root |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.safe-error-report` and `version` is `1`. The contract count is fifty-six. `infer-safe-error` is the report domain. Prompt text is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the safe error extensions are empty. An empty or oversized request id says the field is empty or outside its bounds. A request id with any other character says the request id is a token. An unknown code says the field has an unsupported value. A message other than the code says the safe message is the stable code. A retryable flag that disagrees with the code says retryability follows the stable code. An attempt other than 1 says the safe error attempt is one. `promptPresent` true says a safe error omits the prompt. `secretPresent` true says a safe error omits secrets. A partial receipt other than `absent` or a digest says a partial receipt is absent or a root. A partial root equal to the engine build says the partial receipt repeats the engine build. A partial root equal to the placement says the partial receipt repeats the placement.

`SERVICE_DRAINING`, `WORKER_LOST`, `WORKER_START_FAILED`, `RECEIPT_PERSIST_FAILED`, `INSUFFICIENT_MEMORY`, `CUDA_OOM`, and `REQUEST_TIMEOUT` are retryable. The other stable codes are not.

## Measurement

`measure_safe_error` takes the placement plan and one observation. It returns the report. It does not read a prompt.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the safe error report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the safe error report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the safe error report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says safe error concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says safe error run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says safe error warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says safe error request count is one.
12. The report checks in the Report section, in the order written there.

`verify_safe_error` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the safe error validation did not match.

## Files

`write_safe_error_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the safe error directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the safe error path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the safe error output already exists.

The report is created with `create_new`, written, and `fsync`ed. The prompt is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_safe_error`. The redacted log record is KIP-INFER-0063. Computing the Ed25519 curve has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
