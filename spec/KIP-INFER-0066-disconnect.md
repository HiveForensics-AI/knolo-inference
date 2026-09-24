# KIP-INFER-0066 — Client disconnect

Status: `measure_disconnect` records one client disconnect for a cold micro fixture. It returns a `knolo.infer.disconnect-report`. The stage is `queue`, `prefill`, `decode`, or `stream`. The outcome is `cancelled` and the code is `REQUEST_CANCELLED`. The listener stays up. The record does not close the socket. A queued disconnect has no output tokens. A prompt plus output count above 16 is `CONTEXT_LIMIT_EXCEEDED`. It does not close a connection. `measure_rollback` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Request

The request id is 1 through 64 characters of ASCII letters, digits, and hyphens. The prompt count is 1 through 16. A queued disconnect records zero output tokens. Prefill, decode, and stream may record output tokens when the sum stays within 16.

## Report

The report is a versioned contract. Its identity root is `H(infer-disconnect, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `requestId` | a request token |
| `stage` | `queue`, `prefill`, `decode`, or `stream` |
| `outcome` | `cancelled` |
| `code` | `REQUEST_CANCELLED` |
| `listenerUp` | `true` |
| `socketClosed` | `false` |
| `promptTokens` | `1` through `16` |
| `outputTokens` | `0` for `queue`, otherwise a count that keeps the sum within `16` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.disconnect-report` and `version` is `1`. The contract count is sixty. `infer-disconnect` is the report domain. Prompt text is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the disconnect extensions are empty. An empty or oversized request id says the field is empty or outside its bounds. A request id with any other character says the request id is a token. A stage outside the four names says the field has an unsupported value. An outcome other than `cancelled` says a client disconnect cancels the request. A code other than `REQUEST_CANCELLED` says a client disconnect is REQUEST_CANCELLED. `listenerUp` false says the listener stays up. `socketClosed` true says a disconnect record does not close the socket. A prompt count of zero says the disconnect has no prompt. A prompt count above 16, or a prompt count plus an output count above 16, is `CONTEXT_LIMIT_EXCEEDED`, and the message says the disconnect is the micro fixture. A queued disconnect with any output token says a queued disconnect has no output.

## Measurement

`measure_disconnect` takes the placement plan and one observation. It returns the report. It does not close a socket.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the disconnect report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the disconnect report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the disconnect report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says disconnect concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says disconnect run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says disconnect warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says disconnect request count is one.
12. The report checks in the Report section, in the order written there.

`verify_disconnect` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the disconnect validation did not match.

## Files

`write_disconnect_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the disconnect directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the disconnect path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the disconnect output already exists.

The report is created with `create_new`, written, and `fsync`ed. The socket is not closed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_disconnect`. The receipt-store failure is KIP-INFER-0067. The base-point record is KIP-INFER-0068. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
