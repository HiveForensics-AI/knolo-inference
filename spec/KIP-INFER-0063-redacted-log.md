# KIP-INFER-0063 — Redacted log

Status: `measure_redacted_log` records one completion log line for a cold micro fixture. It returns a `knolo.infer.redaction-report`. The stage is `api`, `prompt`, `admission`, `prefill`, `decode`, or `finalize`. The format is one JSON line. The line is redacted. Prompt text, output text, token ids, the alias, and a path are `omitted`. It does not write a log line. `measure_safe_error` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Line

The request id is 1 through 64 characters of ASCII letters, digits, and hyphens. The prompt-plan root differs from the engine build root. The receipt root differs from both.

## Report

The report is a versioned contract. Its identity root is `H(infer-redaction, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `stage` | `api`, `prompt`, `admission`, `prefill`, `decode`, or `finalize` |
| `requestId` | a request token |
| `promptPlanRoot` | root of the prompt plan |
| `receiptRoot` | root of the inference receipt |
| `format` | `json-line` |
| `redacted` | `true` |
| `prompt` | `omitted` |
| `output` | `omitted` |
| `tokenIds` | `omitted` |
| `alias` | `omitted` |
| `path` | `omitted` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.redaction-report` and `version` is `1`. The contract count is fifty-six. `infer-redaction` is the report domain. Prompt text is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the redaction extensions are empty. A stage outside the six names says the field has an unsupported value. An empty or oversized request id says the field is empty or outside its bounds. A request id with any other character says the request id is a token. A prompt-plan root equal to the engine build says the prompt plan repeats the engine build. A receipt root equal to the engine build says the receipt repeats the engine build. A receipt root equal to the prompt plan says the receipt repeats the prompt plan. A format other than `json-line` says a completion log is one json line. `redacted` false says completion logs are redacted. A prompt other than `omitted` says ordinary logs omit prompt text. An output other than `omitted` says ordinary logs omit output text. Token ids other than `omitted` say ordinary logs omit token ids. An alias other than `omitted` says ordinary logs omit the alias. A path other than `omitted` says ordinary logs omit a path.

## Measurement

`measure_redacted_log` takes the placement plan and one observation. It returns the report. It does not append a JSON line.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the redaction report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the redaction report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the redaction report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says redaction concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says redaction run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says redaction warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says redaction request count is one.
12. The report checks in the Report section, in the order written there.

`verify_redacted_log` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the redaction validation did not match.

## Files

`write_redaction_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the redaction directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the redaction path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the redaction output already exists.

The report is created with `create_new`, written, and `fsync`ed. The log line is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_redacted_log`. The curve record is KIP-INFER-0064. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
