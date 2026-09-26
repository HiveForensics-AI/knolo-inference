# KIP-INFER-0053 — Binary hash inventory

Status: `measure_binary_inventory` records the supervisor and worker hashes for one cold micro fixture. It returns a `knolo.infer.binary-report`. The supervisor name is `knolo-infer`. The worker name is `knolo-infer-worker`. The two hashes differ, and neither hash repeats the engine build root. It does not open either binary. `measure_release_manifest` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Inventory

A byte count of zero is `CONTRACT_INVALID`. A byte count above 64 MiB is `INSUFFICIENT_MEMORY` during measurement and issues no report. A stored report above that cap is `CONTRACT_INVALID`. Exactly 64 MiB is recorded. The message names the supervisor binary or the worker binary.

## Report

The report is a versioned contract. Its identity root is `H(infer-binary, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `supervisorName` | `knolo-infer` |
| `supervisorHash` | raw SHA-256 of the supervisor bytes |
| `supervisorBytes` | size of that binary |
| `workerName` | `knolo-infer-worker` |
| `workerHash` | raw SHA-256 of the worker bytes |
| `workerBytes` | size of that binary |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.binary-report` and `version` is `1`. The contract count is forty-eight. `infer-binary` is the report domain. Paths are not fields.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the binary extensions are empty. A supervisor name other than `knolo-infer` says the supervisor binary is knolo-infer. A worker name other than `knolo-infer-worker` says the worker binary is knolo-infer-worker. Equal hashes say the worker hash repeats the supervisor. A hash equal to the engine build says the binary hash repeats the engine build. A zero size says the supervisor binary is empty, or the worker binary is empty. A stored size above 64 MiB says that binary exceeds 64 MiB.

## Measurement

`measure_binary_inventory` takes the placement plan and one observation. It returns the report. It does not open a binary.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the binary report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the binary report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the binary report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says binary concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says binary run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says binary warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says binary request count is one.
12. `supervisorBytes` is above 64 MiB: `INSUFFICIENT_MEMORY`, and the message says the supervisor binary exceeds 64 MiB.
13. `workerBytes` is above 64 MiB: `INSUFFICIENT_MEMORY`, and the message says the worker binary exceeds 64 MiB.
14. The report checks in the Report section, in the order written there.

`verify_binary_inventory` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the binary validation did not match.

## Files

`write_binary_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the binary directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the binary path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the binary output already exists.

The report is created with `create_new`, written, and `fsync`ed. The binaries are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_binary_inventory`. The reproducible-build record is KIP-INFER-0054. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred. Ed25519 signatures stay shape-checked.
