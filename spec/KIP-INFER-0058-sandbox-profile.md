# KIP-INFER-0058 — Worker sandbox profile

Status: `measure_sandbox_profile` records the Linux worker profile for one cold micro fixture. It returns a `knolo.infer.sandbox-report`. The user is `unprivileged`. The network is `none`. The model CAS is `read-only`. Scratch is `worker-only`. Seccomp is `deferred`. The process group is `isolated`. The worker lifetime is `socket-eof`. Arguments are a `direct-array`. It does not change uid, apply seccomp, or spawn a worker. `measure_receipt_key` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Profile

The memory limit is 1 byte through 64 MiB. Exactly 64 MiB is recorded. Shared memory is 0 through 4096 bytes. Zero shared memory is recorded. A parent-death signal is not the worker lifetime.

## Report

The report is a versioned contract. Its identity root is `H(infer-sandbox, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `userClass` | `unprivileged` |
| `network` | `none` |
| `modelCas` | `read-only` |
| `scratch` | `worker-only` |
| `seccomp` | `deferred` |
| `memoryLimitBytes` | `1` through 64 MiB |
| `processGroup` | `isolated` |
| `parentDeath` | `socket-eof` |
| `sharedMemoryBytes` | `0` through `4096` |
| `arguments` | `direct-array` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.sandbox-report` and `version` is `1`. The contract count is fifty-two. `infer-sandbox` is the report domain. A uid and a seccomp program are not fields.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the sandbox extensions are empty. A user class other than `unprivileged` says the worker user is unprivileged. A network other than `none` says the worker has no external network. A model CAS other than `read-only` says the model cas is read-only. Scratch other than `worker-only` says worker scratch is the only writable path. Seccomp other than `deferred` says seccomp stays deferred until the profile is stable. A memory limit of zero says the worker memory limit is zero. A stored memory limit above 64 MiB says worker memory exceeds 64 MiB. A process group other than `isolated` says the worker process group is isolated. A parent death other than `socket-eof` says the worker lifetime is the supervisor socket. Shared memory above 4096 says shared memory exceeds the worker bound. Arguments other than `direct-array` say the worker arguments are a direct array.

## Measurement

`measure_sandbox_profile` takes the placement plan and one observation. It returns the report. It does not apply the profile.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the sandbox report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the sandbox report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the sandbox report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says sandbox concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says sandbox run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says sandbox warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says sandbox request count is one.
12. `memoryLimitBytes` is above 64 MiB: `INSUFFICIENT_MEMORY`, and the message says worker memory exceeds 64 MiB.
13. The report checks in the Report section, in the order written there.

`verify_sandbox_profile` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the sandbox validation did not match.

## Files

`write_sandbox_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the sandbox directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the sandbox path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the sandbox output already exists.

The report is created with `create_new`, written, and `fsync`ed. The profile is not applied.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_sandbox_profile`. The API boundary is KIP-INFER-0059. Evaluating the Ed25519 equation has not started. Seccomp stays deferred. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
