# KIP-INFER-0054 — Reproducible build record

Status: `measure_reproducible_build` records the lock root and the instruction source for one cold micro fixture. It returns a `knolo.infer.reproducible-report`. The caller supplies both roots and the instruction count. `featureSet` is `cpu` or `cuda`. `buildProfile` is `debug` or `release`. It does not run Cargo, does not read `Cargo.lock`, and does not rebuild the binary. `measure_binary_inventory` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Pin

The lock root differs from the engine build root. The source root differs from both. The instruction count is `1` through `32`. A `cuda` feature set does not by itself name Candle; that mark stays on the notice report.

## Report

The report is a versioned contract. Its identity root is `H(infer-reproducible, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `lockRoot` | root of the pinned lock bytes |
| `sourceRoot` | root of the instruction text |
| `featureSet` | `cpu` or `cuda` |
| `buildProfile` | `debug` or `release` |
| `instructionCount` | pinned instruction steps |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.reproducible-report` and `version` is `1`. The contract count is forty-eight. `infer-reproducible` is the report domain. The instruction text is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the reproducible extensions are empty. Any other feature set or build profile says the field has an unsupported value. A lock root equal to the engine build says the lock repeats the engine build. A source root equal to the engine build says the source repeats the engine build. A source root equal to the lock says the source repeats the lock. A count of zero says the build names no instruction. A count above 32 says the build instructions are too large.

## Measurement

`measure_reproducible_build` takes the placement plan and one observation. It returns the report. It does not create an instruction file.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the reproducible report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the reproducible report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the reproducible report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says reproducible concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says reproducible run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says reproducible warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says reproducible request count is one.
12. The report checks in the Report section, in the order written there.

`verify_reproducible_build` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the reproducible validation did not match.

## Files

`write_reproducible_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the reproducible directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the reproducible path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the reproducible output already exists.

The report is created with `create_new`, written, and `fsync`ed. The instruction text is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_reproducible_build`. The signature shape gate is KIP-INFER-0055. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred. Ed25519 signatures stay shape-checked.
