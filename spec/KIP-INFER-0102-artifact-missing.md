# KIP-INFER-0102 — Missing artifact

Status: `measure_artifact_missing` records one pinned artifact the loader did not find for a cold micro fixture. It returns a `knolo.infer.artifact-missing-report`. The reason is `lock`, `alias`, `image`, or `weights`. The code is `MODEL_ARTIFACT_MISSING` and it is not retryable. The source provider is `local`. A missing lockfile is not present, has no pinned alias, and opens neither the image nor the weights. An unpinned alias was read from the lockfile and opens neither file. A missing image names a pinned alias and is not opened. A missing weight artifact opened the image and is not itself opened. `downloaded` stays false. The forward does not run and no receipt is stored. It does not open a file and it does not download. `measure_cofactor` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-artifact-missing, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `imageRoot` | root distinct from the engine build and the placement |
| `artifactRoot` | root distinct from the engine build, the placement, and the image |
| `reason` | `lock`, `alias`, `image`, or `weights` |
| `code` | `MODEL_ARTIFACT_MISSING` |
| `retryable` | `false` |
| `sourceProvider` | `local` |
| `lockPresent` | `false` for `lock`; `true` otherwise |
| `aliasPinned` | `true` for `image` and `weights`; `false` otherwise |
| `imageOpened` | `true` for `weights`; `false` otherwise |
| `weightsOpened` | `false` |
| `downloaded` | `false` |
| `forwardRan` | `false` |
| `receiptStored` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.artifact-missing-report` and `version` is `1`. The contract count is ninety-eight. `infer-artifact-missing` is the report domain. A path is not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the artifact-missing extensions are empty. An image root equal to the engine build says the image repeats the engine build. An image root equal to the placement says the image repeats the placement. An artifact root equal to the engine build says the artifact repeats the engine build. An artifact root equal to the placement says the artifact repeats the placement. An artifact root equal to the image says the artifact repeats the image. A code other than `MODEL_ARTIFACT_MISSING` says a missing artifact is MODEL_ARTIFACT_MISSING. `retryable` true says a missing artifact is not retryable. A source provider other than `local` says the source provider is local. `downloaded` true says pull does not download. `forwardRan` true says a missing artifact does not run the forward. `receiptStored` true says a missing artifact stores no receipt. A lock reason whose lock is present says a missing lockfile is not present. A lock reason with a pinned alias says a missing lockfile has no pinned alias. A lock reason that opened the image says a missing lockfile does not open the image. A lock reason that opened the weights says a missing lockfile does not open the weights. An alias reason whose lock is absent says an unpinned alias was read from the lockfile. An alias reason that is pinned says an unpinned alias is not pinned. An image reason that opened the image says a missing image is not opened. A weights reason that did not open the image says a missing weight artifact opened the image. A weights reason that opened the weights says a missing weight artifact is not opened.

## Measurement

`measure_artifact_missing` takes the placement plan and one observation. It returns the report. It does not open the lockfile, the image, or the weights.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the artifact-missing report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the artifact-missing report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the artifact-missing report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says artifact-missing concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says artifact-missing run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says artifact-missing warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says artifact-missing request count is one.
12. The report checks in the Report section, in the order written there.

`verify_artifact_missing` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the artifact-missing validation did not match.

## Files

`write_artifact_missing_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the artifact-missing directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the artifact-missing path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the artifact-missing output already exists.

The report is created with `create_new`, written, and `fsync`ed. No artifact bytes are read.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_artifact_missing`. Receipt signing has not started. A canonical-CBOR record has not started. A contract-invalid record has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred. `pull` still returns `MODEL_ARTIFACT_MISSING`.
