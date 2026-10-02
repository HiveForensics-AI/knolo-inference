# KIP-INFER-0052 — Signed release manifest

Status: `measure_release_manifest` records one release manifest for a cold micro fixture. It returns a `knolo.infer.release-report`. The caller supplies the notice root, the SBOM root, the binary-set root, and the signature status. `unsigned-local` carries no signature. `shape-checked` carries one signature block. It does not write a manifest file, does not read a binary, and does not verify an Ed25519 key. `measure_supply_notice` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Manifest

The notice root differs from the SBOM root. The binary-set root differs from the engine build, the notice, and the SBOM. `signatureStatus` is `unsigned-local` or `shape-checked`. An unsigned release has `signatureCount` `0`. A shape-checked release has `signatureCount` `1`.

## Report

The report is a versioned contract. Its identity root is `H(infer-release, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `noticeRoot` | root of the NOTICE inventory |
| `sbomRoot` | root of the SBOM inventory |
| `binarySetRoot` | root of the binary inventory |
| `signatureStatus` | `unsigned-local` or `shape-checked` |
| `signatureCount` | `0` or `1`, matching the status |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.release-report` and `version` is `1`. The contract count is forty-eight. `infer-release` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the release extensions are empty. A repeated SBOM root says the sbom repeats the notice. A binary-set root equal to the engine build says the binary set repeats the engine build. One equal to the notice says the binary set repeats the notice. One equal to the SBOM says the binary set repeats the sbom. An unsigned release with a non-zero count says an unsigned release carries a signature. A shape-checked release whose count is not one says a shape-checked release carries one signature. Any other status says the field has an unsupported value.

## Measurement

`measure_release_manifest` takes the placement plan and one observation. It returns the report. It does not create a manifest file.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the release report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the release report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the release report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says release concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says release run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says release warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says release request count is one.
12. The report checks in the Report section, in the order written there.

`verify_release_manifest` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the release validation did not match.

## Files

`write_release_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the release directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the release path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the release output already exists.

The report is created with `create_new`, written, and `fsync`ed. The manifest text is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_release_manifest`. The binary inventory is KIP-INFER-0053. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred. Ed25519 signatures stay shape-checked.
