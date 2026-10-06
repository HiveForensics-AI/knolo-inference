# KIP-INFER-0051 — Supply notice

Status: `measure_supply_notice` records the NOTICE and SBOM identity for the engine build of one cold micro fixture. It returns a `knolo.infer.notice-report`. The caller supplies the roots and the component count. A `cpu` inventory does not name Candle. A `cuda` inventory does. It does not read a lockfile, does not write an SBOM, and does not verify an Ed25519 signature. `measure_hub_install` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Inventory

The notice root differs from the SBOM root. The component count is `1` through `256`. `featureSet` is `cpu` or `cuda`. The default binary stays on `cpu`, so that inventory records `candleNamed` false. The measurement does not link Candle.

## Report

The report is a versioned contract. Its identity root is `H(infer-notice, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `noticeRoot` | root of the NOTICE inventory |
| `sbomRoot` | root of the SBOM inventory |
| `componentCount` | named third-party components |
| `featureSet` | `cpu` or `cuda` |
| `candleNamed` | whether the inventory names Candle |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | an empty map |

`kind` is `knolo.infer.notice-report` and `version` is `1`. The contract count is forty-four. `infer-notice` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the notice extensions are empty. A repeated SBOM root says the sbom repeats the notice. A component count of zero says the notice names no component. A count above 256 says the notice inventory is too large. A `cpu` inventory that names Candle says the cpu notice names candle. A `cuda` inventory that omits Candle says the cuda notice omits candle. Any other feature set says the field has an unsupported value.

## Measurement

`measure_supply_notice` takes the placement plan and one observation. It returns the report. It does not create a file.

Device `cpu` and device `slot-0` both record a report when the feature set matches the Candle mark. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the notice report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the notice report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the notice report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says notice concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says notice run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says notice warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says notice request count is one.
12. The report checks in the Report section, in the order written there.

`verify_supply_notice` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the notice validation did not match.

## Files

`write_notice_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the notice directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the notice path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the notice output already exists.

The report is created with `create_new`, written, and `fsync`ed. The NOTICE text and the SBOM are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_supply_notice`. The signed release manifest is KIP-INFER-0052. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred. Ed25519 signatures stay shape-checked.
