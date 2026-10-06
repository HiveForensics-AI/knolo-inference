# KIP-INFER-0046 — Hub model record

Status: `measure_hub_record` records Hub metadata for one cold micro fixture. It returns a `knolo.infer.hub-report`. It does not upload, does not download weights, does not open a weight file, and does not write the weight bytes into the report. `measure_agent_effect` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Native support

`nativeSupported` is true only when `distribution` is `active`, greedy token parity is true, and `supportLevel` is `conformant` or `blessed`. An `experimental` mark is recorded and is not native-supported. A `yanked` record is not native-supported. A `blessed` conformance receipt without parity fails that receipt's own check, and the message says blessed support requires greedy token parity. `conformant` without parity is `CONTRACT_INVALID`, and the message says conformant support requires greedy token parity. A stored Hub report that says `blessed` without parity is `CONTRACT_INVALID`, and the message says fast execution is not blessed. The source provider is `local`. Any other provider is `CONTRACT_INVALID`, and the message says the hub record does not download weights.

## Report

The report is a versioned contract. Its identity root is `H(infer-hub, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `publisher` | publisher name, at most 128 bytes |
| `modelImageRoot` | root of the micro model image |
| `artifactRoot` | root of the weight artifact |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `adapterId` | `knolo.micro.v1` |
| `quantization` | `f32` |
| `licenseId` | license id, at most 128 bytes |
| `sourceProvider` | `local` |
| `conformanceRoot` | root of the `ModelConformanceReceiptV1` |
| `benchmarkRoot` | root of the benchmark receipt, different from the conformance root |
| `supportLevel` | the conformance receipt's `experimental`, `conformant`, or `blessed` |
| `greedyTokenParity` | the conformance receipt's parity |
| `distribution` | `active` or `yanked` |
| `nativeSupported` | the computed flag |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.hub-report` and `version` is `1`. The contract count is forty. `infer-hub` is the report domain. Weight bytes are not a field. A signature stays on the model image and is not copied here.

A stored `nativeSupported` that differs from the rule is `CONTRACT_INVALID`, and the message says native support does not match. An adapter other than `knolo.micro.v1` on the observation is `UNSUPPORTED_ARCHITECTURE`, and the message says that architecture adapter is not compiled in. A stored adapter or quantization other than the micro fixture is `CONTRACT_INVALID`, and the message says the hub record is the micro fixture.

## Measurement

`measure_hub_record` takes the placement plan, one observation, and the conformance receipt. It returns the report. It does not create a file.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the hub report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the hub report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the hub report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says hub concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says hub run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says hub warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says hub request count is one.
12. `adapterId` is not `knolo.micro.v1`: `UNSUPPORTED_ARCHITECTURE`, and the message says that architecture adapter is not compiled in.
13. `quantization` is not `f32`: `CONTRACT_INVALID`, and the message says the hub record is the micro fixture.
14. `sourceProvider` is not `local`: `CONTRACT_INVALID`, and the message says the hub record does not download weights.
15. The conformance receipt fails `validate`: that failure is returned. A blessed mark without parity uses that receipt's message, which says blessed support requires greedy token parity.
16. The conformance adapter differs: `CONTRACT_INVALID`, and the message says the adapter id does not match.
17. The conformance engine build root differs: `CONTRACT_INVALID`, and the message says the engine build root does not match.
18. The benchmark root equals the conformance root: `CONTRACT_INVALID`, and the message says the hub record repeats a receipt.
19. The support level is `conformant` and parity is false: `CONTRACT_INVALID`, and the message says conformant support requires greedy token parity. A stored report whose support level is `blessed` without parity says fast execution is not blessed.

`verify_hub_record` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the hub validation did not match.

## Files

`write_hub_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the hub directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the hub path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the hub output already exists.

The receipt is created with `create_new`, written, and `fsync`ed. The conformance receipt and the weights are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_hub_record`. Hub does not store weight blobs. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred. The receipt view is KIP-INFER-0047.
