# KIP-INFER-0043 — Recipe status

Status: `measure_recipe_status` records the support mark of one cold micro-fixture recipe. It returns a `knolo.infer.recipe-report`. It does not open a model file, does not run a model, does not write a serve journal, does not read a GGUF payload, does not allocate a KV pool, and does not write a weight file. `measure_llama_comparison`, `measure_vllm_comparison`, and `measure_mistral_comparison` do not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Marks

The support level is `experimental`, `conformant`, or `blessed`. The micro fixture records `experimental`. `conformant` and `blessed` require greedy token parity on the linked `ModelConformanceReceiptV1`. A blessed mark whose parity is false says fast execution is not blessed. A blessed mark also requires the conformance, security, stability, and benchmark roots to be pairwise distinct. Fast execution without that parity is not blessed. The adapter is `knolo.micro.v1`. Any other adapter, including `knolo.llama.v1`, is `UNSUPPORTED_ARCHITECTURE` before a report is issued.

## Report

The report is a versioned contract. Its identity root is `H(infer-recipe, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `modelImageRoot` | root of the micro model image |
| `artifactRoot` | root of the weight artifact |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` that ran |
| `placementRoot` | root of the `PlacementPlanV1` |
| `adapterId` | `knolo.micro.v1` |
| `conformanceRoot` | root of the `ModelConformanceReceiptV1` |
| `securityRoot` | root of the security receipt |
| `stabilityRoot` | root of the stability receipt |
| `benchmarkRoot` | root of the benchmark receipt |
| `supportLevel` | `experimental`, `conformant`, or `blessed` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.recipe-report` and `version` is `1`. The contract count is thirty-six. `infer-recipe` is the report domain.

The stored conformance receipt's support level equals the recipe support level. Its engine build root equals the recipe engine build root. Its adapter id equals the recipe adapter id.

## Measurement

`measure_recipe_status` takes the placement plan, one observation, and the conformance receipt. It returns the report. It does not create a file and it does not call the model.

The plan is the micro fixture plan: context reservation 16, KV block size 16, and KV precision `f32`. Graph capture stays `off`. Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the recipe report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the recipe report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the recipe report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says recipe concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says recipe run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says recipe warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says recipe request count is one.
12. `adapterId` is not `knolo.micro.v1`: `UNSUPPORTED_ARCHITECTURE`, and the message says that architecture adapter is not compiled in.
13. `supportLevel` is not one of the three marks: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
14. The mark is `blessed` and greedy token parity is false: `CONTRACT_INVALID`, and the message says fast execution is not blessed.
15. The mark is `conformant` and greedy token parity is false: `CONTRACT_INVALID`, and the message says conformant support requires greedy token parity.
16. The conformance receipt fails `validate`: that failure is returned.
17. The conformance adapter differs: `CONTRACT_INVALID`, and the message says the adapter id does not match.
18. The conformance engine build root differs: `CONTRACT_INVALID`, and the message says the engine build root does not match.
19. The conformance support level differs: `CONTRACT_INVALID`, and the message says the support level does not match.
20. The mark is `blessed` and two of the conformance, security, stability, and benchmark roots are equal: `CONTRACT_INVALID`, and the message says the blessed recipe repeats a receipt.

`verify_recipe_status` checks the stored report and recomputes the measurement. A root or count that differs says that field does not match. A recomputed byte mismatch says the recipe validation did not match.

`measure_recipe_status` runs that verify before it returns. A verify failure returns no report.

## Files

`write_recipe_report` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the recipe directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the recipe path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the recipe output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The plan and the conformance receipt are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_recipe_status`. Token ids are unchanged. The micro fixture stays `experimental` unless the caller supplies a matching conformance receipt. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter. The dense Llama-family adapter stays deferred. Phase 5 product integration stays the contract package.
