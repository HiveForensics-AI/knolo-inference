# KIP-INFER-0050 — Hub installation

Status: `measure_hub_install` records that every pinned artifact of one cold micro fixture is present. It returns a `knolo.infer.install-report`. The four artifacts are the `.kmodel`, the weight artifact, the tokenizer, and the template. It does not open those files, does not download weights, and does not promote a CAS entry. `measure_hub_record` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Install

The source provider is `local`. A missing artifact is `MODEL_ARTIFACT_MISSING`. The weight artifact differs from the model image. The template differs from the tokenizer. Weight bytes are not a field. The artifact count is `4`.

## Report

The report is a versioned contract. Its identity root is `H(infer-install, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `modelImageRoot` | root of the `.kmodel` |
| `artifactRoot` | root of the weight artifact |
| `tokenizerRoot` | root of the tokenizer |
| `templateRoot` | root of the template |
| `publisher` | publisher, one line, at most 128 bytes |
| `licenseId` | license id, one line, at most 128 bytes |
| `sourceProvider` | `local` |
| `artifactCount` | `4` |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `verified` |
| `extensions` | an empty map |

`kind` is `knolo.infer.install-report` and `version` is `1`. The contract count is forty-four. `infer-install` is the report domain. Weight bytes are not a field.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the hub install extensions are empty. A source other than `local` is `CONTRACT_INVALID`, and the message says the hub install does not download weights. An artifact count other than four says the hub install verifies four artifacts. A repeated weight root says the weight artifact repeats the model image. A repeated template root says the template repeats the tokenizer. An empty or oversized publisher or license id says the field is empty or outside its bounds.

## Measurement

`measure_hub_install` takes the placement plan and one observation. The observation names which of the four artifacts are present. It returns the report. It does not create a file.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the install report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the install report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the install report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says install concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says install run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says install warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says install request count is one.
12. `sourceProvider` is not `local`: `CONTRACT_INVALID`, and the message says the hub install does not download weights.
13. The model image is absent: `MODEL_ARTIFACT_MISSING`, and the message says the hub install is missing the model image.
14. The weight artifact is absent: `MODEL_ARTIFACT_MISSING`, and the message says the hub install is missing the weight artifact.
15. The tokenizer is absent: `MODEL_ARTIFACT_MISSING`, and the message says the hub install is missing the tokenizer.
16. The template is absent: `MODEL_ARTIFACT_MISSING`, and the message says the hub install is missing the template.
17. The report checks in the Report section, in the order written there.

`verify_hub_install` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the install validation did not match.

## Files

`write_install_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the install directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the install path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the install output already exists.

The report is created with `create_new`, written, and `fsync`ed. Weight bytes are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_hub_install`. The NOTICE inventory is KIP-INFER-0051. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
