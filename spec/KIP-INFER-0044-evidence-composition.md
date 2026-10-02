# KIP-INFER-0044 — Core and Reflex composition

Status: `measure_evidence_composition` records one host-supplied Knowledge Image binding for a cold micro fixture. It returns a `knolo.infer.composition-report`. It does not open a `.knolo` image, does not query Core, does not call Reflex, does not invent an evidence id, and does not write a weight file. `measure_recipe_status` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Binding

The host passes an `EvidenceBindingV1`. The knowledge image root and the knowledge commit root are both required, and they differ. At least one query receipt, one Reflex receipt, and one ordered evidence id are required. The engine copies those roots and the counts. It does not replace them.

## Report

The report is a versioned contract. Its identity root is `H(infer-composition, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `modelImageRoot` | root of the micro model image |
| `artifactRoot` | root of the weight artifact |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` that ran |
| `placementRoot` | root of the `PlacementPlanV1` |
| `evidenceRoot` | root of the `EvidenceBindingV1` |
| `knowledgeImageRoot` | root supplied by the host |
| `knowledgeCommitRoot` | commit root supplied by the host |
| `contextRoot` | context root supplied by the host |
| `queryReceiptCount` | number of query receipt ids |
| `reflexReceiptCount` | number of Reflex receipt ids |
| `orderedEvidenceCount` | number of ordered evidence ids |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `recorded` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.composition-report` and `version` is `1`. The contract count is forty. `infer-composition` is the report domain.

## Measurement

`measure_evidence_composition` takes the placement plan, one observation, and the binding. It returns the report. It does not create a file.

The plan is the micro fixture plan. Device `cpu` and device `slot-0` both record a report. Token ids are unchanged.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the composition report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the composition report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the composition report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says composition concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says composition run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says composition warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says composition request count is one.
12. The binding fails `validate`: that failure is returned.
13. The knowledge image root is absent: `CONTRACT_INVALID`, and the message says the composition has no knowledge image.
14. The knowledge commit root is absent: `CONTRACT_INVALID`, and the message says the composition has no knowledge commit.
15. The two roots are equal: `CONTRACT_INVALID`, and the message says the knowledge commit repeats the image.
16. The query receipt list is empty: `CONTRACT_INVALID`, and the message says the composition has no query receipt.
17. The Reflex receipt list is empty: `CONTRACT_INVALID`, and the message says the composition has no reflex receipt.
18. The ordered evidence list is empty: `CONTRACT_INVALID`, and the message says the composition has no evidence id.
19. An observation root or count differs from the binding: `CONTRACT_INVALID`, and the message says that field does not match.

A count above 256 query receipts, 256 Reflex receipts, or 4096 evidence ids is `CONTRACT_INVALID`, and the message says the evidence list is too large.

`verify_evidence_composition` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the composition validation did not match.

`measure_evidence_composition` runs that verify before it returns. A verify failure returns no report.

## Files

`write_composition_report` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the composition directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the composition path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the composition output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The binding is not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_evidence_composition`. Token ids are unchanged. The engine does not mutate a Knowledge Image. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred. The Agents host effect is KIP-INFER-0045.
