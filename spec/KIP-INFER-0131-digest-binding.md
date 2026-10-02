# KIP-INFER-0131 — Digest binding

Status: `measure_binding` records one host-supplied digest binding for a cold micro fixture. It returns a `knolo.infer.binding-report`. `bound` and `rejected` say the host bound the digest. `unsigned-local` does not. A public key is 32 bytes, a signature is 64 bytes, and a scalar is 32 bytes when the host bound them. Key bytes are not a field. `evidenceBound` stays false. It does not bind the digest and does not bind the evidence. `measure_payload` does not call it. `knolo-infer run` and `knolo-infer serve` do not call it. The `throughput` execution mode stays `BACKEND_NOT_ALLOWED`. A failed measurement issues no report.

## Report

The report is a versioned contract. Its identity root is `H(infer-binding, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` |
| `placementRoot` | root of the `PlacementPlanV1` |
| `digestRoot` | root distinct from the engine build and the placement |
| `evidenceRoot` | root distinct from the digest, the engine build, and the placement |
| `bindStatus` | `bound`, `rejected`, or `unsigned-local` |
| `digestBound` | `true` for `bound` and `rejected`; `false` for `unsigned-local` |
| `evidenceBound` | `false` |
| `publicKeyBytes` | `32` when bound; `0` when unsigned |
| `signatureBytes` | `64` when bound; `0` when unsigned |
| `scalarBytes` | `32` when bound; `0` when unsigned |
| `keyMaterialPresent` | `false` |
| `executionMode` | `isolated-replay` or `pinned` |
| `cachePolicy` | `off` |
| `concurrency` | `1` |
| `runCount` | `1` |
| `warmState` | `cold` |
| `requestCount` | `1` |
| `validationResult` | `verified` for `bound`; `recorded` otherwise |
| `extensions` | an empty map |

`kind` is `knolo.infer.binding-report` and `version` is `1`. The contract count is one hundred thirty-three. `infer-binding` is the report domain.

A non-empty extension map is `CONTRACT_INVALID`, and the message says the binding extensions are empty. A digest root equal to the engine build says the bound digest repeats the engine build. A digest root equal to the placement says the bound digest repeats the placement. An evidence root equal to the engine build says the evidence root repeats the engine build. An evidence root equal to the placement says the evidence root repeats the placement. An evidence root equal to the digest says the evidence root repeats the digest. `keyMaterialPresent` true says key material stays in host storage. `evidenceBound` true says evidence stays unbound. An unsigned release with `digestBound` true says an unsigned release binds the digest. A bound status with `digestBound` false says a bound digest records the binding. A rejected status with `digestBound` false says a rejected digest records the binding. A bound status stores `validationResult` `verified`, and any other result says a bound digest is verified. An unsigned or rejected status stores `recorded`, and any other result says only a bound digest is verified.

## Measurement

`measure_binding` takes the placement plan and one observation. It returns the report. The digest is not bound.

Device `cpu` and device `slot-0` both record a report. Token ids are unchanged by the measurement.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The plan fails `validate`: that failure is returned.
2. `graphCaptureMode` is not `off`: `CONTRACT_INVALID`, and the message says graph capture is off for the binding report.
3. `rejectionReason` is present: `PLACEMENT_UNSATISFIABLE`, and the message says the placement was already rejected.
4. The context reservation, KV block size, or KV precision is not the micro fixture: `CONTRACT_INVALID`, and the message says the binding report is the micro fixture.
5. `executionMode` is `throughput`: `BACKEND_NOT_ALLOWED`, and the message says the throughput execution mode is not enabled.
6. `executionMode` is neither `isolated-replay` nor `pinned`: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
7. `cachePolicy` is not `off`: `CONTRACT_INVALID`, and the message says prefix cache is off for the binding report.
8. `concurrency` is not `1`: `CONTRACT_INVALID`, and the message says binding concurrency is one.
9. `runCount` is not `1`: `CONTRACT_INVALID`, and the message says binding run count is one.
10. `warmState` is not `cold`: `CONTRACT_INVALID`, and the message says binding warm state is cold.
11. `requestCount` is not `1`: `CONTRACT_INVALID`, and the message says binding request count is one.
12. The report checks in the Report section, in the order written there.

`verify_binding` checks the stored report and recomputes the measurement. A recomputed byte mismatch says the binding validation did not match.

## Files

`write_binding_report` verifies the value again, then writes the canonical CBOR into a directory the caller already created. A missing directory is `CONTRACT_INVALID`, and the message says the binding directory does not exist. A symlink component that leaves the directory is `CONTRACT_INVALID`, and the message says the binding path leaves the directory. An existing output, including a symlink, is `CONTRACT_INVALID`, and the message says the binding output already exists.

The report is created with `create_new`, written, and `fsync`ed.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_binding`. Multimodal input has not started. Large expert placement has not started. Extreme-context cache has not started. Vision projector has not started. Positional extension has not started. Exact attention has not started. RMSNorm has not started. RoPE has not started. Fused residual has not started. Metal backend has not started. KV scatter has not started. Disaggregated serving has not started. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The dense Llama-family adapter stays deferred.
