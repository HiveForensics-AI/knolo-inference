# KIP-INFER-0143 — Evidence binding

Status: `knolo-infer run` and a native `complete` store an optional Knowledge Image root, query receipt ids, and Reflex receipt ids on `InferenceReceiptV1.knowledge`. The engine does not open a `.knolo` image, invent ids, or call `measure_evidence_composition`. Omitted binding keeps `knowledge` absent, so a micro receipt that does not pass these fields stays valid. This slice does not add a versioned contract.

## Binding

The caller supplies digests. A value that is not `sha256-` plus 64 lowercase hex is `DIGEST_INVALID`, and nothing is stored. `contextRoot` is `H(infer-evidence, payload)` over this text, in caller order:

```text
knowledge:<digest or empty>
query:<id>
reflex:<id>
```

`knowledgeCommitRoot` stays absent. `orderedEvidenceIds` and `extensions` stay empty. The binding root is `H(infer-evidence, document)` over that canonical object. The intent stores that root in `evidenceBindingRoot` only when a binding is present. The prompt compiler still leaves its own evidence field empty, so token ids do not change.

`run` accepts repeatable `--knowledge-image`, `--query-receipt`, and `--reflex-receipt`. Native JSON accepts `evidence` with `knowledgeImageRoot`, `queryReceiptIds`, and `reflexReceiptIds`. The OpenAI path does not accept that object. A native body that names `evidence` and then supplies no digest is `CONTRACT_INVALID`.

`receipt verify` with the same flags recomputes the binding and compares it to `knowledge`. A missing or different binding is `CONTRACT_INVALID`. Flags that are all omitted do not require a binding.

## TypeScript

`@knolo/infer` exports `acceptReceipt`. It accepts a receipt when `modelRuntimeRoot` matches, `knowledgeImageRoot` matches when the caller required one, and `assurance` matches. The reasons are `model root`, `knowledge image`, `assurance`, and `accepted`. It does not import Agents.

## Out of this slice

Receipt signatures are KIP-INFER-0144. Serve assurance is KIP-INFER-0145. The dense Llama-family adapter is KIP-INFER-0146.
