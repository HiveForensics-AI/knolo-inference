# KIP-INFER-0145 — Serve assurance

Status: an isolated pinned completion from `knolo-infer serve` stamps `assurance` as `same_build_replayable`. The receipt carries the same evidence binding and the same signature as `knolo-infer run` when the request and the signing key supply them. `knolo-infer replay` is still what sets `exact_replay_verified`. Token ids for the micro fixture stay the same. This slice does not add a versioned contract.

## Receipt

`same_build_replayable` means the pinned request can be replayed on a matching build. Serve does not rerun the forward to label the first response. `throughput` stays `BACKEND_NOT_ALLOWED`.

The receipt policy strings stay separate from assurance. A native JSON completion is `atomic-verified`. A native event stream is `durable-stream`. `/v1/chat/completions` is `compatibility`. The OpenAI path does not take an evidence object. Its assurance is still `same_build_replayable` when a receipt is stored.

A native completion accepts `evidence` before the prompt is compiled. A bad digest is `DIGEST_INVALID` and does not open a journal. The stored receipt's `knowledge` field is that binding. `--sign-key` on `serve` signs the receipt id after `computed_id`.

The cold `corruption_campaign` receipt keeps `assurance` `compatibility`. That campaign is not a serve completion.

## Out of this slice

The dense Llama-family adapter is KIP-INFER-0146. A `format: gguf` manifest stays `MODEL_IMAGE_INVALID` until KIP-INFER-0147. Metal, KV scatter, and disaggregated serving stay follow-on.
