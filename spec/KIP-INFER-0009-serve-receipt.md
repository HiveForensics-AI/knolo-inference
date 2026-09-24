# KIP-INFER-0009 — Serve-path receipt journal

Status: `knolo-infer serve` journals every admitted completion and publishes a receipt for `stop` and `length`. The listener stays on `127.0.0.1`. The worker still owns the weights, the page pool, and the scheduler. Without the `cuda` feature it does not link Candle. `knolo-infer run` stays one isolated sequence with `same_build_replayable`. The `cuda` feature of the worker is KIP-INFER-0023.

## Journal

The home directory is `--home`, or `~/.knolo/infer` when the flag is omitted. Tests pass `--home`.

```text
<home>/journals/<request-id>/000000.accepted.cbor
<home>/journals/<request-id>/NNNNNN.<event>.cbor
<home>/journals/<request-id>/sampler-plan.cbor
<home>/journals/<request-id>/execution-plan.cbor
<home>/receipts/sha256/<64 hex>/receipt.cbor
```

`accepted` is fsynced before the supervisor sends `submit`. Its payload root is the intent root. A second use of the same request id fails with `RECEIPT_PERSIST_FAILED` and does not rewrite the directory.

The execution plan records `schedulingMode = continuous`, `prefillChunkTokens = 4`, `mode = pinned`, and prefix cache off. `receiptPolicy` is `atomic-verified` for a native JSON completion, `durable-stream` for a native event stream, and `compatibility` for `/v1/chat/completions`. The sidecar files are not events. The event chain is the one from KIP-INFER-0004.

A `stop` or `length` result appends `prefill`, one `token` event per output id, and `completed`, then stores the receipt. The receipt is stored before the HTTP body, or before `knolo.receipt` on a stream. A cancellation appends `cancelled` and does not store a receipt. A worker rejection, a worker loss, a timeout, or a token mismatch appends `failed` and does not store a receipt. `cancelled` is a terminal journal name, with `completed` and `failed`.

Prompt text and output text stay out of the journal and the receipt. The receipt stores roots, counts, and ids.

## Engine identity

The default worker runs the `reference-f32` oracle. The receipt says so. `tensorBackend` may be `reference-f32` in addition to `candle-cpu` and `candle-cuda`. The kernel plan root is the digest of the bytes `reference-f32:rmsnorm,rope,attention,swiglu,gemv`. The binary hash is `knolo-infer-worker`, not the supervisor. Placement stays `cpu`. A visible GPU may be recorded and is not selected. The `cuda` feature replaces this identity with the bundle in KIP-INFER-0023.

`queueMicros` and `prefillMicros` are 0. `decodeMicros` and `totalMicros` are the supervisor's elapsed time from the accepted write until the worker result. The worker does not report a split clock.

Assurance is `compatibility`. This path does not rerun the sequence, so it does not claim `same_build_replayable`. `exact_replay_verified` remains the result of `knolo-infer replay` on a `knolo-infer run` receipt.

## HTTP

`X-Knolo-Receipt` is the receipt id when the body is the completion, `pending` on an event stream (the id is not known until the last event), and `absent` when no receipt was stored. `X-Knolo-Request-Id` is present on those responses.

A native JSON completion carries:

```json
"receipt": {"assurance": "compatibility", "receiptRoot": "sha256-..."}
```

A native stream keeps `knolo.accepted`, `knolo.delta`, and `knolo.usage`. `knolo.receipt` follows `knolo.usage` for `stop` and `length` and carries `assurance`, `receiptRoot`, and `requestId`. A cancellation still ends at `knolo.usage` and omits `knolo.receipt`.

An OpenAI completion adds top-level `knolo_receipt` with the same id. A streamed completion puts that field on the final choice chunk, before `data: [DONE]`. The header on that stream is `pending`.

```text
GET /knolo/infer/v1/receipts/sha256-<64 hex>
```

The body is JSON: `receiptRoot`, `assurance`, and `requestId`. The canonical CBOR file under the home directory remains authoritative. A missing file is `404` `RECEIPT_REQUIRED`. Any other digest is `400`.

Token ids for a given prompt, sampler, and service class still match a run of that request alone.

## Out of this slice

Signing, `exact_replay_verified` for a served request, authentication, and CUDA stay out. Metrics are specified in KIP-INFER-0010. A serve receipt is not a substitute for `knolo-infer replay`.
