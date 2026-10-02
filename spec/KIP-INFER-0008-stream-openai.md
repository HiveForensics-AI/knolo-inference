# KIP-INFER-0008 — Native stream and OpenAI chat subset

Status: `knolo-infer serve` can stream output tokens and can answer a small OpenAI chat subset. The listener stays on `127.0.0.1`. The worker still owns the weights, the page pool, and the scheduler. `knolo-infer run` stays one isolated sequence. There is still no serve-path receipt, no CUDA kernel, and no claim of full OpenAI compatibility.

## IPC

Protocol version stays 1. Two new worker frames are added. Unknown fields and unknown kinds still close the connection.

`admitted` carries the request id. The worker sends it after the scheduler accepts the request and before the next forward. A refusal stays a `reject` and does not also send `admitted`.

`delta` carries the request id, the output index, and one token id. The index is the count of output tokens already produced, starting at 0. The worker sends a `delta` for each sampled token before it sends the terminal `result`. Prefill, cancellation, and a failed step do not send a `delta`. The `result` token list is still the full output, in order, so a client that ignores `delta` frames sees the same completion as KIP-INFER-0007.

## Native event stream

`POST /knolo/infer/v1/complete` accepts the KIP-INFER-0007 body plus a boolean `stream`. Omission and `false` return the JSON completion from that KIP. `true` returns `text/event-stream`.

`stream` is not `generation.stream`. The generation field remains the Philox stream index. A greedy plan still rejects that index and still rejects a seed.

The response headers include `X-Knolo-Request-Id` and `X-Knolo-Receipt: absent`. The receipt header is the literal word `absent`. This slice does not mint a receipt root.

Events, each one JSON object and one line:

```text
knolo.accepted
knolo.delta
knolo.usage
knolo.error
```

`knolo.accepted` follows admission and precedes any token. `knolo.delta` carries `index`, `tokenId`, and the decoded text of that single id. Indexes are contiguous from 0. `knolo.usage` is the terminal event for `stop`, `length`, and `cancelled`. It carries the prompt token count, the output token count, and the finish reason. It does not repeat the output text. `knolo.error` is the terminal event for a worker loss, a timeout, or a delta list that does not match the result. There is no `knolo.receipt` event.

The decoded pieces concatenate to the same text as the JSON completion. The token ids match that completion and match the same request run through the scheduler alone.

A stream write that fails removes the request and sends `cancel`. The listener stays up. A parent-death signal is still not used.

Admission failure is a JSON error, the same status and body as a non-streaming refusal, because the worker has not sent `admitted`.

## OpenAI subset

```text
GET  /v1/models
POST /v1/chat/completions
```

`GET /v1/models` returns the one alias the supervisor was started with. The object is a list of one model whose `id` is that alias and whose `owned_by` is `knolo`.

`POST /v1/chat/completions` accepts only these fields:

```text
model
messages
stream
max_tokens
temperature
top_p
seed
n
```

Unknown fields are refused. `n` may be omitted or `1`. Any other `n` is refused. `messages` has the same shape as the native route: one to 32 objects, each with string `role` and string `content`. Array content parts, tool calls, and multimodal parts are refused because they are not those two fields.

`model` must be the served alias. `max_tokens` maps to `maxOutputTokens`. Omitted sampler fields stay at the model-image defaults, including `topK`, which this subset does not expose. Nothing that is present is ignored.

`temperature` and `top_p` are the only decimals. A decimal has one to six fractional digits and no exponent. The scale is one million, so `0.5` is `500000` micros and `1` top-p is `1000000` millionths. `temperature` must be from 0 through 2 inclusive. `top_p` must be greater than 0 and at most 1. Every other number must be an unsigned integer. A non-zero temperature still requires `seed`. Temperature 0 still rejects `seed`.

`stream: false` returns one JSON object with `id`, `object` `chat.completion`, `model`, one choice, and `usage`. `finish_reason` is `stop` or `length`. The choice text is the decoded output. Prompt and completion token counts are the compiled prompt length and the output length.

`stream: true` uses unnamed `data:` frames. The first choice delta sets `role` to `assistant`. Each output token is a `content` delta. The last choice frame sets `finish_reason` and is followed by `data: [DONE]`. A cancellation, a worker loss, or a token mismatch sends one `error` object and does not send `[DONE]`. The same receipt and request-id headers are present. `X-Knolo-Request-Id` is the native id, so `POST /knolo/infer/v1/cancel` can name it.

HTTP failures on `/v1` use `{"error":{"message","type","code"}}`. `code` is the Infer error code. `type` is `server_error` for worker loss, worker start failure, and timeout, and `invalid_request_error` otherwise. Native routes keep their existing error object.

## Compatibility matrix

| Input | Result |
| --- | --- |
| `model`, `messages`, `max_tokens`, `temperature`, `top_p`, `seed`, `n=1`, `stream` | accepted when the sampler plan is valid |
| any other field, including `stop`, `tools`, `logprobs`, `user`, `response_format` | `CONTRACT_INVALID` |
| `temperature` outside 0 through 2, or an exponent | `CONTRACT_INVALID` |
| non-zero `temperature` without `seed` | `CONTRACT_INVALID` |
| `GET /v1/models` | the served alias only |
| `POST /v1/embeddings`, `POST /v1/responses` | `404` |
| receipt identity | `X-Knolo-Receipt: absent` |

Token ids for a given prompt, sampler, and service class match a non-streaming native completion and the scheduler run alone. The OpenAI route uses service class `standard`.

## Out of this slice

The serve-path journal, `knolo.receipt`, and `GET /knolo/infer/v1/receipts/{digest}` are KIP-INFER-0009. A completion on this path is not replay-verified. Metrics, authentication, embeddings, the responses API, stop strings, and CUDA stay out.
