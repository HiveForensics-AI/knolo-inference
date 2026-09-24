# KIP-INFER-0011 — Redacted tracing

Status: `knolo-infer serve` writes a structured trace for each completion. The worker still owns the weights, the page pool, and the scheduler, and it still does not link Candle. `knolo-infer run` stays one isolated sequence. The trace is not a rooted contract and is not part of the receipt. There is still no CUDA kernel.

## Files

```text
{home}/traces/by-id/<request-id>.jsonl
{home}/traces/rejected.jsonl
```

`<request-id>` is the completion id. It matches the existing request-id grammar, so the path stays inside `by-id`. `rejected.jsonl` is only for a completion that fails before that file is opened. One JSON object per line. Keys are sorted. Integers are whole. There are no floats.

A trace write failure does not change the HTTP status or the token ids. The scrape and the health check do not append a completion line.

## Allowlist

| Field | Where | Value |
|---|---|---|
| `stage` | every line | `api`, `prompt`, `admission`, `prefill`, `decode`, `finalize` |
| `requestId` | request file | the completion id |
| `class` | `api` | `interactive`, `standard`, `batch`, `background` |
| `stream` | `api` | boolean |
| `promptRoot` | successful `prompt` | prompt-plan root, `sha256-` plus 64 lowercase hex digits |
| `promptTokens` | successful `prompt` | prompt token count |
| `outcome` | terminal stage lines | `admitted`, `rejected`, `stop`, `length`, `cancelled`, `error` |
| `code` | `rejected` or `error` | a stable error name |
| `chunk` | `prefill` | zero-based chunk index |
| `tokens` | `prefill` | token count in that chunk |
| `index` | `decode` | output index |
| `receiptRoot` | `stop` or `length` | receipt id |

A prompt, output, token id, model alias, filesystem path, seed, or free-form message is not a field. A root that is not a digest is not written. `rejected.jsonl` has `stage`, `outcome`, and `code` only.

## Stages

```text
api → prompt → admission → prefill chunks → decode iterations → finalize
```

`api` is written when planning starts. `prompt` carries the prompt-plan root and the prompt token count, or `outcome="rejected"` and the compiler's code. `admission` is `admitted` or `rejected`. One `prefill` line is one scheduler chunk. Its `tokens` values sum to `promptTokens`. One `decode` line is one sampled output token, without the token id. `finalize` is the last line.

`stop` and `length` include `receiptRoot` and no `code`. `cancelled` has neither. `rejected` and `error` include `code` and no `receiptRoot`. A failure before planning writes one `api` line to `rejected.jsonl` and does not open a request file.

The worker tells the supervisor about a prefill chunk with a `prefill` frame: `chunk`, `requestId`, and `tokens`. That frame has no token-id array. Decode still uses the existing `delta` frame. The trace keeps the index and drops the token id. Neither frame changes which token is sampled.

## Out of this slice

Authentication, drain, unload, and CUDA stay out. The trace is not served over HTTP. Prefix cache stays off.
