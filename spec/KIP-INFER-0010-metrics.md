# KIP-INFER-0010 — Prometheus metrics

Status: `knolo-infer serve` exposes Prometheus text on the loopback listener. The worker still owns the weights, the page pool, and the scheduler, and it still does not link Candle. `knolo-infer run` stays one isolated sequence. There is still no CUDA kernel and no redacted log stream.

## Exposition

```text
GET /metrics
GET /knolo/infer/v1/metrics
```

Both paths return the same body. The status is 200. `Content-Type` is `text/plain; version=0.0.4; charset=utf-8`. Any other method is 405. The scrape does not start a worker and does not admit a request. A scrape that arrives while the worker is between scheduler iterations is answered before the next forward.

Labels are a fixed set: `class` (`interactive`, `standard`, `batch`, `background`), `outcome` (`stop`, `length`, `cancelled`, `error`, `rejected`), `code` (a stable error name), `state` (`total`, `free`, `pinned`), `kind` (`memory`, `cuda`), and `le`. A request id, prompt, output, model alias, path, or tenant is not a label.

## Series

| Series | Kind | Source |
|---|---|---|
| `knolo_infer_worker_up` | gauge | 1 when the worker is ready |
| `knolo_infer_metrics_stale` | gauge | 1 when this scrape kept the previous worker snapshot |
| `knolo_infer_worker_restarts_total` | counter | a ready worker left the ready state |
| `knolo_infer_queue_depth{class}` | gauge | sequences still in the queue |
| `knolo_infer_admission_rejected_total{code}` | counter | scheduler and supervisor admission failures |
| `knolo_infer_model_load_seconds` | gauge | weight verification and resident build |
| `knolo_infer_verified_bytes` | gauge | weight bytes accepted after the file hash |
| `knolo_infer_time_to_first_token_seconds` | histogram, by class | submit until the first output token |
| `knolo_infer_prefill_tokens_total` | counter | prompt tokens the scheduler processed |
| `knolo_infer_prefill_seconds_total` | counter | time in prefill forwards |
| `knolo_infer_decode_tokens_total` | counter | sampled output tokens |
| `knolo_infer_decode_seconds_total` | counter | time spent sampling those tokens |
| `knolo_infer_active_sequences` | gauge | sequences that have not finished |
| `knolo_infer_retained_sequences` | gauge | slots still held, including finished sequences not yet dropped |
| `knolo_infer_kv_pages{state}` | gauge | pool total, free, and pinned |
| `knolo_infer_prefix_cache_lookups_total` | counter | stays 0; prefix cache is not allocated |
| `knolo_infer_prefix_cache_hits_total` | counter | stays 0 |
| `knolo_infer_prefix_reused_tokens_total` | counter | stays 0 |
| `knolo_infer_oom_total{kind}` | counter | `memory` counts `INSUFFICIENT_MEMORY`; `cuda` stays 0 |
| `knolo_infer_receipt_finalize_seconds` | histogram | worker result until the receipt is stored |
| `knolo_infer_cancellations_total` | counter | sequences retired as `cancelled` |
| `knolo_infer_request_duration_seconds` | histogram, by class and outcome | plan acceptance until the supervisor finishes |
| `knolo_infer_scheduler_iteration_seconds` | histogram | one `CpuScheduler::step` |

Histogram buckets for request, first-token, and receipt time are `0.001`, `0.005`, `0.01`, `0.025`, `0.05`, `0.1`, `0.25`, `0.5`, `1`, `2.5`, `5`, `10`, and `+Inf` seconds. Scheduler iterations use `0.0001`, `0.0005`, `0.001`, `0.005`, `0.01`, `0.05`, `0.1`, `0.5`, `1`, `5`, and `+Inf`. Bucket counts are cumulative. Sums are seconds with a nine-digit fraction.

Admission codes that have a series are `CONTRACT_INVALID`, `CONTEXT_LIMIT_EXCEEDED`, `INSUFFICIENT_MEMORY`, `PROMPT_COMPILATION_FAILED`, and `OTHER`. A malformed HTTP body is not an admission rejection. `CONTRACT_INVALID` here is a duplicate request id. `INSUFFICIENT_MEMORY` from the page pool also increments `knolo_infer_oom_total{kind="memory"}`.

Pinned pages are the pages held by active sequences. An inactive sequence can be evicted, so its pages are not pinned. Pages reserved by admission and not yet written stay in `free`.

## Worker snapshot

The supervisor sends `stats` on the existing length-prefixed CBOR socket. The worker replies with integers only: queue depth, admission counters, token totals, iteration buckets, the page census, retained slots, prefix zeros, model-load nanoseconds, and verified bytes. The reply is not a rooted contract. Unknown fields are rejected. The snapshot is not part of the receipt and does not change token ids.

`knolo_infer_worker_up` is 0 while the worker is down. Live gauges (queue, active sequences, retained sequences, pages, model-load time, verified bytes) are 0 then. Counters keep the last snapshot. The next ready worker replaces that snapshot; a Prometheus `rate` treats the drop as a counter reset. A scrape that does not receive a snapshot within two seconds sets `knolo_infer_metrics_stale` to 1 and keeps the previous snapshot. Supervisor shutdown is not a restart. A ready worker that exits is one restart.

A stats request does not sample a token and does not change a later completion of the same prompt.

## Out of this slice

Redacted tracing, authentication, drain, unload, and CUDA stay out. `kind="cuda"` remains 0 on this machine. Prefix series remain 0 until a later slice allocates a prefix cache.
