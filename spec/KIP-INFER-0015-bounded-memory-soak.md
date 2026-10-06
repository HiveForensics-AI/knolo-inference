# KIP-INFER-0015 — Bounded-memory soak

Status: a long run of `knolo-infer serve` keeps the KV pool that was allocated when the worker started, and it drops each finished request after the result is written. `knolo-infer run` stays one isolated sequence. There is still no CUDA kernel. Prefix cache stays off. The soak does not cap journal or trace disk.

## Pool

The worker builds one page pool before it accepts a request. That pool is eight pages of 16 tokens on the micro-model, the same `CPU_KV_PAGE_POOL` as KIP-INFER-0005. Its key and value bytes are fixed at construction. Admitting, finishing, and reusing sequences does not add a page and does not reallocate those bytes. A pool whose key and value bytes would exceed 64 MiB is still `INSUFFICIENT_MEMORY` at construction.

`knolo_infer_kv_pages{state="total"}` stays at that page count for the life of the worker. When the worker is idle, `free` equals `total` and `pinned` is 0. An active sequence is not evicted to make room. A request that does not fit is still `INSUFFICIENT_MEMORY`.

## Reap

The scheduler keeps a slot until the result has been delivered. The slot holds the prompt token ids, the sampled token ids, and the logits. After the worker writes the terminal result on the supervisor socket, it drops that slot. Those bytes leave the scheduler. The request id can be admitted again.

A call that drops a request which has not finished is `CONTRACT_INVALID`. The slot stays, and the request still runs. Dropping an id the scheduler does not hold is the same error.

`knolo-infer serve` still creates one journal directory per request id. A second completion with that id is `RECEIPT_PERSIST_FAILED` and is not submitted. The journal rule is unchanged. The soak uses a new id for each completion.

## Soak

Thirty-two sequential completions of the same prompt and the same sampler return the same token ids as that request run alone. After each one:

- the page total is the pool size;
- free pages equal that total;
- pinned pages are 0;
- active sequences are 0;
- retained sequences are 0;
- the worker process is the same process.

The worker's resident set is sampled after the first of those completions, when the model is already loaded. After the thirty-second completion it is at most 8 MiB higher. The listener stays up. This call does not count a restart.

Journals and traces still grow by one directory or file per new request id. That growth is on disk. It is outside this bound.

## Metrics

`knolo_infer_retained_sequences` is a gauge. It counts slots the scheduler still holds, including a finished slot that has not been dropped. It is 0 when the worker is down. A request id, prompt, or alias is not a label. A scrape does not sample a token.

## Out of this slice

Authentication and CUDA stay out. Prefix cache stays off. The soak does not delete old journals, does not recover a journal left at `accepted` by a lost worker, and does not change the pin.
