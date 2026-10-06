# KIP-INFER-0006 — CPU continuous scheduler

Status: in-process scheduler for one f32 model and one KV pool. `knolo-infer run` stays one isolated sequence, as KIP-INFER-0004 requires. There is no worker process and no HTTP server.

The scheduler admits several requests onto the pool from KIP-INFER-0005. Each iteration is one prefill chunk, or one new token for every selected decode sequence. The micro-model forward stays single-sequence. The batch is the scheduler's selection, executed in order. A shared run and a run of the same request alone produce the same token ids and the same finish reason.

## Admission

A request carries a request id, the model runtime root, the prompt token ids, a sampler plan, and a service class. The request id is 1 to 64 ASCII letters, digits, `_`, or `-`, and it is unique among slots the scheduler still holds. A finished slot stays until `reap` drops it. KIP-INFER-0015 is what calls `reap` after the result has been delivered. Sequence ids start at 1. A rejected request does not take an id or a page.

The prompt must be non-empty. `prompt length + max_output_tokens` must fit in the configured context. A request that does not fit is `CONTEXT_LIMIT_EXCEEDED`. The scheduler does not drop prompt tokens and does not lower the token budget.

Pages reserved are `ceil((prompt length + max_output_tokens) / block_size)`. The block size is a power of two from 16 to 65536. The reservation is the worst case, so a request that later stops on an end-of-sequence id still held its full budget until it finished. When the reservation plus the pages already held exceeds the pool, admission is `INSUFFICIENT_MEMORY`. An active sequence is not released to make room. A runtime root other than the scheduler's root is `CONTRACT_INVALID`. The scheduler does not switch models.

Service classes and their weights are `interactive` 8, `standard` 4, `batch` 2, and `background` 1. A new request starts at the lowest virtual time of the runnable sequences, or at 0 when none are runnable.

## Iterations

Cancellation is applied before a forward. One cancelled request is retired per iteration. Its KV sequence is released, its pages return to the pool, and its finish reason is `cancelled`. Tokens already sampled stay on the result. No further token is produced.

Otherwise the runnable sequence with the lowest virtual time runs. Equal times keep the lower sequence id. Virtual time increases by `tokens * (8 / weight)`.

If that sequence still has prompt tokens, the iteration prefills at most `prefill_chunk_tokens` of them. The chunk size is part of the scheduler configuration and is at least 1. The last chunk's logits are the prefill logits. The first output token is sampled on a later iteration.

If that sequence is already decoding, the iteration is a decode batch. The batch contains every decode-ready sequence whose virtual time is less than or equal to the lowest virtual time of a sequence that still has prompt tokens. When every runnable sequence is decoding, the batch is all of them. Each member generates one token, in virtual-time order. The token is drawn with the sampler plan: temperature 0 is lowest-id argmax, and a non-zero temperature uses that request's Philox seed, stream, and token step. End of sequence finishes with `stop`. The token budget finishes with `length`.

A forward error retires that sequence with finish reason `error` and the error code. Other sequences stay admitted. A bad prompt does not change their token ids.

## Out of this slice

Prefix cache stays off. Grammar masks, speculative decoding, and CUDA graphs are not selected. The scheduler does not evict an active sequence, and it does not call the store's inactive-sequence reclaim on purpose. The micro adapter still rejects a context past 16 tokens, so a scheduled micro sequence occupies one page.

`knolo-infer run` still records `schedulingMode = isolated` and prefills the whole prompt in one call. The worker process and the loopback HTTP API are KIP-INFER-0007.
