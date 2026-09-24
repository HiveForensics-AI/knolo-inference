# KIP-INFER-0005 — Paged KV on CPU

Status: the CPU run path allocates from this pool. The continuous scheduler is KIP-INFER-0006. There is no HTTP server.

`knolo-infer run` uses `PagedKv`. Oracle tests still use `SingleBlockKv`. Both stores implement `KvStore`. On the checked-in micro-model fixtures, prefill logits, greedy token ids, and the f32 KV slab match exactly, and the block table is one page.

## Pool

A page holds `block_size` tokens for every layer, with separate f32 key and value slabs. `block_size` is a power of two from 16 to 65536. The micro-model placement uses 16. The run command creates eight pages (`CPU_KV_PAGE_POOL`). The placement's `expectedKvBytes` is that pool, as specified in KIP-INFER-0027. The constructor refuses a pool whose key and value bytes exceed 64 MiB with `INSUFFICIENT_MEMORY`.

Page ids are the lowest free id. `begin_sequence` records an active sequence and does not take a page. The first write of a new page reserves one. `commit_token` appends that page to the block table when the token is the first slot of the page, then increments the committed length. Later tokens in the same page write into the committed page. A commit at a page boundary with no reservation is `CONTRACT_INVALID`.

`abort_pending` returns the reserved page to the free list and leaves the committed length unchanged. `greedy_generate` and `generate_samples` call it when prefill or decode returns an error. A read of an unreserved open token is `CONTRACT_INVALID`. A read past the open token is the same error.

## Reuse and eviction

`release_sequence` zeros every page that sequence held, then returns those ids to the free list. The next reservation takes the lowest free id, so a new sequence does not observe the previous sequence's keys or values.

If the free list is empty, the allocator releases the lowest inactive sequence id other than the caller and tries again. An active sequence is never evicted. When every remaining holder is active, the reservation fails with `INSUFFICIENT_MEMORY` and the caller's committed length stays put. `deactivate` marks a sequence inactive without freeing its pages. The run command does not deactivate, so a live request is not evicted to admit another one.

## Out of this slice

The micro adapter still rejects a context past one page of 16. Multi-page reads are tested on the store directly: token 0 and token 16 land on different pages and keep different values.

There is no prefix-cache index. Contract version 1 still rejects an enabled prefix cache. This slice does not add a worker process or a CUDA allocation. Graph capture stays `off`.
