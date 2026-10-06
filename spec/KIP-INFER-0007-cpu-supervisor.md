# KIP-INFER-0007 — CPU supervisor

Status: one supervisor, one worker process, and a loopback HTTP API for `knolo.micro.v1`. `knolo-infer run` stays one isolated sequence, as KIP-INFER-0004 requires. There is no receipt on this path and no CUDA kernel. An omitted `stream` flag returns the JSON completion defined here. `stream: true` and the OpenAI routes are KIP-INFER-0008.

The supervisor listens on `127.0.0.1` only. It verifies the pinned model image, compiles the prompt, and spawns `knolo-infer-worker`. The worker loads the weights, owns the KV pool and the `CpuScheduler` from KIP-INFER-0006, and has no listening socket. A worker exit does not stop the listener. The request that was in flight fails with `WORKER_LOST`. The next request starts a new worker.

## IPC

The supervisor binds a Unix socket in a directory mode `0700`. The socket itself is mode `0600`. The frame is a big-endian `u32` length followed by one canonical CBOR value. A length of 0 or a length above 1 MiB closes the connection. The worker is the client. It connects only after the weights are loaded, sends `ready`, and then reads frames.

Each value is a map with `version` 1 and a `kind`. Unknown fields and unknown kinds close the connection. The kinds are `ready`, `submit`, `result`, `reject`, `ack`, `cancel`, `release`, and `shutdown`. `submit` carries the request id, the prompt token ids, the canonical sampler plan, and the service class. The worker does not see prompt text.

`release` is accepted only when the worker was started paused. Tests use that pause so a cancel or a crash lands before the forward. A normal `knolo-infer serve` process does not pause.

The worker applies every frame that is already buffered before it steps. Cancellation therefore lands before the forward when the cancel frame is already queued. The scheduler rule is unchanged: one cancelled sequence is retired per iteration, its pages return to the pool, and its finish reason is `cancelled`.

## HTTP

The listener is HTTP/1.1, one request per connection, `Content-Length` required on `POST`. Bodies above 64 KiB are refused. JSON is the strict subset used for model manifests: no floats, no duplicate keys. Unknown fields are refused. Nothing in the body is silently ignored.

```text
GET  /knolo/infer/v1/health
POST /knolo/infer/v1/complete
POST /knolo/infer/v1/cancel
```

`health` answers while the listener is up. `worker` is `ready`, `starting`, or `down`. `status` is `ok` when the worker is ready and `degraded` otherwise. A down worker does not by itself fail the health request.

`complete` accepts `model`, `messages`, optional `generation`, optional `serviceClass`, optional `requestId`, and optional `stream`. `stream` is a boolean. `false` and omission return the JSON object below. `true` is the event stream in KIP-INFER-0008. This flag is not the Philox `generation.stream` index. `model` must be the alias the supervisor was started with. `generation` uses the same fixed-point names as the sampler plan. Omitted sampler fields stay at the model-image defaults. Temperature 0 rejects a seed. A non-zero temperature requires a seed. The service class is `interactive`, `standard`, `batch`, or `background`. The default is `standard`.

The response carries the request id, the decoded output text, the output token ids, and the finish reason. Finish reasons are `stop`, `length`, `cancelled`, and `error`. The body also carries the alias and the model runtime root. It does not carry a receipt, and it does not say the output is replay-verified.

`cancel` takes the request id of an in-flight `complete`. The waiting `complete` then finishes with `cancelled`.

Admission failures use the scheduler codes. A prompt that does not fit is `CONTEXT_LIMIT_EXCEEDED`. A request that needs a page the active sequences already hold is `INSUFFICIENT_MEMORY`. More than 64 in-flight HTTP completions is also `INSUFFICIENT_MEMORY`. The supervisor does not shorten the prompt or the budget, and it does not evict an active sequence.

## Worker scheduler

The worker builds one scheduler for the pinned runtime root. The pool is eight pages of 16 tokens. The prefill chunk is 4. The micro-model forward stays single-sequence. A shared run and a run of the same request alone produce the same token ids and the same finish reason.

The worker exits when the supervisor closes the socket. That includes the supervisor process exiting, because its socket descriptors close with it. A parent-death signal is not used: on Linux it fires when the spawning thread exits, and a completion handler is that thread. The worker does not restart itself, bind a TCP port, or open a weight path it was not given.

## Out of this slice

Prefix cache stays off. Grammar masks, speculative decoding, and CUDA graphs are not selected. Authentication and the serve-path receipt journal are later. `knolo-infer run` still records `schedulingMode = isolated` and prefills the whole prompt in one call. The event stream and the OpenAI chat subset are KIP-INFER-0008.
