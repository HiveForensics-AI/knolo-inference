# Infer security model

This note constrains the contract and model-image milestones. The design's trust boundaries stay in force as the engine is built.

## Untrusted until verified

Downloaded weights, GGUF and safetensors metadata, tokenizer JSON, chat templates, model configs, API requests, and Hub manifests are untrusted. Parsers enforce size, depth, and item limits. They reject indefinite CBOR, floats in rooted objects, duplicate keys, path segments `.` and `..`, and absolute paths.

A model repository cannot supply executable code. `trust_remote_code` is not a contract field. Architecture adapters are compiled into the binary. Unknown adapter ids fail with `UNSUPPORTED_ARCHITECTURE`.

## After verification

A `.kmodel`, artifact root, lockfile pin, prompt plan, or placement plan is trusted only after its canonical bytes re-encode and its digest matches. Missing security-critical fields fail. Defaults are written into the normalized plan. Readers do not fill them in quietly.

Weight identity is the sorted list of path, size, and raw SHA-256. A name, tag, or URL is not identity.

## Receipts and hardware

Receipts store roots, counts, ids, finish reason, and timing. They do not store prompt text or output text. Hardware probes record a device slot of the form `cpu` or `slot-N`. Model strings that look like UUIDs are rejected so a serial number is not a normal field.

Ed25519 signature blocks are retained and excluded from the model-image root and the receipt id. The signature report checks their shape. It does not verify the signature against a key. An empty signature list is an unsigned local document and must be treated as unsigned. A signature report with `keyVerified` true is rejected. The host-key report records a host-supplied match. `keyVerified` is true only when that match is `matched`, the key id is present, and the block is 64 bytes. Key bytes are not a field, and the Ed25519 equation is not evaluated. The receipt-key report records host storage, and a serialized key is rejected. The worker sandbox report records an unprivileged user, no external network, a read-only model CAS, worker-only scratch, deferred seccomp, a socket lifetime, and direct arguments. It does not apply that profile. The API boundary report records localhost or an explicit remote bind, authentication as a hook, and logs and metrics without user content. It does not bind a socket. The cache side-channel report keeps sharing isolated, hides whether another tenant's prefix exists, and does not allocate the prefix index. The equation report records a host-supplied Ed25519 status and does not compute the curve. The safe error report stores the stable code and omits the prompt. The redaction report records one completion line with prompt text, output text, and paths omitted. It does not write that line. The curve report records a host-supplied on-curve, off-curve, or unsigned result and does not multiply the base point. The rollback report records a distinct incoming pin and does not rewrite a lockfile. The disconnect report records a cancelled request and does not close a socket. The receipt-store report records `RECEIPT_PERSIST_FAILED` and does not write a receipt. The base-point report records a host-supplied multiplication and does not add the public-key point. The disk-full report records a full store and does not write the target file. The queued-unload report records admitted work finishing while queued work stays unstarted, and it does not unload a worker. The daemon-restart report records a new owner and does not spawn a process. The point-addition report records a host-supplied addition and does not compare the points. The duplicate-request report records `CONTRACT_INVALID` and does not start the second request. The concurrent-load report records one worker and does not replace a live worker. The prefix-eviction report records zero evicted pages and does not allocate a prefix index. The receipt chain, the evidence-to-output check, the Hub installation check, the NOTICE inventory, the release manifest, the binary inventory, the reproducible-build record, and these reports store roots and counts. They do not open a Knowledge Image, download weights, open a release binary, run Cargo, or write an SBOM file.

## Model images and weight headers

Authoring JSON and the YAML subset are untrusted. Both reject floats, nulls, and duplicate keys. The YAML subset also rejects tabs, anchors, tags, and aliases, so a manifest cannot smuggle a second interpretation through an anchor.

A path in a manifest or a model image is relative POSIX. The compiler and the verifier resolve it and require the canonical file to remain inside the base directory. Symlinks do not widen that directory. Weight bytes are hashed before the safetensors header is trusted. The header is bounded to 16 MiB, shapes are overflow-checked, and tensor names must match the inventory exactly. The CPU reference copies a tensor body only after that same hash matches, and only for an in-memory file of at most 32 MiB. A digest mismatch is returned before a broken header is parsed.

`knolo.infer.lock.json` is replaced by `fsync` and rename in the same directory. It pins roots and a relative path. It is not a license to fetch weights. `pull` fails closed and does not open a network connection. A GGUF file is hashed before its header is parsed. The parser bounds counts, rejects duplicate keys, and does not run tokenizer or chat-template text stored in the file. `Q4_K`, `Q5_K`, and `Q6_K` dequant allocate a new `f32` buffer and do not write that buffer back over the payload. The quantized product allocates its output vector and one decoded block, and it does not write that product over the payload. An explicit conversion writes a new `f32` file and a conversion receipt in a caller-supplied directory. It does not open the source path for writing, and it does not replace an output that already exists. A manifest that asks for GGUF is still rejected rather than compiled.

## Execution

An architecture adapter is selected from the compiled set. `knolo.micro.v1` is the only adapter in this slice. A different id fails with `UNSUPPORTED_ARCHITECTURE` before weight files are opened. The forward pass does not load Python, does not fetch weights, and does not execute the chat template as a general program. The template grammar is one message loop and two field lookups. Includes, filters, environment access, and any other tag fail with `TEMPLATE_INVALID`. Role and content bytes are copied into the output and are not scanned again. A prompt plus its reserved generation tokens that does not fit the context fails with `CONTEXT_LIMIT_EXCEEDED`; the engine does not drop tokens to make it fit. The oracle tests use one block per sequence. `knolo-infer run` draws from a fixed page pool. A page is reserved until the token commits, and a released page is zeroed before it can be reused. The pool is not a prefix cache. The default build keeps that pool on `cpu`. The `cuda` feature places the weights on `slot-0` and still keeps the pages on the host, including when the serve worker is built with that feature.

`accepted` is written and fsynced before the forward pass. The receipt stores roots, counts, ids, and the finish reason. It does not store the prompt or the output text. The journal directory is append-only: each event is a new file. `exact_replay_verified` is produced only by `replay` after a second process repeats the token ids.

## What stays outside the engine

Credentials, network policy, tool implementations, and agent authority are host concerns. The contract types can name them by digest. They cannot carry secret bytes.

Prefix cache, when it exists, will be off unless a plan sets it, and contract version 1 rejects an enabled prefix cache. `measure_prefix_reuse` records zero lookups, hits, misses, and reused tokens for one cold micro run. Version 1 also rejects speculative decoding. Those features need their own receipt fields before they can run.

`knolo-infer serve` accepts HTTP only from `127.0.0.1`. The native body is strict JSON: no floats, no duplicate keys, and unknown fields fail. The OpenAI chat subset allows a decimal only for `temperature` and `top_p`, with at most six fractional digits and no exponent, and it still rejects unknown fields. The supervisor compiles the prompt and does not open weight bytes. It fsyncs an `accepted` journal event before the worker runs, and it stores a receipt of roots for a finished completion. The default receipt names the reference oracle. The `cuda` feature names `candle-cuda` on `slot-0`. `knolo-infer-worker` connects to a mode `0600` Unix socket in a mode `0700` directory, receives token ids, and does not listen. A worker exit does not stop the listener. Event streams carry token text after admission. This slice does not authenticate callers. `X-Knolo-Receipt: absent` means that response has no stored receipt. `pending` means the id arrives on the final event.

`GET /metrics` is Prometheus text. Its labels are the fixed class, outcome, error code, page state, and OOM kind. The exposition does not carry a request id, prompt, output, model alias, or filesystem path. Prefix-cache counters stay at 0 while prefix cache is not allocated, and they are not broken out by tenant.

A completion trace is a separate JSON-lines file. Its fields are the stage, request id, service class, prompt-plan root, receipt root, and counts. It does not store the prompt, the output, a token id, the model alias, or a path. The file is not a receipt and is not returned by the HTTP API. A completion that fails before a request id is opened records only the error code.

Drain refuses a new completion before the body is parsed. The refusal records `SERVICE_DRAINING` and does not record the body. A completion already admitted still finishes and can still be cancelled. Drain does not unload the model and does not stop the listener.

Unload refuses a new completion before the body is parsed. The refusal records `SERVICE_UNLOADED` and does not record the body. A completion already admitted still finishes and can still be cancelled. When no completion remains, the worker zeros its KV pages and exits. The listener stays up. A later completion does not start a worker.

Load starts that same pin again. The body is `{}`. A load while admitted work is still finishing does not cancel that work and does not start a second worker. Load does not clear drain. The new worker has a fresh page pool, and the call does not count a restart.

One serve process owns `{home}/daemon.lock`. The owner file records a pid and a kernel start time. It does not record a prompt, a request id, or a path. A second process does not listen while that owner is alive. A lock left by a dead process is taken, and a journal that stopped after `accepted` is sealed `failed` before the new process accepts a request. The sealed failure does not store a receipt.
