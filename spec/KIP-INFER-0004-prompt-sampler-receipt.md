# KIP-INFER-0004 — Prompt, sampler, and CPU receipt

Status: milestone 1 behavior locked by `cargo test` and `npm test`.

This slice compiles a prompt, samples tokens, and writes a receipt for `knolo.micro.v1` on CPU. It does not start a server and it does not build a CUDA kernel.

## Tokenizer

Embedded tokenizer bytes are UTF-8 JSON:

```json
{"kind":"knolo.micro.tokens.v1","tokens":["<pad>","<bos>","<eos>","\n"," ","a","e","h","i","n","o","r","s","t","u",":"],"version":1}
```

The array index is the token id. The length must equal the model vocabulary (16). Entries must be unique, non-empty, and at most 64 characters. Encoding is greedy longest match; equal lengths keep the lower id. A byte with no match is `TOKENIZER_INVALID`. The error names the byte offset and does not echo the prompt. There is no unknown-token fallback.

`<bos>`, `<eos>`, `<pad>`, and `<unk>` must sit at the ids declared by the model image when those ids are present. The compiler prepends the BOS id. It does not append EOS.

## Template

The renderer accepts one grammar:

- literal text
- exactly one `{% for message in messages %} ... {% endfor %}`
- inside the loop, `{{ message.role }}` and `{{ message.content }}`

Anything else, including `include`, `extends`, filters, macros, and a second loop, is `TEMPLATE_INVALID`. There is no filesystem loader, no environment lookup, and no recursion. Substituted field values are copied as text and are not scanned again. The rendered result is capped at 1 MiB. The checked-in micro template is:

```text
{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}
```

The newline inside the loop and the newline after `endfor` are both literal. Rust and TypeScript implement this grammar directly. minijinja is not linked; a larger template language would have to keep these rendered bytes stable.

`TruncationV1.strategy` for this slice is `none`. If the token ids plus the reserved generation budget exceed the context, compilation returns `CONTEXT_LIMIT_EXCEEDED` and does not drop tokens. The micro context is 16. The checked-in generation default reserves 4 tokens so a short user message still fits.

`PromptPlanV1.token_id_root` is `H("infer-prompt-tokens", [token ids])`. That root is what the model consumes. `conformance/prompt/expected.json` is rewritten by `cargo test -p infer-prompt` and checked by `npm test`.

## Sampler

Processing order is the contract list: logits, banned-token mask, grammar mask, repetition penalty, presence penalty, frequency penalty, temperature, top-k, top-p, min-p, sample, stop.

Version 1 has no banned-token list and no grammar mask. Those steps are present and empty.

Penalties use the fixed-point fields divided by 1_000_000. Repetition penalty `1000000` is identity. For a token that already appears in the prompt or the tokens sampled so far: a positive logit is divided by the penalty and any other logit is multiplied by it. Presence subtracts its penalty once. Frequency subtracts penalty times the count. A penalty of zero sets that logit to zero.

Temperature 0 does not call an RNG. After penalties and top-k it takes the lowest token id among the maximum logits. Top-p and min-p apply only when temperature is non-zero.

Random mode is `philox-4x32-v1`. The key is the seed as two little-endian `u32` words. The counter is `(step, stream low, stream high, 0)`. Step 0 is the first sampled token. The draw is the first output word divided by 2^32, in `[0, 1)`. The cumulative distribution walks token ids in ascending order. The published Random123 zero-key vector is the test lock:

```text
philox4x32_10(counter = 0, key = 0) =
  6627e8d5 e169c58d bc57ac4c 9b00dbd8
```

Generation stops when the sampled id is an EOS id (`stop`) or the budget is spent (`length`). The output text is the concatenation of the sampled token strings, including `<eos>` when that id was sampled. The receipt stores the text root, not the text.

## Journal and receipt

The home directory defaults to `~/.knolo/infer` and tests pass `--home`.

```text
<home>/journals/<request-id>/000000.accepted.cbor
<home>/journals/<request-id>/NNNNNN.<event>.cbor
<home>/journals/<request-id>/sampler-plan.cbor
<home>/receipts/sha256/<64 hex>/receipt.cbor
```

`accepted` is fsynced before the forward pass. Its payload root is the intent root. After a successful pair of runs, the journal appends `prefill`, one `token` event per output id, and `completed`. Each event's chain root is `H("infer-execution-event", { previousEventRoot, event })`. A failed forward pass appends `failed` and does not publish a receipt.

The receipt assurance is `same_build_replayable` only after a second in-process run on the same build, placement, and sampler reproduces the token ids. `exact_replay_verified` is not set on that receipt. `knolo-infer replay` runs again in a new process and writes a `ReplayCheckReceiptV1`. A match sets `exact_replay_verified`. A different prompt, model, engine build, or placement is `REPLAY_ENVIRONMENT_MISMATCH`. Different token ids are `REPLAY_OUTPUT_MISMATCH`.

The engine build records the `knolo-infer` binary hash, `HEAD` when git can see it, the compiled-in `Cargo.lock` root, the rustc version, the host triple, and `tensor_backend = candle-cpu`. The CPU kernel bundle's toolkit version is `none` and its source root is the digest of the bytes `no-cuda-kernels`. A visible GPU may be recorded on the hardware probe. The placement device is `cpu`. The forward pass takes KV pages from the pool in KIP-INFER-0005. The micro context still fits in one page of 16. The `cuda` feature of this command is KIP-INFER-0022. Without that feature the device stays `cpu`.

`throughput` is `BACKEND_NOT_ALLOWED`. `pinned` and `isolated-replay` both run one isolated sequence with receipt policy `atomic-verified`.

## Commands

```bash
knolo-infer run --model <alias> --prompt <text> --mode pinned --receipt <file.cbor>
knolo-infer receipt verify <file.cbor>
knolo-infer replay <file.cbor> --model <alias> --prompt <text>
```

`receipt verify --weights` re-hashes the weight files. A flipped weight byte is `MODEL_DIGEST_MISMATCH`. A flipped receipt byte fails the canonical check. The process does not shell out to another model server.
