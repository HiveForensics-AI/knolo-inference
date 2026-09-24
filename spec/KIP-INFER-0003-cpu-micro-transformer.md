# KIP-INFER-0003: CPU micro-transformer

Status: implemented for the CPU reference slice  
Applies to: `infer-engine`, `infer-native`

The numerical authority is the plain f32 oracle in `infer-engine`. `infer-native` runs the same architecture on Candle CPU. This slice does not start a server, does not hash the `knolo-infer` binary, and does not enable CUDA.

## Architecture

`knolo.micro.v1` has fixed shapes. The model image does not carry a free hyperparameter map. `family` must be `micro`, `format` must be `safetensors`, and `precisions` must be exactly `f32`. The tensor inventory must equal the list below. Any other adapter id, including a well-formed `knolo.llama.v1`, is `UNSUPPORTED_ARCHITECTURE`.

| Field | Value |
| --- | --- |
| vocab | 16 |
| hidden | 8 |
| layers | 2 |
| heads | 2 |
| kv heads | 1 |
| head dim | 4 |
| intermediate | 16 |
| context | 16 |
| KV block | 16 |
| RMSNorm epsilon | `1e-5` |
| RoPE theta | `10000` |

Each block is pre-norm, residual, grouped-query attention, then SwiGLU. There are no biases. Weights are row-major f32, shape `[out, in]`, so `y[o] = sum_i W[o, i] * x[i]`.

RMSNorm:

```text
scale = 1 / sqrt(sum(x^2) / hidden + epsilon)
y = x * scale * weight
```

RoPE is Llama rotate-half. For pair `i` in `0 .. head_dim/2`:

```text
angle = position * theta^(-2i / head_dim)
out[i] = x1 * cos(angle) - x2 * sin(angle)
out[half + i] = x2 * cos(angle) + x1 * sin(angle)
```

Position 0 is the identity. Keys are rotated before they are stored. Values are not rotated. Head `h` reads KV head `h / (heads / kv_heads)`. Scores are `dot(q, k) / sqrt(head_dim)` over keys `0..=position`, then a stable softmax. The causal mask drops future keys instead of writing a negative infinity.

SwiGLU is `down(silu(gate(x)) * up(x))` with `silu(z) = z * sigmoid(z)`. The oracle uses a sign-stable sigmoid. Candle uses its own `silu`.

The logits are `lm_head(rmsnorm(x))` at the last token. Greedy decoding takes the lowest index on a tie. It does not stop on the end-of-sequence id. The image records that id for the later prompt compiler.

## Tensors

Names sort by raw bytes. Every tensor is `f32`.

```text
embed.weight                         [16, 8]
final_norm.weight                    [8]
layers.{0,1}.attn.k.weight           [4, 8]
layers.{0,1}.attn.o.weight           [8, 8]
layers.{0,1}.attn.q.weight           [8, 8]
layers.{0,1}.attn.v.weight           [4, 8]
layers.{0,1}.attn_norm.weight        [8]
layers.{0,1}.mlp.down.weight         [8, 16]
layers.{0,1}.mlp.gate.weight         [16, 8]
layers.{0,1}.mlp.up.weight           [16, 8]
layers.{0,1}.mlp_norm.weight         [8]
lm_head.weight                       [16, 8]
```

`models/micro-transformer/` is the checked-in synthetic model. `cargo test -p infer-engine` rewrites `manifest.json`, `weights.safetensors`, `micro.kmodel`, `tokenizer.json`, and `template.jinja` from the generator below. The tokenizer and template are embedded in the image. Prompt compilation parses them; see KIP-INFER-0004.

The generator starts at seed `0x6D6963726F310001` and, for each element in name order, updates

```text
state ^= state << 13
state ^= state >> 7
state ^= state << 17
signed = (((state >> 11) & 0xFFFFFF) / 16777216) * 2 - 1
```

A `norm.weight` element is `1 + 0.05 * signed`. `lm_head.weight` is `0.75 * signed`. Every other element is `0.20 * signed`. Values are little-endian f32. A non-finite value is `MODEL_IMAGE_INVALID`.

## Load order

`load_verified_micro` reads the `.kmodel` and checks its canonical bytes before it opens a weight file. An unknown adapter fails before those files are opened. Each weight file is size-checked, hashed, and compared with the image. Only then is the safetensors header parsed and the body copied. A digest mismatch is `MODEL_DIGEST_MISMATCH` even when the header is also corrupt. Files above 32 MiB are not loaded into memory.

The CPU placement must name device `cpu`, f32 compute and storage, f32 KV, block size 16, graph capture `off`, the model's runtime root, the measured weight bytes, and the single-block KV bytes (`layers * block * kv_heads * head_dim * 2 * 4`). Anything else is `PLACEMENT_UNSATISFIABLE`. The reference backend id is `reference-f32`. `candle-cpu` is built only by `infer-native`. Another backend is `UNSUPPORTED_KERNEL`.

## KV and generation

`SingleBlockKv` keeps one block per sequence and at most eight sequences. The block table has one id. The oracle tests still use that store. `knolo-infer run` takes pages from the pool in KIP-INFER-0005. A prefill that does not fit, or a token id outside `0..16`, writes nothing. An empty prompt is `PROMPT_COMPILATION_FAILED`. A sequence that does not fit the reserved context is `CONTEXT_LIMIT_EXCEEDED`; tokens are not dropped. A failed step leaves the committed length unchanged. Prefill of a whole prompt and decode of those same tokens one by one produce the same f32 logits.

`conformance/micro-model/expected.json` stores the oracle's prefill logits as lowercase hex of the f32 bits, the greedy token ids, and the model-image, artifact, and runtime roots. `cargo test -p infer-engine` rewrites it. `cargo test -p infer-native` compares a live Candle CPU run with the live oracle. Prefill logits and the f32 KV slab must agree within absolute tolerance `1e-4` (`logitAbsToleranceMillionths` 100). Greedy token ids must match exactly. The fixture prompts keep a top-1 margin of at least `1e-2`, so that tolerance cannot flip the first greedy id. Support level is `experimental`.

The engine build root is not in this file. `knolo-infer run` hashes the binary, as specified in KIP-INFER-0004.

## Limits

No prompt rendering, no sampler other than argmax, no receipt journal, no HTTP server, no CUDA kernel, no GGUF, and no prefix cache. Paged allocation for `knolo-infer run` is KIP-INFER-0005. Candle 0.8.4 is the CPU tensor backend. Bitwise equality with Candle is not required.
