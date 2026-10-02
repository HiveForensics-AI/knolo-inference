# KIP-INFER-0146 — Llama adapter

Status: `knolo.llama.v1` is compiled into the engine. A verified image with family `llama`, format `safetensors`, and one precision of `f32`, `f16`, or `bf16` loads, places, and runs. Storage is decoded to `f32` before the forward. Compute stays `f32`. This slice does not add a versioned contract.

## Shapes

The vocabulary is 32. Hidden size is 8. There are two layers, two heads, one KV head, head dimension 4, and intermediate size 16. Context is 16 tokens. The KV cache is one `f32` block of 16. RMS epsilon is `1e-5`. RoPE theta is `10000`. The page pool stays eight pages of 16.

Tensor names match `knolo.micro.v1`. The embedding and `lm_head` have 32 rows. A directory loads when every tensor name, shape, and dtype is on that list.

`load_verified_model` verifies the image, then dispatches. `knolo.micro.v1` stays on `load_verified_micro`. `knolo.llama.v1` decodes after that check. Any other adapter, including `knolo.missing.v1`, is `UNSUPPORTED_ARCHITECTURE` before a weight file is opened. `load_verified_micro` refuses a llama image with `knolo.micro.v1 loader does not open knolo.llama.v1` before the weight file is opened.

## Fixture

`models/llama-tiny/` is generated. The name is `knolo/llama-tiny`. The `f32` image is `llama.kmodel` with `weights.safetensors`. `weights-f16.safetensors` and `weights-bf16.safetensors` are sidecars, and the `f32` manifest does not name them. Values sit on a grid that is exact in `f16` and `bf16`. The seed is `0x6C6C616D61320002`.

`conformance/llama-tiny/expected.json` records the `f32` oracle. Greedy ids are exact. The logit absolute tolerance is `1e-4`. The fixture margin is at least `1e-2`. `f16` and `bf16` decode to the same `f32` values and the same greedy ids.

The tokenizer kind is `knolo.llama.tokens.v1`. The first 16 tokens match `knolo.micro.tokens.v1`, so the one-loop template still encodes `user: hi`. Token ids of the micro fixture stay the same. Placement bytes of the micro fixture stay the same. A llama placement records the image precision. `knolo-infer run` and `knolo-infer serve` accept the adapter. Serve receipts use that placement and that storage precision. Candle CPU matches the oracle within `1e-4`.

Cold `measure_*` reports still refuse `knolo.llama.v1`. `throughput` stays `BACKEND_NOT_ALLOWED`.

## Out of this slice

A `format: gguf` manifest stays `MODEL_IMAGE_INVALID` until KIP-INFER-0147. The §44 `daily` pin and the README update follow that authoring.
