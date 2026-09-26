# KIP-INFER-0018 — Q6_K CPU dequant

Status: `Q6_K` joins the GGUF tensor allowlist from KIP-INFER-0017. A `Q6_K` payload dequants to finite `f32` on the CPU. The buffer is new. The payload bytes and the tensor type stay as they were. CPU quantized GEMM is specified in KIP-INFER-0024. The conversion receipt is KIP-INFER-0025. There is still no CUDA quantized kernel. The perplexity report is KIP-INFER-0026. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Block

`Q6_K` is ggml type `14`. One super-block holds 256 elements in 210 bytes. The first dimension is a multiple of 256. The byte length is `(ne[0] / 256) * 210 * ne[1] * …`. A product that overflows is `MODEL_IMAGE_INVALID`. A first dimension that is not a multiple of 256 is the same code.

| Offset | Length | Field |
| --- | --- | --- |
| 0 | 128 | `ql`, lower 4 bits of each 6-bit code |
| 128 | 64 | `qh`, upper 2 bits of each 6-bit code |
| 192 | 16 | `scales`, one `i8` per 16 elements |
| 208 | 2 | `d`, little-endian IEEE binary16 super-block scale |

The 256 outputs are two halves of 128. Half `h` uses `ql[64h .. 64h+64]`, `qh[32h .. 32h+32]`, and `scales[8h .. 8h+8]`.

For `l` in `0..32`, with `is = l / 16`, the four 6-bit codes in that row are:

| Code | Low 4 bits | High 2 bits | Output in the half |
| --- | --- | --- | --- |
| `q1` | `ql[l]` bits 0..4 | `qh[l]` bits 0..2 | `l` |
| `q2` | `ql[l + 32]` bits 0..4 | `qh[l]` bits 2..4 | `l + 32` |
| `q3` | `ql[l]` bits 4..8 | `qh[l]` bits 4..6 | `l + 64` |
| `q4` | `ql[l + 32]` bits 4..8 | `qh[l]` bits 6..8 | `l + 96` |

A code is an integer in `0..=63`. The quant is that code minus 32, so the signed range is `-32..=31`.

The scale for those four outputs, indexed inside the half's eight `i8` values, is `scales[is]`, `scales[is + 2]`, `scales[is + 4]`, and `scales[is + 6]`. Across a super-block that is one scale for each run of 16 elements, in this order: `0, 1, 2, 3, 4, 5, 6, 7` for the first half and `8 .. 16` for the second.

## Dequant

`dequant_gguf` reads the payload and returns a new `f32` buffer. It does not write a file, does not change the payload, and does not change the tensor type.

Each element is `f32(d) * f32(scale) * f32(quant)`, multiplied in that order. `d` is IEEE binary16, including subnormals, then the finite check. A non-finite `d` or a non-finite product is `MODEL_IMAGE_INVALID`. An `i8` scale of 0 yields zero for every quant in its group. A negative quant keeps that zero negative.

The output size is `element_count * 4`, the same rule as KIP-INFER-0017. An output above 32 MiB is `INSUFFICIENT_MEMORY` and returns no buffer. A payload whose length is not a whole number of 210-byte blocks is `MODEL_IMAGE_INVALID`.

A `Q6_K` tensor requires `general.quantization_version` to be the `u32` `2`, the same rule KIP-INFER-0017 states for `Q8_0`. A missing key is `MODEL_IMAGE_INVALID`. Any other value is `UNSUPPORTED_QUANTIZATION`. `general.file_type` does not change the tensor type. The architecture string does not select an adapter.

## Out of this slice

`Q5_K` dequant is KIP-INFER-0019. `Q4_K` dequant is KIP-INFER-0020. Every other ggml type stays `UNSUPPORTED_QUANTIZATION`. CPU quantized GEMM is KIP-INFER-0024. The conversion receipt is KIP-INFER-0025. A CUDA quantized kernel stays later. The perplexity report is KIP-INFER-0026. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`.
