# KIP-INFER-0019 — Q5_K CPU dequant

Status: `Q5_K` joins the GGUF tensor allowlist from KIP-INFER-0017. A `Q5_K` payload dequants to finite `f32` on the CPU. The buffer is new. The payload bytes and the tensor type stay as they were. CPU quantized GEMM is specified in KIP-INFER-0024. The conversion receipt is KIP-INFER-0025. There is still no CUDA quantized kernel. The perplexity report is KIP-INFER-0026. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Block

`Q5_K` is ggml type `13`. One super-block holds 256 elements in 176 bytes. The first dimension is a multiple of 256. The byte length is `(ne[0] / 256) * 176 * ne[1] * …`. A product that overflows is `MODEL_IMAGE_INVALID`. A first dimension that is not a multiple of 256 is the same code.

| Offset | Length | Field |
| --- | --- | --- |
| 0 | 2 | `d`, little-endian IEEE binary16 super-block scale |
| 2 | 2 | `dmin`, little-endian IEEE binary16 super-block min scale |
| 4 | 12 | `scales`, eight 6-bit scales and eight 6-bit mins |
| 16 | 32 | `qh`, the high bit of each 5-bit code |
| 48 | 128 | `qs`, the low 4 bits of each 5-bit code |

The 256 outputs are four chunks of 64. Chunk `n` in `0..4` uses `qs[32n .. 32n+32]` and the same `qh[0..32]`. The high-bit masks start at `1` and `2` and shift left by 2 after each chunk. `qh` is not advanced.

For `l` in `0..32`:

| Code | Low 4 bits | High bit | Output in the chunk |
| --- | --- | --- | --- |
| `q_lo` | `qs[l]` bits 0..4 | `qh[l]` masked by the chunk's first bit | `l` |
| `q_hi` | `qs[l]` bits 4..8 | `qh[l]` masked by the chunk's second bit | `l + 32` |

A code is an integer in `0..=31`. The high bit contributes 16 when it is set and 0 when it is clear.

Chunk `n` uses scale groups `2n` and `2n+1`, in that order, for `q_lo` and `q_hi`. Group `j` in `0..8` reads a 6-bit scale and a 6-bit min from the 12 scale bytes:

| Group | Scale | Min |
| --- | --- | --- |
| `j < 4` | `scales[j]` bits 0..6 | `scales[j + 4]` bits 0..6 |
| `j >= 4` | low 4 bits of `scales[j + 4]`, then bits 6..8 of `scales[j - 4]` as bits 4..6 | high 4 bits of `scales[j + 4]`, then bits 6..8 of `scales[j]` as bits 4..6 |

Each 6-bit value is an integer in `0..=63`.

## Dequant

`dequant_gguf` reads the payload and returns a new `f32` buffer. It does not write a file, does not change the payload, and does not change the tensor type.

`d` and `dmin` are IEEE binary16, including subnormals, then the finite check. A non-finite `d` or `dmin` is `MODEL_IMAGE_INVALID`.

For a code whose group has scale `sc` and min `m`:

```text
(d * f32(sc)) * f32(code) - (dmin * f32(m))
```

The products are formed in that order, then subtracted. A non-finite result is `MODEL_IMAGE_INVALID`. A scale of 0 makes the code term zero, and the min term is still subtracted. A min of 0 leaves the code term unchanged. A code is never negative, so a zero code term is non-negative zero before the subtraction.

The output size is `element_count * 4`, the same rule as KIP-INFER-0017. An output above 32 MiB is `INSUFFICIENT_MEMORY` and returns no buffer. A payload whose length is not a whole number of 176-byte blocks is `MODEL_IMAGE_INVALID`.

A `Q5_K` tensor requires `general.quantization_version` to be the `u32` `2`, the same rule KIP-INFER-0017 states for `Q8_0` and `Q6_K`. A missing key is `MODEL_IMAGE_INVALID`. Any other value is `UNSUPPORTED_QUANTIZATION`. `general.file_type` does not change the tensor type. The architecture string does not select an adapter.

## Out of this slice

`Q4_K` dequant is KIP-INFER-0020. Every other ggml type stays `UNSUPPORTED_QUANTIZATION`. CPU quantized GEMM is KIP-INFER-0024. The conversion receipt is KIP-INFER-0025. A CUDA quantized kernel stays later. The perplexity report is KIP-INFER-0026. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`.
