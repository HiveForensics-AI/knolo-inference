# KIP-INFER-0017 — Bounded GGUF parser and CPU dequant

Status: a GGUF file is parsed only after its size and SHA-256 match, and only inside a 32 MiB cap. The tensor allowlist is `F32`, `F16`, `Q8_0`, `Q4_K`, `Q5_K`, and `Q6_K`. `F16`, `Q8_0`, `Q4_K`, `Q5_K`, and `Q6_K` dequant to finite `f32` on the CPU. `Q6_K` is specified in KIP-INFER-0018. `Q5_K` is specified in KIP-INFER-0019. `Q4_K` is specified in KIP-INFER-0020. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`. CPU quantized GEMM is specified in KIP-INFER-0024. The conversion receipt is KIP-INFER-0025. There is still no CUDA quantized kernel. The perplexity report is KIP-INFER-0026.

## File

The file is little-endian GGUF version 3. The first four bytes are `GGUF`. A later version, an earlier version, and any other magic are `MODEL_IMAGE_INVALID`. Big-endian layouts are not detected and do not parse.

The header is the magic, a `u32` version, a `u64` tensor count, and a `u64` metadata count. Metadata pairs follow, with no padding between fields. Each pair is a string key, a `u32` value type, and the value. Tensor infos follow: a string name, a `u32` dimension count, that many `u64` dimensions, a `u32` tensor type, and a `u64` offset. The offset is relative to the tensor data section, not to the start of the file.

`general.alignment` is a `u32`. When it is absent the alignment is 32. When it is present it must be a multiple of 8 in `8..=4096`. The data section starts at the next multiple of that alignment after the tensor infos. Those padding bytes are `0x00`. Each tensor offset is a multiple of the same alignment. Bytes in the data section that no tensor claims, including bytes after the last tensor, are `0x00`.

## Bounds

| Limit | Bound |
| --- | --- |
| File | 32 MiB |
| Metadata pairs | 4096 |
| Metadata key | 1..=65535 bytes |
| String value | at most 1 MiB |
| Array elements | at most 1,048,576 |
| Array nesting | 4 |
| Tensors | 100,000 |
| Tensor name | 1..=64 bytes |
| Dimensions | 1..=4, each at least 1 |

A count is rejected before a buffer of that count is reserved when the remaining bytes cannot hold it. A short read is `MODEL_IMAGE_INVALID` and the message says the file is truncated.

Keys are ASCII, with at least two dot-separated segments. Each segment is non-empty and uses only `a-z`, digits, and `_`. Duplicate keys are `MODEL_IMAGE_INVALID`. A string must be UTF-8 and must not contain a NUL byte. A boolean is the byte `0` or `1`. Nested arrays are allowed up to the depth above. An array has one element type.

`general.architecture` is required. It is a string of 1..=64 bytes from `a-z` and digits. Other metadata is kept and is not executed. `tokenizer.chat_template` and `tokenizer.ggml.*` stay bytes on the parsed value. `general.file_type` does not change a tensor type.

Tensor names are unique. A tensor type outside `F32` (`0`), `F16` (`1`), `Q8_0` (`8`), `Q4_K` (`12`), `Q5_K` (`13`), and `Q6_K` (`14`) is `UNSUPPORTED_QUANTIZATION`. `F32` is four little-endian bytes per element. `F16` is two. `Q8_0` is a 34-byte block: one little-endian binary16 scale, then 32 `i8` values. The first dimension of a `Q8_0` tensor is a multiple of 32. The byte length is `(ne[0] / 32) * 34 * ne[1] * …`. A product that overflows is `MODEL_IMAGE_INVALID`. The `Q6_K` block is specified in KIP-INFER-0018. The `Q5_K` block is specified in KIP-INFER-0019. The `Q4_K` block is specified in KIP-INFER-0020.

Ranges are half-open. An offset that is not aligned, a range past the data section, or two ranges that overlap is `MODEL_IMAGE_INVALID`.

A `Q8_0`, `Q4_K`, `Q5_K`, or `Q6_K` tensor requires `general.quantization_version` to be the `u32` `2`. A missing key is `MODEL_IMAGE_INVALID`. Any other value is `UNSUPPORTED_QUANTIZATION`. An `F32` or `F16` file may omit the key.

## Digest first

`read_verified_gguf` takes a path, a size, and a `sha256-` digest.

1. A missing path is `MODEL_ARTIFACT_MISSING`. The header is not read.
2. A path that is not a regular file, including a symlink, or whose size differs, is `MODEL_DIGEST_MISMATCH`. The body is not read.
3. A matching size above 32 MiB is `MODEL_IMAGE_INVALID`. The body is not read.
4. A digest mismatch is `MODEL_DIGEST_MISMATCH`. The header is not parsed.
5. Header, metadata, and tensor checks run on the bytes just hashed.

`parse_gguf_bytes` is the same parser for a buffer the caller has already bounded and hashed. It refuses a buffer above 32 MiB.

## CPU dequant

Dequant reads a payload and returns a new `f32` buffer. It does not write a file, does not change the payload, and does not change the tensor type. That is not a conversion. No conversion receipt is issued. The explicit conversion is KIP-INFER-0025.

`F32` copies little-endian finite values. `F16` is IEEE binary16, including subnormals, infinities, and NaNs, then the finite check. `Q8_0` expands each block in order as `f32(scale) * f32(i8)`. A non-finite scale or product is `MODEL_IMAGE_INVALID`.

The output size is `element_count * 4`. `dequant_output_bytes` is that size. `dequant_gguf` allocates that many `f32` values and no more. An output above 32 MiB is `INSUFFICIENT_MEMORY` and returns no buffer. A payload whose length is not a whole number of blocks is `MODEL_IMAGE_INVALID`.

Binary16 values used by the conformance test:

| `u16` bits | `f32` bits |
| --- | --- |
| `0x0000` | `0x00000000` |
| `0x8000` | `0x80000000` |
| `0x0001` | `0x33800000` |
| `0x03ff` | `0x387fc000` |
| `0x0400` | `0x38800000` |
| `0x3800` | `0x3f000000` |
| `0x3c00` | `0x3f800000` |
| `0xc000` | `0xc0000000` |
| `0x7bff` | `0x477fe000` |

`0x7c00` and `0x7e00` are infinite and NaN. Dequant rejects them.

## Out of this slice

`Q6_K` dequant is KIP-INFER-0018. `Q5_K` dequant is KIP-INFER-0019. `Q4_K` dequant is KIP-INFER-0020. Every other ggml type stays `UNSUPPORTED_QUANTIZATION`. CPU quantized GEMM is KIP-INFER-0024. The conversion receipt is KIP-INFER-0025. A CUDA quantized kernel stays later. The perplexity report is KIP-INFER-0026. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`, as KIP-INFER-0002 specifies. The architecture string in the GGUF metadata does not select an adapter.
