# KIP-INFER-0024 — CPU quantized GEMM

Status: `quant_gemm` multiplies an allowlisted GGUF payload by `f32` activations on the CPU. The payload bytes and the tensor type stay as they were. No weight file is written. When `dequant_gguf` accepts the same payload, each output vector matches the micro-model `gemv` on that expanded matrix, bit for bit. A product whose `f32` matrix would exceed 32 MiB still runs when the output vector stays inside that cap. There is still no CUDA quantized kernel. The perplexity report is KIP-INFER-0026. The conversion receipt is KIP-INFER-0025. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Layout

`cols` is GGUF `ne[0]`, the blocked dimension. `rows` is the product of the later dimensions. Row `r` is packed contiguously. Its first element is logical column 0. `F32` and `F16` have a block of one element. `Q8_0` has a block of 32. `Q4_K`, `Q5_K`, and `Q6_K` have a block of 256. A `cols` that is not a multiple of that block is `MODEL_IMAGE_INVALID`, and the message says the dimension is not a multiple of the block.

`n` is the number of activation vectors. `x` stores vector `k` at `x[k * cols .. (k + 1) * cols]`. The result stores vector `k` at `y[k * rows .. (k + 1) * rows]`. `n = 1` is the matrix-vector product.

The packed length is `(cols / block_elements) * type_size * rows`. A product that overflows, including an output count of `rows * n` or a `cols * n` that does not fit in `usize`, is `MODEL_IMAGE_INVALID`, and the message says the byte length overflows. Any other length is `MODEL_IMAGE_INVALID`, and the message says the payload length does not match its type. `rows`, `cols`, or `n` of 0, or an `x` whose length is not `cols * n`, is `CONTRACT_INVALID`, and the message says the quantized matrix shape does not match its input.

Checks run in this order: a zero dimension, the block multiple, packed overflow, payload length, output size, then `x`'s length. The first failure is the one returned.

## Product

Each block is expanded with the dequant rules in KIP-INFER-0017, KIP-INFER-0018, KIP-INFER-0019, and KIP-INFER-0020. The expanded block is a temporary of at most 256 `f32` values. It is not appended to a matrix, and it is not written back over the payload.

For output vector `k` and row `r`, the accumulator starts at zero and then, for `c` in `0..cols`:

```text
acc += w[r, c] * x[k, c]
```

`w[r, c]` is the `f32` value dequant would store at index `r * cols + c`. The multiply-add order is that column order. A non-finite expanded weight is `MODEL_IMAGE_INVALID`, and the message says dequant produced a non-finite value. A non-finite accumulator is `CONTRACT_INVALID`, and the message says the quantized matrix product is non-finite. Either failure returns no buffer.

`quant_gemm_output_bytes(rows, n)` is `rows * n * 4`. `quant_gemm` allocates that many `f32` values and one block temporary. An output above 32 MiB is `INSUFFICIENT_MEMORY`, the message says the quantized product exceeds 32 MiB, and the activation is not read. A payload whose expanded matrix would exceed 32 MiB is still multiplied when this output cap holds. `dequant_gguf` on that same payload remains `INSUFFICIENT_MEMORY`.

The byte match with `gemv` holds for each activation when `dequant_gguf` succeeds. The two paths then form the same products in the same order. The micro-model weights stay `f32` safetensors. This function is not selected by `knolo-infer run` or `knolo-infer serve`. The kernel plan roots stay the strings those paths already record.

## Out of this slice

A CUDA quantized kernel stays a later Phase 4 slice. The perplexity report is KIP-INFER-0026. The conversion receipt is KIP-INFER-0025. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter.
