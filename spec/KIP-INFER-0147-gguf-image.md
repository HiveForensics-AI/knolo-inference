# KIP-INFER-0147 — GGUF image

Status: a manifest with `format: gguf` compiles when the architecture adapter is `knolo.llama.v1` and every tensor is `F32`, `F16`, `Q8_0`, `Q4_K`, `Q5_K`, or `Q6_K`. The existing parser and CPU dequant supply the f32 forward. This slice does not add a versioned contract.

## Authoring

`general.architecture` inside the GGUF file does not select an adapter. The manifest does. A micro manifest with `format: gguf` stays `MODEL_IMAGE_INVALID`. Any other weight format stays `MODEL_IMAGE_INVALID`.

The image declares one precision. The Knolo names are `f32`, `f16`, `q8_0`, `q4_k_m`, `q5_k_m`, and `q6_k`. A ggml type outside those six is `UNSUPPORTED_QUANTIZATION` before the forward. The tensor inventory must name the llama shapes. A leading dimension that is not a multiple of the quant block stays `MODEL_IMAGE_INVALID`.

`F32` and `F16` cover every tensor in `models/llama-tiny`. An `F16` GGUF image dequants to the f32 oracle, and the greedy ids match. Load does not write a converted artifact. An explicit conversion is still the only writer of a new weight file.

## Out of this slice

The §44 `daily` pin and the README update follow this authoring. Quantized GEMM stays a separate CPU helper. A CUDA quantized kernel stays off.
