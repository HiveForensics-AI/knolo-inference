# KIP-INFER-0021 — Candle CUDA single request

Status: the `cuda` feature of `infer-native` runs `knolo.micro.v1` on Candle device ordinal 0. Weights are f32 on that device. The KV block stays the host `SingleBlockKv`. Greedy token ids match the f32 oracle. Prefill logits and the KV slab stay within absolute tolerance `1e-4`. `knolo-infer run` selects that device only when its own `cuda` feature is on, as specified in KIP-INFER-0022. `knolo-infer serve` selects it only when that same feature is on, as specified in KIP-INFER-0023. There is no quantized GEMM and no CUDA graph.

## Placement

`cuda_placement` is the CPU byte plan with device `slot-0` on the plan and on the single tensor group. Compute, storage, and KV precision stay `f32`. The block size stays 16. Graph capture stays `off`. The expected byte totals match the CPU plan.

`accept_cuda_placement` checks that plan. A CPU plan is `PLACEMENT_UNSATISFIABLE`. `accept_cpu_placement` still rejects `slot-0`.

The backend id is `candle-cuda` and its device is `slot-0`. Ordinal 0 is that slot. A device that does not open is `PLACEMENT_UNSATISFIABLE`. Without the `cuda` feature, that backend is `UNSUPPORTED_KERNEL`.

## Forward

The steps are the Candle CPU graph: an embedding row, RMSNorm, GEMV, host RoPE, a host KV write, attention, SwiGLU, and the final GEMV. GEMV, RMSNorm, SiLU, softmax, and the attention value product run on the CUDA device. RoPE stays the host f32 loop. Each GEMV result is copied back before RoPE and before the KV write. A non-finite logit is `CONTRACT_INVALID`.

This slice does not dequant on the device and does not read a GGUF payload.

## Parity

`cargo test -p infer-native --features cuda` loads the synthetic micro model and compares Candle CUDA with the live oracle on the fixture prompts. Greedy ids match. Prefill logits and the f32 KV slab agree within `1e-4`. The oracle's top-1 margin stays at least `1e-2`.

`cargo test --workspace` does not enable the feature and does not require `nvcc`.

## Probe

When `nvcc --version` prints a release token such as `V12.2.140`, a visible GPU records `runtimeVersion` `12.2.140`. When `nvcc` is absent, that field stays `none`. The probe does not select the device.

## Out of this slice

`knolo-infer run` takes this placement only with the `cuda` feature, specified in KIP-INFER-0022. The serve worker takes it only with that feature, specified in KIP-INFER-0023. CPU quantized GEMM is the separate kernel in KIP-INFER-0024. This path multiplies f32 weights. The conversion receipt is KIP-INFER-0025. The perplexity report is KIP-INFER-0026. CUDA graphs stay off.
