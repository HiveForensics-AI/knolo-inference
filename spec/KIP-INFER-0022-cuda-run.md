# KIP-INFER-0022 — CUDA run placement

Status: `knolo-infer run` built with the `cuda` feature places `knolo.micro.v1` on device `slot-0`. The tensor backend is `candle-cuda`. The receipt names that device, and the kernel bundle names the toolkit and the architecture of `slot-0`. Greedy token ids match the f32 oracle. The KV pool stays the host page pool from KIP-INFER-0005. Without the feature, the run stays the CPU path in KIP-INFER-0004. `knolo-infer serve` selects the same device only when its `cuda` feature is on, as specified in KIP-INFER-0023. There is no quantized GEMM and no CUDA graph.

## Selection

The `cuda` feature of `infer-cli` enables the same feature on `infer-receipt` and `infer-native`. `cargo test --workspace` does not enable it and does not require `nvcc`. A build of the feature needs `CUDA_ROOT` and `CUDA_HOME` set to the toolkit prefix, and `LD_LIBRARY_PATH` including that prefix's `lib` directory. `cudarc` reads `CUDA_ROOT` and loads `libcublas` from `LD_LIBRARY_PATH`. `nvcc` must be on `PATH`.

With the feature, `run` and `replay` both select `cuda_placement` and `CandleCudaBackend`. A CPU plan is not used. There is no fallback to CPU when the device does not open. Without the feature, `candle-cuda` is not linked and the placement device stays `cpu`.

`throughput` stays `BACKEND_NOT_ALLOWED`.

## Probe

Before `accepted` is written, the run requires a visible GPU `slot-0` and an `nvcc --version` release token. The architecture is that GPU's compute capability, `major.minor`. The toolkit version is the same release the hardware probe stores as `runtimeVersion`, such as `12.2.140`. A missing GPU, a capability that is not `major.minor`, or a toolkit version of `none` is `PLACEMENT_UNSATISFIABLE`. No journal is created.

Opening Candle device ordinal 0 happens on the forward pass. A failure there is `PLACEMENT_UNSATISFIABLE`, the journal records `failed`, and no receipt is stored.

## Kernel bundle

`buildMode` is `cuda`. `featureSet` on the engine build is `cuda`. `tensorBackend` is `candle-cuda` and its version is the Candle version `0.8.4`.

| Field | Bytes or value |
| --- | --- |
| `cudaToolkitVersion` | the `nvcc` release, not `none` |
| `cudaArchitectures` | one entry, the `slot-0` compute capability |
| `sourceRoot` | digest of the bytes `candle-cuda-kernels` |
| `codeObjectRoot` | digest of the bytes `candle-cuda-` plus the Candle version |
| kernel plan root | digest of the bytes `candle-cuda:rmsnorm,rope,attention,swiglu,gemv` |

`compilerFlags` is empty. `jit` is absent. The domain for each digest is `infer-kernel-bundle`. The CPU bundle's `no-cuda-kernels` source is not reused.

The receipt field `deviceSlot` is `slot-0`. `backend` stays `native`. The engine-build root, kernel-bundle root, and kernel-plan root are the documents above. A visible GPU is still recorded on the hardware probe. Graph capture stays `off`.

## Forward

Weights are f32 on device ordinal 0, as in KIP-INFER-0021. RoPE and the KV writes stay on the host. The run allocates eight host pages of 16 tokens. The micro context still fits in one page. The same prompt is executed twice. Matching token ids set `same_build_replayable`. `replay` uses the same CUDA placement and the same bundle. A CPU binary replaying a CUDA receipt is `REPLAY_ENVIRONMENT_MISMATCH`.

`cargo test -p infer-receipt --features cuda` compares those token ids with the f32 oracle on the same prompt, the same sampler, and a host page pool.

## Out of this slice

`knolo-infer serve` selects this device only with the `cuda` feature, specified in KIP-INFER-0023. A GGUF payload is not read on this path. CPU quantized GEMM is KIP-INFER-0024. The conversion receipt is KIP-INFER-0025. The perplexity report is KIP-INFER-0026. CUDA graphs stay off.
