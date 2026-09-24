# KIP-INFER-0023 — CUDA serve worker

Status: `knolo-infer-worker` built with the `cuda` feature places `knolo.micro.v1` on device `slot-0`. The tensor backend is `candle-cuda`. The serve receipt names that device, and the kernel bundle is the one from KIP-INFER-0022. Greedy token ids match the f32 oracle. The KV pool stays the host page pool from KIP-INFER-0005. Assurance stays `compatibility`. Without the feature, the worker stays the reference oracle in KIP-INFER-0009. There is no quantized GEMM and no CUDA graph.

## Selection

The `cuda` feature of `infer-cli` enables the same feature on `infer-serve`, `infer-receipt`, and `infer-native`. `cargo test --workspace` does not enable it and does not require `nvcc`. A build of the feature needs `CUDA_ROOT` and `CUDA_HOME` set to the toolkit prefix, and `LD_LIBRARY_PATH` including that prefix's `lib` directory. `nvcc` must be on `PATH`. `knolo-infer` and `knolo-infer-worker` are built together with that feature. The default binaries stay on `cpu`.

With the feature, the worker selects `cuda_placement` and `CandleCudaBackend`. The supervisor's receipt uses that same placement. A CPU plan is not used. There is no fallback to CPU when the device does not open. Without the feature, `candle-cuda` is not linked into the worker and the placement device stays `cpu`.

`knolo-infer run` is unchanged. This path still does not rerun the sequence, so it does not claim `same_build_replayable`.

## Probe

`Supervisor::start` requires a visible GPU `slot-0` and an `nvcc --version` release token before it binds and before any request is accepted. The architecture is that GPU's compute capability, `major.minor`. The toolkit version is the same release the hardware probe stores as `runtimeVersion`, such as `12.2.140`. A missing GPU, a capability that is not `major.minor`, or a toolkit version of `none` is `PLACEMENT_UNSATISFIABLE`. No request journal is created.

The worker repeats that check before it builds the model. Opening Candle device ordinal 0 happens in that build, after `accepted` is fsynced and before the first token. A failure there exits the worker before it becomes ready. The supervisor seals that journal `failed` and stores no receipt.

## Kernel bundle

The bundle fields are the KIP-INFER-0022 table. `buildMode` is `cuda`. `featureSet` on the engine build is `cuda`. `tensorBackend` is `candle-cuda` and its version is the Candle version `0.8.4`. The receipt field `deviceSlot` is `slot-0`. `backend` stays `native`. `backendVersion` is `0.8.4`. The binary hash is `knolo-infer-worker`, not the supervisor. Graph capture stays `off`.

## Forward

Weights are f32 on device ordinal 0, as in KIP-INFER-0021. RoPE and the KV writes stay on the host. The worker allocates eight host pages of 16 tokens. The micro context still fits in one page. The scheduler, the prefill chunk of 4, and the HTTP routes stay as they are. `cargo test -p infer-serve --features cuda` compares a served completion with the f32 oracle on the same prompt, the same sampler, and a host page pool.

## Out of this slice

A GGUF payload is not read on this path. CPU quantized GEMM is KIP-INFER-0024. The conversion receipt is KIP-INFER-0025. The perplexity report is KIP-INFER-0026. CUDA graphs stay off. Authentication stays a hook. A served request is still not `exact_replay_verified`.
