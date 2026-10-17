# Release notes

These notes are for the first `v0.1.0` tag. Creating or pushing that tag is a separate step. The release workflow runs only on a tag that matches `v[0-9]+.[0-9]+.[0-9]+`.

## Binaries

The workflow builds the CPU binaries with `cargo build --release --locked` and uploads:

- `knolo-infer-linux-x86_64.tar.gz`, containing `knolo-infer` and `knolo-infer-worker`
- `SHA256SUMS`, with the CPU archive line and the lines `cpu/knolo-infer` and `cpu/knolo-infer-worker`

A maintainer machine then builds both programs with `cargo build --release --locked --features cuda -p infer-cli -p infer-serve` and attaches `knolo-infer-linux-x86_64-cuda.tar.gz`. The same `SHA256SUMS` gains that archive line and the lines `cuda/knolo-infer` and `cuda/knolo-infer-worker`. The CUDA binary loads CUDA 12.2 libraries at run time: cuBLAS, the CUDA runtime, and nvRTC. A driver without those libraries cannot run it. The workflow does not compile CUDA. `ubuntu-latest` has no CUDA toolkit.

The installer, `scripts/install.sh`, checks `SHA256SUMS` and installs both binaries into `~/.local/bin`. It refuses to run as root. When `nvidia-smi` is on `PATH` and the CUDA archive is in the release, it installs that archive and checks the `cuda/` lines. If the file is missing, it prints that and installs the CPU archive, checking the `cpu/` lines. When `amd-smi`, `rocm-smi`, or `npu-smi` is on `PATH` and `nvidia-smi` is not, the installer still uses the CPU archive and prints that the ROCm or Ascend archive is not published. A detected AMD or Huawei device is not treated as NVIDIA. Those kernels are not compiled in.

Checksums are produced when the tag is pushed and when the CUDA archive is attached. This note does not record them in advance.

From a checkout, the same two programs install with the commands below. Pass `--features cuda` to both when the build should place a model on `slot-0`.

```bash
cargo install --path crates/infer-cli --locked
cargo install --path crates/infer-serve --locked
```

Workspace crates stay `publish = false`.

## Runnable models

`knolo-infer library` lists the checked-in catalog. These rows can be pulled and run:

- `llama-3.2-1b-instruct`, Llama 3.2 1B Instruct, `Q4_K_M`
- `llama-3.2-1b-uncensored`, a public Llama 3.2 1B model tagged `uncensored`, `Q4_K_M`, still under the Llama 3.2 license
- `llama-3.2-3b-instruct`, Llama 3.2 3B Instruct, `Q4_K_M`
- `llama-3.2-3b-uncensored`, a public Llama 3.2 3B model tagged `uncensored`, `Q4_K_M`, still under the Llama 3.2 license

`pull` prints the license and waits unless `--yes` is set. Weights stay on Hugging Face.

`library --tag new` lists five models from July through October 2026: `nemotron-3.5-bolt`, `granite-4.2-3b`, `qwen3.8-27b`, `gemma-4-12b`, and `muse-glimmer-30b`. Each row is `not_supported`. `pull` refuses on memory or architecture before any download. The table prints the blocker under the name.

## Known limits

- A CUDA build places a real Llama image on `slot-0`. Quantized weights stay packed, and the KV cache and attention for the reserved context stay on the device. The default build stays on CPU. The llama-tiny fixture still places on `slot-0` only when the binary is built with `--features cuda`.
- The context reservation is at most 8192 tokens, rounded down to a multiple of 16. A reservation whose device bytes exceed probed available memory is `INSUFFICIENT_MEMORY` before the forward. A prompt that does not fit is `CONTEXT_LIMIT_EXCEEDED`.
- The chat template has system, user, and assistant turns. It has no tools and no images.
- A quantized token embedding stays packed. The lookup dequantizes the selected row. A dense embedding stays dense. Other quantized matrices stay packed and are multiplied one block at a time.
- `knolo-infer serve` stops generation on the GGUF `tokenizer.ggml.eos_token_id`. An end-of-sequence id on the model image wins when it is set.
- The OpenAI route ignores unused keys, including `stop`, `tools`, and penalty fields. The native route still refuses unknown fields.
- `run --prompt` is one user message. The terminal loop keeps the model loaded and sends the earlier turns. A line that does not fit is not kept.
- No phone build is shipped.
- `pull` of a lockfile alias copies a local pin and does not download. `pull` of a catalog id uses the host allowlist in `docs/LIBRARY.md`.
