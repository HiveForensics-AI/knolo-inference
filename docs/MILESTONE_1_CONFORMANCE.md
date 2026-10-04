# Milestone 1 conformance review

Reviewed on 2026-09-22 on this machine: Rust 1.98.1, 16 cores, 38 GB RAM, NVIDIA GeForce RTX 2060 with Max-Q Design, compute capability 7.5, driver 535.309.01, 6144 MiB. `nvcc` is not installed.

The prior session's compile failures in `infer-receipt` (`?` outside `Result`, missing `build`, `probe.gpus` not mutable), the `f32: From<u32>` errors, and the failed `cli_runs_verifies_and_replays_on_cpu` test were fixed before that session ended. This review re-ran the gates rather than treating that report as current.

## Gates

| Gate | Result |
| --- | --- |
| Rust and TypeScript contract roots match, and malformed vectors fail | Pass. `cargo test --workspace --offline` and `npm test` passed. The `@knolo/core` canonical-CBOR cross-check ran and was not skipped. |
| `.kmodel` round-trip, closed corruption cases, and `pull` refused | Pass. Model-image tests and `cli_builds_verifies_pins_and_refuses_pull` passed. |
| Micro-model logits and greedy tokens, no network | Pass. `conformance/micro-model/expected.json` matches the f32 oracle. Candle CPU greedy ids match. Logit absolute tolerance remains `1e-4`. |
| Same prompt token-id root in Rust and TypeScript, sandbox rejection | Pass. `conformance/prompt/expected.json` is shared. A template that leaves the sandbox is `TEMPLATE_INVALID`. |
| Fresh-process receipt verify, tamper fails, no other model server | Pass. `cli_runs_verifies_and_replays_on_cpu` pins, runs, verifies, and replays. A flipped weight byte is `MODEL_DIGEST_MISMATCH`. A different prompt is `REPLAY_ENVIRONMENT_MISMATCH`. A flipped embedded template byte fails canonical encode. `throughput` is `BACKEND_NOT_ALLOWED`. |

`cargo clippy --workspace --all-targets --offline -- -D warnings` passed in the same review.

## Limits that stay open

- Support level of `knolo.micro.v1` is `experimental`. `ModelConformanceReceiptV1` round-trips in the contract tests. This review does not issue a blessed conformance receipt.
- Signature blocks are shape-checked. The signature is not verified against a key.
- `cargo-fuzz` is not installed, and the only toolchain here is stable. Parser fuzz targets are not in the tree.
- The dense Llama-family adapter is still deferred.
- CUDA single-request cannot be built here until a toolkit provides `nvcc`. Placement stays on `cpu`. A visible 2060 is recorded and not selected.
- Prefix cache and speculative decoding stay rejected in contract version 1.

Milestone 1's five gates pass. The supervisor and the CUDA kernel are not part of this milestone.

## Current status (2026-10-04)

The 2026-09-22 review above is the record of that day. The limits that have changed since KIP-INFER-0141 through KIP-INFER-0147:

- `knolo-infer pull` of a lockfile alias copies a pinned local image and its weight files after the digests match and does not open a network connection. `pull` of a catalog id streams that row from `huggingface.co`, `cdn-lfs.huggingface.co`, or a host equal to `cdn.hf.co` or ending in `.cdn.hf.co`, hashes the bytes while they download, and leaves the previous pin when the digest does not match.
- `knolo-infer run` and `knolo-infer serve` sign a receipt when `--sign-key` is set. `receipt verify --public-key` and `model verify --public-key` check that signature with `ed25519-dalek`. The cold signature reports stay shape-only.
- `knolo.llama.v1` loads the checked-in `models/llama-tiny/` image (vocabulary 32, hidden size 8, two layers, context 16) and a GGUF whose layer count, hidden size, heads, KV heads, head dimension, intermediate size, and vocabulary come from that file. The run reserves at most 2048 tokens. The CPU build runs that image. A CUDA build refuses a non-toy Llama with `UNSUPPORTED_KERNEL` and still places the llama-tiny fixture on `slot-0`.
- A `knolo.llama.v1` manifest with `format: gguf` compiles for `f32`, `f16`, `q8_0`, `q4_k_m`, `q5_k_m`, and `q6_k`. A micro manifest with that format stays `MODEL_IMAGE_INVALID`.
- Prefix cache and speculative decoding stay off. `knolo.micro.v1` stays `experimental`. This note still does not issue a blessed conformance receipt.

The table below is the recheck from earlier on 2026-10-04, before the catalog library and the wider Llama loader. It is the record of that run. Rust 1.98.1, Node 22.14.0, CUDA toolkit 12.2.140 at `~/.local/cuda-12.2`:

| Check | Result |
| --- | --- |
| `cargo test --workspace --offline` | Pass |
| `cargo clippy --workspace --all-targets --offline -- -D warnings` | Pass |
| `npm test`, including the `@knolo/core` CBOR cross-check | Pass |
| `infer-native`, `infer-receipt`, `infer-cli`, and `infer-serve` with `--features cuda` | Pass |
| CPU smoke of `models/llama-tiny`: verify, pin `daily`, pull, plan, run, and one served chat completion | Pass. `plan` and `run` placed on `cpu`. The run assurance was `same_build_replayable`. |

The default `knolo-infer` binary does not enable `cuda`. A debug binary last built with `--features cuda` needs `LD_LIBRARY_PATH` to include the toolkit `lib` directory or `run` cannot load `libcublas`.

Rechecked after the catalog library and the wider Llama loader, still on 2026-10-04:

| Check | Result |
| --- | --- |
| `cargo test --workspace --offline` | Pass |
| `cargo clippy --workspace --all-targets --offline -- -D warnings` | Pass |
| `npm test`, including the `@knolo/core` CBOR cross-check | Pass |
| `infer-native`, `infer-receipt`, `infer-cli`, and `infer-serve` with `--features cuda` | Pass. `PATH` includes the toolkit `bin` and `nvvm/bin`, which is where `nvcc` finds `cicc`. |
| Llama 3.2 1B Instruct `Q4_K_M`, one greedy CPU token | Pass. Operator path under `/tmp/knolo-operator`, not CI. `pull` pinned `sha256-559938b99e5096fae39c1a3b26c4c7fd7eb4ebb59b20cdc67e6f764646522a9a`. `run --prompt "Hello" --max-tokens 1` returned token 3923. `receipt verify` passed with `same_build_replayable`. |
