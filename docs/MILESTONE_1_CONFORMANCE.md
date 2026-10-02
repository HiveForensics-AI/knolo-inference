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
