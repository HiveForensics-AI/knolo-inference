# Release notes

These notes are for the first `v0.1.0` tag. Creating or pushing that tag is a separate step. The release workflow runs only on a tag that matches `v[0-9]+.[0-9]+.[0-9]+`.

## Binaries

The workflow builds the CPU binaries with `cargo build --release --locked` and uploads:

- `knolo-infer-linux-x86_64.tar.gz`, containing `knolo-infer` and `knolo-infer-worker`
- `SHA256SUMS`, with one line for the archive and one line for each binary

The installer, `scripts/install.sh`, downloads that CPU archive, checks the archive line in `SHA256SUMS`, and installs both binaries into `~/.local/bin`. It refuses to run as root. It does not select a CUDA build. A CUDA binary is not part of this tag.

Checksums are produced by the workflow when the tag is pushed. This note does not record them in advance.

From a checkout, the same two programs install with:

```bash
cargo install --path crates/infer-cli --locked
cargo install --path crates/infer-serve --locked
```

Workspace crates stay `publish = false`.

## Runnable models

`knolo-infer library` lists the checked-in catalog. These three rows can be pulled and run on CPU:

- `llama-3.2-1b-instruct`, Llama 3.2 1B Instruct, `Q4_K_M`
- `llama-3.2-1b-uncensored`, a public Llama 3.2 1B model tagged `uncensored`, `Q4_K_M`, still under the Llama 3.2 license
- `llama-3.2-3b-instruct`, Llama 3.2 3B Instruct, `Q4_K_M`

`pull` prints the license and waits unless `--yes` is set. Weights stay on Hugging Face.

`qwen3-4b-instruct` and `qwen3-8b-abliterated` are listed so the catalog is not only the models this build runs. `pull` refuses them before any download.

## Known limits

- A real Llama image runs on CPU. The CUDA build returns `UNSUPPORTED_KERNEL` for any `knolo.llama.v1` image above the llama-tiny fixture. The fixture itself still places on `slot-0` when the binary is built with `--features cuda`.
- The context reservation is at most 2048 tokens, rounded down to a multiple of 16. A prompt that does not fit is `CONTEXT_LIMIT_EXCEEDED`.
- The chat template has system, user, and assistant turns. It has no tools and no images.
- The token embedding is expanded to f32. Other quantized matrices stay packed and are multiplied one block at a time. A 1B prompt on CPU is slow. On this machine, `run llama-3.2-1b-instruct --prompt "Hello" --max-tokens 1` finished one greedy token on CPU in about two minutes, and `receipt verify` passed with `same_build_replayable`.
- `pull` of a lockfile alias copies a local pin and does not download. `pull` of a catalog id uses the host allowlist in `docs/LIBRARY.md`.
