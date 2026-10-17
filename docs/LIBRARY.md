# Model library

`catalog/library.json` is the curated list. `knolo-infer library` reads that file. It does not search Hugging Face, and it does not run `trust_remote_code`. Adding a model is a reviewed edit of the file.

The document kind is `knolo.infer.library` and the version is `1`. Each object in `models` uses these fields:

| Field | Meaning |
| --- | --- |
| `id` | Alias, 1–64 characters, starting with a letter or digit, then lowercase letters, digits, `.`, `_`, or `-` |
| `displayName` | Name printed in the table and in errors |
| `architecture` | Family name, such as `llama` |
| `adapter` | Compiled adapter id. A runnable row is `knolo.llama.v1` |
| `repo` | Hugging Face repo, exactly `owner/name` |
| `revision` | Pinned commit, 1–64 characters from `[A-Za-z0-9._-]`. The CLI does not follow `main` |
| `filename` | Relative POSIX path of the weight file |
| `sizeBytes` | Exact file size |
| `sha256` | `sha256-` plus 64 lowercase hex digits. A row with an empty digest cannot be pulled |
| `quant` | Label. The first runnable rows use `Q4_K_M` |
| `contextReservation` | Tokens reserved at run time. A runnable row uses a multiple of 16 from 16 through 8192. Above probed available memory, `run` and `serve` return `INSUFFICIENT_MEMORY` before the forward |
| `minRamBytes` | Host RAM the row expects. Above `MemTotal`, the table says `too big for this machine` and `pull` returns `INSUFFICIENT_MEMORY` with both sizes |
| `minVramBytes` | Device memory the weight file needs. Zero means no VRAM floor. Above the first accelerator from `nvidia-smi`, `amd-smi` or `rocm-smi`, or `npu-smi`, the same memory refusal wins over an architecture refusal. A Huawei device is named an NPU in that refusal |
| `licenseId` | License name printed before download |
| `licenseUrl` | Page printed with that name |
| `tags` | Any of `instruct`, `base`, `uncensored`, `new` |
| `gated` | When true, `pull` needs a non-empty `HF_TOKEN` and fails before any connection when it is missing |
| `status` | `runnable` or `not_supported`. The table prints `not_supported` as `not supported yet` |
| `blocker` | One line on a `not_supported` row. The table prints it under the name. `pull` includes it in `UNSUPPORTED_ARCHITECTURE` when the file fits. A runnable row omits it |
| `chatTemplate` | Required on a runnable row. It must be the Llama 3 template in `LLAMA_CHAT_TEMPLATE`: one message loop, system, user, and assistant turns, no tools and no images |
| `tokenizer` | Required on a runnable row: `repo`, `revision`, `filename`, `sizeBytes`, and `sha256` for a Hugging Face `tokenizer.json`. The digest is the raw file. After it matches, `pull` embeds a `huggingface.tokenizers.v1` envelope around that object |

A row that is `not_supported` omits `chatTemplate` and `tokenizer`. The catalog parser rejects JSON null. `pull` checks `minVramBytes` against the first accelerator from `nvidia-smi`, `amd-smi` or `rocm-smi`, or `npu-smi`, then `minRamBytes` against `MemTotal`, and only then the architecture. A Huawei refusal says `this NPU reports`. Either refusal happens before the license prompt and before a connection. A `not_supported` row that fits names its `blocker`.

`uncensored` is a tag. It does not change the weights, and Knolo does not abliterate or retrain a model. The Llama 3.2 uncensored row keeps `licenseId` `llama3.2` and the Llama 3.2 license URL. A `mit` string inside that GGUF does not replace the Llama license.

`pull` prints `licenseId` and `licenseUrl` and waits for `y`. `--yes` accepts the license. Without a terminal, `pull` refuses until `--yes` is set. Declining leaves nothing pinned.

Weight and tokenizer URLs are `https://huggingface.co/<repo>/resolve/<revision>/<file>`. Redirects are followed one hop at a time. A host is accepted when it is `huggingface.co`, `cdn-lfs.huggingface.co`, `cdn.hf.co`, or a name ending in `.cdn.hf.co`. A live resolve of the 1B `Q4_K_M` file redirects to `us.aws.cdn.hf.co`. `example.com`, `notcdn.hf.co`, and a suffix such as `us.aws.cdn.hf.co.example.com` are refused. The allowlist is not widened by an environment variable. Bytes are hashed while they download. A size or digest mismatch deletes the staging file and leaves the previous pin. An interrupted file is resumed by size from `~/.knolo/infer/staging/<id>/`.

`library refresh` reads `raw/<revision>/<filename>` for each row that has a digest, at that pinned revision. It expects a Git LFS pointer and compares `oid sha256` and `size` with the catalog. A difference is printed. The command does not write `catalog/library.json`, and `cargo test` does not write it either. Tests serve a fixture on `127.0.0.1` and do not contact Hugging Face.

Files for a catalog pull land in `$KNOLO_INFER_HOME`, default `~/.knolo/infer`, and the pin is `models/<id>/model.kmodel` in `$KNOLO_INFER_HOME/knolo.infer.lock.json`. `pull` of a lockfile alias that is not a catalog id still copies the local pin and does not open a network connection.

## What can be installed

`knolo-infer library` is the shelf. `pull <id>` installs one row. A row is installable only when `status` is `runnable`.

Installable now, each one dense Llama 3.2 text at `Q4_K_M`, using the Llama 3 chat template:

| id | fits a 6 GB GPU |
| --- | --- |
| `llama-3.2-1b-instruct` | yes |
| `llama-3.2-1b-uncensored` | yes |
| `llama-3.2-3b-instruct` | yes |
| `llama-3.2-3b-uncensored` | yes |

`library --tag new` is the July through October 2026 shelf. Every one of those rows is `not_supported`. On a 6144 MiB GPU, Bolt, Qwen3.8 27B, Gemma 4 12B, and Muse Glimmer fail the memory check. Granite 4.2 3B passes that check and `pull` returns `UNSUPPORTED_ARCHITECTURE` with `Architecture is granite, not Llama`.

| id | released | file | why it does not run |
| --- | --- | --- | --- |
| `nemotron-3.5-bolt` | 15 Oct 2026 | `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`, 22018326528 bytes, IQ2 | Mamba-2, LatentMoE, IQ2, and a vision encoder. The file does not fit 6144 MiB |
| `granite-4.2-3b` | 25 Aug 2026 | `granite-4.2-3b-Q4_K_M.gguf`, 2244011552 bytes | Architecture is `granite`. This is the only new file that fits a 6 GB GPU |
| `qwen3.8-27b` | Aug 2026 | `Qwen3.8-27B-Q4_K_M.gguf`, 17442399968 bytes | Qwen adapter does not exist, and the file does not fit 6144 MiB |
| `gemma-4-12b` | Jul 2026 | `gemma-4-12B-it-Q4_K_M.gguf`, 7662533088 bytes | Gemma adapter does not exist, and the file does not fit 6144 MiB |
| `muse-glimmer-30b` | Aug 2026 | `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`, 16756683904 bytes | Architecture is `muse_glimmer`, plus a vision tower, and the file does not fit 6144 MiB |

Digests are the Git LFS `oid` and `size` at the pinned revision. `library refresh` checks them and does not edit the file.

Llama 3 8B, Llama 3.1 8B, and Llama 3.3 70B are not rows. The loader accepts the standard Llama tensor set plus `rope_freqs.weight`. A quantized token embedding stays packed.

## Still ahead

The shelf and the pull-time memory check are in place. The working Llama rows reserve 8192 tokens. A placement whose reserved bytes exceed probed available memory is `INSUFFICIENT_MEMORY` before the forward. 32768 and 128k stay out of scope. The terminal loop keeps the message list and the loaded model, and it prints each visible token as it is sampled. The OpenAI route ignores unused keys. The installer takes the CUDA archive when `nvidia-smi` is on `PATH` and the file is present. The receipt stays. One item remains before a 2026 model can replace Llama 3.2 as something this engine runs:

1. Write the Granite 4.2 3B adapter. One greedy `Hello` has to match a known-good decode before that row becomes `runnable`. Bolt stays listed until Mamba-2, LatentMoE, IQ2, and a vision encoder exist. Do not vendor llama.cpp.

A later model is a reviewed edit of `catalog/library.json`:

1. A dense Llama GGUF whose tensor names are the set the loader already accepts, with the Llama 3 chat template. Any other architecture is `not_supported` until its adapter exists.
2. One `Q4_K_M` file from a public ungated repo. Pin `revision`, `sizeBytes`, and `sha256` from the Git LFS pointer. `library refresh` checks that pin and does not edit the file.
3. The file fits the GPU the row claims. On a 6 GB card that is the 1B or 3B Llama 3.2 shape.
4. One greedy `Hello` on that file matches a known-good decode before the row becomes `runnable`.
5. `pull` prints the base-model license. A finetune does not replace the Llama 3.2 license. Knolo does not abliterate or retrain.
