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
| `contextReservation` | Tokens reserved at run time. A runnable row uses a multiple of 16 from 16 through 2048 |
| `minRamBytes` | Host RAM the row expects. Above the detected RAM, the table says `too big for this machine` and `pull` returns `INSUFFICIENT_MEMORY` |
| `licenseId` | License name printed before download |
| `licenseUrl` | Page printed with that name |
| `tags` | Any of `instruct`, `base`, `uncensored`, `new` |
| `gated` | When true, `pull` needs a non-empty `HF_TOKEN` and fails before any connection when it is missing |
| `status` | `runnable` or `not_supported`. The table prints `not_supported` as `not supported yet` |
| `chatTemplate` | Required on a runnable row. It must be the Llama 3 template in `LLAMA_CHAT_TEMPLATE`: one message loop, system, user, and assistant turns, no tools and no images |
| `tokenizer` | Required on a runnable row: `repo`, `revision`, `filename`, `sizeBytes`, and `sha256` for a Hugging Face `tokenizer.json`. The digest is the raw file. After it matches, `pull` embeds a `huggingface.tokenizers.v1` envelope around that object |

A row that is `not_supported` omits `chatTemplate` and `tokenizer`. The catalog parser rejects JSON null. Qwen rows use adapter `knolo.qwen.v1` and status `not_supported`, so `pull` returns `UNSUPPORTED_ARCHITECTURE` and does not open a connection.

`uncensored` is a tag. It does not change the weights, and Knolo does not abliterate or retrain a model. The Llama 3.2 uncensored row keeps `licenseId` `llama3.2` and the Llama 3.2 license URL. A `mit` string inside that GGUF does not replace the Llama license.

`pull` prints `licenseId` and `licenseUrl` and waits for `y`. `--yes` accepts the license. Without a terminal, `pull` refuses until `--yes` is set. Declining leaves nothing pinned.

Weight and tokenizer URLs are `https://huggingface.co/<repo>/resolve/<revision>/<file>`. Redirects are followed one hop at a time. A host is accepted when it is `huggingface.co`, `cdn-lfs.huggingface.co`, `cdn.hf.co`, or a name ending in `.cdn.hf.co`. A live resolve of the 1B `Q4_K_M` file redirects to `us.aws.cdn.hf.co`. `example.com`, `notcdn.hf.co`, and a suffix such as `us.aws.cdn.hf.co.example.com` are refused. The allowlist is not widened by an environment variable. Bytes are hashed while they download. A size or digest mismatch deletes the staging file and leaves the previous pin. An interrupted file is resumed by size from `~/.knolo/infer/staging/<id>/`.

`library refresh` reads `raw/<revision>/<filename>` for each row that has a digest, at that pinned revision. It expects a Git LFS pointer and compares `oid sha256` and `size` with the catalog. A difference is printed. The command does not write `catalog/library.json`, and `cargo test` does not write it either. Tests serve a fixture on `127.0.0.1` and do not contact Hugging Face.

Files for a catalog pull land in `$KNOLO_INFER_HOME`, default `~/.knolo/infer`, and the pin is `models/<id>/model.kmodel` in `$KNOLO_INFER_HOME/knolo.infer.lock.json`. `pull` of a lockfile alias that is not a catalog id still copies the local pin and does not open a network connection.
