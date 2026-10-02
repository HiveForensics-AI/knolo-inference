# KIP-INFER-0142 — Local pull

Status: `knolo-infer pull <alias>` copies the pinned `.kmodel` and its weight files from paths already on disk. It hashes those bytes before any promote. It does not open a network connection and it does not resume across hosts. This is a live command. It does not add a versioned contract.

## Command

```text
knolo-infer pull <alias> [--lock <file>] [--weights <dir>] [--json]
```

The lock pin names a relative model-image path. The weight directory is `--weights`, or the parent of the `.kmodel` when the flag is omitted. Each path is resolved inside that directory. A symlink that leaves the directory is rejected. A missing lockfile, a missing alias, or a missing source file is `MODEL_ARTIFACT_MISSING`.

The image is verified before any write. A pin whose `modelImageRoot` or `artifactRoot` does not match the file is `MODEL_DIGEST_MISMATCH`, and the destination is left untouched. Weight files are read fully into memory, checked with the existing inventory, and only then written. The promote uses the same temporary file, `fsync`, and rename as the lockfile writer. The lockfile itself is not rewritten.

`--json` prints `alias`, `artifactRoot`, `modelImagePath`, and `modelImageRoot`. `model verify` of the promoted image reports the same image root and artifact root as the pin.

`unsupported_pull` remains the library helper that returns `MODEL_ARTIFACT_MISSING` for a caller that asks for the old refusal. The CLI does not call it.

## Out of this slice

There is no Hugging Face fetch and no cross-host resume. Evidence binding is KIP-INFER-0143. Receipt signatures are KIP-INFER-0144. Serve assurance is KIP-INFER-0145. The dense Llama-family adapter is KIP-INFER-0146.
