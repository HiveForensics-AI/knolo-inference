# KIP-INFER-0002: model image, safetensors inventory, and infer lockfile

Status: implemented for the model-image milestone  
Applies to: `infer-artifact`, `knolo-infer`

This note freezes the behavior tested by `crates/infer-artifact` and `crates/infer-cli`. Contract encoding stays in KIP-INFER-0001. The authoritative model image is still canonical CBOR `ModelImageV1`. JSON and YAML are authoring input only.

## Authoring

`knolo-infer model build <manifest> --out <file.kmodel>` reads a `.json`, `.yaml`, or `.yml` manifest whose parent directory is the base for every relative path.

JSON is unsigned integers, strings, booleans, arrays, and objects. Floats, null, duplicate keys, trailing commas, and a leading BOM are rejected. YAML accepts the same values through a subset: space indentation, `key: value` maps, `- item` lists, `[]`, `{}`, JSON double quotes, single quotes, `true`, `false`, and unsigned integers. Tabs, anchors, tags, document markers, nulls, and floats are rejected. A key colon is followed by a space or by a nested block. Continuation keys of a list item align two columns past the dash.

`sizeBytes: 0`, or an omitted size, means the compiler measures the file. A non-zero size or any declared `sha256` must match the bytes on disk. The built image stores the measured size and the raw SHA-256. `weights.format` in this milestone is `safetensors`. A GGUF manifest is `MODEL_IMAGE_INVALID`.

The compiler sorts capabilities, precisions, tensor names, and weight paths. Duplicate names and paths are `MODEL_IMAGE_INVALID`. Embedded tokenizer and template files are hashed with `infer-tokenizer` and `infer-template`. Their bytes are stored in the image. Weight bytes are not.

Paths are relative POSIX, at most 512 bytes, with no empty, `.`, or `..` segments. The opened file's canonical path must stay inside the manifest directory. A symlink that resolves outside that directory is rejected.

## Verification

`model inspect` and `model verify` decode the `.kmodel`, require canonical bytes, and recompute `modelImageRoot`, `artifactRoot`, and `runtimeRoot`. An empty file, a truncated file, a file larger than 32 MiB, or any other contract kind is `MODEL_IMAGE_INVALID`. Tokenizer, template, architecture, and digest failures keep `TOKENIZER_INVALID`, `TEMPLATE_INVALID`, `UNSUPPORTED_ARCHITECTURE`, and `DIGEST_INVALID`.

`model verify` without `--weights` does not open weight files. With `--weights <dir>`, each descriptor is resolved inside that directory, the size is compared, the file is hashed, and only then is the safetensors header read. A missing file is `MODEL_ARTIFACT_MISSING`. A size or digest mismatch is `MODEL_DIGEST_MISMATCH`.

`--json` prints `weightsChecked`, the three roots, and the weight descriptors. Human output says whether weight files were checked.

## Safetensors inventory

The header length is a little-endian `u64`. It must be non-zero and at most 16 MiB, and it must fit in the file. The header is strict JSON. `__metadata__` may be an object of at most 64 short strings. Every other entry is a tensor with `dtype`, `shape`, and `data_offsets`.

Allowed dtypes are `F32`, `F16`, `BF16`, `I32`, and `U8`, mapped to `f32`, `f16`, `bf16`, `i32`, and `u8`. Any other dtype is `UNSUPPORTED_QUANTIZATION`. Shapes are non-zero `u32` dimensions, at most eight, and the product times the dtype width must equal `end - start` without overflow.

Offsets are relative to the byte region after the header. Each start is a multiple of 8. Tensors may have 0–7 padding bytes between them so the next start is aligned. Larger gaps, overlaps, and a data region that does not end with the last tensor are `MODEL_IMAGE_INVALID`. Duplicate tensor names are rejected. The set of names, shapes, and dtypes must equal the model-image inventory. A missing tensor is `MODEL_ARTIFACT_MISSING`. An unexpected tensor, or a shape or dtype mismatch, is `MODEL_IMAGE_INVALID`. A tensor dtype must also appear in the image's precisions list.

The tensor bodies are not interpreted. The hash check reads the file; the inventory check reads the header.

Shard index files are not a separate document. Each tensor name occurs in one weight file. The union of those headers is the shard index.

## Lockfile

`knolo-infer pin <alias> <file.kmodel>` writes `knolo.infer.lock.json` in the working directory, or the file given by `--lock`. The document is:

```json
{
  "kind": "knolo.infer.lock",
  "version": 1,
  "models": {
    "daily": {
      "modelImageRoot": "sha256-...",
      "artifactRoot": "sha256-...",
      "modelImagePath": "models/daily.kmodel"
    }
  },
  "engine": {
    "channel": "native",
    "buildRoot": "sha256-..."
  },
  "profiles": {}
}
```

The model image is verified before the pin is written. `--weights` also runs the inventory check. The stored path is relative POSIX from the working directory. Aliases are 1–64 characters: a leading letter or digit, then lowercase letters, digits, `.`, `_`, or `-`. Pinning an existing alias replaces that alias and leaves the others. The channel is `native`. `--build-root` sets `buildRoot`; otherwise a new lockfile hashes the `knolo-infer` executable, and an existing lockfile keeps its engine block.

The writer creates a temporary file in the same directory, `fsync`s it, and renames it over the lockfile. Unknown fields, duplicate keys, floats, and a non-native channel are `CONTRACT_INVALID`. This file is not `knolo.lock.json`. Core's parser is unchanged.

## Pull

`knolo-infer pull` exits non-zero with `MODEL_ARTIFACT_MISSING` and the message that pull is unsupported until download staging exists. It does not open a network connection.

## Fixture

`conformance/model-image/` holds the micro authoring manifests, tokenizer, and template. `cargo test` rewrites `weights.safetensors`, `micro.kmodel`, and `expected.json` from those inputs. The JSON and YAML manifests compile to the same canonical bytes.

## Limits

A GGUF manifest is still rejected in this image milestone. The bounded parser is KIP-INFER-0017. There is no network fetch. Ed25519 signature blocks remain shape checks. `read_verified_tensors` returns tensor bodies only after the file hash matches, and it refuses a file above 32 MiB. Executing `knolo.micro.v1` is specified in KIP-INFER-0003. The `conformance/model-image/` fixture stays the one-tensor authoring fixture.
