# KIP-INFER-0025 — Conversion receipt

Status: an explicit conversion expands one allowlisted GGUF tensor payload into a new little-endian `f32` artifact and a `knolo.infer.conversion-receipt`. The source bytes are not opened for writing. `dequant_gguf` and `quant_gemm` do not call it, and they still write no weight file. `knolo-infer run` and `knolo-infer serve` do not call it. There is still no CUDA quantized kernel. The perplexity report is KIP-INFER-0026. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Receipt

The receipt is a versioned contract. Its identity root is `H(infer-conversion, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `sourceArtifactRoot` | `H(infer-model-artifact, { files })` of the one source file |
| `converterBuildRoot` | root of the `EngineBuildDescriptorV1` that converted |
| `conversionConfigRoot` | `H(infer-conversion-config, config)` |
| `destinationArtifactRoot` | `H(infer-model-artifact, { files })` of the one destination file |
| `validationResult` | `matched` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.conversion-receipt` and `version` is `1`. Any other validation result is `CONTRACT_INVALID`, and the message says the field has an unsupported value. A receipt is not issued for a conversion that fails.

Each artifact file is one `ArtifactFileV1`: the caller’s relative path, the byte length, and the raw SHA-256 of those bytes. The source file describes the unchanged payload. The destination file describes the new `f32` body. The two paths differ, so the artifact roots differ even when an `F32` payload is copied through unchanged.

The configuration map is not a contract kind. Its keys are `destinationDtype` (`f32`), `ne` (the GGUF dimensions), `operation` (`dequant`), and `sourceType` (`F32`, `F16`, `Q8_0`, `Q4_K`, `Q5_K`, or `Q6_K`).

## Conversion

`convert_gguf_tensor` takes the tensor type, `ne`, the payload, the source path, the destination path, and the converter build root. It returns the destination bytes and the receipt. It does not create a file.

Checks run in this order. The first failure is the one returned, and it returns no receipt.

1. The source path is not relative POSIX: `CONTRACT_INVALID`, and the message says the artifact path is not relative POSIX.
2. The destination path fails that same check.
3. The two paths are equal: `CONTRACT_INVALID`, and the message says the conversion destination matches the source path.
4. `ne` is empty, longer than 4, contains 0, has a blocked dimension that is not a multiple of the block, or the packed length overflows. These are the `MODEL_IMAGE_INVALID` results of `GgufTensorType::nbytes`.
5. The payload length is not that packed length: `MODEL_IMAGE_INVALID`, and the message says the payload length does not match its type.
6. `dequant_gguf` rejects the payload. An output above 32 MiB is `INSUFFICIENT_MEMORY`. A non-finite value is `MODEL_IMAGE_INVALID`.

The destination body is the little-endian `f32` bytes of that dequant buffer, in element order. The payload slice is not written back.

`verify_gguf_conversion` recomputes the four roots and expands the payload again. Checks run in this order:

1. `validationResult` is not `matched`.
2. The receipt's converter build root differs from the root recorded on the conversion: `CONTRACT_INVALID`, and the message says the converter build root does not match.
3. The configuration root differs: the message says the conversion configuration root does not match.
4. The source artifact root differs: the message says the source artifact root does not match.
5. The destination artifact root differs: the message says the destination artifact root does not match.
6. `dequant_gguf` fails, and that failure is returned.
7. The destination bytes differ from the dequant buffer: `CONTRACT_INVALID`, and the message says the conversion validation did not match.

`convert_gguf_tensor` runs that verify before it returns. A verify failure returns no receipt.

## Files

`write_gguf_conversion` verifies the value again, then writes the destination and the receipt into a directory the caller already created. The receipt path is a third relative POSIX path. It matches neither artifact path. A match is `CONTRACT_INVALID`, and the message says the conversion receipt path matches an artifact path.

The directory is canonicalized. A missing directory, or a missing parent of an output path, is `CONTRACT_INVALID`, and the message says the conversion directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the conversion path leaves the directory. An output path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the conversion output already exists. The existing bytes are not opened for writing.

The destination is created with `create_new`, written, and `fsync`ed. The receipt is the canonical CBOR of the receipt document, written the same way. If the receipt write fails, the destination file is removed. The source path is not opened.

## Out of this slice

The perplexity report is KIP-INFER-0026. A CUDA quantized kernel stays off. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter.
