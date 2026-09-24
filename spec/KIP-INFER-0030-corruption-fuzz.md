# KIP-INFER-0030 — Corruption fuzzing

Status: `measure_corruption_fuzz` runs one bounded campaign over nine parser targets and returns a `knolo.infer.fuzz-report`. It does not use libFuzzer, does not require nightly, and does not install `cargo-fuzz`. It does not run a model, does not allocate a KV pool, and does not write a weight file. `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`, `perplexity_delta`, `measure_placement_memory`, `measure_micro_throughput`, `measure_micro_latency`, and `measure_cancellation_latency` do not call it. `knolo-infer run` and `knolo-infer serve` do not call it. A mutation that keeps the seed identity issues no report. There is still no CUDA quantized kernel. A GGUF authoring manifest is still `MODEL_IMAGE_INVALID`.

## Campaign

The targets, in this order, are `cbor`, `kmodel`, `gguf`, `safetensors`, `template`, `tokenizer`, `api`, `ipc`, and `receipt`. The campaign has one seed for each target. The seed label equals the target name. A seed is 1 to 4096 bytes. A longer seed is `INSUFFICIENT_MEMORY`, and the message says the fuzz seed exceeds the campaign bound. An empty seed is `CONTRACT_INVALID`, and the message says the fuzz seed is empty.

Each seed is mutated six times. The seed itself is not a case. The mutations, in order, are:

1. Replace the seed with an empty buffer.
2. Drop the last byte.
3. XOR the first byte with `0x01`.
4. XOR the last byte with `0x01`.
5. Append `0xff`.
6. XOR the byte at `len / 2` with `0xff`.

That is 54 cases. A probe identifies a buffer. `Ok` is the semantic identity of one accepted document. `Err` rejects the buffer. A probe must not panic. The seed must be accepted. A rejected seed is `CONTRACT_INVALID`, and the message says the fuzz seed was rejected for that target. A mutation whose identity equals the seed identity is `CONTRACT_INVALID`, and the message says corruption was accepted for that target. No report is issued. Any other mutation is either rejected or distinguished. The two counts sum to 54.

`parse_safetensors_bytes` refuses a non-zero byte in the 0–7 alignment slack. The streaming inventory still does not read the tensor body. The safetensors probe uses the in-memory parse, and its identity is the tensor names, dtypes, shapes, and payload bytes. A tokenizer seed in the campaign test has no trailing newline, because the JSON parser skips trailing whitespace.

The corpus root is `H(infer-fuzz-corpus, [ [target, label, seed bytes], ... ])` in campaign order. `infer-fuzz-corpus` is not a contract kind.

## Report

The report is a versioned contract. Its identity root is `H(infer-fuzz, document)`. Version 1 accepts only these fields:

| Field | Value |
| --- | --- |
| `engineBuildRoot` | root of the `EngineBuildDescriptorV1` that records the campaign |
| `corpusRoot` | the corpus root |
| `targetCount` | `9` |
| `seedCount` | `9` |
| `mutationCount` | `54` |
| `rejectedCount` | mutations the probe rejected |
| `distinguishedCount` | mutations whose identity differed |
| `acceptedCount` | `0` |
| `validationResult` | `fail-closed` |
| `extensions` | a map, empty in this slice |

`kind` is `knolo.infer.fuzz-report` and `version` is `1`. The contract count is twenty-three. `infer-fuzz` is the report domain.

Any other validation result is `CONTRACT_INVALID`, and the message says the field has an unsupported value. A target count, seed count, or mutation count other than the values above is `CONTRACT_INVALID`, and the message says the fuzz campaign is not the nine-target slice. An accepted count other than zero is `CONTRACT_INVALID`, and the message says corruption was accepted. A rejected count plus a distinguished count that is not 54 is `CONTRACT_INVALID`, and the message says the fuzz case count does not match.

## Measurement

`measure_corruption_fuzz` takes the observation and a probe. The observation carries the engine build root and the nine seeds. It returns the report. It does not create a file. The caller supplies the seeds and the probe.

Checks run in this order. The first failure is the one returned, and it returns no report.

1. The seed count is not 9: `CONTRACT_INVALID`, and the message says the fuzz campaign is not the nine-target slice.
2. A target name is not one of the nine: `CONTRACT_INVALID`, and the message says the field has an unsupported value.
3. The nine names are not in campaign order: `CONTRACT_INVALID`, and the message says the fuzz targets are not the nine-target campaign.
4. A label differs from its target: `CONTRACT_INVALID`, and the message says the fuzz label does not match the target.
5. A seed is empty: `CONTRACT_INVALID`, and the message says the fuzz seed is empty.
6. A seed is longer than 4096 bytes: `INSUFFICIENT_MEMORY`, and the message says the fuzz seed exceeds the campaign bound.
7. The probe rejects a seed: `CONTRACT_INVALID`, and the message says the fuzz seed was rejected for that target.
8. The probe accepts a mutation as the seed identity: `CONTRACT_INVALID`, and the message says corruption was accepted for that target.

`verify_corruption_fuzz` checks the stored report and reruns the probe. Checks run in this order:

1. The report fails `validate`, and that failure is returned.
2. The engine build root differs. The message says that root does not match.
3. The corpus root differs. The message says that root does not match.
4. Rerunning the campaign fails, and that failure is returned.
5. The recomputed report bytes differ: `CONTRACT_INVALID`, and the message says the fuzz validation did not match.

`measure_corruption_fuzz` runs that verify before it returns. A verify failure returns no report.

## Files

`write_fuzz_report` verifies the value again, then writes the canonical CBOR of the report into a directory the caller already created. The receipt path is relative POSIX. The directory is canonicalized. A missing directory, or a missing parent of the receipt path, is `CONTRACT_INVALID`, and the message says the fuzz directory does not exist. A symlink component, or a canonical parent outside that directory, is `CONTRACT_INVALID`, and the message says the fuzz path leaves the directory. A receipt path that already exists, including as a symlink, is `CONTRACT_INVALID`, and the message says the fuzz output already exists. The existing bytes are not opened for writing.

The receipt is created with `create_new`, written, and `fsync`ed. The seeds are not written.

## Out of this slice

`knolo-infer run` and `knolo-infer serve` still do not call `measure_corruption_fuzz`. Cancellation latency is KIP-INFER-0031. Receipt finalization latency stays later. CUDA graphs stay off. The default `knolo-infer` binary stays on `cpu`. A CUDA quantized kernel stays off. Prefix cache stays off. Compiling a `format: gguf` manifest into a `.kmodel` stays `MODEL_IMAGE_INVALID`. The architecture string in the GGUF metadata does not select an adapter. `cargo-fuzz` stays uninstalled.
