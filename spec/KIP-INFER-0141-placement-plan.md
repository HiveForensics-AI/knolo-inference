# KIP-INFER-0141 — Placement plan

Status: `knolo-infer plan <alias>` reads a pinned model image, probes the machine, and prints the placement root. It does not allocate the page pool and it does not run a forward. This is a live command. It does not add a versioned contract.

## Command

```text
knolo-infer plan <alias> [--intent interactive] [--lock <file>] [--weights <dir>] [--json]
```

The alias is required. The lockfile defaults to `knolo.infer.lock.json` in the working directory. A missing lockfile or a missing alias is `MODEL_ARTIFACT_MISSING`. The pin is opened with the same weight check as `run`. A micro image uses the existing placement: eight KV pages of 16. KV stays on `cpu` unless the binary was built with `cuda` and the probe can select `slot-0`. There is no CPU fallback when that feature is on and `slot-0` cannot be selected.

`--intent` defaults to `interactive`. `throughput` is `BACKEND_NOT_ALLOWED` and the message says the throughput execution mode is not enabled. Any other intent is a usage error. `plan` does not accept run, evidence, or key flags.

`--json` prints `device`, `expectedTotalBytes`, `intent`, `kvBlockSize`, and `placementRoot`. Text output prints the placement root, the device, and the intent.

## Out of this slice

`plan` does not admit a request and does not write a receipt. `pull` is KIP-INFER-0142. Evidence binding is KIP-INFER-0143. Receipt signatures are KIP-INFER-0144. Serve assurance is KIP-INFER-0145. The dense Llama-family adapter is KIP-INFER-0146. A `format: gguf` manifest stays `MODEL_IMAGE_INVALID` until KIP-INFER-0147.
