# Mobile

A phone would run the model in the app process. The React Native module would call the C ABI in `crates/infer-ffi`. It would not open a socket to `knolo-infer serve`, and it would not spawn `knolo-infer-worker`.

A phone build is not shipped. This machine has no Android NDK and no iOS toolchain. The symbols below are proven on the Linux CPU oracle with the llama-tiny fixture. No catalog row runs on a phone. No phone GPU, NNAPI, Core ML, or Metal backend is claimed.

`@knolo/infer` stays the Node contract verifier. It imports `node:crypto` and does not load weights. The mobile caller is this ABI, not that package.

## Symbols

| Symbol | Behavior |
| --- | --- |
| `knolo_infer_version` | Pointer to the static version string `0.1.0` |
| `knolo_infer_last_error` | Pointer to the last error. Empty after a successful greedy call. Valid until the next `knolo_infer_tiny_greedy` |
| `knolo_infer_tiny_greedy` | Greedy ids for the fixed llama-tiny prompt `1, 4, 7` |

`knolo_infer_tiny_greedy(max_tokens, out_tokens, out_cap, out_len)` writes `max_tokens` new ids into `out_tokens`.

- `0` means `*out_len` ids were written.
- `1` means a pointer was null, or `out_cap` was too small. A short buffer sets `*out_len` to the number of ids that were produced and leaves `out_tokens` unchanged. A null pointer is not written through.
- `2` means the engine refused the run. `knolo_infer_last_error` has the message. The llama-tiny context is 16 tokens, including the 3 prompt ids, so a large `max_tokens` returns `2`.

The fixture is the compiled llama-tiny generator. The call places it on `cpu` and uses the reference `f32` backend. It does not read `~/.knolo`, `/proc`, or a GPU tool.

## Still ahead

Android arm64 needs the NDK and a static library smoke on an emulator. The React Native module comes after that, Android first. iOS waits for a Mac. A catalog model waits until a real phone matches this oracle and the weight, the `f32` embedding, and the KV cache fit the free RAM.
