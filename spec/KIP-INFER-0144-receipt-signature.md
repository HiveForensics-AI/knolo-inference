# KIP-INFER-0144 — Receipt signature

Status: `knolo-infer run` and `knolo-infer serve` sign a completed receipt with Ed25519 when the operator passes a signing key. `receipt verify` and `model verify` check a signature when the operator passes a public key. Missing key flags leave the receipt unsigned. `measure_signature_*` stays uncalled. This slice does not add an error code or a versioned contract. The curve is `ed25519-dalek` 2.2.0 with its default features.

## Keys and message

A signing key file is exactly 32 raw seed bytes. A public key file is exactly 32 bytes. Any other length, or a file that cannot be read, is `CONTRACT_INVALID`. There is no key generation command.

The signed message is the label, a zero byte, and the digest text. The receipt label is `knolo.infer.receipt-id`. The image label is `knolo.infer.model-image-root`. `keyId` is `H(infer-host-key, public key)`. The signature is 64 bytes. The block stays outside `content_cbor`, so the receipt id and the model-image root do not include it. Signing the receipt id does not cycle.

## Verify

`run` takes `--sign-key`. `serve` takes `--sign-key` and puts one signature on each completed receipt. `receipt verify --public-key` requires exactly one signature whose key id matches and whose equation holds. A failure is `CONTRACT_INVALID` and the message names the receipt signature. Without `--public-key`, an empty signature list stays valid.

`model verify --public-key` accepts an empty image signature list. A non-empty list must verify, or the code is `MODEL_IMAGE_SIGNATURE_INVALID`.

## Out of this slice

Serve assurance is KIP-INFER-0145. The host-supplied signature reports stay cold. The dense Llama-family adapter is KIP-INFER-0146.
