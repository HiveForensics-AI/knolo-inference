# KIP-INFER-0001: canonical contracts

Status: implemented for the contract milestone  
Applies to: `infer-contracts`, `@knolo/infer`

## Encoding

Infer rooted objects use the KIP-0003 CBOR subset:

- definite lengths only
- null, booleans, integers, byte strings, UTF-8 text, arrays, and maps
- map keys are text and are sorted by their UTF-8 bytes
- duplicate keys are rejected
- integers use the shortest encoding
- a decoded value must re-encode to the same bytes
- floats, tags, indefinite lengths, and other simple values are rejected

A document larger than 32 MiB, nested deeper than 32 levels, or containing a collection longer than 1,048,576 items is rejected with `CANONICAL_CBOR_INVALID`.

## Digests

```text
H(domain, payload) = SHA256(UTF8("knolo:" + domain + ":v1\0") || payload)
```

Rendering is `sha256-` plus 64 lowercase hexadecimal characters. `payload` is canonical CBOR unless a rule below names raw file bytes.

`digest_domain` accepts only the domains in this specification. File content hashes that are not domain-separated are raw SHA-256 values rendered with the same `sha256-` prefix. Weight descriptors and binary hashes use that raw form.

Map keys in contract documents are camelCase. That matches Knolo Core's CBOR maps and the JSON examples in the Infer design. Rust fields are snake_case and convert at the CBOR boundary.

## Domains

Reserved from the design, plus the roots required by model identity and prompt identity:

`infer-agent-effect`, `infer-api`, `infer-architecture`, `infer-artifact-missing`, `infer-attention`, `infer-backend`, `infer-base`, `infer-binary`, `infer-cache-channel`, `infer-cancellation`, `infer-canonical-cbor`, `infer-capacity`, `infer-chain`, `infer-challenge`, `infer-cofactor`, `infer-composition`, `infer-concurrent-load`, `infer-config`, `infer-conformance`, `infer-context-limit`, `infer-contract-invalid`, `infer-conversion`, `infer-conversion-config`, `infer-curve`, `infer-digest-invalid`, `infer-digest-mismatch`, `infer-disconnect`, `infer-disk`, `infer-domain`, `infer-draining`, `infer-duplicate`, `infer-engine-build`, `infer-equality`, `infer-equation`, `infer-eviction`, `infer-evidence`, `infer-evidence-output`, `infer-execution-event`, `infer-execution-plan`, `infer-execution-trace`, `infer-expert-placement`, `infer-fault`, `infer-finalization`, `infer-fuzz`, `infer-fuzz-corpus`, `infer-glm`, `infer-grammar`, `infer-grammar-refusal`, `infer-graph`, `infer-grouped-kernel`, `infer-hardware`, `infer-host-key`, `infer-hub`, `infer-image-invalid`, `infer-image-signature`, `infer-install`, `infer-kernel`, `infer-kernel-bundle`, `infer-kv`, `infer-latency`, `infer-limits`, `infer-llama`, `infer-load`, `infer-logits`, `infer-memory`, `infer-memory-refusal`, `infer-mistral`, `infer-mixed-placement`, `infer-model-artifact`, `infer-model-image`, `infer-model-runtime`, `infer-moe`, `infer-multi-model`, `infer-notice`, `infer-oom`, `infer-output-text`, `infer-output-tokens`, `infer-overhead`, `infer-peak`, `infer-perplexity`, `infer-placement`, `infer-placement-refusal`, `infer-point`, `infer-prefix`, `infer-prompt-compilation`, `infer-prompt-input`, `infer-prompt-plan`, `infer-prompt-tokens`, `infer-public`, `infer-quantization`, `infer-receipt`, `infer-receipt-key`, `infer-receipt-required`, `infer-receipt-sign`, `infer-receipt-store`, `infer-receipt-verify`, `infer-recipe`, `infer-redaction`, `infer-release`, `infer-rendered-text`, `infer-replay-check`, `infer-replay-environment`, `infer-replay-output`, `infer-reproducible`, `infer-request-intent`, `infer-restart`, `infer-rollback`, `infer-router`, `infer-safe-error`, `infer-sampler`, `infer-sandbox`, `infer-scalar`, `infer-secondary`, `infer-signature`, `infer-signature-check`, `infer-special-tokens`, `infer-speculative`, `infer-stop-strings`, `infer-studio`, `infer-swap`, `infer-template`, `infer-template-invalid`, `infer-throughput`, `infer-timeout`, `infer-tokenizer`, `infer-tokenizer-invalid`, `infer-tool-refusal`, `infer-tools`, `infer-truncation`, `infer-unload`, `infer-verification`, `infer-vllm`, `infer-worker-lost`, `infer-worker-start`, `infer-workstation`.

`infer-model-runtime`, `infer-rendered-text`, `infer-special-tokens`, `infer-truncation`, `infer-stop-strings`, `infer-limits`, and `infer-prompt-plan` are additions. The design's domain list said "at least" the original names, and sections 10.4 and 16 need distinct domains for these roots. `infer-conversion` and `infer-conversion-config` are the Phase 4 conversion receipt and its configuration map from design section 18.3. The configuration map is not a versioned contract kind. `infer-perplexity` is the perplexity report from KIP-INFER-0026. `infer-logits` hashes the raw little-endian `f32` matrix and is not a contract kind. `infer-memory` is the memory estimate from KIP-INFER-0027. `infer-throughput` is the throughput report from KIP-INFER-0028. `infer-latency` is the latency report from KIP-INFER-0029. `infer-fuzz` is the corruption campaign report from KIP-INFER-0030. `infer-fuzz-corpus` hashes the campaign seeds and is not a contract kind. `infer-cancellation` is the cancellation latency report from KIP-INFER-0031. `infer-finalization` is the receipt finalization report from KIP-INFER-0032. `infer-overhead` is the receipt overhead report from KIP-INFER-0033. `infer-swap` is the model swap report from KIP-INFER-0034. `infer-verification` is the model verification report from KIP-INFER-0035. `infer-load` is the model load report from KIP-INFER-0036. `infer-peak` is the peak RAM and VRAM report from KIP-INFER-0037. `infer-kv` is the KV utilization report from KIP-INFER-0038. `infer-prefix` is the prefix reuse report from KIP-INFER-0039. `infer-llama` is the llama.cpp comparison from KIP-INFER-0040. `infer-vllm` is the vLLM comparison from KIP-INFER-0041. `infer-mistral` is the mistral.rs comparison from KIP-INFER-0042. `infer-recipe` is the recipe mark from KIP-INFER-0043. `infer-composition` is the Core/Reflex binding from KIP-INFER-0044. `infer-agent-effect` is the Agents host-effect decision from KIP-INFER-0045. `infer-hub` is the Hub metadata record from KIP-INFER-0046. `infer-studio` is the receipt view from KIP-INFER-0047. `infer-chain` is the receipt chain from KIP-INFER-0048. `infer-evidence-output` is the evidence-to-output check from KIP-INFER-0049. `infer-install` is the Hub installation check from KIP-INFER-0050. `infer-notice` is the NOTICE and SBOM inventory from KIP-INFER-0051. `infer-release` is the signed release manifest from KIP-INFER-0052. `infer-binary` is the binary hash inventory from KIP-INFER-0053. `infer-reproducible` is the reproducible-build record from KIP-INFER-0054. `infer-signature` is the signature shape gate from KIP-INFER-0055. `infer-host-key` is the host-key check from KIP-INFER-0056. `infer-receipt-key` is the receipt-key custody record from KIP-INFER-0057. `infer-sandbox` is the worker sandbox profile from KIP-INFER-0058. `infer-api` is the API boundary from KIP-INFER-0059. `infer-cache-channel` is the cache side-channel report from KIP-INFER-0060. `infer-equation` is the signature equation record from KIP-INFER-0061. `infer-safe-error` is the safe error envelope from KIP-INFER-0062. `infer-redaction` is the redacted log record from KIP-INFER-0063. `infer-curve` is the Ed25519 curve record from KIP-INFER-0064. `infer-rollback` is the rollback record from KIP-INFER-0065. `infer-disconnect` is the client disconnect record from KIP-INFER-0066. `infer-receipt-store` is the receipt-store failure from KIP-INFER-0067. `infer-base` is the base-point record from KIP-INFER-0068. `infer-disk` is the disk-full record from KIP-INFER-0069. `infer-unload` is the queued unload from KIP-INFER-0070. `infer-restart` is the daemon restart from KIP-INFER-0071. `infer-point` is the point-addition record from KIP-INFER-0072. `infer-duplicate` is the duplicate request id from KIP-INFER-0073. `infer-concurrent-load` is the concurrent load from KIP-INFER-0074. `infer-eviction` is the prefix-eviction record from KIP-INFER-0075. `infer-equality` is the point-comparison record from KIP-INFER-0076. `infer-oom` is the CUDA out-of-memory record from KIP-INFER-0077. `infer-fault` is the CUDA fault record from KIP-INFER-0078. `infer-timeout` is the request-timeout record from KIP-INFER-0079. `infer-worker-start` is the worker-start record from KIP-INFER-0080. `infer-challenge` is the challenge-hash record from KIP-INFER-0081. `infer-replay-environment` is the replay environment mismatch from KIP-INFER-0082. `infer-replay-output` is the replay output mismatch from KIP-INFER-0083. `infer-worker-lost` is the worker-lost record from KIP-INFER-0084. `infer-draining` is the draining refusal from KIP-INFER-0085. `infer-scalar` is the scalar-reduction record from KIP-INFER-0086. `infer-digest-mismatch` is the digest-mismatch record from KIP-INFER-0087. `infer-tokenizer-invalid` is the tokenizer refusal from KIP-INFER-0088. `infer-template-invalid` is the template refusal from KIP-INFER-0089. `infer-architecture` is the unsupported-architecture record from KIP-INFER-0090. `infer-public` is the public-key multiplication from KIP-INFER-0091. `infer-quantization` is the unsupported-quantization record from KIP-INFER-0092. `infer-kernel` is the unsupported-kernel record from KIP-INFER-0093. `infer-placement-refusal` is the unsatisfiable-placement record from KIP-INFER-0094. `infer-memory-refusal` is the insufficient-memory record from KIP-INFER-0095. `infer-signature-check` is the signature check from KIP-INFER-0096. `infer-context-limit` is the context-limit record from KIP-INFER-0097. `infer-prompt-compilation` is the prompt-compilation record from KIP-INFER-0098. `infer-image-invalid` is the invalid-image record from KIP-INFER-0099. `infer-image-signature` is the invalid signature block from KIP-INFER-0100. `infer-cofactor` is the cofactor clear from KIP-INFER-0101. `infer-artifact-missing` is the missing-artifact record from KIP-INFER-0102. `infer-receipt-required` is the receipt-required record from KIP-INFER-0103. `infer-backend` is the backend refusal from KIP-INFER-0104. `infer-digest-invalid` is the invalid-digest record from KIP-INFER-0105. `infer-receipt-sign` is the receipt signature from KIP-INFER-0106. `infer-canonical-cbor` is the canonical-CBOR refusal from KIP-INFER-0107. `infer-contract-invalid` is the contract-invalid record from KIP-INFER-0108. `infer-grammar-refusal` is the grammar refusal from KIP-INFER-0109. `infer-tool-refusal` is the tool refusal from KIP-INFER-0110. `infer-receipt-verify` is the receipt verification from KIP-INFER-0111. `infer-speculative` is the speculative-decoding record from KIP-INFER-0112. `infer-graph` is the CUDA graph record from KIP-INFER-0113. `infer-multi-model` is the multi-model record from KIP-INFER-0114. `infer-secondary` is the secondary-service record from KIP-INFER-0115. `infer-domain` is the domain-separation record from KIP-INFER-0116. `infer-attention` is the attention-approximation record from KIP-INFER-0117. `infer-moe` is the mixture-of-experts record from KIP-INFER-0118. `infer-expert-placement` is the expert-placement record from KIP-INFER-0119. `infer-grouped-kernel` is the grouped-kernel record from KIP-INFER-0120. `infer-router` is the router-conformance record from KIP-INFER-0121. `infer-glm` is the GLM-4.7 adapter record from KIP-INFER-0122. `infer-workstation` is the workstation-recipe record from KIP-INFER-0123. `infer-mixed-placement` is the mixed-placement record from KIP-INFER-0124. `infer-capacity` is the expert-capacity record from KIP-INFER-0125.

## Versioned objects

Every object has `kind` and `version`. Version 1 is the only accepted version. Unknown top-level fields are rejected. Optional fields are omitted, never encoded as null. `extensions` is required and may be empty. Extension keys are dotted lowercase namespaces, at most 32 of them, and they do not change execution unless a later version defines them.

| Kind | Domain for its identity root |
|---|---|
| `knolo.infer.model-image` | `infer-model-image`, over the document with `signatures` removed |
| `knolo.infer.model-artifact-set` | `infer-model-artifact`, over `{ files }` only |
| `knolo.infer.engine-build` | `infer-engine-build`, over the document |
| `knolo.infer.kernel-bundle` | `infer-kernel-bundle`, over the document |
| `knolo.infer.hardware-probe` | `infer-hardware`, over the document |
| `knolo.infer.placement-plan` | `infer-placement`, over the document |
| `knolo.infer.prompt-input` | `infer-prompt-input`, over the document. The messages root hashes only the messages array |
| `knolo.infer.prompt-plan` | `infer-prompt-plan`, over the document |
| `knolo.infer.evidence-binding` | `infer-evidence`, over the document |
| `knolo.infer.sampler-plan` | `infer-sampler`, over the document |
| `knolo.infer.grammar-plan` | `infer-grammar`, over the document |
| `knolo.infer.inference-intent` | `infer-request-intent`, over the document |
| `knolo.infer.execution-plan` | `infer-execution-plan`, over the document |
| `knolo.infer.inference-event` | `infer-execution-event`, over `{ previousEventRoot, event }` |
| `knolo.infer.inference-receipt` | `infer-receipt`, over the document with `receiptId` and `signatures` removed |
| `knolo.infer.replay-check` | `infer-replay-check`, over the document |
| `knolo.infer.model-conformance` | `infer-conformance`, over the document |
| `knolo.infer.conversion-receipt` | `infer-conversion`, over the document. Added by KIP-INFER-0025 |
| `knolo.infer.perplexity-report` | `infer-perplexity`, over the document. Added by KIP-INFER-0026 |
| `knolo.infer.memory-estimate` | `infer-memory`, over the document. Added by KIP-INFER-0027 |
| `knolo.infer.throughput-report` | `infer-throughput`, over the document. Added by KIP-INFER-0028 |
| `knolo.infer.latency-report` | `infer-latency`, over the document. Added by KIP-INFER-0029 |
| `knolo.infer.fuzz-report` | `infer-fuzz`, over the document. Added by KIP-INFER-0030 |
| `knolo.infer.cancellation-report` | `infer-cancellation`, over the document. Added by KIP-INFER-0031 |
| `knolo.infer.finalization-report` | `infer-finalization`, over the document. Added by KIP-INFER-0032 |
| `knolo.infer.overhead-report` | `infer-overhead`, over the document. Added by KIP-INFER-0033 |
| `knolo.infer.swap-report` | `infer-swap`, over the document. Added by KIP-INFER-0034 |
| `knolo.infer.verification-report` | `infer-verification`, over the document. Added by KIP-INFER-0035 |
| `knolo.infer.load-report` | `infer-load`, over the document. Added by KIP-INFER-0036 |
| `knolo.infer.peak-report` | `infer-peak`, over the document. Added by KIP-INFER-0037 |
| `knolo.infer.kv-report` | `infer-kv`, over the document. Added by KIP-INFER-0038 |
| `knolo.infer.prefix-report` | `infer-prefix`, over the document. Added by KIP-INFER-0039 |
| `knolo.infer.llama-report` | `infer-llama`, over the document. Added by KIP-INFER-0040 |
| `knolo.infer.vllm-report` | `infer-vllm`, over the document. Added by KIP-INFER-0041 |
| `knolo.infer.mistral-report` | `infer-mistral`, over the document. Added by KIP-INFER-0042 |
| `knolo.infer.recipe-report` | `infer-recipe`, over the document. Added by KIP-INFER-0043 |
| `knolo.infer.composition-report` | `infer-composition`, over the document. Added by KIP-INFER-0044 |
| `knolo.infer.agent-effect-report` | `infer-agent-effect`, over the document. Added by KIP-INFER-0045 |
| `knolo.infer.hub-report` | `infer-hub`, over the document. Added by KIP-INFER-0046 |
| `knolo.infer.studio-report` | `infer-studio`, over the document. Added by KIP-INFER-0047 |
| `knolo.infer.chain-report` | `infer-chain`, over the document. Added by KIP-INFER-0048 |
| `knolo.infer.evidence-output-report` | `infer-evidence-output`, over the document. Added by KIP-INFER-0049 |
| `knolo.infer.install-report` | `infer-install`, over the document. Added by KIP-INFER-0050 |
| `knolo.infer.notice-report` | `infer-notice`, over the document. Added by KIP-INFER-0051 |
| `knolo.infer.release-report` | `infer-release`, over the document. Added by KIP-INFER-0052 |
| `knolo.infer.binary-report` | `infer-binary`, over the document. Added by KIP-INFER-0053 |
| `knolo.infer.reproducible-report` | `infer-reproducible`, over the document. Added by KIP-INFER-0054 |
| `knolo.infer.signature-report` | `infer-signature`, over the document. Added by KIP-INFER-0055 |
| `knolo.infer.host-key-report` | `infer-host-key`, over the document. Added by KIP-INFER-0056 |
| `knolo.infer.receipt-key-report` | `infer-receipt-key`, over the document. Added by KIP-INFER-0057 |
| `knolo.infer.sandbox-report` | `infer-sandbox`, over the document. Added by KIP-INFER-0058 |
| `knolo.infer.api-report` | `infer-api`, over the document. Added by KIP-INFER-0059 |
| `knolo.infer.cache-channel-report` | `infer-cache-channel`, over the document. Added by KIP-INFER-0060 |
| `knolo.infer.equation-report` | `infer-equation`, over the document. Added by KIP-INFER-0061 |
| `knolo.infer.safe-error-report` | `infer-safe-error`, over the document. Added by KIP-INFER-0062 |
| `knolo.infer.redaction-report` | `infer-redaction`, over the document. Added by KIP-INFER-0063 |
| `knolo.infer.curve-report` | `infer-curve`, over the document. Added by KIP-INFER-0064 |
| `knolo.infer.rollback-report` | `infer-rollback`, over the document. Added by KIP-INFER-0065 |
| `knolo.infer.disconnect-report` | `infer-disconnect`, over the document. Added by KIP-INFER-0066 |
| `knolo.infer.receipt-store-report` | `infer-receipt-store`, over the document. Added by KIP-INFER-0067 |
| `knolo.infer.base-report` | `infer-base`, over the document. Added by KIP-INFER-0068 |
| `knolo.infer.disk-report` | `infer-disk`, over the document. Added by KIP-INFER-0069 |
| `knolo.infer.unload-report` | `infer-unload`, over the document. Added by KIP-INFER-0070 |
| `knolo.infer.restart-report` | `infer-restart`, over the document. Added by KIP-INFER-0071 |
| `knolo.infer.point-report` | `infer-point`, over the document. Added by KIP-INFER-0072 |
| `knolo.infer.duplicate-report` | `infer-duplicate`, over the document. Added by KIP-INFER-0073 |
| `knolo.infer.concurrent-load-report` | `infer-concurrent-load`, over the document. Added by KIP-INFER-0074 |
| `knolo.infer.eviction-report` | `infer-eviction`, over the document. Added by KIP-INFER-0075 |
| `knolo.infer.equality-report` | `infer-equality`, over the document. Added by KIP-INFER-0076 |
| `knolo.infer.oom-report` | `infer-oom`, over the document. Added by KIP-INFER-0077 |
| `knolo.infer.fault-report` | `infer-fault`, over the document. Added by KIP-INFER-0078 |
| `knolo.infer.timeout-report` | `infer-timeout`, over the document. Added by KIP-INFER-0079 |
| `knolo.infer.worker-start-report` | `infer-worker-start`, over the document. Added by KIP-INFER-0080 |
| `knolo.infer.challenge-report` | `infer-challenge`, over the document. Added by KIP-INFER-0081 |
| `knolo.infer.replay-environment-report` | `infer-replay-environment`, over the document. Added by KIP-INFER-0082 |
| `knolo.infer.replay-output-report` | `infer-replay-output`, over the document. Added by KIP-INFER-0083 |
| `knolo.infer.worker-lost-report` | `infer-worker-lost`, over the document. Added by KIP-INFER-0084 |
| `knolo.infer.draining-report` | `infer-draining`, over the document. Added by KIP-INFER-0085 |
| `knolo.infer.scalar-report` | `infer-scalar`, over the document. Added by KIP-INFER-0086 |
| `knolo.infer.digest-mismatch-report` | `infer-digest-mismatch`, over the document. Added by KIP-INFER-0087 |
| `knolo.infer.tokenizer-invalid-report` | `infer-tokenizer-invalid`, over the document. Added by KIP-INFER-0088 |
| `knolo.infer.template-invalid-report` | `infer-template-invalid`, over the document. Added by KIP-INFER-0089 |
| `knolo.infer.architecture-report` | `infer-architecture`, over the document. Added by KIP-INFER-0090 |
| `knolo.infer.public-report` | `infer-public`, over the document. Added by KIP-INFER-0091 |
| `knolo.infer.quantization-report` | `infer-quantization`, over the document. Added by KIP-INFER-0092 |
| `knolo.infer.kernel-report` | `infer-kernel`, over the document. Added by KIP-INFER-0093 |
| `knolo.infer.placement-refusal-report` | `infer-placement-refusal`, over the document. Added by KIP-INFER-0094 |
| `knolo.infer.memory-refusal-report` | `infer-memory-refusal`, over the document. Added by KIP-INFER-0095 |
| `knolo.infer.signature-check-report` | `infer-signature-check`, over the document. Added by KIP-INFER-0096 |
| `knolo.infer.context-limit-report` | `infer-context-limit`, over the document. Added by KIP-INFER-0097 |
| `knolo.infer.prompt-compilation-report` | `infer-prompt-compilation`, over the document. Added by KIP-INFER-0098 |
| `knolo.infer.image-invalid-report` | `infer-image-invalid`, over the document. Added by KIP-INFER-0099 |
| `knolo.infer.image-signature-report` | `infer-image-signature`, over the document. Added by KIP-INFER-0100 |
| `knolo.infer.cofactor-report` | `infer-cofactor`, over the document. Added by KIP-INFER-0101 |
| `knolo.infer.artifact-missing-report` | `infer-artifact-missing`, over the document. Added by KIP-INFER-0102 |
| `knolo.infer.receipt-required-report` | `infer-receipt-required`, over the document. Added by KIP-INFER-0103 |
| `knolo.infer.backend-report` | `infer-backend`, over the document. Added by KIP-INFER-0104 |
| `knolo.infer.digest-invalid-report` | `infer-digest-invalid`, over the document. Added by KIP-INFER-0105 |
| `knolo.infer.receipt-sign-report` | `infer-receipt-sign`, over the document. Added by KIP-INFER-0106 |
| `knolo.infer.canonical-cbor-report` | `infer-canonical-cbor`, over the document. Added by KIP-INFER-0107 |
| `knolo.infer.contract-invalid-report` | `infer-contract-invalid`, over the document. Added by KIP-INFER-0108 |
| `knolo.infer.grammar-refusal-report` | `infer-grammar-refusal`, over the document. Added by KIP-INFER-0109 |
| `knolo.infer.tool-refusal-report` | `infer-tool-refusal`, over the document. Added by KIP-INFER-0110 |
| `knolo.infer.receipt-verify-report` | `infer-receipt-verify`, over the document. Added by KIP-INFER-0111 |
| `knolo.infer.speculative-report` | `infer-speculative`, over the document. Added by KIP-INFER-0112 |
| `knolo.infer.graph-report` | `infer-graph`, over the document. Added by KIP-INFER-0113 |
| `knolo.infer.multi-model-report` | `infer-multi-model`, over the document. Added by KIP-INFER-0114 |
| `knolo.infer.secondary-report` | `infer-secondary`, over the document. Added by KIP-INFER-0115 |
| `knolo.infer.domain-report` | `infer-domain`, over the document. Added by KIP-INFER-0116 |
| `knolo.infer.attention-report` | `infer-attention`, over the document. Added by KIP-INFER-0117 |
| `knolo.infer.moe-report` | `infer-moe`, over the document. Added by KIP-INFER-0118 |
| `knolo.infer.expert-placement-report` | `infer-expert-placement`, over the document. Added by KIP-INFER-0119 |
| `knolo.infer.grouped-kernel-report` | `infer-grouped-kernel`, over the document. Added by KIP-INFER-0120 |
| `knolo.infer.router-report` | `infer-router`, over the document. Added by KIP-INFER-0121 |
| `knolo.infer.glm-report` | `infer-glm`, over the document. Added by KIP-INFER-0122 |
| `knolo.infer.workstation-report` | `infer-workstation`, over the document. Added by KIP-INFER-0123 |
| `knolo.infer.mixed-placement-report` | `infer-mixed-placement`, over the document. Added by KIP-INFER-0124 |
| `knolo.infer.capacity-report` | `infer-capacity`, over the document. Added by KIP-INFER-0125 |

Embedded tokenizer and template bytes carry a `root` that must equal `H(infer-tokenizer, byte-string)` or `H(infer-template, byte-string)`. A mismatch is `TOKENIZER_INVALID` or `TEMPLATE_INVALID`.

The model runtime root is:

```text
H(infer-model-runtime, {
  architectureAdapterRoot,
  artifactRoot,
  configRoot,
  modelImageRoot,
  templateRoot,
  tokenizerRoot
})
```

`architectureAdapterRoot` is `H(infer-config, { adapter })`. `configRoot` hashes architecture, capabilities, generation defaults, precisions, requirements, and special tokens. `artifactRoot` hashes the sorted weight-file list. Paths are relative POSIX, with no `.` or `..` segments.

## Numbers

Sampler and limit fields are integers. Temperature, repetition, presence, and frequency penalties are micros. `topP` and `minP` are millionths. Logit tolerance on a conformance receipt is millionths of a logit. Perplexity on a perplexity report is micros, and its logit delta is millionths. Throughput on a throughput report is tokens or requests per second in micros. The rate is the count times `1_000_000` times `1_000_000_000`, divided by the elapsed nanoseconds, and the remainder is discarded. Latency on a latency report is nanoseconds. Time to first token is the prefill duration. Time per output token is the decode duration divided by the decode token count, and the remainder is discarded. With one sample, the p50, p95, and p99 are that request's prefill plus decode. There is no floating-point value in a rooted object.

Temperature `0` is greedy: `rng` is `none`, and `seed` and `stream` are omitted. A positive temperature requires `rng` `philox-4x32-v1` plus `seed` and `stream`. The version-1 processing order is fixed in `SAMPLER_ORDER_V1`. Ties break toward the lowest token id.

## What this milestone verifies

`conformance/contracts/vectors.json` is produced by the Rust fixtures. `@knolo/infer` re-encodes every valid vector to the same bytes, recomputes every digest, and returns the same error code for every rejected vector.

The TypeScript package enforces canonical CBOR, digest bytes, `kind`, `version`, and the allowed key set. The Rust crate also enforces bounds, sorted lists, fixed-point ranges, and cross-field rules such as placement byte totals and receipt ids. Cryptographic checking of Ed25519 signatures is not part of this milestone. A signature block is structural: algorithm `ed25519`, a key id, and 64 signature bytes. Empty signatures are the unsigned local form.

The shared CBOR subset was checked against `@knolo/core`'s `canonicalCbor` and `digestDomain` for null, booleans, integers through 2^32, text, bytes, arrays, and maps.

## Error codes

`CANONICAL_CBOR_INVALID`, `CONTRACT_INVALID`, and `DIGEST_INVALID` are emitted here. The serving codes from the design (`MODEL_IMAGE_INVALID` through `REPLAY_OUTPUT_MISMATCH`) are reserved on `ErrorCode` so later milestones do not rename them. Retryable codes are `INSUFFICIENT_MEMORY`, `RECEIPT_PERSIST_FAILED`, `WORKER_START_FAILED`, `WORKER_LOST`, `CUDA_OOM`, and `REQUEST_TIMEOUT`.
