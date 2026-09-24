import { createHash } from "node:crypto";

import { InferFailure } from "./cbor.js";

export const INFER_DOMAINS = [
  "infer-agent-effect",
  "infer-api",
  "infer-base",
  "infer-binary",
  "infer-cache-channel",
  "infer-cancellation",
  "infer-chain",
  "infer-composition",
  "infer-concurrent-load",
  "infer-config",
  "infer-conformance",
  "infer-conversion",
  "infer-conversion-config",
  "infer-curve",
  "infer-disconnect",
  "infer-disk",
  "infer-duplicate",
  "infer-engine-build",
  "infer-equation",
  "infer-eviction",
  "infer-evidence",
  "infer-evidence-output",
  "infer-execution-event",
  "infer-execution-plan",
  "infer-execution-trace",
  "infer-finalization",
  "infer-fuzz",
  "infer-fuzz-corpus",
  "infer-grammar",
  "infer-hardware",
  "infer-host-key",
  "infer-hub",
  "infer-install",
  "infer-kernel-bundle",
  "infer-kv",
  "infer-latency",
  "infer-limits",
  "infer-llama",
  "infer-load",
  "infer-logits",
  "infer-memory",
  "infer-mistral",
  "infer-model-artifact",
  "infer-model-image",
  "infer-model-runtime",
  "infer-notice",
  "infer-output-text",
  "infer-output-tokens",
  "infer-overhead",
  "infer-peak",
  "infer-perplexity",
  "infer-placement",
  "infer-point",
  "infer-prefix",
  "infer-prompt-input",
  "infer-prompt-plan",
  "infer-prompt-tokens",
  "infer-receipt",
  "infer-receipt-key",
  "infer-receipt-store",
  "infer-recipe",
  "infer-redaction",
  "infer-release",
  "infer-rendered-text",
  "infer-replay-check",
  "infer-reproducible",
  "infer-request-intent",
  "infer-restart",
  "infer-rollback",
  "infer-safe-error",
  "infer-sampler",
  "infer-sandbox",
  "infer-signature",
  "infer-special-tokens",
  "infer-stop-strings",
  "infer-studio",
  "infer-swap",
  "infer-template",
  "infer-throughput",
  "infer-tokenizer",
  "infer-tools",
  "infer-truncation",
  "infer-unload",
  "infer-verification",
  "infer-vllm",
] as const;

export function digestDomain(domain: string, payload: Uint8Array): string {
  if (!INFER_DOMAINS.includes(domain as (typeof INFER_DOMAINS)[number])) {
    throw new InferFailure("DIGEST_INVALID", "unknown digest domain");
  }
  return hashDomain(domain, payload);
}

/** Same prefix rule as KIP-0003, including domains owned by other Knolo products. */
export function hashDomain(domain: string, payload: Uint8Array): string {
  const hash = createHash("sha256");
  hash.update(Buffer.from(`knolo:${domain}:v1\0`, "utf8"));
  hash.update(payload);
  return `sha256-${hash.digest("hex")}`;
}
