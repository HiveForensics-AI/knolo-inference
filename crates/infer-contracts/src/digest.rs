use sha2::{Digest, Sha256};

use crate::cbor::CborValue;
use crate::error::{fail, ErrorCode, InferFailure};

pub const DIGEST_PREFIX: &str = "sha256-";

/// Domains reserved by the Infer design, plus the roots that sections 10.4
/// and 16 require and that the §9.2 "at least" list did not name.
pub const INFER_DOMAINS: &[&str] = &[
    "infer-agent-effect",
    "infer-api",
    "infer-architecture",
    "infer-artifact-missing",
    "infer-attention",
    "infer-backend",
    "infer-base",
    "infer-binary",
    "infer-cache-channel",
    "infer-cancellation",
    "infer-canonical-cbor",
    "infer-capacity",
    "infer-chain",
    "infer-challenge",
    "infer-cofactor",
    "infer-composition",
    "infer-concurrent-load",
    "infer-config",
    "infer-conformance",
    "infer-context-limit",
    "infer-contract-invalid",
    "infer-conversion",
    "infer-conversion-config",
    "infer-curve",
    "infer-digest-invalid",
    "infer-digest-mismatch",
    "infer-disconnect",
    "infer-disk",
    "infer-domain",
    "infer-draining",
    "infer-duplicate",
    "infer-engine-build",
    "infer-equality",
    "infer-equation",
    "infer-eviction",
    "infer-evidence",
    "infer-evidence-output",
    "infer-execution-event",
    "infer-execution-plan",
    "infer-execution-trace",
    "infer-expert-placement",
    "infer-fault",
    "infer-finalization",
    "infer-fuzz",
    "infer-fuzz-corpus",
    "infer-glm",
    "infer-grammar",
    "infer-grammar-refusal",
    "infer-graph",
    "infer-grouped-kernel",
    "infer-hardware",
    "infer-host-key",
    "infer-hub",
    "infer-image-invalid",
    "infer-image-signature",
    "infer-install",
    "infer-kernel",
    "infer-kernel-bundle",
    "infer-kv",
    "infer-latency",
    "infer-limits",
    "infer-llama",
    "infer-load",
    "infer-logits",
    "infer-memory",
    "infer-memory-refusal",
    "infer-mistral",
    "infer-mixed-placement",
    "infer-model-artifact",
    "infer-model-image",
    "infer-model-runtime",
    "infer-moe",
    "infer-multi-model",
    "infer-notice",
    "infer-oom",
    "infer-output-text",
    "infer-output-tokens",
    "infer-overhead",
    "infer-peak",
    "infer-perplexity",
    "infer-placement",
    "infer-placement-refusal",
    "infer-point",
    "infer-prefix",
    "infer-prompt-compilation",
    "infer-prompt-input",
    "infer-prompt-plan",
    "infer-prompt-tokens",
    "infer-public",
    "infer-quantization",
    "infer-receipt",
    "infer-receipt-key",
    "infer-receipt-required",
    "infer-receipt-sign",
    "infer-receipt-store",
    "infer-receipt-verify",
    "infer-recipe",
    "infer-redaction",
    "infer-release",
    "infer-rendered-text",
    "infer-replay-check",
    "infer-replay-environment",
    "infer-replay-output",
    "infer-reproducible",
    "infer-request-intent",
    "infer-restart",
    "infer-rollback",
    "infer-router",
    "infer-safe-error",
    "infer-sampler",
    "infer-sandbox",
    "infer-scalar",
    "infer-secondary",
    "infer-signature",
    "infer-signature-check",
    "infer-special-tokens",
    "infer-speculative",
    "infer-stop-strings",
    "infer-studio",
    "infer-swap",
    "infer-template",
    "infer-template-invalid",
    "infer-throughput",
    "infer-timeout",
    "infer-tokenizer",
    "infer-tokenizer-invalid",
    "infer-tool-refusal",
    "infer-tools",
    "infer-truncation",
    "infer-unload",
    "infer-verification",
    "infer-vllm",
    "infer-worker-lost",
    "infer-worker-start",
    "infer-workstation",
];

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DigestHex(String);

impl DigestHex {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn parse(value: &str) -> Result<Self, InferFailure> {
        validate_digest(value)?;
        Ok(Self(value.to_string()))
    }
}

impl std::fmt::Display for DigestHex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

pub fn validate_digest(value: &str) -> Result<(), InferFailure> {
    let Some(hex) = value.strip_prefix(DIGEST_PREFIX) else {
        return Err(fail(
            ErrorCode::DigestInvalid,
            "digest must start with sha256-",
        ));
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(fail(
            ErrorCode::DigestInvalid,
            "digest must be sha256- plus 64 lowercase hex characters",
        ));
    }
    Ok(())
}

pub fn validate_domain(domain: &str) -> Result<(), InferFailure> {
    if !INFER_DOMAINS.contains(&domain) {
        return Err(fail(ErrorCode::DigestInvalid, "unknown digest domain"));
    }
    Ok(())
}

pub fn digest_bytes(domain: &str, payload: &[u8]) -> Result<DigestHex, InferFailure> {
    validate_domain(domain)?;
    let mut hasher = Sha256::new();
    hasher.update(format!("knolo:{domain}:v1\0").as_bytes());
    hasher.update(payload);
    let hashed = hasher.finalize();
    let mut hex = String::with_capacity(DIGEST_PREFIX.len() + 64);
    hex.push_str(DIGEST_PREFIX);
    for byte in hashed {
        hex.push_str(&format!("{byte:02x}"));
    }
    Ok(DigestHex(hex))
}

pub fn digest_value(domain: &str, value: &CborValue) -> Result<DigestHex, InferFailure> {
    digest_bytes(domain, &value.to_bytes())
}

pub fn sha256_prefixed(bytes: &[u8]) -> DigestHex {
    let hashed = Sha256::digest(bytes);
    let mut hex = String::with_capacity(DIGEST_PREFIX.len() + 64);
    hex.push_str(DIGEST_PREFIX);
    for byte in hashed {
        hex.push_str(&format!("{byte:02x}"));
    }
    DigestHex(hex)
}

pub fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn decode_hex(value: &str) -> Result<Vec<u8>, InferFailure> {
    if value.len() % 2 != 0 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(fail(ErrorCode::ContractInvalid, "invalid hexadecimal"));
    }
    let mut out = Vec::with_capacity(value.len() / 2);
    let bytes = value.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let hi = hex_nibble(bytes[i])?;
        let lo = hex_nibble(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Ok(out)
}

fn hex_nibble(byte: u8) -> Result<u8, InferFailure> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(fail(ErrorCode::ContractInvalid, "invalid hexadecimal")),
    }
}
