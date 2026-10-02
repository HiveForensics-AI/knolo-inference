//! Ed25519 signatures over a receipt id or a model-image root.
//!
//! The signed bytes are the label, a zero byte, and the digest text. The
//! signature block stays outside the receipt id and outside the model-image
//! root. A missing key file is the caller's choice; this module does not
//! invent a key.

use std::fs;
use std::path::Path;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, SignatureV1};

const RECEIPT_LABEL: &str = "knolo.infer.receipt-id";
const IMAGE_LABEL: &str = "knolo.infer.model-image-root";

pub fn load_ed25519_seed(path: &Path) -> Result<[u8; 32], InferFailure> {
    load_key(path, "ed25519 signing key is 32 bytes")
}

pub fn load_ed25519_public(path: &Path) -> Result<[u8; 32], InferFailure> {
    load_key(path, "ed25519 public key is 32 bytes")
}

pub fn ed25519_public(seed: &[u8; 32]) -> [u8; 32] {
    SigningKey::from_bytes(seed).verifying_key().to_bytes()
}

pub fn sign_receipt_id(seed: &[u8; 32], receipt_id: &str) -> Result<SignatureV1, InferFailure> {
    sign_label(seed, RECEIPT_LABEL, receipt_id)
}

pub fn verify_receipt_signature(
    receipt_id: &str,
    signatures: &[SignatureV1],
    public: &[u8; 32],
) -> Result<(), InferFailure> {
    verify_label(
        receipt_id,
        signatures,
        public,
        RECEIPT_LABEL,
        ErrorCode::ContractInvalid,
        "receipt signature",
    )
}

pub fn verify_image_signatures(
    image_root: &DigestHex,
    signatures: &[SignatureV1],
    public: &[u8; 32],
) -> Result<(), InferFailure> {
    if signatures.is_empty() {
        return Ok(());
    }
    verify_label(
        image_root.as_str(),
        signatures,
        public,
        IMAGE_LABEL,
        ErrorCode::ModelImageSignatureInvalid,
        "model image signature",
    )
}

fn sign_label(seed: &[u8; 32], label: &str, id: &str) -> Result<SignatureV1, InferFailure> {
    let key = SigningKey::from_bytes(seed);
    let public = key.verifying_key().to_bytes();
    let signature = key.sign(&domain_message(label, id));
    Ok(SignatureV1 {
        algorithm: "ed25519".into(),
        key_id: key_id(&public)?,
        signature: signature.to_bytes().to_vec(),
    })
}

fn verify_label(
    id: &str,
    signatures: &[SignatureV1],
    public: &[u8; 32],
    label: &str,
    code: ErrorCode,
    what: &str,
) -> Result<(), InferFailure> {
    if signatures.len() != 1 {
        return Err(fail(code, format!("{what} count must be one")));
    }
    let signature = &signatures[0];
    signature
        .validate()
        .map_err(|_| fail(code, format!("{what} is not a 64-byte ed25519 block")))?;
    if signature.key_id != key_id(public)? {
        return Err(fail(code, format!("{what} key does not match")));
    }
    let verifying = VerifyingKey::from_bytes(public)
        .map_err(|_| fail(code, format!("{what} public key is not an ed25519 point")))?;
    let parsed = Signature::from_slice(&signature.signature)
        .map_err(|_| fail(code, format!("{what} is not a 64-byte ed25519 block")))?;
    verifying
        .verify(&domain_message(label, id), &parsed)
        .map_err(|_| fail(code, format!("{what} did not verify")))
}

fn key_id(public: &[u8; 32]) -> Result<String, InferFailure> {
    Ok(infer_contracts::digest_bytes("infer-host-key", public)?.to_string())
}

fn domain_message(label: &str, id: &str) -> Vec<u8> {
    let mut message = Vec::with_capacity(label.len() + 1 + id.len());
    message.extend(label.as_bytes());
    message.push(0);
    message.extend(id.as_bytes());
    message
}

fn load_key(path: &Path, message: &str) -> Result<[u8; 32], InferFailure> {
    let bytes = fs::read(path).map_err(|err| {
        fail(
            ErrorCode::ContractInvalid,
            format!("cannot read key file: {err}"),
        )
    })?;
    if bytes.len() != 32 {
        return Err(fail(ErrorCode::ContractInvalid, message));
    }
    let mut key = [0u8; 32];
    key.copy_from_slice(&bytes);
    Ok(key)
}
