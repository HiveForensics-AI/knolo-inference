//! Canonical contracts for Knolo Infer.
//!
//! Rooted objects use definite-length canonical CBOR and domain-separated
//! SHA-256 digests. Integer sampler settings stay fixed-point until a
//! verified plan is executed.

mod cbor;
mod contracts;
mod digest;
mod error;
mod fields;

pub use cbor::{decode_canonical, encode_canonical, CborValue, MAX_DEPTH, MAX_DOCUMENT_BYTES};
pub use contracts::*;
pub use digest::{
    decode_hex, digest_bytes, digest_value, encode_hex, sha256_prefixed, validate_digest,
    DigestHex, INFER_DOMAINS,
};
pub use error::{fail, ErrorCode, InferFailure};

pub fn decode_contract(bytes: &[u8]) -> Result<Contract, InferFailure> {
    Contract::from_bytes(bytes)
}
