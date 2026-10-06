//! Compile and verify Knolo model images.
//!
//! Authoring JSON or the supported YAML subset becomes a canonical `.kmodel`.
//! Weight files stay outside the image. This crate hashes them, checks the
//! safetensors header against the tensor inventory, parses a bounded GGUF
//! file, and writes `knolo.infer.lock.json`. Tensor bodies are returned only
//! after that hash matches. `pull` of a lockfile alias copies a pinned local
//! image into place after the digest matches and does not open a network
//! connection. A catalog id is streamed from an allowlisted host, hashed
//! while it downloads, and promoted only after the digest matches.

mod authoring;
mod catalog;
mod compile;
mod gguf;
mod io;
mod json;
mod lockfile;
mod paths;
mod pull;
mod safetensors;
mod sign;
mod verify;
mod yaml;

pub use catalog::{
    checked_in_catalog, display_status, find_row, gpu_label, host_allowed, list_installed,
    load_catalog, load_catalog_file, pull_catalog_row, refresh_catalog, remove_installed, Catalog,
    CatalogRow, FetchPolicy, InstalledModel, PullOutcome, PullRequest, RefreshChange,
    LLAMA_CHAT_TEMPLATE,
};
pub use compile::{compile_manifest, compile_manifest_capped, write_model_image, CompiledModel};
pub use gguf::{
    canonical_llama_tensor, encode_gguf, parse_gguf_bytes, parse_gguf_bytes_capped,
    read_gguf_tensors, read_gguf_tensors_capped, read_llama_gguf, read_verified_gguf,
    read_verified_gguf_capped, GgufArray, GgufFile, GgufMetadata, GgufTensor, GgufTensorDraft,
    GgufTensorType, GgufValue, GgufValueType, GgufWeightRead, GGUF_DEFAULT_ALIGNMENT,
    GGUF_QUANT_VERSION, GGUF_VERSION, MAX_GGUF_BYTES, RUN_MAX_GGUF_BYTES,
    RUN_MAX_GGUF_TENSOR_BYTES,
};
pub use infer_contracts::{
    sha256_prefixed, DigestHex, ErrorCode, InferFailure, MAX_DOCUMENT_BYTES,
};
pub use io::{hash_current_executable, hash_regular_file, write_atomic};
pub use json::{parse_strict_json, parse_strict_json_allowing_null};
pub use lockfile::{
    new_lockfile, pin_alias, read_lockfile, unsupported_pull, write_lockfile, EnginePin, Lockfile,
    ModelPin, ProfilePin,
};
pub use paths::{portable_model_path, resolve_lock_relative};
pub use pull::{pull_alias, pull_destination, PullReport};
pub use safetensors::{
    encode_safetensors, parse_safetensors_bytes, read_safetensors_inventory, read_verified_tensors,
    TensorBytes, TensorView, MAX_IN_MEMORY_WEIGHT, MAX_SAFETENSORS_HEADER,
};
pub use sign::{
    ed25519_public, load_ed25519_public, load_ed25519_seed, sign_receipt_id,
    verify_image_signatures, verify_receipt_signature,
};
pub use verify::{verify_image, verify_weights, verify_weights_capped, Verification};

pub(crate) fn map_image(err: InferFailure) -> InferFailure {
    match err.code {
        ErrorCode::CanonicalCborInvalid | ErrorCode::ContractInvalid => {
            InferFailure::new(ErrorCode::ModelImageInvalid, err.message)
        }
        _ => err,
    }
}
