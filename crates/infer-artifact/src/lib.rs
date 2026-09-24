//! Compile and verify Knolo model images.
//!
//! Authoring JSON or the supported YAML subset becomes a canonical `.kmodel`.
//! Weight files stay outside the image. This crate hashes them, checks the
//! safetensors header against the tensor inventory, parses a bounded GGUF
//! file, and writes `knolo.infer.lock.json`. Tensor bodies are returned only
//! after that hash matches. It does not download weights.

mod authoring;
mod compile;
mod gguf;
mod io;
mod json;
mod lockfile;
mod paths;
mod safetensors;
mod verify;
mod yaml;

pub use compile::{compile_manifest, write_model_image, CompiledModel};
pub use gguf::{
    encode_gguf, parse_gguf_bytes, read_verified_gguf, GgufArray, GgufFile, GgufMetadata,
    GgufTensor, GgufTensorDraft, GgufTensorType, GgufValue, GgufValueType, GGUF_DEFAULT_ALIGNMENT,
    GGUF_QUANT_VERSION, GGUF_VERSION, MAX_GGUF_BYTES,
};
pub use infer_contracts::{
    sha256_prefixed, DigestHex, ErrorCode, InferFailure, MAX_DOCUMENT_BYTES,
};
pub use io::{hash_current_executable, hash_regular_file, write_atomic};
pub use json::parse_strict_json;
pub use lockfile::{
    new_lockfile, pin_alias, read_lockfile, unsupported_pull, write_lockfile, EnginePin, Lockfile,
    ModelPin, ProfilePin,
};
pub use paths::portable_model_path;
pub use safetensors::{
    encode_safetensors, parse_safetensors_bytes, read_safetensors_inventory, read_verified_tensors,
    TensorBytes, TensorView, MAX_IN_MEMORY_WEIGHT, MAX_SAFETENSORS_HEADER,
};
pub use verify::{verify_image, verify_weights, Verification};

pub(crate) fn map_image(err: InferFailure) -> InferFailure {
    match err.code {
        ErrorCode::CanonicalCborInvalid | ErrorCode::ContractInvalid => {
            InferFailure::new(ErrorCode::ModelImageInvalid, err.message)
        }
        _ => err,
    }
}
