//! Verify canonical model-image bytes and, when asked, the weight files they name.

use std::path::Path;

use infer_contracts::{
    decode_canonical, fail, ErrorCode, InferFailure, ModelImageV1, MAX_DOCUMENT_BYTES,
};

use crate::map_image;
use crate::paths::resolve_inside;
use crate::safetensors::{inventory_verified_file, require_inventory};

use infer_contracts::DigestHex;

#[derive(Debug)]
pub struct Verification {
    pub image: ModelImageV1,
    pub image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub runtime_root: DigestHex,
    pub weights_checked: bool,
}

pub fn verify_image(bytes: &[u8]) -> Result<Verification, InferFailure> {
    if bytes.is_empty() {
        return Err(fail(ErrorCode::ModelImageInvalid, "model image is empty"));
    }
    if bytes.len() > MAX_DOCUMENT_BYTES {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "model image exceeds 32 MiB",
        ));
    }
    let value = decode_canonical(bytes).map_err(map_image)?;
    let image = ModelImageV1::from_cbor(&value).map_err(map_image)?;
    let encoded = image.to_bytes().map_err(map_image)?;
    if encoded != bytes {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "model image is not canonical",
        ));
    }
    Ok(Verification {
        image_root: image.image_root().map_err(map_image)?,
        artifact_root: image.artifact_root().map_err(map_image)?,
        runtime_root: image.runtime_root().map_err(map_image)?,
        weights_checked: false,
        image,
    })
}

pub fn verify_weights(image: &ModelImageV1, weights_dir: &Path) -> Result<(), InferFailure> {
    if image.format != "safetensors" {
        return Err(fail(
            ErrorCode::ModelImageInvalid,
            "only safetensors inventories are verified in this milestone",
        ));
    }
    let mut found = Vec::new();
    for file in &image.files {
        let path = resolve_inside(
            weights_dir,
            &file.path,
            ErrorCode::ModelImageInvalid,
            ErrorCode::ModelArtifactMissing,
        )?;
        let mut tensors =
            inventory_verified_file(&path, file.size_bytes, &file.sha256, &file.path)?;
        for tensor in &mut tensors {
            tensor.file = file.path.clone();
        }
        found.extend(tensors);
    }
    require_inventory(&image.tensor_inventory, &image.precisions, &found)
}
