//! Pinned model image and the sampler plan for one completion.
//!
//! The supervisor reads the image and does not open weight bytes. The worker
//! opens the weights after the image roots match the lockfile.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use infer_artifact::{read_lockfile, verify_image};
use infer_contracts::{
    fail, DigestHex, ErrorCode, FixedPointSamplerV1, InferFailure, ModelImageV1, SamplerPlanV1,
};

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub temperature_micros: Option<u32>,
    pub top_p_millionths: Option<u32>,
    pub min_p_millionths: Option<u32>,
    pub repetition_penalty_micros: Option<u32>,
    pub presence_penalty_micros: Option<u32>,
    pub frequency_penalty_micros: Option<u32>,
    pub top_k: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub seed: Option<u64>,
    pub stream: Option<u64>,
}

impl GenerationRequest {
    pub fn omitted() -> Self {
        Self {
            temperature_micros: None,
            top_p_millionths: None,
            min_p_millionths: None,
            repetition_penalty_micros: None,
            presence_penalty_micros: None,
            frequency_penalty_micros: None,
            top_k: None,
            max_output_tokens: None,
            seed: None,
            stream: None,
        }
    }
}

#[derive(Clone)]
pub struct PinnedModel {
    pub image: ModelImageV1,
    pub image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub runtime_root: String,
    pub weight_bytes: u64,
    pub kmodel: PathBuf,
    pub weights: PathBuf,
}

pub fn resolve_pin(
    work_dir: &Path,
    lock_path: &Path,
    alias: &str,
    weights_dir: Option<&Path>,
) -> Result<PinnedModel, InferFailure> {
    let lock_path = if lock_path.is_absolute() {
        lock_path.to_path_buf()
    } else {
        work_dir.join(lock_path)
    };
    let lock = read_lockfile(&lock_path)?
        .ok_or_else(|| fail(ErrorCode::ModelArtifactMissing, "infer lockfile is missing"))?;
    let pin = lock
        .models
        .get(alias)
        .ok_or_else(|| fail(ErrorCode::ModelArtifactMissing, "alias is not pinned"))?;
    let kmodel = work_dir.join(&pin.model_image_path);
    let weights = match weights_dir {
        Some(dir) => dir.to_path_buf(),
        None => kmodel
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
    };
    let bytes = fs::read(&kmodel).map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            fail(ErrorCode::ModelArtifactMissing, "model image is missing")
        } else {
            fail(
                ErrorCode::ModelImageInvalid,
                format!("model image could not be read: {err}"),
            )
        }
    })?;
    let verification = verify_image(&bytes)?;
    if verification.image_root.as_str() != pin.model_image_root
        || verification.artifact_root.as_str() != pin.artifact_root
    {
        return Err(fail(
            ErrorCode::ModelDigestMismatch,
            "pinned roots do not match the model image",
        ));
    }
    let mut weight_bytes = 0u64;
    for file in &verification.image.files {
        weight_bytes = weight_bytes
            .checked_add(file.size_bytes)
            .ok_or_else(|| fail(ErrorCode::ModelImageInvalid, "weight byte total overflows"))?;
    }
    Ok(PinnedModel {
        image: verification.image,
        image_root: verification.image_root,
        artifact_root: verification.artifact_root,
        runtime_root: verification.runtime_root.to_string(),
        weight_bytes,
        kmodel,
        weights,
    })
}

pub fn build_sampler(
    image: &ModelImageV1,
    generation: &GenerationRequest,
) -> Result<SamplerPlanV1, InferFailure> {
    let mut settings: FixedPointSamplerV1 = image.generation_defaults.clone();
    apply(
        &mut settings.temperature_micros,
        generation.temperature_micros,
    );
    apply(&mut settings.top_p_millionths, generation.top_p_millionths);
    apply(&mut settings.min_p_millionths, generation.min_p_millionths);
    apply(
        &mut settings.repetition_penalty_micros,
        generation.repetition_penalty_micros,
    );
    apply(
        &mut settings.presence_penalty_micros,
        generation.presence_penalty_micros,
    );
    apply(
        &mut settings.frequency_penalty_micros,
        generation.frequency_penalty_micros,
    );
    apply(&mut settings.top_k, generation.top_k);
    apply(
        &mut settings.max_output_tokens,
        generation.max_output_tokens,
    );
    settings.validate()?;
    let greedy = settings.temperature_micros == 0;
    if greedy && (generation.seed.is_some() || generation.stream.is_some()) {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "greedy sampler cannot carry an RNG seed",
        ));
    }
    if !greedy && generation.seed.is_none() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "sampled generation requires a seed",
        ));
    }
    let mut eos = Vec::new();
    if let Some(id) = image.special_tokens.eos {
        eos.push(id);
    }
    eos.sort_unstable();
    eos.dedup();
    let plan = SamplerPlanV1 {
        settings,
        rng: if greedy {
            "none".into()
        } else {
            "philox-4x32-v1".into()
        },
        seed: if greedy { None } else { generation.seed },
        stream: if greedy {
            None
        } else {
            Some(generation.stream.unwrap_or(0))
        },
        tie_break: "lowest-token-id".into(),
        eos_token_ids: eos,
        stop_string_roots: Vec::new(),
        extensions: BTreeMap::new(),
    };
    plan.validate()?;
    Ok(plan)
}

fn apply(slot: &mut u32, override_value: Option<u32>) {
    if let Some(value) = override_value {
        *slot = value;
    }
}
