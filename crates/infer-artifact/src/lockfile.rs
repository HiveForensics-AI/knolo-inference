//! `knolo.infer.lock.json`. The writer replaces the file by rename after fsync.
//! Core's `knolo.lock.json` is not read or written.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, ModelImageV1};

use crate::io::{read_utf8_limited, write_atomic};
use crate::json::parse_strict_json;
use crate::paths::check_relative_posix;

pub const LOCK_KIND: &str = "knolo.infer.lock";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Lockfile {
    pub kind: String,
    pub version: u32,
    pub models: BTreeMap<String, ModelPin>,
    pub engine: EnginePin,
    pub profiles: BTreeMap<String, ProfilePin>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelPin {
    pub model_image_root: String,
    pub artifact_root: String,
    pub model_image_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EnginePin {
    pub channel: String,
    pub build_root: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProfilePin {
    pub placement_root: String,
}

pub fn new_lockfile(build_root: &DigestHex) -> Lockfile {
    Lockfile {
        kind: LOCK_KIND.into(),
        version: 1,
        models: BTreeMap::new(),
        engine: EnginePin {
            channel: "native".into(),
            build_root: build_root.as_str().to_string(),
        },
        profiles: BTreeMap::new(),
    }
}

pub fn read_lockfile(path: &Path) -> Result<Option<Lockfile>, InferFailure> {
    if !path.exists() {
        return Ok(None);
    }
    let text = read_utf8_limited(path, ErrorCode::ContractInvalid)?;
    let value = parse_strict_json(&text, ErrorCode::ContractInvalid)?;
    let lock: Lockfile = serde_json::from_value(value)
        .map_err(|err| fail(ErrorCode::ContractInvalid, format!("infer lockfile: {err}")))?;
    lock.validate()?;
    Ok(Some(lock))
}

pub fn write_lockfile(path: &Path, lock: &Lockfile) -> Result<(), InferFailure> {
    lock.validate()?;
    let mut text = serde_json::to_string_pretty(lock).map_err(|_| {
        fail(
            ErrorCode::ContractInvalid,
            "infer lockfile could not be encoded",
        )
    })?;
    text.push('\n');
    write_atomic(path, text.as_bytes(), ErrorCode::ContractInvalid)
}

pub fn pin_alias(
    lock: &mut Lockfile,
    alias: &str,
    model_image_path: &str,
    image: &ModelImageV1,
) -> Result<(), InferFailure> {
    valid_alias(alias)?;
    check_relative_posix(model_image_path, ErrorCode::ContractInvalid)?;
    if lock.models.len() >= 1024 && !lock.models.contains_key(alias) {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "infer lockfile has too many model aliases",
        ));
    }
    lock.models.insert(
        alias.to_string(),
        ModelPin {
            model_image_root: image.image_root()?.to_string(),
            artifact_root: image.artifact_root()?.to_string(),
            model_image_path: model_image_path.to_string(),
        },
    );
    Ok(())
}

pub fn unsupported_pull() -> InferFailure {
    fail(
        ErrorCode::ModelArtifactMissing,
        "pull is unsupported until download staging exists",
    )
}

impl Lockfile {
    fn validate(&self) -> Result<(), InferFailure> {
        if self.kind != LOCK_KIND || self.version != 1 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "infer lockfile kind and version must be knolo.infer.lock 1",
            ));
        }
        if self.models.len() > 1024 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "infer lockfile has too many model aliases",
            ));
        }
        if self.profiles.len() > 64 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "infer lockfile has too many profiles",
            ));
        }
        if self.engine.channel != "native" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "lockfile engine channel must be native",
            ));
        }
        DigestHex::parse(&self.engine.build_root)?;
        for (alias, pin) in &self.models {
            valid_alias(alias)?;
            DigestHex::parse(&pin.model_image_root)?;
            DigestHex::parse(&pin.artifact_root)?;
            check_relative_posix(&pin.model_image_path, ErrorCode::ContractInvalid)?;
        }
        for (name, profile) in &self.profiles {
            valid_alias(name)?;
            DigestHex::parse(&profile.placement_root)?;
        }
        Ok(())
    }
}

fn valid_alias(alias: &str) -> Result<(), InferFailure> {
    let bytes = alias.as_bytes();
    if !(1..=64).contains(&bytes.len())
        || !bytes[0].is_ascii_alphanumeric()
        || !bytes.iter().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_' | b'-')
        })
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "alias must be 1-64 lowercase letters, digits, '.', '_', or '-'",
        ));
    }
    Ok(())
}
