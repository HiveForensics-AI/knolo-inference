//! Bounded corruption campaign over the nine parser targets.
//!
//! The campaign does not run a model and does not read a weight file on its
//! own. The caller supplies one seed per target and a probe that identifies a
//! buffer. `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`,
//! `perplexity_delta`, `measure_placement_memory`, `measure_micro_throughput`,
//! `measure_micro_latency`, and `measure_cancellation_latency` do not call
//! this path. The layout is specified in `spec/KIP-INFER-0030-corruption-fuzz.md`.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use infer_contracts::{
    fail, CborValue, DigestHex, ErrorCode, FuzzReportV1, InferFailure, FUZZ_MUTATIONS_PER_SEED,
    FUZZ_MUTATION_COUNT, FUZZ_SEED_COUNT, FUZZ_TARGETS,
};

/// Largest seed this campaign will mutate.
pub const MAX_FUZZ_SEED_BYTES: usize = 4096;

/// One original document for one parser target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuzzSeed {
    pub target: String,
    pub label: String,
    pub bytes: Vec<u8>,
}

/// The nine seeds and the engine build that records the campaign.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorruptionObservation {
    pub engine_build_root: DigestHex,
    pub seeds: Vec<FuzzSeed>,
}

/// One campaign and the report that records it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorruptionFuzz {
    pub observed: CorruptionObservation,
    pub report: FuzzReportV1,
}

/// Identify one buffer for a named target.
///
/// `Ok` is the semantic identity of an accepted document. `Err` rejects the
/// buffer. A probe must not panic.
pub trait CorruptionProbe {
    fn identify(&self, target: &str, bytes: &[u8]) -> Result<Vec<u8>, InferFailure>;
}

/// Six mutations of one seed. The seed itself is not a mutation.
pub fn fuzz_mutations(seed: &[u8]) -> Vec<Vec<u8>> {
    let mut truncated = seed.to_vec();
    truncated.pop();
    let mut first = seed.to_vec();
    if let Some(byte) = first.first_mut() {
        *byte ^= 0x01;
    }
    let mut last = seed.to_vec();
    if let Some(byte) = last.last_mut() {
        *byte ^= 0x01;
    }
    let mut appended = seed.to_vec();
    appended.push(0xff);
    let mut middle = seed.to_vec();
    if !middle.is_empty() {
        let index = middle.len() / 2;
        middle[index] ^= 0xff;
    }
    vec![Vec::new(), truncated, first, last, appended, middle]
}

/// Run the nine-target campaign. No file is created.
pub fn measure_corruption_fuzz(
    observed: &CorruptionObservation,
    probe: &impl CorruptionProbe,
) -> Result<CorruptionFuzz, InferFailure> {
    let produced = assemble(observed, probe)?;
    verify_corruption_fuzz(&produced, probe)?;
    Ok(produced)
}

/// Recompute the report from the stored seeds and the same probe.
pub fn verify_corruption_fuzz(
    measurement: &CorruptionFuzz,
    probe: &impl CorruptionProbe,
) -> Result<(), InferFailure> {
    let report = &measurement.report;
    report.validate()?;
    same_root(
        "engine build root does not match",
        &report.engine_build_root,
        &measurement.observed.engine_build_root,
    )?;
    same_root(
        "corpus root does not match",
        &report.corpus_root,
        &corpus_root(&measurement.observed.seeds)?,
    )?;
    let recomputed = assemble(&measurement.observed, probe)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "fuzz validation did not match",
        ));
    }
    Ok(())
}

/// Write the report. The seeds are not written.
pub fn write_fuzz_report(
    base: &Path,
    measurement: &CorruptionFuzz,
    probe: &impl CorruptionProbe,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_corruption_fuzz(measurement, probe)?;
    let receipt_bytes = measurement.report.to_bytes()?;
    let receipt = output_path(base, receipt_path)?;
    write_new(&receipt, &receipt_bytes)
}

fn assemble(
    observed: &CorruptionObservation,
    probe: &impl CorruptionProbe,
) -> Result<CorruptionFuzz, InferFailure> {
    if observed.seeds.len() != FUZZ_TARGETS.len() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "fuzz campaign is not the nine-target slice",
        ));
    }
    for (seed, expected) in observed.seeds.iter().zip(FUZZ_TARGETS) {
        if !FUZZ_TARGETS.contains(&seed.target.as_str()) {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "field target has an unsupported value",
            ));
        }
        if seed.target != expected {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "fuzz targets are not the nine-target campaign",
            ));
        }
        if seed.label != seed.target {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "fuzz label does not match the target",
            ));
        }
        if seed.bytes.is_empty() {
            return Err(fail(ErrorCode::ContractInvalid, "fuzz seed is empty"));
        }
        if seed.bytes.len() > MAX_FUZZ_SEED_BYTES {
            return Err(fail(
                ErrorCode::InsufficientMemory,
                "fuzz seed exceeds the campaign bound",
            ));
        }
    }
    let mut rejected_count = 0u32;
    let mut distinguished_count = 0u32;
    for seed in &observed.seeds {
        let origin = probe.identify(&seed.target, &seed.bytes).map_err(|_| {
            fail(
                ErrorCode::ContractInvalid,
                format!("fuzz seed was rejected for {}", seed.target),
            )
        })?;
        let mutations = fuzz_mutations(&seed.bytes);
        if mutations.len() != FUZZ_MUTATIONS_PER_SEED as usize {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "fuzz campaign is not the nine-target slice",
            ));
        }
        for mutation in mutations {
            match probe.identify(&seed.target, &mutation) {
                Err(_) => rejected_count += 1,
                Ok(identity) if identity == origin => {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        format!("corruption was accepted for {}", seed.target),
                    ));
                }
                Ok(_) => distinguished_count += 1,
            }
        }
    }
    let report = FuzzReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        corpus_root: corpus_root(&observed.seeds)?,
        target_count: FUZZ_SEED_COUNT,
        seed_count: FUZZ_SEED_COUNT,
        mutation_count: FUZZ_MUTATION_COUNT,
        rejected_count,
        distinguished_count,
        accepted_count: 0,
        validation_result: "fail-closed".into(),
        extensions: std::collections::BTreeMap::new(),
    };
    report.validate()?;
    Ok(CorruptionFuzz {
        observed: observed.clone(),
        report,
    })
}

fn corpus_root(seeds: &[FuzzSeed]) -> Result<DigestHex, InferFailure> {
    let mut items = Vec::with_capacity(seeds.len());
    for seed in seeds {
        items.push(CborValue::Array(vec![
            CborValue::Text(seed.target.clone()),
            CborValue::Text(seed.label.clone()),
            CborValue::Bytes(seed.bytes.clone()),
        ]));
    }
    infer_contracts::digest_value("infer-fuzz-corpus", &CborValue::Array(items))
}

fn same_root(message: &str, left: &DigestHex, right: &DigestHex) -> Result<(), InferFailure> {
    if left == right {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn output_path(base: &Path, relative: &str) -> Result<PathBuf, InferFailure> {
    infer_contracts::validate_relative_path(relative)?;
    let base = fs::canonicalize(base).map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            fail(ErrorCode::ContractInvalid, "fuzz directory does not exist")
        } else {
            fail(ErrorCode::ContractInvalid, format!("fuzz directory: {err}"))
        }
    })?;
    if !base.is_dir() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "fuzz directory does not exist",
        ));
    }
    let mut parent = base.clone();
    let mut parts = relative.split('/');
    let file_name = parts.next_back().expect("relative path has a file name");
    for segment in parts {
        parent.push(segment);
        let meta = match fs::symlink_metadata(&parent) {
            Ok(meta) => meta,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "fuzz directory does not exist",
                ));
            }
            Err(err) => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    format!("fuzz directory: {err}"),
                ));
            }
        };
        if meta.file_type().is_symlink() || !meta.is_dir() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "fuzz path leaves the directory",
            ));
        }
    }
    let parent = fs::canonicalize(&parent)
        .map_err(|err| fail(ErrorCode::ContractInvalid, format!("fuzz directory: {err}")))?;
    if parent != base && !parent.starts_with(&base) {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "fuzz path leaves the directory",
        ));
    }
    let path = parent.join(file_name);
    match fs::symlink_metadata(&path) {
        Ok(_) => Err(fail(
            ErrorCode::ContractInvalid,
            "fuzz output already exists",
        )),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(path),
        Err(err) => Err(fail(
            ErrorCode::ContractInvalid,
            format!("fuzz output: {err}"),
        )),
    }
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), InferFailure> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| {
            if err.kind() == std::io::ErrorKind::AlreadyExists {
                fail(ErrorCode::ContractInvalid, "fuzz output already exists")
            } else {
                fail(ErrorCode::ContractInvalid, format!("write: {err}"))
            }
        })?;
    file.write_all(bytes)
        .map_err(|err| fail(ErrorCode::ContractInvalid, format!("write: {err}")))?;
    file.sync_all()
        .map_err(|err| fail(ErrorCode::ContractInvalid, format!("write: {err}")))?;
    drop(file);
    sync_parent(path);
    Ok(())
}

fn sync_parent(path: &Path) {
    let Some(parent) = path.parent() else {
        return;
    };
    if let Ok(dir) = File::open(parent) {
        let _ = dir.sync_all();
    }
}
