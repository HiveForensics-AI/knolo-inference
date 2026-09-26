//! Perplexity and accuracy delta between two f32 logit matrices.
//!
//! The measurement does not run a model and does not read a GGUF payload.
//! `dequant_gguf`, `quant_gemm`, and `convert_gguf_tensor` do not call this
//! path. The layout is specified in `spec/KIP-INFER-0026-perplexity-report.md`.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use infer_contracts::{
    fail, logit_root, output_token_root, DigestHex, ErrorCode, InferFailure, PerplexityReportV1,
};

use crate::greedy::argmax;

const MAX_PERPLEXITY_BYTES: u64 = 32 * 1024 * 1024;
const LOG_PROBABILITY_CLAMP: f64 = 1e-6;

/// One scored pair of logit matrices and the report that records the delta.
#[derive(Debug, Clone, PartialEq)]
pub struct PerplexityMeasurement {
    pub vocab: u32,
    pub targets: Vec<u32>,
    pub reference: Vec<f32>,
    pub candidate: Vec<f32>,
    pub reporter_build_root: DigestHex,
    pub report: PerplexityReportV1,
}

/// Score the candidate logits against the reference and the target ids.
///
/// No file is created. The logit slices are left unchanged.
pub fn perplexity_delta(
    vocab: u32,
    targets: &[u32],
    reference: &[f32],
    candidate: &[f32],
    reporter_build_root: &DigestHex,
) -> Result<PerplexityMeasurement, InferFailure> {
    let produced = measure(vocab, targets, reference, candidate, reporter_build_root)?;
    verify_perplexity_report(&produced)?;
    Ok(produced)
}

/// Recompute the report from the stored matrices.
pub fn verify_perplexity_report(measurement: &PerplexityMeasurement) -> Result<(), InferFailure> {
    let report = &measurement.report;
    report.validate()?;
    if report.reporter_build_root != measurement.reporter_build_root {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "reporter build root does not match",
        ));
    }
    let token_count = u32::try_from(measurement.targets.len()).map_err(|_| {
        fail(
            ErrorCode::ContractInvalid,
            "perplexity token count does not match",
        )
    })?;
    if report.token_count != token_count {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "perplexity token count does not match",
        ));
    }
    if report.reference_logit_root != logit_root(&measurement.reference)? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "reference logit root does not match",
        ));
    }
    if report.candidate_logit_root != logit_root(&measurement.candidate)? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "candidate logit root does not match",
        ));
    }
    if report.target_token_root != output_token_root(&measurement.targets)? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "target token root does not match",
        ));
    }
    let recomputed = measure(
        measurement.vocab,
        &measurement.targets,
        &measurement.reference,
        &measurement.candidate,
        &measurement.reporter_build_root,
    )?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "perplexity validation did not match",
        ));
    }
    Ok(())
}

/// Write the report. The logit matrices are not written.
pub fn write_perplexity_report(
    base: &Path,
    measurement: &PerplexityMeasurement,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_perplexity_report(measurement)?;
    let receipt_bytes = measurement.report.to_bytes()?;
    let receipt = output_path(base, receipt_path)?;
    write_new(&receipt, &receipt_bytes)
}

fn measure(
    vocab: u32,
    targets: &[u32],
    reference: &[f32],
    candidate: &[f32],
    reporter_build_root: &DigestHex,
) -> Result<PerplexityMeasurement, InferFailure> {
    if vocab == 0 {
        return Err(fail(ErrorCode::ContractInvalid, "perplexity vocab is zero"));
    }
    if targets.is_empty() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "perplexity token count is zero",
        ));
    }
    let token_count = u32::try_from(targets.len()).map_err(|_| {
        fail(
            ErrorCode::ContractInvalid,
            "perplexity logit length overflows",
        )
    })?;
    let count = u64::from(vocab)
        .checked_mul(u64::from(token_count))
        .ok_or_else(|| {
            fail(
                ErrorCode::ContractInvalid,
                "perplexity logit length overflows",
            )
        })?;
    let bytes = count.checked_mul(4).ok_or_else(|| {
        fail(
            ErrorCode::ContractInvalid,
            "perplexity logit length overflows",
        )
    })?;
    if bytes > MAX_PERPLEXITY_BYTES {
        return Err(fail(
            ErrorCode::InsufficientMemory,
            "perplexity logit matrix exceeds 32 MiB",
        ));
    }
    let Some(count_usize) = usize::try_from(count).ok() else {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "perplexity logit length overflows",
        ));
    };
    let vocab_usize = vocab as usize;
    for target in targets {
        if *target >= vocab {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "perplexity target is outside the vocab",
            ));
        }
    }
    if reference.len() != count_usize {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "perplexity reference length does not match",
        ));
    }
    if candidate.len() != count_usize {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "perplexity candidate length does not match",
        ));
    }
    require_finite(reference)?;
    require_finite(candidate)?;

    let mut reference_nll = 0.0f64;
    let mut candidate_nll = 0.0f64;
    let mut matches = 0u32;
    let mut parity = true;
    for (row, target) in targets.iter().copied().enumerate() {
        let start = row * vocab_usize;
        let end = start + vocab_usize;
        let reference_row = &reference[start..end];
        let candidate_row = &candidate[start..end];
        reference_nll += row_nll(reference_row, target as usize)?;
        candidate_nll += row_nll(candidate_row, target as usize)?;
        let reference_id = argmax(reference_row)?;
        let candidate_id = argmax(candidate_row)?;
        if reference_id != candidate_id {
            parity = false;
        }
        if candidate_id == target {
            matches += 1;
        }
    }
    let reference_perplexity = perplexity_micros(reference_nll, token_count)?;
    let candidate_perplexity = perplexity_micros(candidate_nll, token_count)?;
    let delta = i128::from(candidate_perplexity) - i128::from(reference_perplexity);
    let delta = i64::try_from(delta).map_err(|_| out_of_bounds())?;
    let accuracy = accuracy_millionths(matches, token_count);
    let max_abs = max_abs_millionths(reference, candidate)?;
    let report = PerplexityReportV1 {
        reference_logit_root: logit_root(reference)?,
        candidate_logit_root: logit_root(candidate)?,
        target_token_root: output_token_root(targets)?,
        reporter_build_root: reporter_build_root.clone(),
        token_count,
        reference_perplexity_micros: reference_perplexity,
        candidate_perplexity_micros: candidate_perplexity,
        perplexity_delta_micros: delta,
        greedy_parity: parity,
        target_accuracy_millionths: accuracy,
        max_abs_logit_delta_millionths: max_abs,
        validation_result: "recorded".into(),
        extensions: Default::default(),
    };
    Ok(PerplexityMeasurement {
        vocab,
        targets: targets.to_vec(),
        reference: reference.to_vec(),
        candidate: candidate.to_vec(),
        reporter_build_root: reporter_build_root.clone(),
        report,
    })
}

fn require_finite(values: &[f32]) -> Result<(), InferFailure> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "perplexity logit is not finite",
        ));
    }
    Ok(())
}

fn row_nll(row: &[f32], target: usize) -> Result<f64, InferFailure> {
    let mut max = f64::NEG_INFINITY;
    for value in row {
        let logit = f64::from(*value);
        if logit > max {
            max = logit;
        }
    }
    let mut sum = 0.0f64;
    for value in row {
        sum += (f64::from(*value) - max).exp();
    }
    if !sum.is_finite() || sum == 0.0 {
        return Err(out_of_bounds());
    }
    let log_z = max + sum.ln();
    let mut log_p = f64::from(row[target]) - log_z;
    if log_p > 0.0 {
        if log_p > LOG_PROBABILITY_CLAMP {
            return Err(out_of_bounds());
        }
        log_p = 0.0;
    }
    let nll = -log_p;
    if !nll.is_finite() || nll < 0.0 {
        return Err(out_of_bounds());
    }
    Ok(nll)
}

fn perplexity_micros(total_nll: f64, token_count: u32) -> Result<u64, InferFailure> {
    let mean = total_nll / f64::from(token_count);
    let perplexity = mean.exp();
    fixed_micros(perplexity)
}

fn max_abs_millionths(reference: &[f32], candidate: &[f32]) -> Result<u64, InferFailure> {
    let mut max = 0.0f64;
    for (left, right) in reference.iter().zip(candidate) {
        let delta = (f64::from(*left) - f64::from(*right)).abs();
        if !delta.is_finite() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "perplexity logit is not finite",
            ));
        }
        if delta > max {
            max = delta;
        }
    }
    fixed_micros(max)
}

fn fixed_micros(value: f64) -> Result<u64, InferFailure> {
    if !value.is_finite() || value < 0.0 {
        return Err(out_of_bounds());
    }
    let scaled = value * 1_000_000.0;
    if !scaled.is_finite() || scaled >= u64::MAX as f64 {
        return Err(out_of_bounds());
    }
    let rounded = scaled.round();
    if !rounded.is_finite() || rounded < 0.0 || rounded >= u64::MAX as f64 {
        return Err(out_of_bounds());
    }
    Ok(rounded as u64)
}

fn accuracy_millionths(matches: u32, token_count: u32) -> u32 {
    let numer = u64::from(matches) * 1_000_000;
    let denom = u64::from(token_count);
    ((numer + denom / 2) / denom) as u32
}

fn out_of_bounds() -> InferFailure {
    fail(
        ErrorCode::ContractInvalid,
        "perplexity is outside its bounds",
    )
}

fn output_path(base: &Path, relative: &str) -> Result<PathBuf, InferFailure> {
    infer_contracts::validate_relative_path(relative)?;
    let base = fs::canonicalize(base).map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            fail(
                ErrorCode::ContractInvalid,
                "perplexity directory does not exist",
            )
        } else {
            fail(
                ErrorCode::ContractInvalid,
                format!("perplexity directory: {err}"),
            )
        }
    })?;
    if !base.is_dir() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "perplexity directory does not exist",
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
                    "perplexity directory does not exist",
                ));
            }
            Err(err) => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    format!("perplexity directory: {err}"),
                ));
            }
        };
        if meta.file_type().is_symlink() || !meta.is_dir() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "perplexity path leaves the directory",
            ));
        }
    }
    let parent = fs::canonicalize(&parent).map_err(|err| {
        fail(
            ErrorCode::ContractInvalid,
            format!("perplexity directory: {err}"),
        )
    })?;
    if parent != base && !parent.starts_with(&base) {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "perplexity path leaves the directory",
        ));
    }
    let path = parent.join(file_name);
    match fs::symlink_metadata(&path) {
        Ok(_) => Err(fail(
            ErrorCode::ContractInvalid,
            "perplexity output already exists",
        )),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(path),
        Err(err) => Err(fail(
            ErrorCode::ContractInvalid,
            format!("perplexity output: {err}"),
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
                fail(
                    ErrorCode::ContractInvalid,
                    "perplexity output already exists",
                )
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
