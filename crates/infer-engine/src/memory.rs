//! Planner memory estimate for one placement.
//!
//! The measurement does not allocate a KV pool and does not run a model.
//! `dequant_gguf`, `quant_gemm`, `convert_gguf_tensor`, and `perplexity_delta`
//! do not call this path. The layout is specified in
//! `spec/KIP-INFER-0027-planner-memory.md`.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use infer_contracts::{
    fail, DigestHex, ErrorCode, InferFailure, MemoryEstimateReportV1, PlacementPlanV1,
};

use crate::traits::KvLayout;

const MAX_POOL_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvPoolPlan {
    pub slots: u64,
    pub total_bytes: u64,
}

/// Key and value bytes of an f32 page pool. `PagedKv::new` allocates this size.
pub fn planned_kv_bytes(layout: &KvLayout, page_count: u32) -> Result<u64, InferFailure> {
    Ok(planned_kv_pool(layout, page_count)?.total_bytes)
}

pub(crate) fn planned_kv_pool(
    layout: &KvLayout,
    page_count: u32,
) -> Result<KvPoolPlan, InferFailure> {
    if layout.dtype != "f32"
        || layout.layers == 0
        || layout.kv_heads == 0
        || layout.head_dim == 0
        || page_count == 0
        || !(16..=65_536).contains(&layout.block_size)
        || !layout.block_size.is_power_of_two()
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "kv layout is not a paged f32 pool",
        ));
    }
    let width = (layout.kv_heads as u64)
        .checked_mul(layout.head_dim as u64)
        .ok_or_else(|| fail(ErrorCode::ContractInvalid, "kv width overflows"))?;
    let slots = (layout.layers as u64)
        .checked_mul(layout.block_size as u64)
        .and_then(|count| count.checked_mul(width))
        .ok_or_else(|| fail(ErrorCode::ContractInvalid, "kv page size overflows"))?;
    let page_bytes = slots
        .checked_mul(8)
        .ok_or_else(|| fail(ErrorCode::ContractInvalid, "kv page size overflows"))?;
    let total_bytes = page_bytes
        .checked_mul(u64::from(page_count))
        .ok_or_else(|| fail(ErrorCode::InsufficientMemory, "kv page pool size overflows"))?;
    if total_bytes > MAX_POOL_BYTES {
        return Err(fail(
            ErrorCode::InsufficientMemory,
            "kv page pool exceeds 64 MiB",
        ));
    }
    Ok(KvPoolPlan { slots, total_bytes })
}

/// Byte counts the caller measured. This function does not allocate them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryObservation {
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    pub workspace_bytes: u64,
    pub staging_bytes: u64,
    pub overhead_bytes: u64,
}

/// One placement, the counts that were measured, and the report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryEstimate {
    pub plan: PlacementPlanV1,
    pub observed: MemoryObservation,
    pub estimator_build_root: DigestHex,
    pub report: MemoryEstimateReportV1,
}

/// Record the measured bytes when each count is within the plan.
///
/// No file is created and no pool is allocated.
pub fn measure_placement_memory(
    plan: &PlacementPlanV1,
    observed: &MemoryObservation,
    estimator_build_root: &DigestHex,
) -> Result<MemoryEstimate, InferFailure> {
    let estimate = assemble(plan, observed, estimator_build_root)?;
    verify_memory_estimate(&estimate)?;
    Ok(estimate)
}

/// Recompute the report from the stored plan and counts.
pub fn verify_memory_estimate(estimate: &MemoryEstimate) -> Result<(), InferFailure> {
    let report = &estimate.report;
    report.validate()?;
    if report.estimator_build_root != estimate.estimator_build_root {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "estimator build root does not match",
        ));
    }
    if report.placement_root != estimate.plan.root()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "placement root does not match",
        ));
    }
    same(
        "declared weight bytes do not match",
        report.declared_weight_bytes,
        estimate.plan.expected_weight_bytes,
    )?;
    same(
        "declared kv bytes do not match",
        report.declared_kv_bytes,
        estimate.plan.expected_kv_bytes,
    )?;
    same(
        "declared workspace bytes do not match",
        report.declared_workspace_bytes,
        estimate.plan.expected_workspace_bytes,
    )?;
    same(
        "declared staging bytes do not match",
        report.declared_staging_bytes,
        estimate.plan.expected_staging_bytes,
    )?;
    same(
        "declared overhead bytes do not match",
        report.declared_overhead_bytes,
        estimate.plan.expected_overhead_bytes,
    )?;
    same(
        "declared margin bytes do not match",
        report.declared_margin_bytes,
        estimate.plan.safety_margin_bytes,
    )?;
    same(
        "declared total bytes do not match",
        report.declared_total_bytes,
        estimate.plan.expected_total_bytes,
    )?;
    same(
        "measured weight bytes do not match",
        report.measured_weight_bytes,
        estimate.observed.weight_bytes,
    )?;
    same(
        "measured kv bytes do not match",
        report.measured_kv_bytes,
        estimate.observed.kv_bytes,
    )?;
    same(
        "measured workspace bytes do not match",
        report.measured_workspace_bytes,
        estimate.observed.workspace_bytes,
    )?;
    same(
        "measured staging bytes do not match",
        report.measured_staging_bytes,
        estimate.observed.staging_bytes,
    )?;
    same(
        "measured overhead bytes do not match",
        report.measured_overhead_bytes,
        estimate.observed.overhead_bytes,
    )?;
    same(
        "measured total bytes do not match",
        report.measured_total_bytes,
        measured_total(&estimate.observed)?,
    )?;
    let recomputed = assemble(
        &estimate.plan,
        &estimate.observed,
        &estimate.estimator_build_root,
    )?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "memory estimate validation did not match",
        ));
    }
    Ok(())
}

/// Write the report. The plan and the measured buffers are not written.
pub fn write_memory_estimate(
    base: &Path,
    estimate: &MemoryEstimate,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_memory_estimate(estimate)?;
    let receipt_bytes = estimate.report.to_bytes()?;
    let receipt = output_path(base, receipt_path)?;
    write_new(&receipt, &receipt_bytes)
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &MemoryObservation,
    estimator_build_root: &DigestHex,
) -> Result<MemoryEstimate, InferFailure> {
    plan.validate()?;
    if plan.graph_capture_mode != "off" {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "graph capture is off for the memory estimate",
        ));
    }
    if plan.rejection_reason.is_some() {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "placement was already rejected",
        ));
    }
    exceed(
        observed.weight_bytes,
        plan.expected_weight_bytes,
        "measured weight bytes exceed the placement bound",
    )?;
    exceed(
        observed.kv_bytes,
        plan.expected_kv_bytes,
        "measured kv bytes exceed the placement bound",
    )?;
    exceed(
        observed.workspace_bytes,
        plan.expected_workspace_bytes,
        "measured workspace bytes exceed the placement bound",
    )?;
    exceed(
        observed.staging_bytes,
        plan.expected_staging_bytes,
        "measured staging bytes exceed the placement bound",
    )?;
    exceed(
        observed.overhead_bytes,
        plan.expected_overhead_bytes,
        "measured overhead bytes exceed the placement bound",
    )?;
    let measured_total_bytes = measured_total(observed)?;
    let headroom_bytes = plan
        .expected_total_bytes
        .checked_sub(measured_total_bytes)
        .ok_or_else(|| fail(ErrorCode::ContractInvalid, "memory headroom does not match"))?;
    let report = MemoryEstimateReportV1 {
        placement_root: plan.root()?,
        estimator_build_root: estimator_build_root.clone(),
        declared_weight_bytes: plan.expected_weight_bytes,
        declared_kv_bytes: plan.expected_kv_bytes,
        declared_workspace_bytes: plan.expected_workspace_bytes,
        declared_staging_bytes: plan.expected_staging_bytes,
        declared_overhead_bytes: plan.expected_overhead_bytes,
        declared_margin_bytes: plan.safety_margin_bytes,
        declared_total_bytes: plan.expected_total_bytes,
        measured_weight_bytes: observed.weight_bytes,
        measured_kv_bytes: observed.kv_bytes,
        measured_workspace_bytes: observed.workspace_bytes,
        measured_staging_bytes: observed.staging_bytes,
        measured_overhead_bytes: observed.overhead_bytes,
        measured_total_bytes,
        headroom_bytes,
        validation_result: "within-bounds".into(),
        extensions: std::collections::BTreeMap::new(),
    };
    report.validate()?;
    Ok(MemoryEstimate {
        plan: plan.clone(),
        observed: observed.clone(),
        estimator_build_root: estimator_build_root.clone(),
        report,
    })
}

fn measured_total(observed: &MemoryObservation) -> Result<u64, InferFailure> {
    let mut total = 0u64;
    for value in [
        observed.weight_bytes,
        observed.kv_bytes,
        observed.workspace_bytes,
        observed.staging_bytes,
        observed.overhead_bytes,
    ] {
        total = total
            .checked_add(value)
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "measured bytes overflow"))?;
    }
    Ok(total)
}

fn exceed(measured: u64, declared: u64, message: &str) -> Result<(), InferFailure> {
    if measured > declared {
        Err(fail(ErrorCode::InsufficientMemory, message))
    } else {
        Ok(())
    }
}

fn same(message: &str, left: u64, right: u64) -> Result<(), InferFailure> {
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
            fail(
                ErrorCode::ContractInvalid,
                "memory estimate directory does not exist",
            )
        } else {
            fail(
                ErrorCode::ContractInvalid,
                format!("memory estimate directory: {err}"),
            )
        }
    })?;
    if !base.is_dir() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "memory estimate directory does not exist",
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
                    "memory estimate directory does not exist",
                ));
            }
            Err(err) => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    format!("memory estimate directory: {err}"),
                ));
            }
        };
        if meta.file_type().is_symlink() || !meta.is_dir() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "memory estimate path leaves the directory",
            ));
        }
    }
    let parent = fs::canonicalize(&parent).map_err(|err| {
        fail(
            ErrorCode::ContractInvalid,
            format!("memory estimate directory: {err}"),
        )
    })?;
    if parent != base && !parent.starts_with(&base) {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "memory estimate path leaves the directory",
        ));
    }
    let path = parent.join(file_name);
    match fs::symlink_metadata(&path) {
        Ok(_) => Err(fail(
            ErrorCode::ContractInvalid,
            "memory estimate output already exists",
        )),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(path),
        Err(err) => Err(fail(
            ErrorCode::ContractInvalid,
            format!("memory estimate output: {err}"),
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
                    "memory estimate output already exists",
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
