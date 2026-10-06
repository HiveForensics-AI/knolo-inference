//! NOTICE and SBOM inventory for one cold micro fixture.
//!
//! The report stores roots and a component count. This measurement does not
//! write an SBOM and does not read a lockfile. The layout is specified in
//! `spec/KIP-INFER-0051-supply-notice.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure, NoticeReportV1, PlacementPlanV1};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied inventory for the engine build that would run the fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoticeObservation {
    pub engine_build_root: DigestHex,
    pub notice_root: DigestHex,
    pub sbom_root: DigestHex,
    pub component_count: u32,
    pub feature_set: String,
    pub candle_named: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the notice report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroNotice {
    pub plan: PlacementPlanV1,
    pub observed: NoticeObservation,
    pub report: NoticeReportV1,
}

/// Record the NOTICE and SBOM identity. No inventory file is written.
pub fn measure_supply_notice(
    plan: &PlacementPlanV1,
    observed: &NoticeObservation,
) -> Result<MicroNotice, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_supply_notice(&produced)?;
    Ok(produced)
}

/// Recompute the notice report from the stored plan and observation.
pub fn verify_supply_notice(measurement: &MicroNotice) -> Result<(), InferFailure> {
    let report = &measurement.report;
    let observed = &measurement.observed;
    report.validate()?;
    same_root(
        "engine build root does not match",
        &report.engine_build_root,
        &observed.engine_build_root,
    )?;
    same_root(
        "placement root does not match",
        &report.placement_root,
        &measurement.plan.root()?,
    )?;
    same_root(
        "notice root does not match",
        &report.notice_root,
        &observed.notice_root,
    )?;
    same_root(
        "sbom root does not match",
        &report.sbom_root,
        &observed.sbom_root,
    )?;
    same_u32(
        "component count does not match",
        report.component_count,
        observed.component_count,
    )?;
    same_text(
        "feature set does not match",
        &report.feature_set,
        &observed.feature_set,
    )?;
    same_bool(
        "candle named does not match",
        report.candle_named,
        observed.candle_named,
    )?;
    same_text(
        "execution mode does not match",
        &report.execution_mode,
        &observed.execution_mode,
    )?;
    same_text(
        "cache policy does not match",
        &report.cache_policy,
        &observed.cache_policy,
    )?;
    same_u32(
        "notice concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "notice run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "notice warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "notice request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "notice validation did not match",
        ));
    }
    Ok(())
}

/// Write the notice report. The NOTICE and the SBOM are not written.
pub fn write_notice_report(
    base: &Path,
    measurement: &MicroNotice,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_supply_notice(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "notice",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &NoticeObservation,
) -> Result<MicroNotice, InferFailure> {
    accept_micro_plan(plan, "notice")?;
    accept_cold_single(
        "notice",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    let report = NoticeReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        notice_root: observed.notice_root.clone(),
        sbom_root: observed.sbom_root.clone(),
        component_count: observed.component_count,
        feature_set: observed.feature_set.clone(),
        candle_named: observed.candle_named,
        execution_mode: observed.execution_mode.clone(),
        cache_policy: observed.cache_policy.clone(),
        concurrency: observed.concurrency,
        run_count: observed.run_count,
        warm_state: observed.warm_state.clone(),
        request_count: observed.request_count,
        validation_result: "recorded".into(),
        extensions: BTreeMap::new(),
    };
    report.validate()?;
    Ok(MicroNotice {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
