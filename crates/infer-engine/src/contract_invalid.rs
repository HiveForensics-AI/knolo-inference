//! One contract field the decoder refused for a cold micro fixture.
//!
//! The report stores the reason and the field length. This measurement does
//! not accept the document. The layout is specified in
//! `spec/KIP-INFER-0108-contract-invalid.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, ContractInvalidReportV1, DigestHex, ErrorCode, InferFailure, PlacementPlanV1,
    MAX_FIELD_RECORD_BYTES,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied contract refusal for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractInvalidObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub field_bytes: u32,
    pub code: String,
    pub retryable: bool,
    pub field_present: bool,
    pub type_accepted: bool,
    pub value_accepted: bool,
    pub decoded: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the contract-invalid report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroContractInvalid {
    pub plan: PlacementPlanV1,
    pub observed: ContractInvalidObservation,
    pub report: ContractInvalidReportV1,
}

/// Record the contract refusal. The document is not accepted.
pub fn measure_contract_invalid(
    plan: &PlacementPlanV1,
    observed: &ContractInvalidObservation,
) -> Result<MicroContractInvalid, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_contract_invalid(&produced)?;
    Ok(produced)
}

/// Recompute the contract-invalid report from the stored plan and observation.
pub fn verify_contract_invalid(measurement: &MicroContractInvalid) -> Result<(), InferFailure> {
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
    same_text("reason does not match", &report.reason, &observed.reason)?;
    same_u32(
        "field bytes do not match",
        report.field_bytes,
        observed.field_bytes,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "field present does not match",
        report.field_present,
        observed.field_present,
    )?;
    same_bool(
        "type accepted does not match",
        report.type_accepted,
        observed.type_accepted,
    )?;
    same_bool(
        "value accepted does not match",
        report.value_accepted,
        observed.value_accepted,
    )?;
    same_bool("decoded does not match", report.decoded, observed.decoded)?;
    same_bool(
        "forward ran does not match",
        report.forward_ran,
        observed.forward_ran,
    )?;
    same_bool(
        "receipt stored does not match",
        report.receipt_stored,
        observed.receipt_stored,
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
        "contract-invalid concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "contract-invalid run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "contract-invalid warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "contract-invalid request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "contract-invalid validation did not match",
        ));
    }
    Ok(())
}

/// Write the contract-invalid report. The document is not accepted.
pub fn write_contract_invalid_report(
    base: &Path,
    measurement: &MicroContractInvalid,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_contract_invalid(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "contract-invalid",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &ContractInvalidObservation,
) -> Result<MicroContractInvalid, InferFailure> {
    accept_micro_plan(plan, "contract-invalid")?;
    accept_cold_single(
        "contract-invalid",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if matches!(observed.reason.as_str(), "type" | "unknown" | "value")
        && observed.field_bytes > MAX_FIELD_RECORD_BYTES
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "field exceeds the record cap",
        ));
    }
    let report = ContractInvalidReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        field_bytes: observed.field_bytes,
        code: observed.code.clone(),
        retryable: observed.retryable,
        field_present: observed.field_present,
        type_accepted: observed.type_accepted,
        value_accepted: observed.value_accepted,
        decoded: observed.decoded,
        forward_ran: observed.forward_ran,
        receipt_stored: observed.receipt_stored,
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
    Ok(MicroContractInvalid {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
