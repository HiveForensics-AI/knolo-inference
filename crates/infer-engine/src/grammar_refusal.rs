//! One grammar the compiler did not build for a cold micro fixture.
//!
//! The report stores the reason and the source length. This measurement
//! does not compile a grammar. The layout is specified in
//! `spec/KIP-INFER-0109-grammar-refusal.md`.

use std::collections::BTreeMap;
use std::path::Path;

use infer_contracts::{
    fail, DigestHex, ErrorCode, GrammarRefusalReportV1, InferFailure, PlacementPlanV1,
    MAX_GRAMMAR_SOURCE_BYTES,
};

use crate::report_io::{
    accept_cold_single, accept_micro_plan, same_bool, same_root, same_text, same_u32,
    write_exclusive,
};

/// Caller-supplied grammar refusal for one cold fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarRefusalObservation {
    pub engine_build_root: DigestHex,
    pub reason: String,
    pub source_bytes: u32,
    pub code: String,
    pub retryable: bool,
    pub source_opened: bool,
    pub grammar_rooted: bool,
    pub mask_applied: bool,
    pub grammar_compiled: bool,
    pub forward_ran: bool,
    pub receipt_stored: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
}

/// One micro-fixture observation and the grammar-refusal report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroGrammarRefusal {
    pub plan: PlacementPlanV1,
    pub observed: GrammarRefusalObservation,
    pub report: GrammarRefusalReportV1,
}

/// Record the grammar refusal. The grammar is not compiled.
pub fn measure_grammar_refusal(
    plan: &PlacementPlanV1,
    observed: &GrammarRefusalObservation,
) -> Result<MicroGrammarRefusal, InferFailure> {
    let produced = assemble(plan, observed)?;
    verify_grammar_refusal(&produced)?;
    Ok(produced)
}

/// Recompute the grammar-refusal report from the stored plan and observation.
pub fn verify_grammar_refusal(measurement: &MicroGrammarRefusal) -> Result<(), InferFailure> {
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
        "source bytes do not match",
        report.source_bytes,
        observed.source_bytes,
    )?;
    same_text("code does not match", &report.code, &observed.code)?;
    same_bool(
        "retryable does not match",
        report.retryable,
        observed.retryable,
    )?;
    same_bool(
        "source opened does not match",
        report.source_opened,
        observed.source_opened,
    )?;
    same_bool(
        "grammar rooted does not match",
        report.grammar_rooted,
        observed.grammar_rooted,
    )?;
    same_bool(
        "mask applied does not match",
        report.mask_applied,
        observed.mask_applied,
    )?;
    same_bool(
        "grammar compiled does not match",
        report.grammar_compiled,
        observed.grammar_compiled,
    )?;
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
        "grammar-refusal concurrency does not match",
        report.concurrency,
        observed.concurrency,
    )?;
    same_u32(
        "grammar-refusal run count does not match",
        report.run_count,
        observed.run_count,
    )?;
    same_text(
        "grammar-refusal warm state does not match",
        &report.warm_state,
        &observed.warm_state,
    )?;
    same_u32(
        "grammar-refusal request count does not match",
        report.request_count,
        observed.request_count,
    )?;
    let recomputed = assemble(&measurement.plan, &measurement.observed)?;
    if recomputed.report.to_bytes()? != report.to_bytes()? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "grammar-refusal validation did not match",
        ));
    }
    Ok(())
}

/// Write the grammar-refusal report. The grammar is not compiled.
pub fn write_grammar_refusal_report(
    base: &Path,
    measurement: &MicroGrammarRefusal,
    receipt_path: &str,
) -> Result<(), InferFailure> {
    verify_grammar_refusal(measurement)?;
    write_exclusive(
        base,
        receipt_path,
        &measurement.report.to_bytes()?,
        "grammar-refusal",
    )
}

fn assemble(
    plan: &PlacementPlanV1,
    observed: &GrammarRefusalObservation,
) -> Result<MicroGrammarRefusal, InferFailure> {
    accept_micro_plan(plan, "grammar-refusal")?;
    accept_cold_single(
        "grammar-refusal",
        &observed.execution_mode,
        &observed.cache_policy,
        observed.concurrency,
        observed.run_count,
        &observed.warm_state,
        observed.request_count,
    )?;
    if matches!(observed.reason.as_str(), "automaton" | "mask")
        && observed.source_bytes > MAX_GRAMMAR_SOURCE_BYTES
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "grammar source exceeds the record cap",
        ));
    }
    let report = GrammarRefusalReportV1 {
        engine_build_root: observed.engine_build_root.clone(),
        placement_root: plan.root()?,
        reason: observed.reason.clone(),
        source_bytes: observed.source_bytes,
        code: observed.code.clone(),
        retryable: observed.retryable,
        source_opened: observed.source_opened,
        grammar_rooted: observed.grammar_rooted,
        mask_applied: observed.mask_applied,
        grammar_compiled: observed.grammar_compiled,
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
    Ok(MicroGrammarRefusal {
        plan: plan.clone(),
        observed: observed.clone(),
        report,
    })
}
