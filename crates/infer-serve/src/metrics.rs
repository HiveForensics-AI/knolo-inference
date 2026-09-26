//! Prometheus text for the loopback supervisor.
//!
//! Labels are a fixed set: service class, finish outcome, error code, page
//! state, and OOM kind. A request id, prompt, alias, or path is not a label.
//! Prefix cache is not allocated, so its counters stay at the worker's zeros.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use infer_contracts::ErrorCode;
use infer_engine::{ServiceClass, ITERATION_BUCKETS};

use crate::protocol::WorkerStats;

const CLASSES: usize = 4;
const OUTCOMES: usize = 5;
const LATENCY_BUCKETS: usize = 13;

const CLASS_LABELS: [&str; CLASSES] = ["interactive", "standard", "batch", "background"];
const OUTCOME_LABELS: [&str; OUTCOMES] = ["stop", "length", "cancelled", "error", "rejected"];
const LATENCY_LABELS: [&str; LATENCY_BUCKETS] = [
    "0.001", "0.005", "0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1", "2.5", "5", "10", "+Inf",
];
const LATENCY_BOUND_NANOS: [u64; LATENCY_BUCKETS] = [
    1_000_000,
    5_000_000,
    10_000_000,
    25_000_000,
    50_000_000,
    100_000_000,
    250_000_000,
    500_000_000,
    1_000_000_000,
    2_500_000_000,
    5_000_000_000,
    10_000_000_000,
    u64::MAX,
];
const ITERATION_LABELS: [&str; ITERATION_BUCKETS] = [
    "0.0001", "0.0005", "0.001", "0.005", "0.01", "0.05", "0.1", "0.5", "1", "5", "+Inf",
];

#[derive(Clone, Copy)]
struct Hist {
    buckets: [u64; LATENCY_BUCKETS],
    sum_nanos: u64,
    count: u64,
}

impl Default for Hist {
    fn default() -> Self {
        Self {
            buckets: [0; LATENCY_BUCKETS],
            sum_nanos: 0,
            count: 0,
        }
    }
}

impl Hist {
    fn observe(&mut self, nanos: u64) {
        self.count = self.count.saturating_add(1);
        self.sum_nanos = self.sum_nanos.saturating_add(nanos);
        let index = LATENCY_BOUND_NANOS
            .iter()
            .position(|bound| nanos <= *bound)
            .unwrap_or(LATENCY_BUCKETS - 1);
        self.buckets[index] = self.buckets[index].saturating_add(1);
    }
}

#[derive(Clone, Copy)]
pub enum Rejection {
    Contract,
    Context,
    Memory,
    Prompt,
    Other,
}

#[derive(Clone, Default)]
pub struct HostMetrics {
    pub restarts: u64,
    pub snapshot: Option<WorkerStats>,
    rejected_contract: u64,
    rejected_context: u64,
    rejected_memory: u64,
    rejected_prompt: u64,
    rejected_other: u64,
    requests: [[Hist; OUTCOMES]; CLASSES],
    ttft: [Hist; CLASSES],
    receipt: Hist,
}

impl HostMetrics {
    pub fn note_rejection(&mut self, kind: Rejection) {
        let counter = match kind {
            Rejection::Contract => &mut self.rejected_contract,
            Rejection::Context => &mut self.rejected_context,
            Rejection::Memory => &mut self.rejected_memory,
            Rejection::Prompt => &mut self.rejected_prompt,
            Rejection::Other => &mut self.rejected_other,
        };
        *counter = counter.saturating_add(1);
    }

    fn observe_request(&mut self, class: ServiceClass, outcome: &str, nanos: u64) {
        self.requests[class_index(class)][outcome_index(outcome)].observe(nanos);
    }

    fn observe_ttft(&mut self, class: ServiceClass, nanos: u64) {
        self.ttft[class_index(class)].observe(nanos);
    }

    fn observe_receipt(&mut self, nanos: u64) {
        self.receipt.observe(nanos);
    }
}

pub struct RequestWatch<'a> {
    metrics: &'a Mutex<HostMetrics>,
    class: ServiceClass,
    started: Instant,
    outcome: Option<&'static str>,
    ttft_nanos: Option<u64>,
    receipt_nanos: Option<u64>,
    done: bool,
}

impl<'a> RequestWatch<'a> {
    pub fn new(metrics: &'a Mutex<HostMetrics>, class: ServiceClass, started: Instant) -> Self {
        Self {
            metrics,
            class,
            started,
            outcome: None,
            ttft_nanos: None,
            receipt_nanos: None,
            done: false,
        }
    }

    pub fn set_outcome(&mut self, outcome: &'static str) {
        self.outcome = Some(outcome);
    }

    pub fn outcome_is_unset(&self) -> bool {
        self.outcome.is_none()
    }

    pub fn note_ttft(&mut self) {
        if self.ttft_nanos.is_none() {
            self.ttft_nanos = Some(duration_nanos(self.started.elapsed()));
        }
    }

    pub fn note_receipt(&mut self, elapsed: Duration) {
        self.receipt_nanos = Some(duration_nanos(elapsed));
    }
}

impl Drop for RequestWatch<'_> {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        self.done = true;
        let outcome = self.outcome.unwrap_or("error");
        let elapsed = duration_nanos(self.started.elapsed());
        let mut host = self.metrics.lock().unwrap_or_else(|err| err.into_inner());
        host.observe_request(self.class, outcome, elapsed);
        if let Some(ttft) = self.ttft_nanos {
            host.observe_ttft(self.class, ttft);
        }
        if let Some(receipt) = self.receipt_nanos {
            host.observe_receipt(receipt);
        }
    }
}

pub fn note_compile_failure(metrics: &Mutex<HostMetrics>, code: ErrorCode) {
    let kind = match code {
        ErrorCode::ContextLimitExceeded => Some(Rejection::Context),
        ErrorCode::InsufficientMemory => Some(Rejection::Memory),
        ErrorCode::PromptCompilationFailed => Some(Rejection::Prompt),
        ErrorCode::ServiceDraining | ErrorCode::ServiceUnloaded => Some(Rejection::Other),
        _ => None,
    };
    if let Some(kind) = kind {
        metrics
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .note_rejection(kind);
    }
}

pub fn duration_outcome(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::InsufficientMemory
        | ErrorCode::ContextLimitExceeded
        | ErrorCode::PromptCompilationFailed
        | ErrorCode::ServiceDraining
        | ErrorCode::ServiceUnloaded => "rejected",
        ErrorCode::RequestCancelled => "cancelled",
        _ => "error",
    }
}

pub fn finish_outcome(reason: &str) -> &'static str {
    match reason {
        "stop" => "stop",
        "length" => "length",
        "cancelled" => "cancelled",
        _ => "error",
    }
}

/// `worker_up` is the process that answered, or the last ready check.
/// Live gauges are zero while that process is down. Counters keep the last
/// snapshot so a crash does not pretend the work never happened. A worker
/// restart replaces the snapshot, which a Prometheus `rate` treats as a reset.
pub fn render(host: &HostMetrics, worker_up: bool, stale: bool) -> String {
    let worker = host.snapshot.clone().unwrap_or_default();
    let zero_gauges = !worker_up;
    let mut out = String::new();
    gauge(
        &mut out,
        "knolo_infer_worker_up",
        "1 when the worker process is ready.",
        "",
        u64::from(worker_up),
    );
    gauge(
        &mut out,
        "knolo_infer_metrics_stale",
        "1 when this scrape kept the previous worker snapshot.",
        "",
        u64::from(stale),
    );
    counter(
        &mut out,
        "knolo_infer_worker_restarts_total",
        "Ready workers that left the ready state.",
        "",
        host.restarts,
    );
    help(
        &mut out,
        "knolo_infer_queue_depth",
        "gauge",
        "Sequences still queued, by service class.",
    );
    for (index, class) in CLASS_LABELS.iter().enumerate() {
        let value = if zero_gauges {
            0
        } else {
            match index {
                0 => worker.queue_interactive,
                1 => worker.queue_standard,
                2 => worker.queue_batch,
                _ => worker.queue_background,
            }
        };
        sample(
            &mut out,
            "knolo_infer_queue_depth",
            &format!("{{class=\"{class}\"}}"),
            value,
        );
    }
    help(
        &mut out,
        "knolo_infer_admission_rejected_total",
        "counter",
        "Admission rejections by stable error code.",
    );
    for (code, value) in [
        (
            "CONTRACT_INVALID",
            host.rejected_contract
                .saturating_add(worker.rejected_contract),
        ),
        (
            "CONTEXT_LIMIT_EXCEEDED",
            host.rejected_context
                .saturating_add(worker.rejected_context),
        ),
        (
            "INSUFFICIENT_MEMORY",
            host.rejected_memory.saturating_add(worker.rejected_memory),
        ),
        (
            "PROMPT_COMPILATION_FAILED",
            host.rejected_prompt.saturating_add(worker.rejected_prompt),
        ),
        (
            "OTHER",
            host.rejected_other.saturating_add(worker.rejected_other),
        ),
    ] {
        sample(
            &mut out,
            "knolo_infer_admission_rejected_total",
            &format!("{{code=\"{code}\"}}"),
            value,
        );
    }
    gauge(
        &mut out,
        "knolo_infer_model_load_seconds",
        "Seconds to verify weights and build the resident model.",
        "",
        if zero_gauges {
            "0.000000000".to_string()
        } else {
            format_nanos(worker.model_load_nanos)
        },
    );
    gauge(
        &mut out,
        "knolo_infer_verified_bytes",
        "Weight bytes accepted after the file hash matched.",
        "",
        if zero_gauges {
            0
        } else {
            worker.verified_bytes
        },
    );
    help(
        &mut out,
        "knolo_infer_time_to_first_token_seconds",
        "histogram",
        "Seconds from submit until the first output token.",
    );
    for (index, class) in CLASS_LABELS.iter().enumerate() {
        emit_histogram(
            &mut out,
            "knolo_infer_time_to_first_token_seconds",
            &format!("class=\"{class}\""),
            &host.ttft[index],
        );
    }
    counter(
        &mut out,
        "knolo_infer_prefill_tokens_total",
        "Prompt tokens processed by the scheduler.",
        "",
        worker.prefill_tokens,
    );
    counter(
        &mut out,
        "knolo_infer_prefill_seconds_total",
        "Seconds spent in prefill forwards.",
        "",
        format_nanos(worker.prefill_nanos),
    );
    counter(
        &mut out,
        "knolo_infer_decode_tokens_total",
        "Output tokens sampled by the scheduler.",
        "",
        worker.decode_tokens,
    );
    counter(
        &mut out,
        "knolo_infer_decode_seconds_total",
        "Seconds spent sampling output tokens.",
        "",
        format_nanos(worker.decode_nanos),
    );
    gauge(
        &mut out,
        "knolo_infer_active_sequences",
        "Sequences that have not finished.",
        "",
        if zero_gauges {
            0
        } else {
            worker.active_sequences
        },
    );
    gauge(
        &mut out,
        "knolo_infer_retained_sequences",
        "Sequences the scheduler still holds, including finished sequences not yet dropped.",
        "",
        if zero_gauges {
            0
        } else {
            worker.retained_sequences
        },
    );
    help(
        &mut out,
        "knolo_infer_kv_pages",
        "gauge",
        "KV pages in the worker pool. Pinned pages belong to active sequences.",
    );
    for (state, value) in [
        ("total", worker.kv_total),
        ("free", worker.kv_free),
        ("pinned", worker.kv_pinned),
    ] {
        let value = if zero_gauges { 0 } else { value };
        sample(
            &mut out,
            "knolo_infer_kv_pages",
            &format!("{{state=\"{state}\"}}"),
            value,
        );
    }
    counter(
        &mut out,
        "knolo_infer_prefix_cache_lookups_total",
        "Prefix lookups. Prefix cache is not allocated, so this stays 0.",
        "",
        worker.prefix_lookups,
    );
    counter(
        &mut out,
        "knolo_infer_prefix_cache_hits_total",
        "Prefix hits. Prefix cache is not allocated, so this stays 0.",
        "",
        worker.prefix_hits,
    );
    counter(
        &mut out,
        "knolo_infer_prefix_reused_tokens_total",
        "Tokens reused from a prefix. Prefix cache is not allocated, so this stays 0.",
        "",
        worker.prefix_reused_tokens,
    );
    help(
        &mut out,
        "knolo_infer_oom_total",
        "counter",
        "Memory admission failures and CUDA out-of-memory events.",
    );
    sample(
        &mut out,
        "knolo_infer_oom_total",
        "{kind=\"memory\"}",
        worker.oom,
    );
    sample(&mut out, "knolo_infer_oom_total", "{kind=\"cuda\"}", 0);
    help(
        &mut out,
        "knolo_infer_receipt_finalize_seconds",
        "histogram",
        "Seconds from the worker result until the receipt is stored.",
    );
    emit_histogram(
        &mut out,
        "knolo_infer_receipt_finalize_seconds",
        "",
        &host.receipt,
    );
    counter(
        &mut out,
        "knolo_infer_cancellations_total",
        "Sequences the scheduler retired as cancelled.",
        "",
        worker.cancellations,
    );
    help(
        &mut out,
        "knolo_infer_request_duration_seconds",
        "histogram",
        "Seconds from admission until the supervisor finishes the request.",
    );
    for (class_index, class) in CLASS_LABELS.iter().enumerate() {
        for (outcome_index, outcome) in OUTCOME_LABELS.iter().enumerate() {
            emit_histogram(
                &mut out,
                "knolo_infer_request_duration_seconds",
                &format!("class=\"{class}\",outcome=\"{outcome}\""),
                &host.requests[class_index][outcome_index],
            );
        }
    }
    help(
        &mut out,
        "knolo_infer_scheduler_iteration_seconds",
        "histogram",
        "Seconds spent in one scheduler iteration.",
    );
    emit_iteration(&mut out, &worker);
    out
}

fn emit_iteration(out: &mut String, worker: &WorkerStats) {
    let mut cumulative = 0u64;
    for (label, bin) in ITERATION_LABELS.iter().zip(worker.iteration_buckets) {
        cumulative = cumulative.saturating_add(bin);
        sample(
            out,
            "knolo_infer_scheduler_iteration_seconds_bucket",
            &format!("{{le=\"{label}\"}}"),
            cumulative,
        );
    }
    sample(
        out,
        "knolo_infer_scheduler_iteration_seconds_sum",
        "",
        format_nanos(worker.iteration_nanos),
    );
    sample(
        out,
        "knolo_infer_scheduler_iteration_seconds_count",
        "",
        worker.iterations,
    );
}

fn emit_histogram(out: &mut String, name: &str, labels: &str, hist: &Hist) {
    let mut cumulative = 0u64;
    for (label, bin) in LATENCY_LABELS.iter().zip(hist.buckets) {
        cumulative = cumulative.saturating_add(bin);
        let rendered = if labels.is_empty() {
            format!("{{le=\"{label}\"}}")
        } else {
            format!("{{{labels},le=\"{label}\"}}")
        };
        sample(out, &format!("{name}_bucket"), &rendered, cumulative);
    }
    let suffix = if labels.is_empty() {
        String::new()
    } else {
        format!("{{{labels}}}")
    };
    sample(
        out,
        &format!("{name}_sum"),
        &suffix,
        format_nanos(hist.sum_nanos),
    );
    sample(out, &format!("{name}_count"), &suffix, hist.count);
}

fn help(out: &mut String, name: &str, kind: &str, text: &str) {
    out.push_str("# HELP ");
    out.push_str(name);
    out.push(' ');
    out.push_str(text);
    out.push('\n');
    out.push_str("# TYPE ");
    out.push_str(name);
    out.push(' ');
    out.push_str(kind);
    out.push('\n');
}

fn gauge(out: &mut String, name: &str, text: &str, labels: &str, value: impl RenderNum) {
    help(out, name, "gauge", text);
    sample(out, name, labels, value);
}

fn counter(out: &mut String, name: &str, text: &str, labels: &str, value: impl RenderNum) {
    help(out, name, "counter", text);
    sample(out, name, labels, value);
}

fn sample(out: &mut String, name: &str, labels: &str, value: impl RenderNum) {
    out.push_str(name);
    out.push_str(labels);
    out.push(' ');
    value.push_to(out);
    out.push('\n');
}

trait RenderNum {
    fn push_to(&self, out: &mut String);
}

impl RenderNum for u64 {
    fn push_to(&self, out: &mut String) {
        out.push_str(&self.to_string());
    }
}

impl RenderNum for String {
    fn push_to(&self, out: &mut String) {
        out.push_str(self);
    }
}

fn format_nanos(nanos: u64) -> String {
    let whole = nanos / 1_000_000_000;
    let frac = nanos % 1_000_000_000;
    format!("{whole}.{frac:09}")
}

fn duration_nanos(elapsed: Duration) -> u64 {
    u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

fn class_index(class: ServiceClass) -> usize {
    match class {
        ServiceClass::Interactive => 0,
        ServiceClass::Standard => 1,
        ServiceClass::Batch => 2,
        ServiceClass::Background => 3,
    }
}

fn outcome_index(outcome: &str) -> usize {
    OUTCOME_LABELS
        .iter()
        .position(|label| *label == outcome)
        .unwrap_or(3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposition_uses_fixed_labels_and_cumulative_buckets() {
        let mut host = HostMetrics {
            restarts: 2,
            ..HostMetrics::default()
        };
        host.observe_request(ServiceClass::Interactive, "stop", 2_000_000);
        host.observe_ttft(ServiceClass::Interactive, 500_000);
        host.observe_receipt(20_000_000);
        let mut iteration_buckets = [0; infer_engine::ITERATION_BUCKETS];
        iteration_buckets[0] = 1;
        iteration_buckets[10] = 1;
        let worker = WorkerStats {
            queue_interactive: 1,
            kv_total: 8,
            kv_free: 7,
            kv_pinned: 1,
            verified_bytes: 70,
            model_load_nanos: 1_500_000,
            decode_tokens: 2,
            iteration_buckets,
            iterations: 2,
            rejected_memory: 1,
            oom: 1,
            retained_sequences: 3,
            ..WorkerStats::default()
        };
        let host = HostMetrics {
            snapshot: Some(worker),
            ..host
        };
        let text = render(&host, true, false);
        assert!(text.contains("knolo_infer_worker_up 1\n"));
        assert!(text.contains("knolo_infer_metrics_stale 0\n"));
        assert!(text.contains("knolo_infer_worker_restarts_total 2\n"));
        assert!(text.contains("knolo_infer_queue_depth{class=\"interactive\"} 1\n"));
        assert!(text.contains("knolo_infer_queue_depth{class=\"background\"} 0\n"));
        assert!(
            text.contains("knolo_infer_admission_rejected_total{code=\"INSUFFICIENT_MEMORY\"} 1\n")
        );
        assert!(text.contains("knolo_infer_verified_bytes 70\n"));
        assert!(text.contains("knolo_infer_model_load_seconds 0.001500000\n"));
        assert!(text.contains("knolo_infer_kv_pages{state=\"pinned\"} 1\n"));
        assert!(text.contains("knolo_infer_retained_sequences 3\n"));
        assert!(text.contains("knolo_infer_prefix_reused_tokens_total 0\n"));
        assert!(text.contains("knolo_infer_oom_total{kind=\"cuda\"} 0\n"));
        assert!(text.contains(
            "knolo_infer_request_duration_seconds_bucket{class=\"interactive\",outcome=\"stop\",le=\"0.001\"} 0\n"
        ));
        assert!(text.contains(
            "knolo_infer_request_duration_seconds_bucket{class=\"interactive\",outcome=\"stop\",le=\"0.005\"} 1\n"
        ));
        assert!(text.contains(
            "knolo_infer_request_duration_seconds_count{class=\"interactive\",outcome=\"stop\"} 1\n"
        ));
        assert!(text.contains("knolo_infer_scheduler_iteration_seconds_bucket{le=\"0.0001\"} 1\n"));
        assert!(text.contains("knolo_infer_scheduler_iteration_seconds_bucket{le=\"+Inf\"} 2\n"));
        assert!(!text.contains("job-"));
        assert!(!text.contains("prompt"));
        let down = render(&host, false, false);
        assert!(down.contains("knolo_infer_worker_up 0\n"));
        assert!(down.contains("knolo_infer_queue_depth{class=\"interactive\"} 0\n"));
        assert!(down.contains("knolo_infer_verified_bytes 0\n"));
        assert!(down.contains("knolo_infer_decode_tokens_total 2\n"));
        assert!(down.contains("knolo_infer_kv_pages{state=\"total\"} 0\n"));
        assert!(down.contains("knolo_infer_retained_sequences 0\n"));
    }
}
