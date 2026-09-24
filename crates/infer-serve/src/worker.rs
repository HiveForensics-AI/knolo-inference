//! One model, one KV pool, and the continuous scheduler.
//!
//! Without the `cuda` feature the model is the reference oracle. With that
//! feature the weights run on Candle device ordinal 0 and the KV pages stay
//! on the host. Frames already buffered are applied before the next forward,
//! so a queued cancel retires the sequence before tokens are sampled. The
//! process does not bind a TCP port. Reads use a short timeout so writes stay
//! blocking.

use std::collections::{HashMap, HashSet};
use std::io::{ErrorKind, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use infer_contracts::{fail, CborValue, ErrorCode, InferFailure};
#[cfg(not(feature = "cuda"))]
use infer_engine::{accept_cpu_placement, cpu_placement, MicroAdapter, ReferenceF32Backend};
#[cfg(feature = "cuda")]
use infer_engine::{accept_cuda_placement, cuda_placement, require_cuda_slot0, ADAPTER_ID};
use infer_engine::{
    load_verified_micro, micro_kv_layout, ArchitectureAdapter, CpuScheduler, ExecutableModel,
    PagedKv, ScheduleOp, ScheduleRequest, SchedulerConfig, VerifiedWeightSource, BLOCK_SIZE,
    CPU_KV_PAGE_POOL, MAX_CONTEXT,
};
#[cfg(feature = "cuda")]
use infer_native::{cpu_adapter_by_id, CandleCudaBackend};

use crate::frame::FrameDecoder;
use crate::model::resolve_pin;
use crate::protocol::{
    decode_to_worker, encode_from_worker, write_message, FromWorker, ToWorker, WorkerStats,
};
pub(crate) const PREFILL_CHUNK_TOKENS: u32 = 4;
const GATE_ENV: &str = "KNOLO_INFER_WORKER_GATE";
const POLL: Duration = Duration::from_millis(1);

struct Session {
    scheduler: CpuScheduler,
    model: Box<dyn ExecutableModel>,
    kv: PagedKv,
    runtime_root: String,
    model_load_nanos: u64,
    verified_bytes: u64,
    prefill_chunks: HashMap<String, u32>,
}

pub fn worker_main(args: impl Iterator<Item = String>) -> Result<(), InferFailure> {
    match run_worker(args) {
        Ok(()) => Ok(()),
        Err(err) => {
            eprintln!("knolo-infer-worker: {err}");
            let _ = std::io::stderr().flush();
            Err(err)
        }
    }
}

fn run_worker(args: impl Iterator<Item = String>) -> Result<(), InferFailure> {
    let config = WorkerConfig::parse(args)?;
    let started = Instant::now();
    let pinned = resolve_pin(
        &config.work_dir,
        &config.lock_path,
        &config.alias,
        config.weights_dir.as_deref(),
    )?;
    let source = load_verified_micro(&pinned.kmodel, &pinned.weights)?;
    if source.runtime_root.as_str() != pinned.runtime_root {
        return Err(fail(
            ErrorCode::ModelDigestMismatch,
            "worker runtime root changed while opening weights",
        ));
    }
    let verified_bytes = source.weight_bytes;
    let model = build_model(&source)?;
    let model_load_nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let kv = PagedKv::new(micro_kv_layout(), CPU_KV_PAGE_POOL)?;
    let scheduler = CpuScheduler::new(SchedulerConfig::new(
        source.runtime_root.as_str(),
        MAX_CONTEXT,
        BLOCK_SIZE,
        CPU_KV_PAGE_POOL,
        PREFILL_CHUNK_TOKENS,
    )?);
    let session = Session {
        scheduler,
        model,
        kv,
        runtime_root: source.runtime_root.to_string(),
        model_load_nanos,
        verified_bytes,
        prefill_chunks: HashMap::new(),
    };
    let mut stream = UnixStream::connect(&config.socket).map_err(|err| {
        fail(
            ErrorCode::WorkerStartFailed,
            format!("worker could not connect to the supervisor: {err}"),
        )
    })?;
    let paused = std::env::var(GATE_ENV).ok().as_deref() == Some("1");
    serve_connection(&mut stream, session, paused)
}

#[cfg(not(feature = "cuda"))]
fn build_model(source: &VerifiedWeightSource) -> Result<Box<dyn ExecutableModel>, InferFailure> {
    let placement = cpu_placement(source)?;
    accept_cpu_placement(source, &placement)?;
    MicroAdapter.build(source, &placement, &ReferenceF32Backend)
}

#[cfg(feature = "cuda")]
fn build_model(source: &VerifiedWeightSource) -> Result<Box<dyn ExecutableModel>, InferFailure> {
    require_cuda_slot0()?;
    let placement = cuda_placement(source)?;
    accept_cuda_placement(source, &placement)?;
    cpu_adapter_by_id(ADAPTER_ID)?.build(source, &placement, &CandleCudaBackend)
}

fn serve_connection(
    stream: &mut UnixStream,
    mut session: Session,
    mut paused: bool,
) -> Result<(), InferFailure> {
    let ready = FromWorker::Ready {
        runtime_root: session.runtime_root.clone(),
    };
    write_message(&*stream, &encode_from_worker(&ready)?)?;
    let mut decoder = FrameDecoder::new();
    loop {
        match drain(stream, &mut decoder, &mut session, &mut paused)? {
            Drain::Shutdown | Drain::Eof => return Ok(()),
            Drain::Partial => {
                if !read_blocking(stream, &mut decoder)? {
                    return Ok(());
                }
            }
            Drain::Open => {
                if paused {
                    if !read_blocking(stream, &mut decoder)? {
                        return Ok(());
                    }
                    continue;
                }
                let stepped = session
                    .scheduler
                    .step(&mut *session.model, &mut session.kv)?;
                report_steps(stream, &mut session, stepped.as_ref())?;
                report_new(stream, &mut session)?;
                if stepped.is_none() && !read_blocking(stream, &mut decoder)? {
                    return Ok(());
                }
            }
        }
    }
}

enum Drain {
    Shutdown,
    Eof,
    Partial,
    Open,
}

fn drain(
    stream: &mut UnixStream,
    decoder: &mut FrameDecoder,
    session: &mut Session,
    paused: &mut bool,
) -> Result<Drain, InferFailure> {
    stream.set_read_timeout(Some(POLL)).map_err(io_worker)?;
    loop {
        while let Some(value) = decoder.pop()? {
            if handle_frame(stream, session, paused, &value)? {
                return Ok(Drain::Shutdown);
            }
        }
        let mut buf = [0u8; 8192];
        match stream.read(&mut buf) {
            Ok(0) => return Ok(Drain::Eof),
            Ok(n) => decoder.push(&buf[..n])?,
            Err(err) if is_timeout(err.kind()) => {
                return Ok(if decoder.has_unread() {
                    Drain::Partial
                } else {
                    Drain::Open
                });
            }
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(io_worker(err)),
        }
    }
}

fn read_blocking(
    stream: &mut UnixStream,
    decoder: &mut FrameDecoder,
) -> Result<bool, InferFailure> {
    stream.set_read_timeout(None).map_err(io_worker)?;
    let mut buf = [0u8; 8192];
    loop {
        match stream.read(&mut buf) {
            Ok(0) => return Ok(false),
            Ok(n) => {
                decoder.push(&buf[..n])?;
                return Ok(true);
            }
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(err) => return Err(io_worker(err)),
        }
    }
}

fn handle_frame(
    stream: &mut UnixStream,
    session: &mut Session,
    paused: &mut bool,
    value: &CborValue,
) -> Result<bool, InferFailure> {
    match decode_to_worker(value)? {
        ToWorker::Shutdown => {
            session.kv.invalidate();
            Ok(true)
        }
        ToWorker::Release => {
            if std::env::var(GATE_ENV).ok().as_deref() != Some("1") {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "release is only accepted by a paused worker",
                ));
            }
            *paused = false;
            Ok(false)
        }
        ToWorker::Submit {
            request_id,
            prompt,
            sampler,
            class,
        } => {
            let request = ScheduleRequest {
                request_id: request_id.clone(),
                runtime_root: session.runtime_root.clone(),
                prompt,
                sampler: *sampler,
                class,
            };
            if let Err(err) = session.scheduler.submit(request) {
                send_reject(stream, &request_id, &err)?;
            } else {
                let admitted = FromWorker::Admitted { request_id };
                write_message(&*stream, &encode_from_worker(&admitted)?)?;
            }
            Ok(false)
        }
        ToWorker::Cancel { request_id } => {
            if let Err(err) = session.scheduler.cancel(&request_id) {
                send_reject(stream, &request_id, &err)?;
            } else {
                let ack = FromWorker::Ack { request_id };
                write_message(&*stream, &encode_from_worker(&ack)?)?;
            }
            Ok(false)
        }
        ToWorker::Stats => {
            let message = FromWorker::Stats(Box::new(worker_stats(session)));
            write_message(&*stream, &encode_from_worker(&message)?)?;
            Ok(false)
        }
    }
}

fn worker_stats(session: &Session) -> WorkerStats {
    let counters = session.scheduler.counters();
    let census = session.kv.census();
    WorkerStats {
        queue_interactive: u64::from(counters.queue_interactive),
        queue_standard: u64::from(counters.queue_standard),
        queue_batch: u64::from(counters.queue_batch),
        queue_background: u64::from(counters.queue_background),
        active_sequences: u64::from(counters.active_sequences),
        rejected_contract: counters.rejected_contract,
        rejected_context: counters.rejected_context,
        rejected_memory: counters.rejected_memory,
        rejected_prompt: counters.rejected_prompt,
        rejected_other: counters.rejected_other,
        cancellations: counters.cancellations,
        oom: counters.oom,
        prefill_tokens: counters.prefill_tokens,
        prefill_nanos: counters.prefill_nanos,
        decode_tokens: counters.decode_tokens,
        decode_nanos: counters.decode_nanos,
        iterations: counters.iterations,
        iteration_nanos: counters.iteration_nanos,
        iteration_buckets: counters.iteration_buckets,
        kv_total: u64::from(census.total),
        kv_free: u64::from(census.free),
        kv_pinned: u64::from(census.pinned),
        retained_sequences: u64::try_from(session.scheduler.retained()).unwrap_or(u64::MAX),
        prefix_lookups: counters.prefix_lookups,
        prefix_hits: counters.prefix_hits,
        prefix_reused_tokens: counters.prefix_reused_tokens,
        model_load_nanos: session.model_load_nanos,
        verified_bytes: session.verified_bytes,
    }
}

fn report_steps(
    stream: &mut UnixStream,
    session: &mut Session,
    step: Option<&infer_engine::ScheduleStep>,
) -> Result<(), InferFailure> {
    let Some(step) = step else {
        return Ok(());
    };
    let mut seen = HashSet::new();
    for op in &step.ops {
        match op {
            ScheduleOp::Prefill {
                sequence_id,
                tokens,
            } => {
                let request_id = session.scheduler.request_id(*sequence_id)?;
                let chunk = session
                    .prefill_chunks
                    .get(&request_id)
                    .copied()
                    .unwrap_or(0);
                session
                    .prefill_chunks
                    .insert(request_id.clone(), chunk.saturating_add(1));
                let message = FromWorker::Prefill {
                    request_id,
                    chunk,
                    tokens: *tokens,
                };
                write_message(&*stream, &encode_from_worker(&message)?)?;
            }
            ScheduleOp::Decode {
                sequence_id,
                token_id,
            } => {
                if !seen.insert(*sequence_id) {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "scheduler sampled one sequence twice in one step",
                    ));
                }
                let (request_id, index, issued) = session.scheduler.last_output(*sequence_id)?;
                if issued != *token_id {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "scheduler token does not match the decode step",
                    ));
                }
                let message = FromWorker::Delta {
                    request_id,
                    index,
                    token_id: *token_id,
                };
                write_message(&*stream, &encode_from_worker(&message)?)?;
            }
            ScheduleOp::Cancel { .. } | ScheduleOp::Fail { .. } => {}
        }
    }
    Ok(())
}

fn report_new(stream: &mut UnixStream, session: &mut Session) -> Result<(), InferFailure> {
    let pending = session.scheduler.results();
    for result in pending {
        session.prefill_chunks.remove(&result.request_id);
        let request_id = result.request_id.clone();
        let message = FromWorker::Result {
            request_id: result.request_id,
            tokens: result.tokens,
            finish_reason: result.finish_reason,
            error_code: result.error_code.map(|code| code.as_str().to_string()),
        };
        write_message(&*stream, &encode_from_worker(&message)?)?;
        session.scheduler.reap(&request_id)?;
    }
    Ok(())
}

fn send_reject(
    stream: &mut UnixStream,
    request_id: &str,
    err: &InferFailure,
) -> Result<(), InferFailure> {
    let mut message = err.message.clone();
    if message.len() > 1024 {
        message.truncate(1024);
    }
    if message.is_empty() {
        message = err.code.to_string();
    }
    let reject = FromWorker::Reject {
        request_id: request_id.to_string(),
        code: err.code.as_str().to_string(),
        message,
    };
    write_message(&*stream, &encode_from_worker(&reject)?)
}

fn is_timeout(kind: ErrorKind) -> bool {
    kind == ErrorKind::WouldBlock || kind == ErrorKind::TimedOut
}

fn io_worker(err: std::io::Error) -> InferFailure {
    fail(ErrorCode::WorkerLost, format!("worker ipc failed: {err}"))
}

struct WorkerConfig {
    socket: PathBuf,
    alias: String,
    lock_path: PathBuf,
    work_dir: PathBuf,
    weights_dir: Option<PathBuf>,
}

impl WorkerConfig {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, InferFailure> {
        let mut socket = None;
        let mut alias = None;
        let mut lock_path = None;
        let mut work_dir = None;
        let mut weights_dir = None;
        let mut args = args.peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--socket" => socket = Some(required(&mut args, "--socket")?),
                "--model" => alias = Some(required(&mut args, "--model")?),
                "--lock" => lock_path = Some(required(&mut args, "--lock")?),
                "--work-dir" => work_dir = Some(required(&mut args, "--work-dir")?),
                "--weights" => weights_dir = Some(required(&mut args, "--weights")?),
                other => {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        format!("unknown worker argument {other}"),
                    ))
                }
            }
        }
        Ok(Self {
            socket: PathBuf::from(socket.ok_or_else(|| usage("worker requires --socket"))?),
            alias: alias.ok_or_else(|| usage("worker requires --model"))?,
            lock_path: PathBuf::from(lock_path.ok_or_else(|| usage("worker requires --lock"))?),
            work_dir: PathBuf::from(work_dir.ok_or_else(|| usage("worker requires --work-dir"))?),
            weights_dir: weights_dir.map(PathBuf::from),
        })
    }
}

fn required(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<String, InferFailure> {
    match args.next() {
        Some(value) if !value.starts_with('-') => Ok(value),
        _ => Err(usage(format!("{flag} needs a value"))),
    }
}

fn usage(message: impl Into<String>) -> InferFailure {
    fail(ErrorCode::ContractInvalid, message)
}
