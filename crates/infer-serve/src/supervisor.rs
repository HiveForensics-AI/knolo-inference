//! Loopback supervisor. The listener stays up when the worker exits.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{ErrorKind, Read};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use infer_artifact::parse_strict_json;
use infer_contracts::{fail, ErrorCode, InferFailure};
use infer_engine::{ServiceClass, ADAPTER_ID, MAX_CONTEXT, VOCAB};
use infer_prompt::{compile_model_prompt, decode_tokens, parse_tokenizer};
use serde_json::{json, Map, Value};

use crate::daemon_lock::DaemonLock;
use crate::frame::FrameDecoder;
use crate::http::{self, HttpRequest};
use crate::metrics::{self, duration_outcome, finish_outcome, RequestWatch};
use crate::model::{build_sampler, resolve_pin, GenerationRequest, PinnedModel};
use crate::openai;
use crate::protocol::{
    check_request_id, decode_from_worker, encode_to_worker, parse_class, write_message, FromWorker,
    ToWorker, WorkerStats,
};

static IPC_DIRS: AtomicU64 = AtomicU64::new(0);

const MAX_INFLIGHT: usize = 64;
const MAX_HANDLERS: usize = 128;
const MAX_MESSAGES: usize = 32;
const REQUEST_WAIT: Duration = Duration::from_secs(30);
const ADMIT_WAIT: Duration = Duration::from_secs(5);
const READY_WAIT: Duration = Duration::from_secs(5);
const DRAIN_WAIT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub alias: String,
    pub work_dir: PathBuf,
    pub lock_path: PathBuf,
    pub weights_dir: Option<PathBuf>,
    pub bind: SocketAddr,
    pub worker_bin: PathBuf,
    pub home: PathBuf,
    /// When set, the worker admits requests and does not sample until
    /// [`Supervisor::release_forward`]. `knolo-infer serve` leaves this off.
    pub pause_before_forward: bool,
}

enum WorkerState {
    Starting,
    Ready,
    Down,
}

enum Outcome {
    Result {
        tokens: Vec<u32>,
        finish_reason: String,
        error_code: Option<String>,
    },
    Rejected {
        code: String,
        message: String,
    },
    Ack,
    Lost,
}

enum WorkerEvent {
    Admitted,
    Prefill { chunk: u32, tokens: u32 },
    Delta { index: u32, token_id: u32 },
    Finished(Outcome),
}

struct Plan {
    request_id: String,
    prompt: Vec<u32>,
    prompt_plan: infer_contracts::PromptPlanV1,
    sampler: Box<infer_contracts::SamplerPlanV1>,
    class: ServiceClass,
    prompt_tokens: u32,
    stream: bool,
    trace: crate::trace::RequestTrace,
}

struct WorkerSlot {
    generation: u64,
    state: WorkerState,
    child: Option<Child>,
    writer: Option<UnixStream>,
    reader: Option<JoinHandle<()>>,
    dir: Option<PathBuf>,
    completion: HashMap<String, Sender<WorkerEvent>>,
    cancel: HashMap<String, Sender<Outcome>>,
    stats: Option<Sender<Result<WorkerStats, ()>>>,
    released: bool,
    release_sent: bool,
    draining: bool,
    unloading: bool,
    reserved: usize,
}

struct Inner {
    config: ServeConfig,
    pinned: PinnedModel,
    home: PathBuf,
    identity: crate::receipt::ServeIdentity,
    worker: Mutex<WorkerSlot>,
    metrics: Mutex<metrics::HostMetrics>,
    trace: crate::trace::Trace,
    scrape: Mutex<()>,
    ready: Condvar,
    shutdown: AtomicBool,
    handlers: AtomicUsize,
    next_id: AtomicU64,
}

pub struct Supervisor {
    inner: Arc<Inner>,
    http_thread: Mutex<Option<JoinHandle<()>>>,
    address: SocketAddr,
    daemon_lock: DaemonLock,
}

impl Supervisor {
    pub fn start(mut config: ServeConfig) -> Result<Self, InferFailure> {
        check_loopback(config.bind)?;
        if !config.worker_bin.is_file() {
            return Err(fail(
                ErrorCode::WorkerStartFailed,
                format!(
                    "worker binary is missing at {}",
                    config.worker_bin.display()
                ),
            ));
        }
        config.work_dir = fs::canonicalize(&config.work_dir).map_err(|err| {
            fail(
                ErrorCode::ModelArtifactMissing,
                format!("work directory: {err}"),
            )
        })?;
        if config.lock_path.is_relative() {
            config.lock_path = config.work_dir.join(&config.lock_path);
        }
        if let Some(weights) = config.weights_dir.take() {
            config.weights_dir = Some(if weights.is_relative() {
                config.work_dir.join(weights)
            } else {
                weights
            });
        }
        let pinned = resolve_pin(
            &config.work_dir,
            &config.lock_path,
            &config.alias,
            config.weights_dir.as_deref(),
        )?;
        if pinned.image.architecture.adapter != ADAPTER_ID {
            return Err(fail(
                ErrorCode::UnsupportedArchitecture,
                "serve accepts knolo.micro.v1 only",
            ));
        }
        let home = if config.home.is_relative() {
            config.work_dir.join(&config.home)
        } else {
            config.home.clone()
        };
        fs::create_dir_all(&home).map_err(|err| {
            fail(
                ErrorCode::ReceiptPersistFailed,
                format!("serve home: {err}"),
            )
        })?;
        let home = fs::canonicalize(&home).map_err(|err| {
            fail(
                ErrorCode::ReceiptPersistFailed,
                format!("serve home: {err}"),
            )
        })?;
        config.home = home.clone();
        // The lock is taken before the listener so a held home never binds,
        // and an open journal is sealed before a new request is accepted.
        let daemon_lock = DaemonLock::acquire(&home)?;
        let identity = crate::receipt::prepare_serve(&home, &config.worker_bin, &pinned)?;
        let trace = crate::trace::Trace::open(&home)?;
        let listener = TcpListener::bind(config.bind).map_err(|err| {
            fail(
                ErrorCode::ContractInvalid,
                format!("supervisor listen failed: {err}"),
            )
        })?;
        listener.set_nonblocking(true).map_err(|err| {
            fail(
                ErrorCode::ContractInvalid,
                format!("supervisor listen failed: {err}"),
            )
        })?;
        let address = listener.local_addr().map_err(|err| {
            fail(
                ErrorCode::ContractInvalid,
                format!("supervisor listen failed: {err}"),
            )
        })?;
        let inner = Arc::new(Inner {
            config,
            pinned,
            home,
            identity,
            worker: Mutex::new(WorkerSlot {
                generation: 0,
                state: WorkerState::Down,
                child: None,
                writer: None,
                reader: None,
                dir: None,
                completion: HashMap::new(),
                cancel: HashMap::new(),
                stats: None,
                released: false,
                release_sent: false,
                draining: false,
                unloading: false,
                reserved: 0,
            }),
            metrics: Mutex::new(metrics::HostMetrics::default()),
            trace,
            scrape: Mutex::new(()),
            ready: Condvar::new(),
            shutdown: AtomicBool::new(false),
            handlers: AtomicUsize::new(0),
            next_id: AtomicU64::new(1),
        });
        let http_inner = Arc::clone(&inner);
        let http_thread = thread::spawn(move || accept_loop(listener, http_inner));
        let supervisor = Self {
            inner,
            http_thread: Mutex::new(Some(http_thread)),
            address,
            daemon_lock,
        };
        if let Err(err) = supervisor.inner.ensure_worker() {
            supervisor.shutdown();
            return Err(err);
        }
        Ok(supervisor)
    }

    pub fn address(&self) -> SocketAddr {
        self.address
    }

    pub fn worker_label(&self) -> &'static str {
        match lock_slot(&self.inner).state {
            WorkerState::Ready => "ready",
            WorkerState::Starting => "starting",
            WorkerState::Down => "down",
        }
    }

    pub fn worker_pid(&self) -> Option<u32> {
        lock_slot(&self.inner).child.as_ref().map(Child::id)
    }

    pub fn inflight(&self) -> usize {
        lock_slot(&self.inner).completion.len()
    }

    pub fn release_forward(&self) -> Result<(), InferFailure> {
        let mut slot = lock_slot(&self.inner);
        slot.released = true;
        if matches!(slot.state, WorkerState::Ready) && !slot.release_sent {
            write_worker(&mut slot, &ToWorker::Release)?;
            slot.release_sent = true;
        }
        Ok(())
    }

    pub fn kill_worker(&self) -> Result<(), InferFailure> {
        let mut slot = lock_slot(&self.inner);
        if let Some(child) = slot.child.as_mut() {
            child.kill().map_err(|err| {
                fail(
                    ErrorCode::WorkerLost,
                    format!("worker could not be stopped: {err}"),
                )
            })?;
        }
        Ok(())
    }

    /// Stop admitting completions, then wait for work that was already reserved.
    /// The listener stays up until [`Self::shutdown`].
    pub fn drain_for_shutdown(&self) {
        let _ = begin_drain(&self.inner);
        let deadline = Instant::now() + DRAIN_WAIT;
        while occupancy_of(&self.inner) > 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
    }

    pub fn shutdown(&self) {
        if self.inner.shutdown.swap(true, Ordering::SeqCst) {
            return;
        }
        stop_worker(&self.inner);
        let deadline = Instant::now() + Duration::from_secs(2);
        while self.inner.handlers.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        if let Some(thread) = self
            .http_thread
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .take()
        {
            let _ = thread.join();
        }
        self.daemon_lock.release();
    }
}

impl Drop for Supervisor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl Inner {
    fn ensure_worker(self: &Arc<Self>) -> Result<(), InferFailure> {
        loop {
            let mut slot = lock_slot(self);
            match slot.state {
                WorkerState::Ready => {
                    send_release_if_needed(&mut slot)?;
                    return Ok(());
                }
                WorkerState::Starting => {
                    let (guard, timeout) = self
                        .ready
                        .wait_timeout(slot, READY_WAIT)
                        .unwrap_or_else(|err| err.into_inner());
                    slot = guard;
                    if timeout.timed_out() && matches!(slot.state, WorkerState::Starting) {
                        return Err(fail(
                            ErrorCode::WorkerStartFailed,
                            "worker did not become ready",
                        ));
                    }
                }
                WorkerState::Down => {
                    if slot.unloading {
                        return Err(fail(ErrorCode::ServiceUnloaded, "model is not loaded"));
                    }
                    slot.state = WorkerState::Starting;
                    slot.generation = slot.generation.wrapping_add(1);
                    let generation = slot.generation;
                    let old_reader = slot.reader.take();
                    drop(slot);
                    if let Some(reader) = old_reader {
                        let _ = reader.join();
                    }
                    match spawn_worker(self, generation) {
                        Ok(live) => {
                            let mut slot = lock_slot(self);
                            if slot.generation != generation {
                                drop_live(live);
                                continue;
                            }
                            slot.child = Some(live.child);
                            slot.writer = Some(live.writer);
                            slot.reader = Some(live.reader);
                            slot.dir = Some(live.dir);
                            slot.stats = None;
                            slot.release_sent = false;
                            slot.state = WorkerState::Ready;
                            send_release_if_needed(&mut slot)?;
                            self.ready.notify_all();
                            return Ok(());
                        }
                        Err(err) => {
                            let mut slot = lock_slot(self);
                            if slot.generation == generation {
                                slot.state = WorkerState::Down;
                                self.ready.notify_all();
                            }
                            return Err(err);
                        }
                    }
                }
            }
        }
    }
}

struct Live {
    child: Child,
    writer: UnixStream,
    reader: JoinHandle<()>,
    dir: PathBuf,
}

struct DirGuard(PathBuf);

impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn spawn_worker(inner: &Arc<Inner>, generation: u64) -> Result<Live, InferFailure> {
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-ipc-{}-{}-{generation}",
        std::process::id(),
        IPC_DIRS.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir(&dir).map_err(|err| {
        fail(
            ErrorCode::WorkerStartFailed,
            format!("worker socket directory: {err}"),
        )
    })?;
    let guard = DirGuard(dir.clone());
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).map_err(|err| {
        fail(
            ErrorCode::WorkerStartFailed,
            format!("worker socket directory: {err}"),
        )
    })?;
    let socket_path = dir.join("worker.sock");
    if socket_path.as_os_str().len() >= 100 {
        return Err(fail(
            ErrorCode::WorkerStartFailed,
            "worker socket path is too long",
        ));
    }
    let listener = UnixListener::bind(&socket_path).map_err(|err| {
        fail(
            ErrorCode::WorkerStartFailed,
            format!("worker socket bind failed: {err}"),
        )
    })?;
    fs::set_permissions(&socket_path, fs::Permissions::from_mode(0o600)).map_err(|err| {
        fail(
            ErrorCode::WorkerStartFailed,
            format!("worker socket mode: {err}"),
        )
    })?;
    listener.set_nonblocking(true).map_err(|err| {
        fail(
            ErrorCode::WorkerStartFailed,
            format!("worker socket bind failed: {err}"),
        )
    })?;
    let stderr_path = dir.join("stderr.log");
    let stderr = File::create(&stderr_path)
        .map_err(|err| fail(ErrorCode::WorkerStartFailed, format!("worker log: {err}")))?;
    let mut child = Command::new(&inner.config.worker_bin)
        .arg("--socket")
        .arg(&socket_path)
        .arg("--model")
        .arg(&inner.config.alias)
        .arg("--lock")
        .arg(&inner.config.lock_path)
        .arg("--work-dir")
        .arg(&inner.config.work_dir)
        .args(
            inner
                .config
                .weights_dir
                .iter()
                .flat_map(|path| ["--weights".into(), path.display().to_string()]),
        )
        .env(
            "KNOLO_INFER_WORKER_GATE",
            if inner.config.pause_before_forward {
                "1"
            } else {
                "0"
            },
        )
        .current_dir(&inner.config.work_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|err| {
            fail(
                ErrorCode::WorkerStartFailed,
                format!("worker binary could not be started: {err}"),
            )
        })?;
    let stream = match accept_worker(&listener, &mut child, &stderr_path) {
        Ok(stream) => stream,
        Err(err) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(err);
        }
    };
    drop(listener);
    let ready = match read_ready(&mut child, &stream, &stderr_path) {
        Ok(ready) => ready,
        Err(err) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(err);
        }
    };
    if ready != inner.pinned.runtime_root {
        let _ = child.kill();
        let _ = child.wait();
        return Err(fail(
            ErrorCode::ModelDigestMismatch,
            "worker runtime root does not match the pinned model",
        ));
    }
    let reader_stream = stream.try_clone().map_err(|err| {
        let _ = child.kill();
        fail(
            ErrorCode::WorkerStartFailed,
            format!("worker socket could not be shared: {err}"),
        )
    })?;
    let reader_inner = Arc::clone(inner);
    let reader = thread::spawn(move || reader_loop(reader_stream, reader_inner, generation));
    std::mem::forget(guard);
    Ok(Live {
        child,
        writer: stream,
        reader,
        dir,
    })
}

fn accept_worker(
    listener: &UnixListener,
    child: &mut Child,
    stderr_path: &Path,
) -> Result<UnixStream, InferFailure> {
    let start = Instant::now();
    loop {
        match listener.accept() {
            Ok((stream, _)) => return Ok(stream),
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                if let Some(status) = child.try_wait().map_err(|err| {
                    fail(
                        ErrorCode::WorkerStartFailed,
                        format!("worker status: {err}"),
                    )
                })? {
                    return Err(fail(
                        ErrorCode::WorkerStartFailed,
                        format!(
                            "worker exited {status} before ready{}",
                            stderr_tail(stderr_path)
                        ),
                    ));
                }
                if start.elapsed() > READY_WAIT {
                    return Err(fail(
                        ErrorCode::WorkerStartFailed,
                        format!("worker did not connect{}", stderr_tail(stderr_path)),
                    ));
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(err) => {
                return Err(fail(
                    ErrorCode::WorkerStartFailed,
                    format!("worker socket accept failed: {err}"),
                ))
            }
        }
    }
}

fn read_ready(
    child: &mut Child,
    mut stream: &UnixStream,
    stderr_path: &Path,
) -> Result<String, InferFailure> {
    stream.set_read_timeout(Some(READY_WAIT)).map_err(|err| {
        fail(
            ErrorCode::WorkerStartFailed,
            format!("worker socket: {err}"),
        )
    })?;
    let value = crate::protocol::read_message(&mut stream).map_err(|err| {
        let status = child.try_wait().ok().flatten();
        fail(
            ErrorCode::WorkerStartFailed,
            format!(
                "worker ready frame: {err}{}{}",
                status
                    .map(|status| format!(" exit {status}"))
                    .unwrap_or_default(),
                stderr_tail(stderr_path)
            ),
        )
    })?;
    stream.set_read_timeout(None).map_err(|err| {
        fail(
            ErrorCode::WorkerStartFailed,
            format!("worker socket: {err}"),
        )
    })?;
    match decode_from_worker(&value)? {
        FromWorker::Ready { runtime_root } => Ok(runtime_root),
        _ => Err(fail(
            ErrorCode::WorkerStartFailed,
            "worker did not send ready",
        )),
    }
}

fn reader_loop(mut stream: UnixStream, inner: Arc<Inner>, generation: u64) {
    let mut decoder = FrameDecoder::new();
    let mut buf = [0u8; 8192];
    loop {
        if lock_slot(&inner).generation != generation {
            break;
        }
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                if decoder.push(&buf[..n]).is_err() {
                    break;
                }
                loop {
                    match decoder.pop() {
                        Ok(Some(value)) => {
                            if dispatch(&inner, generation, &value).is_err() {
                                mark_lost(&inner, generation);
                                return;
                            }
                        }
                        Ok(None) => break,
                        Err(_) => {
                            mark_lost(&inner, generation);
                            return;
                        }
                    }
                }
            }
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
    mark_lost(&inner, generation);
}

fn dispatch(inner: &Inner, generation: u64, value: &infer_contracts::CborValue) -> Result<(), ()> {
    let message = decode_from_worker(value).map_err(|_| ())?;
    let mut slot = lock_slot(inner);
    if slot.generation != generation {
        return Ok(());
    }
    match message {
        FromWorker::Ready { .. } => Ok(()),
        FromWorker::Admitted { request_id } => {
            if let Some(sender) = slot.completion.get(&request_id) {
                let _ = sender.send(WorkerEvent::Admitted);
            }
            Ok(())
        }
        FromWorker::Delta {
            request_id,
            index,
            token_id,
        } => {
            if let Some(sender) = slot.completion.get(&request_id) {
                let _ = sender.send(WorkerEvent::Delta { index, token_id });
            }
            Ok(())
        }
        FromWorker::Prefill {
            request_id,
            chunk,
            tokens,
        } => {
            if let Some(sender) = slot.completion.get(&request_id) {
                let _ = sender.send(WorkerEvent::Prefill { chunk, tokens });
            }
            Ok(())
        }
        FromWorker::Result {
            request_id,
            tokens,
            finish_reason,
            error_code,
        } => {
            if let Some(sender) = slot.completion.remove(&request_id) {
                let _ = sender.send(WorkerEvent::Finished(Outcome::Result {
                    tokens,
                    finish_reason,
                    error_code,
                }));
            }
            Ok(())
        }
        FromWorker::Reject {
            request_id,
            code,
            message,
        } => {
            let outcome = Outcome::Rejected { code, message };
            if let Some(sender) = slot.cancel.remove(&request_id) {
                let _ = sender.send(outcome);
            } else if let Some(sender) = slot.completion.remove(&request_id) {
                let _ = sender.send(WorkerEvent::Finished(outcome));
            }
            Ok(())
        }
        FromWorker::Ack { request_id } => {
            if let Some(sender) = slot.cancel.remove(&request_id) {
                let _ = sender.send(Outcome::Ack);
            }
            Ok(())
        }
        FromWorker::Stats(stats) => {
            if let Some(sender) = slot.stats.take() {
                let _ = sender.send(Ok(*stats));
            }
            Ok(())
        }
    }
}

fn mark_lost(inner: &Inner, generation: u64) {
    let was_ready = {
        let mut slot = lock_slot(inner);
        if slot.generation != generation {
            return;
        }
        let was_ready = matches!(slot.state, WorkerState::Ready);
        fail_waiters(&mut slot);
        if let Some(mut child) = slot.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        slot.writer = None;
        slot.state = WorkerState::Down;
        if let Some(dir) = slot.dir.take() {
            let _ = fs::remove_dir_all(dir);
        }
        inner.ready.notify_all();
        was_ready
    };
    if was_ready {
        let mut metrics = lock_metrics(inner);
        metrics.restarts = metrics.restarts.saturating_add(1);
    }
}

fn stop_worker(inner: &Inner) {
    let mut slot = lock_slot(inner);
    slot.generation = slot.generation.wrapping_add(1);
    let child = slot.child.take();
    let writer = slot.writer.take();
    let reader = slot.reader.take();
    let dir = slot.dir.take();
    slot.state = WorkerState::Down;
    fail_waiters(&mut slot);
    drop(slot);
    if let Some(mut writer) = writer {
        let _ = write_worker_stream(&mut writer, &ToWorker::Shutdown);
    }
    if let Some(mut child) = child {
        let deadline = Instant::now() + Duration::from_millis(200);
        loop {
            match child.try_wait() {
                Ok(Some(_)) => break,
                _ if Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break;
                }
                _ => thread::sleep(Duration::from_millis(10)),
            }
        }
    }
    if let Some(reader) = reader {
        if reader.thread().id() == thread::current().id() {
            drop(reader);
        } else {
            let _ = reader.join();
        }
    }
    if let Some(dir) = dir {
        let _ = fs::remove_dir_all(dir);
    }
}

fn fail_waiters(slot: &mut WorkerSlot) {
    if let Some(sender) = slot.stats.take() {
        let _ = sender.send(Err(()));
    }
    for (_, sender) in slot.completion.drain() {
        let _ = sender.send(WorkerEvent::Finished(Outcome::Lost));
    }
    for (_, sender) in slot.cancel.drain() {
        let _ = sender.send(Outcome::Lost);
    }
}

fn drop_live(live: Live) {
    let Live {
        mut child,
        writer,
        reader,
        dir,
    } = live;
    drop(writer);
    let _ = child.kill();
    let _ = child.wait();
    let _ = reader.join();
    let _ = fs::remove_dir_all(dir);
}

fn send_release_if_needed(slot: &mut WorkerSlot) -> Result<(), InferFailure> {
    if slot.released && !slot.release_sent && matches!(slot.state, WorkerState::Ready) {
        write_worker(slot, &ToWorker::Release)?;
        slot.release_sent = true;
    }
    Ok(())
}

fn write_worker(slot: &mut WorkerSlot, message: &ToWorker) -> Result<(), InferFailure> {
    let writer = slot
        .writer
        .as_mut()
        .ok_or_else(|| fail(ErrorCode::WorkerLost, "worker ipc is closed"))?;
    write_worker_stream(writer, message)
}

fn write_worker_stream(writer: &mut UnixStream, message: &ToWorker) -> Result<(), InferFailure> {
    write_message(writer, &encode_to_worker(message)?)
}

fn lock_slot(inner: &Inner) -> std::sync::MutexGuard<'_, WorkerSlot> {
    inner.worker.lock().unwrap_or_else(|err| err.into_inner())
}

fn lock_metrics(inner: &Inner) -> std::sync::MutexGuard<'_, metrics::HostMetrics> {
    inner.metrics.lock().unwrap_or_else(|err| err.into_inner())
}

fn accept_loop(listener: TcpListener, inner: Arc<Inner>) {
    while !inner.shutdown.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                if inner.handlers.load(Ordering::SeqCst) >= MAX_HANDLERS {
                    let body = error_bytes(&fail(
                        ErrorCode::InsufficientMemory,
                        "too many HTTP connections",
                    ));
                    let mut stream = stream;
                    let _ = http::write_response(&mut stream, 503, &body);
                    continue;
                }
                inner.handlers.fetch_add(1, Ordering::SeqCst);
                let inner = Arc::clone(&inner);
                thread::spawn(move || {
                    let _guard = HandlerGuard(&inner.handlers);
                    handle_connection(stream, &inner);
                });
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(20));
            }
            Err(err) if err.kind() == ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
}

struct HandlerGuard<'a>(&'a AtomicUsize);

impl Drop for HandlerGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn handle_connection(mut stream: TcpStream, inner: &Arc<Inner>) {
    if http::prepare_stream(&mut stream).is_err() {
        return;
    }
    let request = match http::read_request(&mut stream) {
        Ok(request) => request,
        Err(err) => {
            let status = http::status_for_failure(&err);
            let _ = http::write_response(&mut stream, status, &error_bytes(&err));
            return;
        }
    };
    if request.path == "/metrics" || request.path == "/knolo/infer/v1/metrics" {
        if request.method != "GET" {
            let err = fail(ErrorCode::ContractInvalid, "HTTP method is not allowed");
            let _ = http::write_response(&mut stream, 405, &error_bytes(&err));
            return;
        }
        let body = metrics_body(inner);
        let _ = http::write_text(
            &mut stream,
            200,
            "text/plain; version=0.0.4; charset=utf-8",
            &body,
        );
        return;
    }
    if request.method == "POST"
        && (request.path == "/knolo/infer/v1/complete" || request.path == "/v1/chat/completions")
    {
        match post_completion(&mut stream, inner, &request) {
            PostReply::Json {
                status,
                body,
                request_id,
                receipt,
            } => {
                let _ = write_json(
                    &mut stream,
                    status,
                    &body,
                    request_id.as_deref(),
                    receipt.as_deref(),
                );
            }
            PostReply::Sent => {}
        }
        return;
    }
    let (status, body) = route(inner, &request);
    let receipt = request.path.starts_with("/v1/").then_some("absent");
    let _ = write_json(&mut stream, status, &body, None, receipt);
}

enum PostReply {
    Json {
        status: u16,
        body: Vec<u8>,
        request_id: Option<String>,
        receipt: Option<String>,
    },
    Sent,
}

fn write_json(
    stream: &mut TcpStream,
    status: u16,
    body: &[u8],
    request_id: Option<&str>,
    receipt: Option<&str>,
) -> Result<(), InferFailure> {
    let mut extra = Vec::new();
    if let Some(receipt) = receipt {
        extra.push(("X-Knolo-Receipt", receipt));
    }
    if let Some(id) = request_id {
        extra.push(("X-Knolo-Request-Id", id));
    }
    if extra.is_empty() {
        return http::write_response(stream, status, body);
    }
    http::write_response_extra(stream, status, body, &extra)
}

fn post_completion(stream: &mut TcpStream, inner: &Arc<Inner>, request: &HttpRequest) -> PostReply {
    let _settle = UnloadGuard(inner);
    let mut hold = match AdmissionHold::reserve(inner) {
        Ok(hold) => hold,
        Err(err) => {
            inner.trace.reject(err.code);
            metrics::note_compile_failure(&inner.metrics, err.code);
            return json_failure(&request.path, None, &err);
        }
    };
    let started = Instant::now();
    let planned = if request.path == "/knolo/infer/v1/complete" {
        plan_native(inner, &request.body).map(|plan| (plan, false))
    } else {
        plan_openai(inner, &request.body).map(|plan| (plan, true))
    };
    let (mut plan, openai) = match planned {
        Ok(plan) => plan,
        Err(err) => {
            metrics::note_compile_failure(&inner.metrics, err.code);
            return json_failure(&request.path, None, &err);
        }
    };
    let mut watch = RequestWatch::new(&inner.metrics, plan.class, started);
    let mut opened = match open_plan(inner, &plan, openai) {
        Ok(opened) => opened,
        Err(err) => {
            watch.set_outcome(duration_outcome(err.code));
            plan.trace
                .finish(duration_outcome(err.code), Some(err.code), None);
            return json_failure(&request.path, Some(&plan.request_id), &err);
        }
    };
    if !plan.stream {
        return match finish_buffered(inner, &mut plan, openai, &mut opened, &mut watch, &mut hold) {
            Ok((body, receipt)) => PostReply::Json {
                status: 200,
                body,
                request_id: Some(plan.request_id.clone()),
                receipt: Some(receipt),
            },
            Err(err) => {
                if watch_outcome_missing(&watch) {
                    watch.set_outcome(duration_outcome(err.code));
                }
                json_failure(&request.path, Some(&plan.request_id), &err)
            }
        };
    }
    match serve_stream(
        stream,
        inner,
        &mut plan,
        openai,
        &mut opened,
        &mut watch,
        &mut hold,
    ) {
        Ok(()) => PostReply::Sent,
        Err(err) => {
            if watch_outcome_missing(&watch) {
                watch.set_outcome(duration_outcome(err.code));
            }
            json_failure(&request.path, Some(plan.request_id.as_str()), &err)
        }
    }
}

fn watch_outcome_missing(watch: &RequestWatch<'_>) -> bool {
    watch.outcome_is_unset()
}

fn open_plan(
    inner: &Inner,
    plan: &Plan,
    openai: bool,
) -> Result<crate::receipt::OpenedRequest, InferFailure> {
    crate::receipt::open_request(
        &inner.home,
        &inner.identity,
        &inner.pinned,
        &crate::receipt::RequestDraft {
            request_id: &plan.request_id,
            prompt: &plan.prompt_plan,
            sampler: &plan.sampler,
            class: crate::protocol::class_name(plan.class),
            openai,
            stream: plan.stream,
        },
    )
}

fn json_failure(path: &str, request_id: Option<&str>, err: &InferFailure) -> PostReply {
    let openai = path.starts_with("/v1/");
    let body = if openai {
        openai::error_document(err.code.as_str(), &err.message)
    } else {
        error_bytes(err)
    };
    let known = request_id.map(str::to_string);
    PostReply::Json {
        status: http::status_for_failure(err),
        body,
        request_id: known,
        receipt: (openai || request_id.is_some()).then(|| "absent".to_string()),
    }
}

fn route(inner: &Arc<Inner>, request: &HttpRequest) -> (u16, Vec<u8>) {
    if let Some(digest) = request.path.strip_prefix("/knolo/infer/v1/receipts/") {
        if request.method != "GET" {
            let err = fail(ErrorCode::ContractInvalid, "HTTP method is not allowed");
            return (405, error_bytes(&err));
        }
        return match crate::receipt::read_receipt(&inner.home, digest) {
            Ok(body) => (200, body),
            Err(err) => (http::status_for_failure(&err), error_bytes(&err)),
        };
    }
    let openai_path = request.path.starts_with("/v1/");
    let result = match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/knolo/infer/v1/health") => Ok(health_body(inner)),
        ("GET", "/v1/models") => Ok(openai::models_document(&inner.config.alias)),
        ("POST", "/knolo/infer/v1/cancel") => cancel(inner, &request.body),
        ("POST", "/knolo/infer/v1/drain") => drain_http(inner, &request.body),
        ("POST", "/knolo/infer/v1/models/unload") => unload_http(inner, &request.body),
        ("POST", "/knolo/infer/v1/models/load") => load_http(inner, &request.body),
        (
            _,
            "/knolo/infer/v1/health"
            | "/knolo/infer/v1/complete"
            | "/knolo/infer/v1/cancel"
            | "/knolo/infer/v1/drain"
            | "/knolo/infer/v1/models/unload"
            | "/knolo/infer/v1/models/load"
            | "/v1/models"
            | "/v1/chat/completions",
        ) => Err(fail(
            ErrorCode::ContractInvalid,
            "HTTP method is not allowed",
        )),
        _ => Err(fail(ErrorCode::ContractInvalid, "HTTP path is unknown")),
    };
    match result {
        Ok(body) => (200, body),
        Err(err) => {
            let mut status = http::status_for_failure(&err);
            if err.message == "HTTP method is not allowed" {
                status = 405;
            } else if err.message == "HTTP path is unknown" {
                status = 404;
            }
            let body = if openai_path {
                openai::error_document(err.code.as_str(), &err.message)
            } else {
                error_bytes(&err)
            };
            (status, body)
        }
    }
}

fn metrics_body(inner: &Inner) -> Vec<u8> {
    let ready = matches!(lock_slot(inner).state, WorkerState::Ready);
    let stale = if ready { !scrape_worker(inner) } else { false };
    let worker_up = matches!(lock_slot(inner).state, WorkerState::Ready);
    let host = lock_metrics(inner).clone();
    metrics::render(&host, worker_up, stale && worker_up).into_bytes()
}

fn scrape_worker(inner: &Inner) -> bool {
    let _guard = inner.scrape.lock().unwrap_or_else(|err| err.into_inner());
    let (sender, receiver) = mpsc::channel();
    {
        let mut slot = lock_slot(inner);
        if !matches!(slot.state, WorkerState::Ready) || slot.stats.is_some() {
            return false;
        }
        slot.stats = Some(sender);
        if write_worker(&mut slot, &ToWorker::Stats).is_err() {
            slot.stats = None;
            return false;
        }
    }
    let stats = match receiver.recv_timeout(Duration::from_secs(2)) {
        Ok(Ok(stats)) => Some(stats),
        _ => {
            let mut slot = lock_slot(inner);
            slot.stats = None;
            None
        }
    };
    match stats {
        Some(stats) => {
            lock_metrics(inner).snapshot = Some(stats);
            true
        }
        None => false,
    }
}

fn health_body(inner: &Inner) -> Vec<u8> {
    let (worker, lifecycle, status) = {
        let slot = lock_slot(inner);
        let worker = match slot.state {
            WorkerState::Ready => "ready",
            WorkerState::Starting => "starting",
            WorkerState::Down => "down",
        };
        let lifecycle = lifecycle_name(&slot);
        let status = match lifecycle {
            "unloading" => "unloading",
            "unloaded" => "unloaded",
            "draining" => "draining",
            "drained" => "drained",
            _ if worker == "ready" => "ok",
            _ => "degraded",
        };
        (worker, lifecycle, status)
    };
    serde_json::to_vec(&json!({
        "alias": inner.config.alias,
        "lifecycle": lifecycle,
        "runtimeRoot": inner.pinned.runtime_root,
        "status": status,
        "worker": worker,
    }))
    .expect("health json")
}

fn drain_http(inner: &Inner, body: &[u8]) -> Result<Vec<u8>, InferFailure> {
    let text = std::str::from_utf8(body)
        .map_err(|_| fail(ErrorCode::ContractInvalid, "HTTP body is not UTF-8"))?;
    let value = parse_strict_json(text, ErrorCode::ContractInvalid)?;
    let object = expect_object(&value)?;
    reject_unknown(object, &[])?;
    let (status, inflight) = begin_drain(inner);
    Ok(serde_json::to_vec(&json!({
        "inflight": inflight,
        "status": status,
    }))
    .expect("drain json"))
}

fn begin_drain(inner: &Inner) -> (&'static str, usize) {
    let mut slot = lock_slot(inner);
    slot.draining = true;
    let inflight = occupancy(&slot);
    let status = if inflight == 0 { "drained" } else { "draining" };
    (status, inflight)
}

fn unload_http(inner: &Inner, body: &[u8]) -> Result<Vec<u8>, InferFailure> {
    let text = std::str::from_utf8(body)
        .map_err(|_| fail(ErrorCode::ContractInvalid, "HTTP body is not UTF-8"))?;
    let value = parse_strict_json(text, ErrorCode::ContractInvalid)?;
    let object = expect_object(&value)?;
    reject_unknown(object, &[])?;
    let (status, inflight) = begin_unload(inner);
    Ok(serde_json::to_vec(&json!({
        "inflight": inflight,
        "status": status,
    }))
    .expect("unload json"))
}

fn lifecycle_name(slot: &WorkerSlot) -> &'static str {
    if slot.unloading {
        if occupancy(slot) == 0 && matches!(slot.state, WorkerState::Down) {
            "unloaded"
        } else {
            "unloading"
        }
    } else if !slot.draining {
        "serving"
    } else if occupancy(slot) == 0 {
        "drained"
    } else {
        "draining"
    }
}

fn load_http(inner: &Arc<Inner>, body: &[u8]) -> Result<Vec<u8>, InferFailure> {
    let text = std::str::from_utf8(body)
        .map_err(|_| fail(ErrorCode::ContractInvalid, "HTTP body is not UTF-8"))?;
    let value = parse_strict_json(text, ErrorCode::ContractInvalid)?;
    let object = expect_object(&value)?;
    reject_unknown(object, &[])?;
    let (status, inflight) = begin_load(inner)?;
    Ok(serde_json::to_vec(&json!({
        "inflight": inflight,
        "status": status,
    }))
    .expect("load json"))
}

fn begin_load(inner: &Arc<Inner>) -> Result<(&'static str, usize), InferFailure> {
    {
        let slot = lock_slot(inner);
        if slot.unloading && occupancy(&slot) > 0 {
            return Ok(("unloading", occupancy(&slot)));
        }
    }
    maybe_finish_unload(inner);
    let leaving_unload = {
        let mut slot = lock_slot(inner);
        if slot.unloading && occupancy(&slot) > 0 {
            return Ok(("unloading", occupancy(&slot)));
        }
        if matches!(slot.state, WorkerState::Ready) {
            return Ok((lifecycle_name(&slot), occupancy(&slot)));
        }
        let leaving = slot.unloading;
        slot.unloading = false;
        leaving
    };
    if let Err(err) = inner.ensure_worker() {
        if leaving_unload {
            let mut slot = lock_slot(inner);
            if matches!(slot.state, WorkerState::Down) && occupancy(&slot) == 0 {
                slot.unloading = true;
            }
        }
        return Err(err);
    }
    let slot = lock_slot(inner);
    Ok((lifecycle_name(&slot), occupancy(&slot)))
}

fn begin_unload(inner: &Inner) -> (&'static str, usize) {
    let inflight = {
        let mut slot = lock_slot(inner);
        slot.unloading = true;
        occupancy(&slot)
    };
    if inflight == 0 {
        maybe_finish_unload(inner);
        ("unloaded", 0)
    } else {
        ("unloading", inflight)
    }
}

fn maybe_finish_unload(inner: &Inner) {
    let should_stop = {
        let slot = lock_slot(inner);
        slot.unloading && occupancy(&slot) == 0 && !matches!(slot.state, WorkerState::Down)
    };
    if should_stop {
        stop_worker(inner);
    }
}

struct UnloadGuard<'a>(&'a Arc<Inner>);

impl Drop for UnloadGuard<'_> {
    fn drop(&mut self) {
        maybe_finish_unload(self.0);
    }
}

fn occupancy(slot: &WorkerSlot) -> usize {
    slot.completion.len().saturating_add(slot.reserved)
}

fn occupancy_of(inner: &Inner) -> usize {
    occupancy(&lock_slot(inner))
}

struct AdmissionHold {
    inner: Arc<Inner>,
    armed: bool,
}

impl AdmissionHold {
    fn reserve(inner: &Arc<Inner>) -> Result<Self, InferFailure> {
        let mut slot = lock_slot(inner);
        if slot.unloading {
            return Err(fail(ErrorCode::ServiceUnloaded, "model is not loaded"));
        }
        if slot.draining {
            return Err(fail(ErrorCode::ServiceDraining, "supervisor is draining"));
        }
        slot.reserved = slot.reserved.saturating_add(1);
        Ok(Self {
            inner: Arc::clone(inner),
            armed: true,
        })
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for AdmissionHold {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.armed = false;
        let mut slot = lock_slot(&self.inner);
        slot.reserved = slot.reserved.saturating_sub(1);
    }
}

fn plan_native(inner: &Arc<Inner>, body: &[u8]) -> Result<Plan, InferFailure> {
    let mut traced = false;
    let result = plan_native_inner(inner, body, &mut traced);
    if let Err(err) = &result {
        if !traced {
            inner.trace.reject(err.code);
        }
    }
    result
}

fn plan_native_inner(
    inner: &Arc<Inner>,
    body: &[u8],
    traced: &mut bool,
) -> Result<Plan, InferFailure> {
    let text = std::str::from_utf8(body)
        .map_err(|_| fail(ErrorCode::ContractInvalid, "HTTP body is not UTF-8"))?;
    let value = parse_strict_json(text, ErrorCode::ContractInvalid)?;
    let object = expect_object(&value)?;
    reject_unknown(
        object,
        &[
            "generation",
            "messages",
            "model",
            "requestId",
            "serviceClass",
            "stream",
        ],
    )?;
    let model = expect_string(object, "model")?;
    if model != inner.config.alias {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "model alias does not match the served model",
        ));
    }
    let messages = chat_messages(object)?;
    let generation = generation_request(object)?;
    let class = match object.get("serviceClass") {
        None => ServiceClass::Standard,
        Some(Value::String(value)) => parse_class(value)?,
        Some(_) => {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "serviceClass must be text",
            ))
        }
    };
    let stream = match object.get("stream") {
        None => false,
        Some(Value::Bool(value)) => *value,
        Some(_) => return Err(fail(ErrorCode::ContractInvalid, "stream must be a boolean")),
    };
    let request_id = match object.get("requestId") {
        None => format!("r{:016x}", inner.next_id.fetch_add(1, Ordering::SeqCst)),
        Some(Value::String(value)) => {
            check_request_id(value)?;
            value.clone()
        }
        Some(_) => return Err(fail(ErrorCode::ContractInvalid, "requestId must be text")),
    };
    assemble_plan(
        inner,
        messages,
        &generation,
        class,
        request_id,
        stream,
        traced,
    )
}

fn plan_openai(inner: &Arc<Inner>, body: &[u8]) -> Result<Plan, InferFailure> {
    let mut traced = false;
    let result = plan_openai_inner(inner, body, &mut traced);
    if let Err(err) = &result {
        if !traced {
            inner.trace.reject(err.code);
        }
    }
    result
}

fn plan_openai_inner(
    inner: &Arc<Inner>,
    body: &[u8],
    traced: &mut bool,
) -> Result<Plan, InferFailure> {
    let text = std::str::from_utf8(body)
        .map_err(|_| fail(ErrorCode::ContractInvalid, "HTTP body is not UTF-8"))?;
    let chat = openai::parse_chat(text)?;
    if chat.model != inner.config.alias {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "model alias does not match the served model",
        ));
    }
    let generation = GenerationRequest {
        temperature_micros: chat.temperature_micros,
        top_p_millionths: chat.top_p_millionths,
        max_output_tokens: chat.max_tokens,
        seed: chat.seed,
        ..GenerationRequest::omitted()
    };
    let messages = chat
        .messages
        .iter()
        .map(|turn| infer_contracts::ChatMessageV1 {
            role: turn.role.clone(),
            content: turn.content.clone(),
        })
        .collect();
    let request_id = format!("r{:016x}", inner.next_id.fetch_add(1, Ordering::SeqCst));
    assemble_plan(
        inner,
        messages,
        &generation,
        ServiceClass::Standard,
        request_id,
        chat.stream,
        traced,
    )
}

fn assemble_plan(
    inner: &Inner,
    messages: Vec<infer_contracts::ChatMessageV1>,
    generation: &GenerationRequest,
    class: ServiceClass,
    request_id: String,
    stream: bool,
    traced: &mut bool,
) -> Result<Plan, InferFailure> {
    *traced = true;
    let mut trace = inner.trace.begin(&request_id, class, stream);
    let sampler = match build_sampler(&inner.pinned.image, generation) {
        Ok(sampler) => sampler,
        Err(err) => {
            trace.finish(duration_outcome(err.code), Some(err.code), None);
            return Err(err);
        }
    };
    let compiled = match compile_model_prompt(
        &inner.pinned.image,
        messages,
        VOCAB as u32,
        MAX_CONTEXT,
        sampler.settings.max_output_tokens,
    ) {
        Ok(compiled) => compiled,
        Err(err) => {
            trace.prompt_rejected(err.code);
            trace.finish(duration_outcome(err.code), Some(err.code), None);
            return Err(err);
        }
    };
    let prompt_tokens = match u32::try_from(compiled.plan.token_ids.len()) {
        Ok(tokens) => tokens,
        Err(_) => {
            let err = fail(
                ErrorCode::ContextLimitExceeded,
                "prompt does not fit in the reserved context",
            );
            trace.prompt_rejected(err.code);
            trace.finish(duration_outcome(err.code), Some(err.code), None);
            return Err(err);
        }
    };
    let prompt_root = match compiled.plan.root() {
        Ok(root) => root,
        Err(err) => {
            trace.prompt_rejected(err.code);
            trace.finish(duration_outcome(err.code), Some(err.code), None);
            return Err(err);
        }
    };
    trace.prompt(prompt_root.as_str(), prompt_tokens);
    let prompt = compiled.plan.token_ids.clone();
    Ok(Plan {
        request_id,
        prompt,
        prompt_plan: compiled.plan,
        sampler: Box::new(sampler),
        class,
        prompt_tokens,
        stream,
        trace,
    })
}

fn finish_buffered(
    inner: &Arc<Inner>,
    plan: &mut Plan,
    openai: bool,
    opened: &mut crate::receipt::OpenedRequest,
    watch: &mut RequestWatch<'_>,
    hold: &mut AdmissionHold,
) -> Result<(Vec<u8>, String), InferFailure> {
    let message = submit_message(plan);
    let outcome = match submit_and_wait(inner, message, &mut plan.trace, watch, hold) {
        Ok(outcome) => outcome,
        Err(err) => {
            watch.set_outcome(duration_outcome(err.code));
            plan.trace
                .finish(duration_outcome(err.code), Some(err.code), None);
            return Err(opened.seal_failed(err));
        }
    };
    let Outcome::Result {
        tokens,
        finish_reason,
        error_code,
    } = outcome
    else {
        let err = outcome_error(outcome);
        watch.set_outcome(duration_outcome(err.code));
        plan.trace
            .finish(duration_outcome(err.code), Some(err.code), None);
        return Err(opened.seal_failed(err));
    };
    if let Some(code) = error_code {
        let err = failure_from_worker(&code, "worker finished with an error");
        watch.set_outcome(duration_outcome(err.code));
        plan.trace
            .finish(duration_outcome(err.code), Some(err.code), None);
        return Err(opened.seal_failed(err));
    }
    let text = match decode_output(inner, &tokens) {
        Ok(text) => text,
        Err(err) => {
            watch.set_outcome(duration_outcome(err.code));
            plan.trace
                .finish(duration_outcome(err.code), Some(err.code), None);
            return Err(opened.seal_failed(err));
        }
    };
    if finish_reason == "cancelled" {
        opened.seal_cancelled()?;
        watch.set_outcome("cancelled");
        plan.trace.finish("cancelled", None, None);
        if openai {
            return Err(fail(
                ErrorCode::RequestCancelled,
                "completion was cancelled",
            ));
        }
        return Ok((
            native_completion(inner, plan, &text, &tokens, &finish_reason, None),
            "absent".into(),
        ));
    }
    let receipt_started = Instant::now();
    let published = match opened.publish(&tokens, &text, &finish_reason, &inner.home) {
        Ok(published) => published,
        Err(err) => {
            watch.set_outcome(duration_outcome(err.code));
            plan.trace
                .finish(duration_outcome(err.code), Some(err.code), None);
            return Err(err);
        }
    };
    watch.note_receipt(receipt_started.elapsed());
    watch.set_outcome(finish_outcome(&finish_reason));
    plan.trace.finish(
        finish_outcome(&finish_reason),
        None,
        Some(&published.receipt_id),
    );
    if openai {
        let finish = openai_finish(&finish_reason)?;
        return Ok((
            openai::completion_document(
                &chat_id(&plan.request_id),
                &inner.config.alias,
                &text,
                finish,
                plan.prompt_tokens,
                output_len(&tokens)?,
                Some(&published.receipt_id),
            ),
            published.receipt_id,
        ));
    }
    Ok((
        native_completion(
            inner,
            plan,
            &text,
            &tokens,
            &finish_reason,
            Some(&published.receipt_id),
        ),
        published.receipt_id,
    ))
}

fn native_completion(
    inner: &Inner,
    plan: &Plan,
    text: &str,
    tokens: &[u32],
    finish_reason: &str,
    receipt: Option<&str>,
) -> Vec<u8> {
    let mut value = json!({
        "model": {
            "alias": inner.config.alias,
            "runtimeRoot": inner.pinned.runtime_root,
        },
        "output": {
            "finishReason": finish_reason,
            "text": text,
            "tokenIds": tokens,
        },
        "requestId": plan.request_id,
    });
    if let Some(receipt) = receipt {
        value["receipt"] = json!({
            "assurance": "compatibility",
            "receiptRoot": receipt,
        });
    }
    serde_json::to_vec(&value).expect("completion json")
}

fn serve_stream(
    stream: &mut TcpStream,
    inner: &Arc<Inner>,
    plan: &mut Plan,
    openai: bool,
    opened: &mut crate::receipt::OpenedRequest,
    watch: &mut RequestWatch<'_>,
    hold: &mut AdmissionHold,
) -> Result<(), InferFailure> {
    let message = submit_message(plan);
    let receiver = match submit(inner, message, hold) {
        Ok(receiver) => receiver,
        Err(err) => {
            watch.set_outcome(duration_outcome(err.code));
            plan.trace
                .finish(duration_outcome(err.code), Some(err.code), None);
            return Err(opened.seal_failed(err));
        }
    };
    if let Err(err) = wait_admitted(inner, &plan.request_id, &receiver, &mut plan.trace) {
        watch.set_outcome(duration_outcome(err.code));
        plan.trace
            .finish(duration_outcome(err.code), Some(err.code), None);
        return Err(opened.seal_failed(err));
    }
    if http::write_sse_headers(stream, &plan.request_id, "pending").is_err() {
        let _ = opened.seal_cancelled();
        abandon(inner, &plan.request_id);
        watch.set_outcome("cancelled");
        plan.trace.finish("cancelled", None, None);
        return Ok(());
    }
    let pumped = if openai {
        pump_openai(stream, inner, plan, &receiver, opened, watch)
    } else {
        pump_native(stream, inner, plan, &receiver, opened, watch)
    };
    if let Err(err) = pumped {
        if watch.outcome_is_unset() {
            watch.set_outcome(duration_outcome(err.code));
        }
        plan.trace
            .finish(duration_outcome(err.code), Some(err.code), None);
        let payload = if openai {
            String::from_utf8(openai::error_document(err.code.as_str(), &err.message))
                .unwrap_or_else(|_| {
                    r#"{"error":{"code":"CONTRACT_INVALID","message":"response encoding failed","type":"invalid_request_error"}}"#
                        .to_string()
                })
        } else {
            native_error_line(&plan.request_id, &err)
        };
        if http::write_sse(stream, (!openai).then_some("knolo.error"), &payload).is_err() {
            let _ = opened.seal_cancelled();
            abandon(inner, &plan.request_id);
            return Ok(());
        }
        let _ = opened.seal_failed(err);
        abandon(inner, &plan.request_id);
    }
    Ok(())
}

fn pump_native(
    stream: &mut TcpStream,
    inner: &Arc<Inner>,
    plan: &mut Plan,
    receiver: &Receiver<WorkerEvent>,
    opened: &mut crate::receipt::OpenedRequest,
    watch: &mut RequestWatch<'_>,
) -> Result<(), InferFailure> {
    write_native(
        stream,
        "knolo.accepted",
        &json!({
            "model": {
                "alias": inner.config.alias,
                "runtimeRoot": inner.pinned.runtime_root,
            },
            "requestId": plan.request_id,
        }),
    )?;
    let mut seen = Vec::new();
    loop {
        match recv_event(receiver)? {
            WorkerEvent::Prefill { chunk, tokens } => plan.trace.prefill(chunk, tokens),
            WorkerEvent::Delta { index, token_id } => {
                watch.note_ttft();
                plan.trace.decode(index);
                if index != seen.len() as u32 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "stream token index is not contiguous",
                    ));
                }
                let text = decode_output(inner, &[token_id])?;
                seen.push(token_id);
                write_native(
                    stream,
                    "knolo.delta",
                    &json!({
                        "index": index,
                        "requestId": plan.request_id,
                        "text": text,
                        "tokenId": token_id,
                    }),
                )?;
            }
            WorkerEvent::Finished(outcome) => {
                let Outcome::Result {
                    tokens,
                    finish_reason,
                    error_code,
                } = outcome
                else {
                    let err = outcome_error(outcome);
                    watch.set_outcome(duration_outcome(err.code));
                    return Err(err);
                };
                if let Some(code) = error_code {
                    let err = failure_from_worker(&code, "worker finished with an error");
                    watch.set_outcome(duration_outcome(err.code));
                    return Err(err);
                }
                if tokens != seen {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "stream tokens do not match the worker result",
                    ));
                }
                let text = decode_output(inner, &tokens)?;
                if finish_reason == "cancelled" {
                    opened.seal_cancelled()?;
                    watch.set_outcome("cancelled");
                    plan.trace.finish("cancelled", None, None);
                } else {
                    let receipt_started = Instant::now();
                    let published =
                        match opened.publish(&tokens, &text, &finish_reason, &inner.home) {
                            Ok(published) => published,
                            Err(err) => {
                                watch.set_outcome(duration_outcome(err.code));
                                plan.trace
                                    .finish(duration_outcome(err.code), Some(err.code), None);
                                return Err(err);
                            }
                        };
                    watch.note_receipt(receipt_started.elapsed());
                    watch.set_outcome(finish_outcome(&finish_reason));
                    plan.trace.finish(
                        finish_outcome(&finish_reason),
                        None,
                        Some(&published.receipt_id),
                    );
                    write_native(
                        stream,
                        "knolo.usage",
                        &json!({
                            "finishReason": finish_reason,
                            "outputTokens": tokens.len(),
                            "promptTokens": plan.prompt_tokens,
                            "requestId": plan.request_id,
                        }),
                    )?;
                    return write_native(
                        stream,
                        "knolo.receipt",
                        &json!({
                            "assurance": "compatibility",
                            "receiptRoot": published.receipt_id,
                            "requestId": plan.request_id,
                        }),
                    );
                }
                return write_native(
                    stream,
                    "knolo.usage",
                    &json!({
                        "finishReason": finish_reason,
                        "outputTokens": tokens.len(),
                        "promptTokens": plan.prompt_tokens,
                        "requestId": plan.request_id,
                    }),
                );
            }
            WorkerEvent::Admitted => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "worker event was not a token",
                ))
            }
        }
    }
}

fn pump_openai(
    stream: &mut TcpStream,
    inner: &Arc<Inner>,
    plan: &mut Plan,
    receiver: &Receiver<WorkerEvent>,
    opened: &mut crate::receipt::OpenedRequest,
    watch: &mut RequestWatch<'_>,
) -> Result<(), InferFailure> {
    let id = chat_id(&plan.request_id);
    let model = inner.config.alias.as_str();
    write_data(
        stream,
        &openai::chunk_document(&id, model, json!({"role": "assistant"}), None, None),
    )?;
    let mut seen = Vec::new();
    loop {
        match recv_event(receiver)? {
            WorkerEvent::Prefill { chunk, tokens } => plan.trace.prefill(chunk, tokens),
            WorkerEvent::Delta { index, token_id } => {
                watch.note_ttft();
                plan.trace.decode(index);
                if index != seen.len() as u32 {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "stream token index is not contiguous",
                    ));
                }
                let text = decode_output(inner, &[token_id])?;
                seen.push(token_id);
                write_data(
                    stream,
                    &openai::chunk_document(&id, model, json!({"content": text}), None, None),
                )?;
            }
            WorkerEvent::Finished(outcome) => {
                let Outcome::Result {
                    tokens,
                    finish_reason,
                    error_code,
                } = outcome
                else {
                    let err = outcome_error(outcome);
                    watch.set_outcome(duration_outcome(err.code));
                    return Err(err);
                };
                if let Some(code) = error_code {
                    let err = failure_from_worker(&code, "worker finished with an error");
                    watch.set_outcome(duration_outcome(err.code));
                    return Err(err);
                }
                if tokens != seen {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "stream tokens do not match the worker result",
                    ));
                }
                if finish_reason == "cancelled" {
                    opened.seal_cancelled()?;
                    watch.set_outcome("cancelled");
                    plan.trace.finish("cancelled", None, None);
                    return Err(fail(
                        ErrorCode::RequestCancelled,
                        "completion was cancelled",
                    ));
                }
                let text = decode_output(inner, &tokens)?;
                let receipt_started = Instant::now();
                let published = match opened.publish(&tokens, &text, &finish_reason, &inner.home) {
                    Ok(published) => published,
                    Err(err) => {
                        watch.set_outcome(duration_outcome(err.code));
                        plan.trace
                            .finish(duration_outcome(err.code), Some(err.code), None);
                        return Err(err);
                    }
                };
                watch.note_receipt(receipt_started.elapsed());
                watch.set_outcome(finish_outcome(&finish_reason));
                plan.trace.finish(
                    finish_outcome(&finish_reason),
                    None,
                    Some(&published.receipt_id),
                );
                let finish = openai_finish(&finish_reason)?;
                write_data(
                    stream,
                    &openai::chunk_document(
                        &id,
                        model,
                        json!({}),
                        Some(finish),
                        Some(&published.receipt_id),
                    ),
                )?;
                return write_data(stream, "[DONE]");
            }
            WorkerEvent::Admitted => {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "worker event was not a token",
                ))
            }
        }
    }
}

fn write_native(stream: &mut TcpStream, event: &str, value: &Value) -> Result<(), InferFailure> {
    let data = serde_json::to_string(value).expect("event json");
    http::write_sse(stream, Some(event), &data)
        .map_err(|_| fail(ErrorCode::ContractInvalid, "event stream write failed"))
}

fn write_data(stream: &mut TcpStream, data: &str) -> Result<(), InferFailure> {
    http::write_sse(stream, None, data)
        .map_err(|_| fail(ErrorCode::ContractInvalid, "event stream write failed"))
}

fn native_error_line(request_id: &str, err: &InferFailure) -> String {
    let message: String = err.message.chars().take(1024).collect();
    serde_json::to_string(&json!({
        "error": {"code": err.code.as_str(), "message": message},
        "requestId": request_id,
    }))
    .unwrap_or_else(|_| {
        r#"{"error":{"code":"CONTRACT_INVALID","message":"response encoding failed"},"requestId":"unknown"}"#
            .to_string()
    })
}

fn recv_event(receiver: &Receiver<WorkerEvent>) -> Result<WorkerEvent, InferFailure> {
    match receiver.recv_timeout(REQUEST_WAIT) {
        Ok(event) => Ok(event),
        Err(RecvTimeoutError::Timeout) => {
            Err(fail(ErrorCode::RequestTimeout, "completion timed out"))
        }
        Err(RecvTimeoutError::Disconnected) => Err(fail(
            ErrorCode::WorkerLost,
            "worker exited during the request",
        )),
    }
}

fn openai_finish(reason: &str) -> Result<&'static str, InferFailure> {
    match reason {
        "stop" => Ok("stop"),
        "length" => Ok("length"),
        "cancelled" => Err(fail(
            ErrorCode::RequestCancelled,
            "completion was cancelled",
        )),
        _ => Err(fail(
            ErrorCode::ContractInvalid,
            "finish reason is not in the OpenAI subset",
        )),
    }
}

fn chat_id(request_id: &str) -> String {
    format!("chatcmpl-{request_id}")
}

fn output_len(tokens: &[u32]) -> Result<u32, InferFailure> {
    u32::try_from(tokens.len())
        .map_err(|_| fail(ErrorCode::ContractInvalid, "output token count overflows"))
}

fn decode_output(inner: &Inner, tokens: &[u32]) -> Result<String, InferFailure> {
    let tokenizer = parse_tokenizer(&inner.pinned.image.tokenizer.bytes, VOCAB as u32)?;
    decode_tokens(&tokenizer, tokens)
}

fn submit_message(plan: &Plan) -> ToWorker {
    ToWorker::Submit {
        request_id: plan.request_id.clone(),
        prompt: plan.prompt.clone(),
        sampler: plan.sampler.clone(),
        class: plan.class,
    }
}

fn outcome_error(outcome: Outcome) -> InferFailure {
    match outcome {
        Outcome::Rejected { code, message } => failure_from_worker(&code, &message),
        Outcome::Lost => fail(ErrorCode::WorkerLost, "worker exited during the request"),
        Outcome::Ack => fail(
            ErrorCode::ContractInvalid,
            "worker ack was not a completion",
        ),
        Outcome::Result { .. } => fail(ErrorCode::ContractInvalid, "worker result was unexpected"),
    }
}

fn cancel(inner: &Arc<Inner>, body: &[u8]) -> Result<Vec<u8>, InferFailure> {
    let text = std::str::from_utf8(body)
        .map_err(|_| fail(ErrorCode::ContractInvalid, "HTTP body is not UTF-8"))?;
    let value = parse_strict_json(text, ErrorCode::ContractInvalid)?;
    let object = expect_object(&value)?;
    reject_unknown(object, &["requestId"])?;
    let request_id = expect_string(object, "requestId")?;
    check_request_id(&request_id)?;
    let (sender, receiver) = mpsc::channel();
    {
        let mut slot = lock_slot(inner);
        if !slot.completion.contains_key(&request_id) {
            return Err(fail(ErrorCode::ContractInvalid, "request is not admitted"));
        }
        if slot.cancel.contains_key(&request_id) {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "request is already cancelled",
            ));
        }
        slot.cancel.insert(request_id.clone(), sender);
        if let Err(err) = write_worker(
            &mut slot,
            &ToWorker::Cancel {
                request_id: request_id.clone(),
            },
        ) {
            slot.cancel.remove(&request_id);
            return Err(err);
        }
    }
    match receiver.recv_timeout(REQUEST_WAIT) {
        Ok(Outcome::Ack) => Ok(serde_json::to_vec(&json!({
            "requestId": request_id,
            "status": "cancelled",
        }))
        .expect("cancel json")),
        Ok(Outcome::Rejected { code, message }) => Err(failure_from_worker(&code, &message)),
        Ok(Outcome::Lost) | Err(RecvTimeoutError::Disconnected) => Err(fail(
            ErrorCode::WorkerLost,
            "worker exited during cancellation",
        )),
        Ok(Outcome::Result { .. }) => Err(fail(
            ErrorCode::ContractInvalid,
            "worker result was not a cancellation",
        )),
        Err(RecvTimeoutError::Timeout) => {
            Err(fail(ErrorCode::RequestTimeout, "cancellation timed out"))
        }
    }
}

fn submit(
    inner: &Arc<Inner>,
    message: ToWorker,
    hold: &mut AdmissionHold,
) -> Result<Receiver<WorkerEvent>, InferFailure> {
    let ToWorker::Submit { request_id, .. } = &message else {
        return Err(fail(ErrorCode::ContractInvalid, "missing submission"));
    };
    let request_id = request_id.clone();
    inner.ensure_worker()?;
    let (sender, receiver) = mpsc::channel();
    let rejection = {
        let mut slot = lock_slot(inner);
        if !matches!(slot.state, WorkerState::Ready) {
            return Err(fail(ErrorCode::WorkerLost, "worker is not ready"));
        }
        if slot.completion.contains_key(&request_id) {
            Some(metrics::Rejection::Contract)
        } else if slot.completion.len() >= MAX_INFLIGHT {
            Some(metrics::Rejection::Memory)
        } else {
            slot.completion.insert(request_id.clone(), sender);
            slot.reserved = slot.reserved.saturating_sub(1);
            hold.disarm();
            if let Err(err) = write_worker(&mut slot, &message) {
                slot.completion.remove(&request_id);
                return Err(err);
            }
            None
        }
    };
    if let Some(kind) = rejection {
        lock_metrics(inner).note_rejection(kind);
        let (code, message) = match kind {
            metrics::Rejection::Memory => {
                (ErrorCode::InsufficientMemory, "too many in-flight requests")
            }
            _ => (ErrorCode::ContractInvalid, "request id is already admitted"),
        };
        return Err(fail(code, message));
    }
    Ok(receiver)
}

fn submit_and_wait(
    inner: &Arc<Inner>,
    message: ToWorker,
    trace: &mut crate::trace::RequestTrace,
    watch: &mut RequestWatch<'_>,
    hold: &mut AdmissionHold,
) -> Result<Outcome, InferFailure> {
    let ToWorker::Submit { request_id, .. } = &message else {
        return Err(fail(ErrorCode::ContractInvalid, "missing submission"));
    };
    let request_id = request_id.clone();
    let receiver = submit(inner, message, hold)?;
    let deadline = Instant::now() + REQUEST_WAIT;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            abandon(inner, &request_id);
            return Err(fail(ErrorCode::RequestTimeout, "completion timed out"));
        }
        match receiver.recv_timeout(remaining) {
            Ok(WorkerEvent::Admitted) => trace.admitted(),
            Ok(WorkerEvent::Prefill { chunk, tokens }) => trace.prefill(chunk, tokens),
            Ok(WorkerEvent::Delta { index, .. }) => {
                watch.note_ttft();
                trace.decode(index);
            }
            Ok(WorkerEvent::Finished(outcome)) => {
                note_terminal(trace, &outcome);
                return Ok(outcome);
            }
            Err(RecvTimeoutError::Timeout) => {
                abandon(inner, &request_id);
                return Err(fail(ErrorCode::RequestTimeout, "completion timed out"));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(fail(
                    ErrorCode::WorkerLost,
                    "worker exited during the request",
                ))
            }
        }
    }
}

fn wait_admitted(
    inner: &Arc<Inner>,
    request_id: &str,
    receiver: &Receiver<WorkerEvent>,
    trace: &mut crate::trace::RequestTrace,
) -> Result<(), InferFailure> {
    match receiver.recv_timeout(ADMIT_WAIT) {
        Ok(WorkerEvent::Admitted) => {
            trace.admitted();
            Ok(())
        }
        Ok(WorkerEvent::Finished(outcome)) => {
            note_terminal(trace, &outcome);
            Err(outcome_error(outcome))
        }
        Ok(WorkerEvent::Prefill { .. } | WorkerEvent::Delta { .. }) => {
            abandon(inner, request_id);
            Err(fail(
                ErrorCode::ContractInvalid,
                "worker sent a token before admission",
            ))
        }
        Err(RecvTimeoutError::Timeout) => {
            abandon(inner, request_id);
            Err(fail(ErrorCode::RequestTimeout, "admission timed out"))
        }
        Err(RecvTimeoutError::Disconnected) => Err(fail(
            ErrorCode::WorkerLost,
            "worker exited during the request",
        )),
    }
}

fn note_terminal(trace: &mut crate::trace::RequestTrace, outcome: &Outcome) {
    if let Outcome::Rejected { code, .. } = outcome {
        if let Some(parsed) = ErrorCode::parse(code) {
            trace.admission_rejected(parsed);
        }
    }
}

fn abandon(inner: &Inner, request_id: &str) {
    let mut slot = lock_slot(inner);
    slot.completion.remove(request_id);
    let _ = write_worker(
        &mut slot,
        &ToWorker::Cancel {
            request_id: request_id.to_string(),
        },
    );
}

fn chat_messages(
    object: &Map<String, Value>,
) -> Result<Vec<infer_contracts::ChatMessageV1>, InferFailure> {
    let Some(value) = object.get("messages") else {
        return Err(fail(ErrorCode::ContractInvalid, "missing field: messages"));
    };
    let Some(items) = value.as_array() else {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "messages must be an array",
        ));
    };
    if items.is_empty() || items.len() > MAX_MESSAGES {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "messages must contain 1 to 32 entries",
        ));
    }
    let mut messages = Vec::with_capacity(items.len());
    for item in items {
        let message = expect_object(item)?;
        reject_unknown(message, &["content", "role"])?;
        messages.push(infer_contracts::ChatMessageV1 {
            role: expect_string(message, "role")?,
            content: expect_string(message, "content")?,
        });
    }
    Ok(messages)
}

fn generation_request(object: &Map<String, Value>) -> Result<GenerationRequest, InferFailure> {
    let Some(value) = object.get("generation") else {
        return Ok(GenerationRequest::omitted());
    };
    let generation = expect_object(value)?;
    reject_unknown(
        generation,
        &[
            "frequencyPenaltyMicros",
            "maxOutputTokens",
            "minPMillionths",
            "presencePenaltyMicros",
            "repetitionPenaltyMicros",
            "seed",
            "stream",
            "temperatureMicros",
            "topK",
            "topPMillionths",
        ],
    )?;
    Ok(GenerationRequest {
        temperature_micros: optional_u32(generation, "temperatureMicros")?,
        top_p_millionths: optional_u32(generation, "topPMillionths")?,
        min_p_millionths: optional_u32(generation, "minPMillionths")?,
        repetition_penalty_micros: optional_u32(generation, "repetitionPenaltyMicros")?,
        presence_penalty_micros: optional_u32(generation, "presencePenaltyMicros")?,
        frequency_penalty_micros: optional_u32(generation, "frequencyPenaltyMicros")?,
        top_k: optional_u32(generation, "topK")?,
        max_output_tokens: optional_u32(generation, "maxOutputTokens")?,
        seed: optional_u64(generation, "seed")?,
        stream: optional_u64(generation, "stream")?,
    })
}

fn expect_object(value: &Value) -> Result<&Map<String, Value>, InferFailure> {
    value
        .as_object()
        .ok_or_else(|| fail(ErrorCode::ContractInvalid, "JSON value must be an object"))
}

fn expect_string(object: &Map<String, Value>, key: &str) -> Result<String, InferFailure> {
    match object.get(key) {
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(fail(
            ErrorCode::ContractInvalid,
            format!("field {key} must be text"),
        )),
        None => Err(fail(
            ErrorCode::ContractInvalid,
            format!("missing field: {key}"),
        )),
    }
}

fn optional_u32(object: &Map<String, Value>, key: &str) -> Result<Option<u32>, InferFailure> {
    optional_u64(object, key)?
        .map(|value| {
            u32::try_from(value).map_err(|_| {
                fail(
                    ErrorCode::ContractInvalid,
                    format!("field {key} exceeds u32"),
                )
            })
        })
        .transpose()
}

fn optional_u64(object: &Map<String, Value>, key: &str) -> Result<Option<u64>, InferFailure> {
    match object.get(key) {
        None => Ok(None),
        Some(Value::Number(number)) => number.as_u64().map(Some).ok_or_else(|| {
            fail(
                ErrorCode::ContractInvalid,
                format!("field {key} must be an integer"),
            )
        }),
        Some(_) => Err(fail(
            ErrorCode::ContractInvalid,
            format!("field {key} must be an integer"),
        )),
    }
}

fn reject_unknown(object: &Map<String, Value>, allowed: &[&str]) -> Result<(), InferFailure> {
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        Err(fail(
            ErrorCode::ContractInvalid,
            format!("unknown field: {key}"),
        ))
    } else {
        Ok(())
    }
}

fn failure_from_worker(code: &str, message: &str) -> InferFailure {
    let code = ErrorCode::parse(code).unwrap_or(ErrorCode::ContractInvalid);
    fail(code, message)
}

fn error_bytes(err: &InferFailure) -> Vec<u8> {
    let mut message = err.message.clone();
    if message.len() > 1024 {
        message.truncate(1024);
    }
    serde_json::to_vec(&json!({
        "error": {"code": err.code.as_str(), "message": message}
    }))
    .unwrap_or_else(|_| {
        br#"{"error":{"code":"CONTRACT_INVALID","message":"response encoding failed"}}"#.to_vec()
    })
}

fn stderr_tail(path: &Path) -> String {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(_) => return String::new(),
    };
    let mut text = String::new();
    if file.read_to_string(&mut text).is_err() {
        return String::new();
    }
    let text = text.trim();
    if text.is_empty() {
        return String::new();
    }
    let tail: String = text
        .chars()
        .rev()
        .take(400)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!(" worker: {}", tail.replace('\n', " "))
}

fn check_loopback(addr: SocketAddr) -> Result<(), InferFailure> {
    match addr.ip() {
        IpAddr::V4(ip) if ip == Ipv4Addr::LOCALHOST => Ok(()),
        _ => Err(fail(
            ErrorCode::ContractInvalid,
            "supervisor listens on 127.0.0.1 only",
        )),
    }
}
