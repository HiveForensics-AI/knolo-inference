use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use infer_artifact::{
    hash_current_executable, new_lockfile, pin_alias, verify_image, write_lockfile, ErrorCode,
};
use infer_contracts::ChatMessageV1;
use infer_engine::{
    cpu_placement, load_verified_micro, micro_kv_layout, write_synthetic_model,
    ArchitectureAdapter, CpuScheduler, MicroAdapter, PagedKv, ReferenceF32Backend, ScheduleRequest,
    SchedulerConfig, ServiceClass, BLOCK_SIZE, CPU_KV_PAGE_POOL, MAX_CONTEXT, VOCAB,
};
use infer_prompt::compile_model_prompt;
use infer_serve::{build_sampler, GenerationRequest, ServeConfig, Supervisor};
use serde_json::Value;

static TEMP: AtomicU64 = AtomicU64::new(0);

struct Reply {
    status: u16,
    body: Value,
}

fn scratch() -> PathBuf {
    let n = TEMP.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "knolo-infer-serve-fixture-{}-{n}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    write_synthetic_model(&dir).unwrap();
    let bytes = fs::read(dir.join("micro.kmodel")).unwrap();
    let verification = verify_image(&bytes).unwrap();
    let mut lock = new_lockfile(&hash_current_executable().unwrap());
    pin_alias(&mut lock, "micro", "micro.kmodel", &verification.image).unwrap();
    write_lockfile(&dir.join("knolo.infer.lock.json"), &lock).unwrap();
    dir
}

fn start(dir: &Path, pause: bool) -> Supervisor {
    Supervisor::start(ServeConfig {
        alias: "micro".into(),
        work_dir: dir.to_path_buf(),
        lock_path: dir.join("knolo.infer.lock.json"),
        weights_dir: None,
        bind: "127.0.0.1:0".parse().unwrap(),
        worker_bin: PathBuf::from(env!("CARGO_BIN_EXE_knolo-infer-worker")),
        home: dir.join("home"),
        pause_before_forward: pause,
    })
    .unwrap_or_else(|err| panic!("supervisor start: {err}"))
}

fn exchange(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> Reply {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let payload = body.unwrap_or("");
    let request = if method == "GET" {
        format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
    } else {
        format!(
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
            payload.len()
        )
    };
    stream.write_all(request.as_bytes()).unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).unwrap();
    let text = String::from_utf8(buf).unwrap();
    let (head, body) = text.split_once("\r\n\r\n").expect("HTTP headers");
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    let parsed = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_str(body).unwrap_or_else(|err| panic!("{err}: {body}"))
    };
    Reply {
        status,
        body: parsed,
    }
}

fn greedy(max_output_tokens: u32) -> GenerationRequest {
    GenerationRequest {
        max_output_tokens: Some(max_output_tokens),
        ..GenerationRequest::omitted()
    }
}

fn completion(content: &str, generation: &str, class: &str) -> String {
    format!(
        r#"{{"generation":{{{generation}}},"messages":[{{"content":"{content}","role":"user"}}],"model":"micro"{class}}}"#
    )
}

fn tokens_of(body: &Value) -> Vec<u32> {
    body["output"]["tokenIds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as u32)
        .collect()
}

fn alone(
    dir: &Path,
    content: &str,
    generation: &GenerationRequest,
    class: ServiceClass,
) -> (Vec<u32>, String) {
    let bytes = fs::read(dir.join("micro.kmodel")).unwrap();
    let verification = verify_image(&bytes).unwrap();
    let sampler = build_sampler(&verification.image, generation).unwrap();
    let compiled = compile_model_prompt(
        &verification.image,
        vec![ChatMessageV1 {
            role: "user".into(),
            content: content.into(),
        }],
        VOCAB as u32,
        MAX_CONTEXT,
        sampler.settings.max_output_tokens,
    )
    .unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), dir).unwrap();
    let placement = cpu_placement(&source).unwrap();
    let mut model = MicroAdapter
        .build(&source, &placement, &ReferenceF32Backend)
        .unwrap();
    let mut kv = PagedKv::new(micro_kv_layout(), CPU_KV_PAGE_POOL).unwrap();
    let mut scheduler = CpuScheduler::new(
        SchedulerConfig::new(
            source.runtime_root.as_str(),
            MAX_CONTEXT,
            BLOCK_SIZE,
            CPU_KV_PAGE_POOL,
            4,
        )
        .unwrap(),
    );
    scheduler
        .submit(ScheduleRequest {
            request_id: "alone".into(),
            runtime_root: source.runtime_root.to_string(),
            prompt: compiled.plan.token_ids,
            sampler,
            class,
        })
        .unwrap();
    let results = scheduler.run_until_idle(&mut *model, &mut kv).unwrap();
    let result = results
        .iter()
        .find(|result| result.request_id == "alone")
        .unwrap();
    (result.tokens.clone(), result.finish_reason.clone())
}

fn listening_tcp_inodes(pid: u32) -> Vec<String> {
    let mut sockets = std::collections::HashSet::new();
    let fd_dir = format!("/proc/{pid}/fd");
    let entries = fs::read_dir(&fd_dir)
        .unwrap_or_else(|err| panic!("worker {pid} file descriptors are not readable: {err}"));
    for entry in entries {
        let path = entry.unwrap().path();
        if let Ok(link) = fs::read_link(path) {
            let text = link.to_string_lossy().into_owned();
            if let Some(rest) = text.strip_prefix("socket:[") {
                if let Some(inode) = rest.strip_suffix(']') {
                    sockets.insert(inode.to_string());
                }
            }
        }
    }
    let mut found = Vec::new();
    for proc in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let Ok(text) = fs::read_to_string(proc) else {
            continue;
        };
        for line in text.lines().skip(1) {
            let cols: Vec<_> = line.split_whitespace().collect();
            if cols.get(3).copied() == Some("0A") {
                if let Some(inode) = cols.get(9) {
                    if sockets.contains(*inode) {
                        found.push((*inode).to_string());
                    }
                }
            }
        }
    }
    found
}

#[test]
fn bind_address_must_be_loopback() {
    let started = Supervisor::start(ServeConfig {
        alias: "micro".into(),
        work_dir: PathBuf::from("."),
        lock_path: PathBuf::from("knolo.infer.lock.json"),
        weights_dir: None,
        bind: "0.0.0.0:9".parse().unwrap(),
        worker_bin: PathBuf::from("knolo-infer-worker"),
        home: PathBuf::from("home"),
        pause_before_forward: false,
    });
    match started {
        Ok(supervisor) => {
            supervisor.shutdown();
            panic!("loopback check returned a supervisor");
        }
        Err(err) => assert_eq!(err.code, ErrorCode::ContractInvalid),
    }
}

#[test]
fn http_completion_matches_the_scheduler_and_the_worker_does_not_listen() {
    let dir = scratch();
    let supervisor = start(&dir, false);
    let worker_pid = supervisor.worker_pid().unwrap();
    assert!(
        listening_tcp_inodes(worker_pid).is_empty(),
        "worker is listening: {:?}",
        listening_tcp_inodes(worker_pid)
    );
    assert!(!listening_tcp_inodes(std::process::id()).is_empty());
    let health = exchange(supervisor.address(), "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.status, 200);
    assert_eq!(health.body["status"], "ok");
    assert_eq!(health.body["worker"], "ready");
    let generation = greedy(2);
    let (expected, finish) = alone(&dir, "hi", &generation, ServiceClass::Standard);
    let reply = exchange(
        supervisor.address(),
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("hi", r#""maxOutputTokens":2"#, "")),
    );
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(tokens_of(&reply.body), expected);
    assert_eq!(reply.body["output"]["finishReason"], finish);
    assert_eq!(reply.body["receipt"]["assurance"], "compatibility");
    let root = reply.body["receipt"]["receiptRoot"].as_str().unwrap();
    let fetched = exchange(
        supervisor.address(),
        "GET",
        &format!("/knolo/infer/v1/receipts/{root}"),
        None,
    );
    assert_eq!(fetched.status, 200, "{}", fetched.body);
    assert_eq!(fetched.body["receiptRoot"], root);
    assert_eq!(fetched.body["requestId"], reply.body["requestId"]);
    let stored = fs::read(
        dir.join("home")
            .join("receipts")
            .join("sha256")
            .join(root.strip_prefix("sha256-").unwrap())
            .join("receipt.cbor"),
    )
    .unwrap();
    let receipt = infer_contracts::InferenceReceiptV1::from_cbor(
        &infer_contracts::decode_canonical(&stored).unwrap(),
    )
    .unwrap();
    assert_eq!(receipt.assurance, "compatibility");
    assert_eq!(receipt.execution.scheduling_mode, "continuous");
    #[cfg(not(feature = "cuda"))]
    assert_eq!(receipt.engine.backend_version, "reference-f32");
    #[cfg(feature = "cuda")]
    assert_eq!(receipt.engine.backend_version, "0.8.4");
    assert_eq!(
        receipt.execution.event_trace_root,
        infer_engine::verify_journal(
            &dir.join("home"),
            fetched.body["requestId"].as_str().unwrap()
        )
        .unwrap()
    );
    let missing = exchange(
        supervisor.address(),
        "GET",
        "/knolo/infer/v1/receipts/sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        None,
    );
    assert_eq!(missing.status, 404, "{}", missing.body);
    let method = exchange(
        supervisor.address(),
        "POST",
        "/knolo/infer/v1/health",
        Some("{}"),
    );
    assert_eq!(method.status, 405);
    let missing = exchange(supervisor.address(), "GET", "/v1/embeddings", None);
    assert_eq!(missing.status, 404);
    assert_eq!(missing.body["error"]["type"], "invalid_request_error");
    supervisor.shutdown();
}

#[test]
fn sampled_completion_matches_the_scheduler() {
    let dir = scratch();
    let supervisor = start(&dir, false);
    let mut generation = greedy(2);
    generation.temperature_micros = Some(100_000);
    generation.top_k = Some(0);
    generation.seed = Some(7);
    let (expected, finish) = alone(&dir, "hi", &generation, ServiceClass::Standard);
    let reply = exchange(
        supervisor.address(),
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion(
            "hi",
            r#""maxOutputTokens":2,"seed":7,"temperatureMicros":100000,"topK":0"#,
            "",
        )),
    );
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(tokens_of(&reply.body), expected);
    assert_eq!(reply.body["output"]["finishReason"], finish);
    supervisor.shutdown();
}

#[test]
fn shared_completions_match_separate_runs() {
    let dir = scratch();
    let supervisor = start(&dir, true);
    let addr = supervisor.address();
    let (left_tx, left_rx) = mpsc::channel();
    let (right_tx, right_rx) = mpsc::channel();
    thread::spawn(move || {
        left_tx
            .send(exchange(
                addr,
                "POST",
                "/knolo/infer/v1/complete",
                Some(&completion(
                    "hi",
                    r#""maxOutputTokens":2"#,
                    r#","serviceClass":"interactive""#,
                )),
            ))
            .unwrap();
    });
    thread::spawn(move || {
        right_tx
            .send(exchange(
                addr,
                "POST",
                "/knolo/infer/v1/complete",
                Some(&completion(
                    "at",
                    r#""maxOutputTokens":2"#,
                    r#","serviceClass":"background""#,
                )),
            ))
            .unwrap();
    });
    let start = std::time::Instant::now();
    while supervisor.inflight() < 2 {
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "both requests were not admitted"
        );
        thread::sleep(Duration::from_millis(10));
    }
    thread::sleep(Duration::from_millis(30));
    assert!(left_rx.try_recv().is_err());
    supervisor.release_forward().unwrap();
    let left = left_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let right = right_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(left.status, 200, "{}", left.body);
    assert_eq!(right.status, 200, "{}", right.body);
    let (hi, hi_finish) = alone(&dir, "hi", &greedy(2), ServiceClass::Interactive);
    let (at, at_finish) = alone(&dir, "at", &greedy(2), ServiceClass::Background);
    let matched = |body: &Value, tokens: &[u32], finish: &str| {
        tokens_of(body) == tokens && body["output"]["finishReason"] == finish
    };
    let ordered = matched(&left.body, &hi, &hi_finish) && matched(&right.body, &at, &at_finish);
    let swapped = matched(&left.body, &at, &at_finish) && matched(&right.body, &hi, &hi_finish);
    assert!(
        ordered || swapped,
        "left {} right {}",
        left.body,
        right.body
    );
    supervisor.shutdown();
}

#[test]
fn a_ninth_admission_is_insufficient_memory() {
    let dir = scratch();
    let supervisor = start(&dir, true);
    let addr = supervisor.address();
    let (tx, rx) = mpsc::channel();
    for _ in 0..9 {
        let tx = tx.clone();
        thread::spawn(move || {
            tx.send(exchange(
                addr,
                "POST",
                "/knolo/infer/v1/complete",
                Some(&completion("hi", r#""maxOutputTokens":1"#, "")),
            ))
            .unwrap();
        });
    }
    drop(tx);
    let start = std::time::Instant::now();
    let mut early = None;
    while early.is_none() || supervisor.inflight() != 8 {
        if let Ok(reply) = rx.try_recv() {
            early = Some(reply);
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "ninth request was not refused, inflight {}",
            supervisor.inflight()
        );
        thread::sleep(Duration::from_millis(10));
    }
    let early = early.unwrap();
    assert_eq!(early.status, 400, "{}", early.body);
    assert_eq!(early.body["error"]["code"], "INSUFFICIENT_MEMORY");
    supervisor.release_forward().unwrap();
    for _ in 0..8 {
        let reply = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(reply.status, 200, "{}", reply.body);
    }
    supervisor.shutdown();
}

#[test]
fn cancel_before_the_forward_finishes_cancelled() {
    let dir = scratch();
    let supervisor = start(&dir, true);
    let addr = supervisor.address();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        tx.send(exchange(
            addr,
            "POST",
            "/knolo/infer/v1/complete",
            Some(
                r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"hi","role":"user"}],"model":"micro","requestId":"job-1"}"#,
            ),
        ))
        .unwrap();
    });
    let start = std::time::Instant::now();
    while supervisor.inflight() < 1 {
        assert!(start.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    }
    let journal = dir.join("home").join("journals").join("job-1");
    assert!(
        journal.join("000000.accepted.cbor").is_file(),
        "accepted is fsynced before the forward"
    );
    assert!(!journal.join("000001.cancelled.cbor").exists());
    let cancel = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/cancel",
        Some(r#"{"requestId":"job-1"}"#),
    );
    assert_eq!(cancel.status, 200, "{}", cancel.body);
    assert_eq!(cancel.body["status"], "cancelled");
    supervisor.release_forward().unwrap();
    let reply = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(reply.body["output"]["finishReason"], "cancelled");
    assert!(tokens_of(&reply.body).is_empty());
    assert!(reply.body.get("receipt").is_none());
    assert!(journal.join("000001.cancelled.cbor").is_file());
    supervisor.shutdown();
}

#[test]
fn a_worker_crash_leaves_the_listener_up() {
    let dir = scratch();
    let supervisor = start(&dir, true);
    let addr = supervisor.address();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        tx.send(exchange(
            addr,
            "POST",
            "/knolo/infer/v1/complete",
            Some(&completion("hi", r#""maxOutputTokens":2"#, "")),
        ))
        .unwrap();
    });
    let start = std::time::Instant::now();
    while supervisor.inflight() < 1 {
        assert!(start.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    }
    supervisor.kill_worker().unwrap();
    let lost = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(lost.status, 503, "{}", lost.body);
    assert_eq!(lost.body["error"]["code"], "WORKER_LOST");
    let start = std::time::Instant::now();
    while supervisor.worker_label() != "down" {
        assert!(start.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    }
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.status, 200);
    assert_eq!(health.body["worker"], "down");
    supervisor.release_forward().unwrap();
    let (expected, _) = alone(&dir, "at", &greedy(2), ServiceClass::Standard);
    let reply = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("at", r#""maxOutputTokens":2"#, "")),
    );
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(tokens_of(&reply.body), expected);
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.body["worker"], "ready");
    supervisor.shutdown();
}

#[test]
fn unknown_fields_floats_and_a_long_prompt_are_refused() {
    let dir = scratch();
    let supervisor = start(&dir, false);
    let addr = supervisor.address();
    let unknown = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(r#"{"messages":[{"content":"hi","role":"user"}],"model":"micro","tools":[]}"#),
    );
    assert_eq!(unknown.status, 400, "{}", unknown.body);
    assert_eq!(unknown.body["error"]["code"], "CONTRACT_INVALID");
    let floaty = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(
            r#"{"generation":{"temperatureMicros":1.5},"messages":[{"content":"hi","role":"user"}],"model":"micro"}"#,
        ),
    );
    assert_eq!(floaty.status, 400, "{}", floaty.body);
    let wrong = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(r#"{"messages":[{"content":"hi","role":"user"}],"model":"other"}"#),
    );
    assert_eq!(wrong.status, 400, "{}", wrong.body);
    let long = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion(&"a".repeat(32), r#""maxOutputTokens":4"#, "")),
    );
    assert_eq!(long.status, 400, "{}", long.body);
    assert_eq!(long.body["error"]["code"], "CONTEXT_LIMIT_EXCEEDED");
    let unseeded = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("hi", r#""temperatureMicros":1000"#, "")),
    );
    assert_eq!(unseeded.status, 400, "{}", unseeded.body);
    supervisor.shutdown();
}

struct Raw {
    status: u16,
    head: String,
    body: String,
}

fn exchange_raw(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> Raw {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let payload = body.unwrap_or("");
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    stream.write_all(request.as_bytes()).unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).unwrap();
    let text = String::from_utf8(buf).unwrap();
    let (head, body) = text.split_once("\r\n\r\n").expect("HTTP headers");
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    Raw {
        status,
        head: head.to_string(),
        body: body.to_string(),
    }
}

fn sse_events(body: &str) -> Vec<(String, String)> {
    body.split("\n\n")
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut name = String::new();
            let mut data = String::new();
            for line in part.split('\n') {
                if let Some(rest) = line.strip_prefix("event: ") {
                    name = rest.to_string();
                } else if let Some(rest) = line.strip_prefix("data: ") {
                    data = rest.to_string();
                }
            }
            (name, data)
        })
        .collect()
}

#[test]
fn native_stream_matches_the_scheduler() {
    let dir = scratch();
    let supervisor = start(&dir, false);
    let (expected, finish) = alone(&dir, "hi", &greedy(2), ServiceClass::Standard);
    let raw = exchange_raw(
        supervisor.address(),
        "POST",
        "/knolo/infer/v1/complete",
        Some(
            r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"hi","role":"user"}],"model":"micro","stream":true}"#,
        ),
    );
    assert_eq!(raw.status, 200, "{}", raw.body);
    assert!(raw.head.to_ascii_lowercase().contains("text/event-stream"));
    assert!(raw.head.contains("X-Knolo-Receipt: pending"));
    assert!(raw.body.contains("knolo.receipt"));
    let events = sse_events(&raw.body);
    assert_eq!(events[0].0, "knolo.accepted");
    let deltas: Vec<_> = events
        .iter()
        .filter(|event| event.0 == "knolo.delta")
        .collect();
    let usage = events
        .iter()
        .find(|event| event.0 == "knolo.usage")
        .unwrap();
    let usage: Value = serde_json::from_str(&usage.1).unwrap();
    assert_eq!(usage["finishReason"], finish);
    assert_eq!(usage["outputTokens"], expected.len());
    let receipt = events
        .iter()
        .find(|event| event.0 == "knolo.receipt")
        .unwrap();
    assert_eq!(events.last().unwrap().0, "knolo.receipt");
    let receipt: Value = serde_json::from_str(&receipt.1).unwrap();
    assert_eq!(receipt["assurance"], "compatibility");
    assert!(receipt["receiptRoot"]
        .as_str()
        .unwrap()
        .starts_with("sha256-"));
    let mut ids = Vec::new();
    let mut text = String::new();
    for (index, event) in deltas.iter().enumerate() {
        let delta: Value = serde_json::from_str(&event.1).unwrap();
        assert_eq!(delta["index"], index);
        ids.push(delta["tokenId"].as_u64().unwrap() as u32);
        text.push_str(delta["text"].as_str().unwrap());
    }
    assert_eq!(ids, expected);
    let json = exchange(
        supervisor.address(),
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("hi", r#""maxOutputTokens":2"#, "")),
    );
    assert_eq!(json.body["output"]["text"], text);
    supervisor.shutdown();
}

#[test]
fn openai_completion_matches_the_scheduler_and_rejects_unknown_fields() {
    let dir = scratch();
    let supervisor = start(&dir, false);
    let addr = supervisor.address();
    let models = exchange(addr, "GET", "/v1/models", None);
    assert_eq!(models.status, 200, "{}", models.body);
    assert_eq!(models.body["data"][0]["id"], "micro");
    assert_eq!(models.body["data"][0]["owned_by"], "knolo");
    let (expected, finish) = alone(&dir, "hi", &greedy(2), ServiceClass::Standard);
    let raw = exchange_raw(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"max_tokens":2,"messages":[{"content":"hi","role":"user"}],"model":"micro"}"#),
    );
    assert_eq!(raw.status, 200, "{}", raw.body);
    assert!(raw.head.contains("X-Knolo-Receipt: sha256-"));
    assert!(raw.head.contains("X-Knolo-Request-Id:"));
    let body: Value = serde_json::from_str(&raw.body).unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert!(body["knolo_receipt"]
        .as_str()
        .unwrap()
        .starts_with("sha256-"));
    assert_eq!(body["choices"][0]["finish_reason"], finish);
    assert_eq!(body["usage"]["completion_tokens"], expected.len());
    let native = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("hi", r#""maxOutputTokens":2"#, "")),
    );
    assert_eq!(
        body["choices"][0]["message"]["content"],
        native.body["output"]["text"]
    );
    let mut generation = greedy(2);
    generation.temperature_micros = Some(500_000);
    generation.seed = Some(7);
    let (sampled, sampled_finish) = alone(&dir, "hi", &generation, ServiceClass::Standard);
    let warm = exchange(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(
            r#"{"max_tokens":2,"messages":[{"content":"hi","role":"user"}],"model":"micro","seed":7,"temperature":0.5}"#,
        ),
    );
    assert_eq!(warm.status, 200, "{}", warm.body);
    assert_eq!(warm.body["choices"][0]["finish_reason"], sampled_finish);
    let warm_native = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion(
            "hi",
            r#""maxOutputTokens":2,"seed":7,"temperatureMicros":500000"#,
            "",
        )),
    );
    assert_eq!(
        warm.body["choices"][0]["message"]["content"],
        warm_native.body["output"]["text"]
    );
    assert_eq!(tokens_of(&warm_native.body), sampled);
    let tools = exchange(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"messages":[{"content":"hi","role":"user"}],"model":"micro","tools":[]}"#),
    );
    assert_eq!(tools.status, 400, "{}", tools.body);
    assert_eq!(tools.body["error"]["code"], "CONTRACT_INVALID");
    assert_eq!(tools.body["error"]["type"], "invalid_request_error");
    let method = exchange(addr, "GET", "/v1/chat/completions", None);
    assert_eq!(method.status, 405, "{}", method.body);
    supervisor.shutdown();
}

#[test]
fn openai_stream_ends_with_done_and_cancel_uses_the_request_id() {
    let dir = scratch();
    let supervisor = start(&dir, false);
    let addr = supervisor.address();
    let raw = exchange_raw(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(
            r#"{"max_tokens":2,"messages":[{"content":"hi","role":"user"}],"model":"micro","stream":true}"#,
        ),
    );
    assert_eq!(raw.status, 200, "{}", raw.body);
    assert!(raw.head.contains("X-Knolo-Receipt: pending"));
    assert!(raw.body.contains("knolo_receipt"));
    let request_id = raw
        .head
        .lines()
        .find_map(|line| line.strip_prefix("X-Knolo-Request-Id: "))
        .unwrap();
    let events = sse_events(&raw.body);
    assert!(events
        .iter()
        .any(|event| event.1.contains("\"role\":\"assistant\"")));
    assert_eq!(events.last().unwrap().1, "[DONE]");
    let joined: String = events
        .iter()
        .filter_map(|event| serde_json::from_str::<Value>(&event.1).ok())
        .filter_map(|value| {
            value["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_string)
        })
        .collect();
    let native = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("hi", r#""maxOutputTokens":2"#, "")),
    );
    assert_eq!(joined, native.body["output"]["text"]);
    assert!(!request_id.is_empty());
    supervisor.shutdown();
}

#[test]
fn a_paused_stream_can_be_cancelled_and_a_dropped_stream_does_not_stop_the_listener() {
    let dir = scratch();
    let supervisor = start(&dir, true);
    let addr = supervisor.address();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        tx.send(exchange_raw(
            addr,
            "POST",
            "/knolo/infer/v1/complete",
            Some(
                r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"hi","role":"user"}],"model":"micro","requestId":"job-stream","stream":true}"#,
            ),
        ))
        .unwrap();
    });
    let start = std::time::Instant::now();
    while supervisor.inflight() < 1 {
        assert!(start.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    }
    let cancel = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/cancel",
        Some(r#"{"requestId":"job-stream"}"#),
    );
    assert_eq!(cancel.status, 200, "{}", cancel.body);
    supervisor.release_forward().unwrap();
    let raw = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(raw.status, 200, "{}", raw.body);
    assert!(raw.body.contains("knolo.usage"));
    assert!(raw.body.contains("\"finishReason\":\"cancelled\""));
    assert!(!raw.body.contains("knolo.delta"));
    let mut dropped = TcpStream::connect(addr).unwrap();
    dropped
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let payload = r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"hi","role":"user"}],"model":"micro","stream":true}"#;
    let request = format!(
        "POST /knolo/infer/v1/complete HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    dropped.write_all(request.as_bytes()).unwrap();
    let mut buf = Vec::new();
    let mut chunk = [0u8; 256];
    while !String::from_utf8_lossy(&buf).contains("knolo.accepted") {
        let n = dropped.read(&mut chunk).unwrap();
        assert!(n > 0, "stream closed before admission");
        buf.extend_from_slice(&chunk[..n]);
    }
    drop(dropped);
    let (expected, _) = alone(&dir, "at", &greedy(2), ServiceClass::Standard);
    let reply = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("at", r#""maxOutputTokens":2"#, "")),
    );
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(tokens_of(&reply.body), expected);
    supervisor.shutdown();
}

fn series(body: &str, name: &str) -> f64 {
    let prefix = format!("{name} ");
    let line = body
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("missing {name}\n{body}"));
    line[prefix.len()..].trim().parse().expect(name)
}

#[test]
fn metrics_are_prometheus_text_without_user_content() {
    let dir = scratch();
    let supervisor = start(&dir, true);
    let addr = supervisor.address();
    let idle = exchange_raw(addr, "GET", "/metrics", None);
    assert_eq!(idle.status, 200);
    assert!(idle.head.to_ascii_lowercase().contains("text/plain"));
    assert!(idle.head.contains("version=0.0.4"));
    let native = exchange_raw(addr, "GET", "/knolo/infer/v1/metrics", None);
    assert_eq!(native.body, idle.body);
    assert_eq!(series(&idle.body, "knolo_infer_worker_up"), 1.0);
    assert_eq!(series(&idle.body, "knolo_infer_metrics_stale"), 0.0);
    assert_eq!(series(&idle.body, "knolo_infer_worker_restarts_total"), 0.0);
    assert_eq!(
        series(&idle.body, "knolo_infer_kv_pages{state=\"total\"}"),
        8.0
    );
    assert_eq!(
        series(&idle.body, "knolo_infer_kv_pages{state=\"free\"}"),
        8.0
    );
    assert_eq!(
        series(&idle.body, "knolo_infer_kv_pages{state=\"pinned\"}"),
        0.0
    );
    assert_eq!(
        series(&idle.body, "knolo_infer_prefix_reused_tokens_total"),
        0.0
    );
    assert_eq!(
        series(&idle.body, "knolo_infer_oom_total{kind=\"cuda\"}"),
        0.0
    );
    assert!(series(&idle.body, "knolo_infer_verified_bytes") > 0.0);
    let refused = exchange(addr, "POST", "/metrics", Some(""));
    assert_eq!(refused.status, 405, "{}", refused.body);

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        tx.send(exchange(
            addr,
            "POST",
            "/knolo/infer/v1/complete",
            Some(
                r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"hi","role":"user"}],"model":"micro","requestId":"job-metric-1","serviceClass":"interactive"}"#,
            ),
        ))
        .unwrap();
    });
    let start = std::time::Instant::now();
    while supervisor.inflight() < 1 {
        assert!(start.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    }
    let queued = exchange_raw(addr, "GET", "/metrics", None).body;
    assert_eq!(
        series(&queued, "knolo_infer_queue_depth{class=\"interactive\"}"),
        1.0
    );
    assert_eq!(series(&queued, "knolo_infer_active_sequences"), 1.0);
    assert_eq!(
        series(&queued, "knolo_infer_kv_pages{state=\"pinned\"}"),
        0.0
    );
    assert!(!queued.contains("job-metric-1"));
    assert!(!queued.contains("\"hi\""));
    let cancel = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/cancel",
        Some(r#"{"requestId":"job-metric-1"}"#),
    );
    assert_eq!(cancel.status, 200, "{}", cancel.body);
    supervisor.release_forward().unwrap();
    let reply = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(reply.body["output"]["finishReason"], "cancelled");
    assert!(tokens_of(&reply.body).is_empty());
    let (expected, finish) = alone(&dir, "hi", &greedy(2), ServiceClass::Interactive);
    let followed = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion(
            "hi",
            r#""maxOutputTokens":2"#,
            r#","serviceClass":"interactive""#,
        )),
    );
    assert_eq!(followed.status, 200, "{}", followed.body);
    assert_eq!(tokens_of(&followed.body), expected);
    assert_eq!(followed.body["output"]["finishReason"], finish);
    let done = exchange_raw(addr, "GET", "/metrics", None).body;
    assert!(series(&done, "knolo_infer_cancellations_total") >= 1.0);
    assert!(!done.contains("job-metric-1"));
    assert!(series(&done, "knolo_infer_decode_tokens_total") >= 1.0);
    assert!(series(&done, "knolo_infer_prefill_tokens_total") >= 1.0);
    assert!(series(&done, "knolo_infer_scheduler_iteration_seconds_count") >= 1.0);
    assert_eq!(series(&done, "knolo_infer_prefix_cache_hits_total"), 0.0);
    assert_eq!(
        series(&done, "knolo_infer_queue_depth{class=\"interactive\"}"),
        0.0
    );
    assert!(
        series(
            &done,
            &format!(
                "knolo_infer_request_duration_seconds_count{{class=\"interactive\",outcome=\"{finish}\"}}"
            ),
        ) >= 1.0
    );
    assert!(
        series(
            &done,
            "knolo_infer_time_to_first_token_seconds_count{class=\"interactive\"}",
        ) >= 1.0
    );
    assert!(series(&done, "knolo_infer_receipt_finalize_seconds_count") >= 1.0);

    let limited = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("hi", r#""maxOutputTokens":16"#, "")),
    );
    assert_eq!(limited.status, 400, "{}", limited.body);
    assert_eq!(limited.body["error"]["code"], "CONTEXT_LIMIT_EXCEEDED");
    let rejected = exchange_raw(addr, "GET", "/metrics", None).body;
    assert!(
        series(
            &rejected,
            "knolo_infer_admission_rejected_total{code=\"CONTEXT_LIMIT_EXCEEDED\"}",
        ) >= 1.0
    );

    supervisor.kill_worker().unwrap();
    let start = std::time::Instant::now();
    while supervisor.worker_label() != "down" {
        assert!(start.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    }
    let start = std::time::Instant::now();
    let down = loop {
        let down = exchange_raw(addr, "GET", "/metrics", None).body;
        if series(&down, "knolo_infer_worker_up") == 0.0
            && series(&down, "knolo_infer_worker_restarts_total") >= 1.0
        {
            break down;
        }
        assert!(start.elapsed() < Duration::from_secs(5), "{down}");
        thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(
        series(&down, "knolo_infer_queue_depth{class=\"interactive\"}"),
        0.0
    );
    assert_eq!(series(&down, "knolo_infer_kv_pages{state=\"total\"}"), 0.0);
    assert!(series(&down, "knolo_infer_decode_tokens_total") >= 1.0);
    let (again, _) = alone(&dir, "at", &greedy(2), ServiceClass::Standard);
    let restarted = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("at", r#""maxOutputTokens":2"#, "")),
    );
    assert_eq!(restarted.status, 200, "{}", restarted.body);
    assert_eq!(tokens_of(&restarted.body), again);
    supervisor.shutdown();
}

fn json_strings(value: &Value) -> Vec<String> {
    let mut found = Vec::new();
    collect_strings(value, &mut found);
    found
}

fn collect_strings(value: &Value, found: &mut Vec<String>) {
    match value {
        Value::String(text) => found.push(text.clone()),
        Value::Array(items) => {
            for item in items {
                collect_strings(item, found);
            }
        }
        Value::Object(map) => {
            for (key, child) in map {
                found.push(key.clone());
                collect_strings(child, found);
            }
        }
        Value::Number(_) | Value::Bool(_) | Value::Null => {}
    }
}

fn assert_redacted(value: &Value, request_id: &str) {
    let allowed = [
        "api",
        "prompt",
        "admission",
        "prefill",
        "decode",
        "finalize",
        "interactive",
        "standard",
        "batch",
        "background",
        "admitted",
        "rejected",
        "stop",
        "length",
        "cancelled",
        "error",
        "class",
        "code",
        "chunk",
        "index",
        "outcome",
        "promptRoot",
        "promptTokens",
        "receiptRoot",
        "requestId",
        "stage",
        "stream",
        "tokens",
    ];
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                assert!(allowed.contains(&key.as_str()), "{key}");
                assert_redacted(child, request_id);
            }
        }
        Value::Array(items) => {
            for child in items {
                assert_redacted(child, request_id);
            }
        }
        Value::String(text) => {
            let digest = text.starts_with("sha256-")
                && text.len() == "sha256-".len() + 64
                && text.as_bytes()["sha256-".len()..]
                    .iter()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase());
            assert!(
                allowed.contains(&text.as_str())
                    || text == request_id
                    || digest
                    || ErrorCode::parse(text).is_some(),
                "{text}"
            );
        }
        Value::Number(number) => assert!(number.is_u64(), "{number}"),
        Value::Bool(_) | Value::Null => {}
    }
}

fn trace_lines(dir: &Path, request_id: &str) -> Vec<Value> {
    let text = fs::read_to_string(
        dir.join("home/traces/by-id")
            .join(format!("{request_id}.jsonl")),
    )
    .unwrap_or_else(|err| panic!("{request_id} trace: {err}"));
    text.lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn traces_follow_the_request_and_omit_content() {
    let dir = scratch();
    let supervisor = start(&dir, false);
    let addr = supervisor.address();
    let by_id = dir.join("home/traces/by-id");
    assert!(fs::read_dir(&by_id).unwrap().next().is_none());
    let _ = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    let _ = exchange_raw(addr, "GET", "/metrics", None);
    assert!(fs::read_dir(&by_id).unwrap().next().is_none());

    let malformed = exchange(addr, "POST", "/knolo/infer/v1/complete", Some("NEEDLE"));
    assert_eq!(malformed.status, 400, "{}", malformed.body);
    let rejected = fs::read_to_string(dir.join("home/traces/rejected.jsonl")).unwrap();
    assert!(!rejected.contains("NEEDLE"));
    assert!(rejected.contains("CONTRACT_INVALID"));
    assert!(fs::read_dir(&by_id).unwrap().next().is_none());

    let unknown = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(
            r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"NEEDLE","role":"user"}],"model":"micro","requestId":"job-trace-bad"}"#,
        ),
    );
    assert_eq!(unknown.status, 400, "{}", unknown.body);
    let bad = trace_lines(&dir, "job-trace-bad");
    let bad_text = fs::read_to_string(by_id.join("job-trace-bad.jsonl")).unwrap();
    assert!(!bad_text.contains("NEEDLE"));
    assert_eq!(bad[0]["stage"], "api");
    assert_eq!(bad[1]["stage"], "prompt");
    assert_eq!(bad[1]["code"], "TOKENIZER_INVALID");
    assert_eq!(bad.last().unwrap()["stage"], "finalize");
    assert_eq!(bad.last().unwrap()["code"], "TOKENIZER_INVALID");
    assert!(bad.last().unwrap().get("receiptRoot").is_none());
    for line in &bad {
        assert_redacted(line, "job-trace-bad");
    }

    let generation = greedy(2);
    let (expected, finish) = alone(&dir, "south", &generation, ServiceClass::Standard);
    let bytes = fs::read(dir.join("micro.kmodel")).unwrap();
    let verification = verify_image(&bytes).unwrap();
    let sampler = build_sampler(&verification.image, &generation).unwrap();
    let compiled = compile_model_prompt(
        &verification.image,
        vec![ChatMessageV1 {
            role: "user".into(),
            content: "south".into(),
        }],
        VOCAB as u32,
        MAX_CONTEXT,
        sampler.settings.max_output_tokens,
    )
    .unwrap();
    let prompt_root = compiled.plan.root().unwrap().to_string();
    let prompt_tokens = u32::try_from(compiled.plan.token_ids.len()).unwrap();
    let reply = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(
            r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"south","role":"user"}],"model":"micro","requestId":"job-trace-1"}"#,
        ),
    );
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(tokens_of(&reply.body), expected);
    assert_eq!(reply.body["output"]["finishReason"], finish);
    let lines = trace_lines(&dir, "job-trace-1");
    let text = fs::read_to_string(by_id.join("job-trace-1.jsonl")).unwrap();
    assert!(!text.contains("south"));
    let output = reply.body["output"]["text"].as_str().unwrap();
    assert!(
        !output.is_empty(),
        "the fixture must decode to text so redaction can be checked"
    );
    assert!(
        lines
            .iter()
            .all(|line| !json_strings(line).iter().any(|text| text == output)),
        "{output}"
    );
    for line in &lines {
        assert_redacted(line, "job-trace-1");
    }
    assert_eq!(lines[0]["stage"], "api");
    assert_eq!(lines[0]["class"], "standard");
    assert_eq!(lines[0]["stream"], false);
    assert_eq!(lines[1]["stage"], "prompt");
    assert_eq!(lines[1]["promptRoot"], prompt_root);
    assert_eq!(lines[1]["promptTokens"], prompt_tokens);
    assert_eq!(lines[2]["stage"], "admission");
    assert_eq!(lines[2]["outcome"], "admitted");
    assert_eq!(lines.last().unwrap()["stage"], "finalize");
    assert_eq!(lines.last().unwrap()["outcome"], finish);
    assert_eq!(
        lines.last().unwrap()["receiptRoot"],
        reply.body["receipt"]["receiptRoot"]
    );
    let middle = &lines[3..lines.len() - 1];
    let prefill: Vec<_> = middle
        .iter()
        .take_while(|line| line["stage"] == "prefill")
        .collect();
    let decode: Vec<_> = middle
        .iter()
        .skip(prefill.len())
        .inspect(|line| assert_eq!(line["stage"], "decode"))
        .collect();
    assert!(!prefill.is_empty());
    let mut covered = 0u32;
    for (index, line) in prefill.iter().enumerate() {
        assert_eq!(line["chunk"], index as u64);
        let tokens = line["tokens"].as_u64().unwrap() as u32;
        assert!(tokens > 0 && tokens <= 4);
        covered += tokens;
    }
    assert_eq!(covered, prompt_tokens);
    assert_eq!(decode.len(), expected.len());
    for (index, line) in decode.iter().enumerate() {
        assert_eq!(line["index"], index as u64);
        assert!(line.get("tokenId").is_none());
    }
    let before = fs::read_to_string(by_id.join("job-trace-1.jsonl")).unwrap();
    let _ = exchange_raw(addr, "GET", "/metrics", None);
    assert_eq!(
        fs::read_to_string(by_id.join("job-trace-1.jsonl")).unwrap(),
        before
    );
    let metrics = exchange_raw(addr, "GET", "/knolo/infer/v1/metrics", None).body;
    assert!(!metrics.contains("job-trace-1"));
    assert!(!metrics.contains("south"));

    let again = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(
            r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"south","role":"user"}],"model":"micro","requestId":"job-trace-2"}"#,
        ),
    );
    assert_eq!(again.status, 200, "{}", again.body);
    assert_eq!(tokens_of(&again.body), expected);
    supervisor.shutdown();
}

fn trace_names(dir: &Path) -> Vec<String> {
    let mut names = fs::read_dir(dir.join("home/traces/by-id"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    names
}

#[test]
fn drain_finishes_admitted_work_and_refuses_new_completions() {
    let dir = scratch();
    let supervisor = start(&dir, true);
    let addr = supervisor.address();
    let worker_pid = supervisor.worker_pid().unwrap();
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.body["status"], "ok");
    assert_eq!(health.body["lifecycle"], "serving");
    let bad = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/drain",
        Some(r#"{"extra":true}"#),
    );
    assert_eq!(bad.status, 400, "{}", bad.body);
    assert_eq!(bad.body["error"]["code"], "CONTRACT_INVALID");
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.body["lifecycle"], "serving");
    let method = exchange(addr, "GET", "/knolo/infer/v1/drain", None);
    assert_eq!(method.status, 405);

    let generation = greedy(2);
    let (expected, finish) = alone(&dir, "hi", &generation, ServiceClass::Standard);
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        tx.send(exchange(
            addr,
            "POST",
            "/knolo/infer/v1/complete",
            Some(
                r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"hi","role":"user"}],"model":"micro","requestId":"job-drain-1"}"#,
            ),
        ))
        .unwrap();
    });
    let start = std::time::Instant::now();
    while supervisor.inflight() < 1 {
        assert!(start.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    }
    let drain = exchange(addr, "POST", "/knolo/infer/v1/drain", Some("{}"));
    assert_eq!(drain.status, 200, "{}", drain.body);
    assert_eq!(drain.body["status"], "draining");
    assert!(drain.body["inflight"].as_u64().unwrap() >= 1);
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.status, 200);
    assert_eq!(health.body["status"], "draining");
    assert_eq!(health.body["lifecycle"], "draining");
    assert_eq!(health.body["worker"], "ready");

    let refused = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(
            r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"NEEDLE-drain","role":"user"}],"model":"micro","requestId":"job-drain-new"}"#,
        ),
    );
    assert_eq!(refused.status, 503, "{}", refused.body);
    assert_eq!(refused.body["error"]["code"], "SERVICE_DRAINING");
    let openai = exchange(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"messages":[{"content":"NEEDLE-drain","role":"user"}],"model":"micro"}"#),
    );
    assert_eq!(openai.status, 503, "{}", openai.body);
    assert_eq!(openai.body["error"]["code"], "SERVICE_DRAINING");
    assert_eq!(openai.body["error"]["type"], "server_error");
    let rejected = fs::read_to_string(dir.join("home/traces/rejected.jsonl")).unwrap();
    assert!(rejected.contains("SERVICE_DRAINING"));
    assert!(!rejected.contains("NEEDLE-drain"));
    assert!(!rejected.contains("job-drain-new"));
    assert_eq!(trace_names(&dir), vec!["job-drain-1.jsonl".to_string()]);
    let metrics = exchange_raw(addr, "GET", "/metrics", None);
    assert_eq!(metrics.status, 200);
    assert_eq!(
        series(&metrics.body, "knolo_infer_worker_restarts_total"),
        0.0
    );
    assert_eq!(series(&metrics.body, "knolo_infer_worker_up"), 1.0);
    assert_eq!(
        series(
            &metrics.body,
            "knolo_infer_admission_rejected_total{code=\"OTHER\"}"
        ),
        2.0
    );
    assert!(!metrics.body.contains("NEEDLE-drain"));
    assert_eq!(supervisor.worker_pid(), Some(worker_pid));

    supervisor.release_forward().unwrap();
    let reply = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(tokens_of(&reply.body), expected);
    assert_eq!(reply.body["output"]["finishReason"], finish);
    assert!(reply.body["receipt"]["receiptRoot"]
        .as_str()
        .unwrap()
        .starts_with("sha256-"));
    let start = std::time::Instant::now();
    loop {
        let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
        if health.body["status"] == "drained" {
            assert_eq!(health.body["lifecycle"], "drained");
            assert_eq!(health.body["worker"], "ready");
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    }
    let again = exchange(addr, "POST", "/knolo/infer/v1/drain", Some("{}"));
    assert_eq!(again.status, 200, "{}", again.body);
    assert_eq!(again.body["status"], "drained");
    assert_eq!(again.body["inflight"], 0);
    let still = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("at", r#""maxOutputTokens":2"#, "")),
    );
    assert_eq!(still.status, 503, "{}", still.body);
    assert_eq!(still.body["error"]["code"], "SERVICE_DRAINING");
    assert_eq!(supervisor.worker_pid(), Some(worker_pid));
    supervisor.drain_for_shutdown();
    let after = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(after.body["lifecycle"], "drained");
    supervisor.shutdown();
}

#[test]
fn unload_when_idle_stops_the_worker_without_a_restart() {
    let dir = scratch();
    let supervisor = start(&dir, false);
    let addr = supervisor.address();
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.body["lifecycle"], "serving");
    assert_eq!(health.body["worker"], "ready");
    let bad = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/models/unload",
        Some(r#"{"extra":true}"#),
    );
    assert_eq!(bad.status, 400, "{}", bad.body);
    assert_eq!(bad.body["error"]["code"], "CONTRACT_INVALID");
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.body["lifecycle"], "serving");
    assert_eq!(health.body["worker"], "ready");
    let method = exchange(addr, "GET", "/knolo/infer/v1/models/unload", None);
    assert_eq!(method.status, 405);

    let unload = exchange(addr, "POST", "/knolo/infer/v1/models/unload", Some("{}"));
    assert_eq!(unload.status, 200, "{}", unload.body);
    assert_eq!(unload.body["status"], "unloaded");
    assert_eq!(unload.body["inflight"], 0);
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.status, 200);
    assert_eq!(health.body["lifecycle"], "unloaded");
    assert_eq!(health.body["status"], "unloaded");
    assert_eq!(health.body["worker"], "down");
    assert!(supervisor.worker_pid().is_none());
    let metrics = exchange_raw(addr, "GET", "/metrics", None);
    assert_eq!(metrics.status, 200);
    assert_eq!(
        series(&metrics.body, "knolo_infer_worker_restarts_total"),
        0.0
    );
    assert_eq!(series(&metrics.body, "knolo_infer_worker_up"), 0.0);
    assert_eq!(
        series(&metrics.body, "knolo_infer_kv_pages{state=\"total\"}"),
        0.0
    );

    let refused = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(
            r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"NEEDLE-unload","role":"user"}],"model":"micro","requestId":"job-unload-new"}"#,
        ),
    );
    assert_eq!(refused.status, 503, "{}", refused.body);
    assert_eq!(refused.body["error"]["code"], "SERVICE_UNLOADED");
    let openai = exchange(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"messages":[{"content":"NEEDLE-unload","role":"user"}],"model":"micro"}"#),
    );
    assert_eq!(openai.status, 503, "{}", openai.body);
    assert_eq!(openai.body["error"]["code"], "SERVICE_UNLOADED");
    assert_eq!(openai.body["error"]["type"], "server_error");
    let rejected = fs::read_to_string(dir.join("home/traces/rejected.jsonl")).unwrap();
    assert!(rejected.contains("SERVICE_UNLOADED"));
    assert!(!rejected.contains("NEEDLE-unload"));
    assert!(!rejected.contains("job-unload-new"));
    assert!(supervisor.worker_pid().is_none());
    assert_eq!(
        series(
            &exchange_raw(addr, "GET", "/metrics", None).body,
            "knolo_infer_admission_rejected_total{code=\"OTHER\"}"
        ),
        2.0
    );

    let again = exchange(addr, "POST", "/knolo/infer/v1/models/unload", Some("{}"));
    assert_eq!(again.status, 200, "{}", again.body);
    assert_eq!(again.body["status"], "unloaded");
    assert_eq!(again.body["inflight"], 0);
    let drain = exchange(addr, "POST", "/knolo/infer/v1/drain", Some("{}"));
    assert_eq!(drain.status, 200, "{}", drain.body);
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.body["lifecycle"], "unloaded");
    assert_eq!(health.body["worker"], "down");
    assert!(supervisor.worker_pid().is_none());
    assert_eq!(
        series(
            &exchange_raw(addr, "GET", "/metrics", None).body,
            "knolo_infer_worker_restarts_total"
        ),
        0.0
    );
    supervisor.shutdown();
}

#[test]
fn unload_finishes_admitted_work_then_drops_the_worker() {
    let dir = scratch();
    let supervisor = start(&dir, true);
    let addr = supervisor.address();
    let worker_pid = supervisor.worker_pid().unwrap();
    let generation = greedy(2);
    let (expected, finish) = alone(&dir, "hi", &generation, ServiceClass::Standard);
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        tx.send(exchange(
            addr,
            "POST",
            "/knolo/infer/v1/complete",
            Some(
                r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"hi","role":"user"}],"model":"micro","requestId":"job-unload-1"}"#,
            ),
        ))
        .unwrap();
    });
    let start = std::time::Instant::now();
    while supervisor.inflight() < 1 {
        assert!(start.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    }
    let (cancel_tx, cancel_rx) = mpsc::channel();
    thread::spawn(move || {
        cancel_tx
            .send(exchange(
                addr,
                "POST",
                "/knolo/infer/v1/complete",
                Some(
                    r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"at","role":"user"}],"model":"micro","requestId":"job-unload-cancel"}"#,
                ),
            ))
            .unwrap();
    });
    let start = std::time::Instant::now();
    while supervisor.inflight() < 2 {
        assert!(start.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    }

    let unload = exchange(addr, "POST", "/knolo/infer/v1/models/unload", Some("{}"));
    assert_eq!(unload.status, 200, "{}", unload.body);
    assert_eq!(unload.body["status"], "unloading");
    assert!(unload.body["inflight"].as_u64().unwrap() >= 2);
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.body["lifecycle"], "unloading");
    assert_eq!(health.body["status"], "unloading");
    assert_eq!(health.body["worker"], "ready");
    assert_eq!(supervisor.worker_pid(), Some(worker_pid));

    let refused = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(
            r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"NEEDLE-unload","role":"user"}],"model":"micro"}"#,
        ),
    );
    assert_eq!(refused.status, 503, "{}", refused.body);
    assert_eq!(refused.body["error"]["code"], "SERVICE_UNLOADED");
    let rejected = fs::read_to_string(dir.join("home/traces/rejected.jsonl")).unwrap();
    assert!(rejected.contains("SERVICE_UNLOADED"));
    assert!(!rejected.contains("NEEDLE-unload"));
    assert_eq!(supervisor.worker_pid(), Some(worker_pid));

    let cancel = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/cancel",
        Some(r#"{"requestId":"job-unload-cancel"}"#),
    );
    assert_eq!(cancel.status, 200, "{}", cancel.body);
    assert_eq!(cancel.body["status"], "cancelled");
    assert_eq!(supervisor.worker_pid(), Some(worker_pid));

    supervisor.release_forward().unwrap();
    let reply = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(tokens_of(&reply.body), expected);
    assert_eq!(reply.body["output"]["finishReason"], finish);
    assert!(reply.body["receipt"]["receiptRoot"]
        .as_str()
        .unwrap()
        .starts_with("sha256-"));
    let cancelled = cancel_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(cancelled.status, 200, "{}", cancelled.body);
    assert_eq!(cancelled.body["output"]["finishReason"], "cancelled");
    assert!(tokens_of(&cancelled.body).is_empty());
    assert!(cancelled.body.get("receipt").is_none());

    let start = std::time::Instant::now();
    loop {
        let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
        if health.body["lifecycle"] == "unloaded" {
            assert_eq!(health.body["status"], "unloaded");
            assert_eq!(health.body["worker"], "down");
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    }
    assert!(supervisor.worker_pid().is_none());
    let metrics = exchange_raw(addr, "GET", "/metrics", None);
    assert_eq!(
        series(&metrics.body, "knolo_infer_worker_restarts_total"),
        0.0
    );
    assert_eq!(series(&metrics.body, "knolo_infer_worker_up"), 0.0);
    assert_eq!(
        series(&metrics.body, "knolo_infer_kv_pages{state=\"total\"}"),
        0.0
    );
    let still = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("at", r#""maxOutputTokens":2"#, "")),
    );
    assert_eq!(still.status, 503, "{}", still.body);
    assert_eq!(still.body["error"]["code"], "SERVICE_UNLOADED");
    assert!(supervisor.worker_pid().is_none());
    supervisor.shutdown();
}

#[test]
fn load_after_unload_restores_the_same_tokens() {
    let dir = scratch();
    let supervisor = start(&dir, false);
    let addr = supervisor.address();
    let (expected, finish) = alone(&dir, "hi", &greedy(2), ServiceClass::Standard);
    let before = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("hi", r#""maxOutputTokens":2"#, "")),
    );
    assert_eq!(before.status, 200, "{}", before.body);
    assert_eq!(tokens_of(&before.body), expected);
    let first_pid = supervisor.worker_pid().unwrap();

    let unload = exchange(addr, "POST", "/knolo/infer/v1/models/unload", Some("{}"));
    assert_eq!(unload.status, 200, "{}", unload.body);
    assert_eq!(unload.body["status"], "unloaded");
    assert!(supervisor.worker_pid().is_none());

    let bad = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/models/load",
        Some(r#"{"extra":true}"#),
    );
    assert_eq!(bad.status, 400, "{}", bad.body);
    assert_eq!(bad.body["error"]["code"], "CONTRACT_INVALID");
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.body["lifecycle"], "unloaded");
    assert_eq!(health.body["worker"], "down");
    let method = exchange(addr, "GET", "/knolo/infer/v1/models/load", None);
    assert_eq!(method.status, 405);
    assert!(supervisor.worker_pid().is_none());

    let load = exchange(addr, "POST", "/knolo/infer/v1/models/load", Some("{}"));
    assert_eq!(load.status, 200, "{}", load.body);
    assert_eq!(load.body["status"], "serving");
    assert_eq!(load.body["inflight"], 0);
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.body["lifecycle"], "serving");
    assert_eq!(health.body["status"], "ok");
    assert_eq!(health.body["worker"], "ready");
    let loaded_pid = supervisor.worker_pid().unwrap();
    assert_ne!(loaded_pid, first_pid);
    assert_eq!(
        series(
            &exchange_raw(addr, "GET", "/metrics", None).body,
            "knolo_infer_worker_restarts_total"
        ),
        0.0
    );

    let again = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("hi", r#""maxOutputTokens":2"#, "")),
    );
    assert_eq!(again.status, 200, "{}", again.body);
    assert_eq!(tokens_of(&again.body), expected);
    assert_eq!(again.body["output"]["finishReason"], finish);
    assert!(again.body["receipt"]["receiptRoot"]
        .as_str()
        .unwrap()
        .starts_with("sha256-"));
    let openai = exchange(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"max_tokens":2,"messages":[{"content":"hi","role":"user"}],"model":"micro"}"#),
    );
    assert_eq!(openai.status, 200, "{}", openai.body);
    assert_eq!(
        openai.body["choices"][0]["message"]["content"],
        again.body["output"]["text"]
    );

    let second = exchange(addr, "POST", "/knolo/infer/v1/models/load", Some("{}"));
    assert_eq!(second.status, 200, "{}", second.body);
    assert_eq!(second.body["status"], "serving");
    assert_eq!(supervisor.worker_pid(), Some(loaded_pid));

    let unload = exchange(addr, "POST", "/knolo/infer/v1/models/unload", Some("{}"));
    assert_eq!(unload.body["status"], "unloaded");
    let drain = exchange(addr, "POST", "/knolo/infer/v1/drain", Some("{}"));
    assert_eq!(drain.status, 200, "{}", drain.body);
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.body["lifecycle"], "unloaded");
    assert!(supervisor.worker_pid().is_none());
    let load = exchange(addr, "POST", "/knolo/infer/v1/models/load", Some("{}"));
    assert_eq!(load.status, 200, "{}", load.body);
    assert_eq!(load.body["status"], "drained");
    assert_eq!(load.body["inflight"], 0);
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.body["lifecycle"], "drained");
    assert_eq!(health.body["status"], "drained");
    assert_eq!(health.body["worker"], "ready");
    assert!(supervisor.worker_pid().is_some());
    let refused = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(
            r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"NEEDLE-load","role":"user"}],"model":"micro"}"#,
        ),
    );
    assert_eq!(refused.status, 503, "{}", refused.body);
    assert_eq!(refused.body["error"]["code"], "SERVICE_DRAINING");
    let rejected = fs::read_to_string(dir.join("home/traces/rejected.jsonl")).unwrap();
    assert!(rejected.contains("SERVICE_DRAINING"));
    assert!(!rejected.contains("NEEDLE-load"));
    assert_eq!(
        series(
            &exchange_raw(addr, "GET", "/metrics", None).body,
            "knolo_infer_worker_restarts_total"
        ),
        0.0
    );
    supervisor.shutdown();
}

#[test]
fn load_during_unload_does_not_replace_the_worker() {
    let dir = scratch();
    let supervisor = start(&dir, true);
    let addr = supervisor.address();
    let worker_pid = supervisor.worker_pid().unwrap();
    let (expected, finish) = alone(&dir, "hi", &greedy(2), ServiceClass::Standard);
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        tx.send(exchange(
            addr,
            "POST",
            "/knolo/infer/v1/complete",
            Some(
                r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"hi","role":"user"}],"model":"micro","requestId":"job-load-1"}"#,
            ),
        ))
        .unwrap();
    });
    let start = std::time::Instant::now();
    while supervisor.inflight() < 1 {
        assert!(start.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    }

    let unload = exchange(addr, "POST", "/knolo/infer/v1/models/unload", Some("{}"));
    assert_eq!(unload.status, 200, "{}", unload.body);
    assert_eq!(unload.body["status"], "unloading");
    let load = exchange(addr, "POST", "/knolo/infer/v1/models/load", Some("{}"));
    assert_eq!(load.status, 200, "{}", load.body);
    assert_eq!(load.body["status"], "unloading");
    assert!(load.body["inflight"].as_u64().unwrap() >= 1);
    assert_eq!(supervisor.worker_pid(), Some(worker_pid));
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.body["lifecycle"], "unloading");
    assert_eq!(health.body["worker"], "ready");

    supervisor.release_forward().unwrap();
    let reply = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(tokens_of(&reply.body), expected);
    assert_eq!(reply.body["output"]["finishReason"], finish);

    let start = std::time::Instant::now();
    loop {
        let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
        if health.body["lifecycle"] == "unloaded" {
            assert_eq!(health.body["worker"], "down");
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(5));
        thread::sleep(Duration::from_millis(10));
    }
    assert!(supervisor.worker_pid().is_none());
    assert_eq!(
        series(
            &exchange_raw(addr, "GET", "/metrics", None).body,
            "knolo_infer_worker_restarts_total"
        ),
        0.0
    );

    let load = exchange(addr, "POST", "/knolo/infer/v1/models/load", Some("{}"));
    assert_eq!(load.status, 200, "{}", load.body);
    assert_eq!(load.body["status"], "serving");
    let health = exchange(addr, "GET", "/knolo/infer/v1/health", None);
    assert_eq!(health.body["lifecycle"], "serving");
    assert_eq!(health.body["worker"], "ready");
    let restored = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("hi", r#""maxOutputTokens":2"#, "")),
    );
    assert_eq!(restored.status, 200, "{}", restored.body);
    assert_eq!(tokens_of(&restored.body), expected);
    assert_ne!(supervisor.worker_pid(), Some(worker_pid));
    assert_eq!(
        series(
            &exchange_raw(addr, "GET", "/metrics", None).body,
            "knolo_infer_worker_restarts_total"
        ),
        0.0
    );
    supervisor.shutdown();
}

fn worker_rss_kib(pid: u32) -> u64 {
    let text = fs::read_to_string(format!("/proc/{pid}/status"))
        .unwrap_or_else(|err| panic!("worker {pid} status is not readable: {err}"));
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest
                .split_whitespace()
                .next()
                .unwrap_or_else(|| panic!("VmRSS has no value: {line}"))
                .parse()
                .unwrap_or_else(|err| panic!("VmRSS is not an integer: {err}"));
        }
    }
    panic!("worker {pid} status has no VmRSS");
}

#[test]
fn a_bounded_memory_soak_keeps_the_page_pool_and_drops_finished_slots() {
    const COMPLETIONS: u32 = 32;
    const RSS_SLACK_KIB: u64 = 8 * 1024;
    let dir = scratch();
    let supervisor = start(&dir, false);
    let addr = supervisor.address();
    let (expected, finish) = alone(&dir, "hi", &greedy(2), ServiceClass::Standard);
    let pid = supervisor.worker_pid().unwrap();
    let mut baseline = None;
    for n in 0..COMPLETIONS {
        let reply = exchange(
            addr,
            "POST",
            "/knolo/infer/v1/complete",
            Some(&format!(
                r#"{{"generation":{{"maxOutputTokens":2}},"messages":[{{"content":"hi","role":"user"}}],"model":"micro","requestId":"soak-{n}"}}"#
            )),
        );
        assert_eq!(reply.status, 200, "{}", reply.body);
        assert_eq!(tokens_of(&reply.body), expected);
        assert_eq!(reply.body["output"]["finishReason"], finish);
        assert_eq!(supervisor.worker_pid(), Some(pid));
        let metrics = exchange_raw(addr, "GET", "/metrics", None);
        assert_eq!(metrics.status, 200);
        assert_eq!(
            series(&metrics.body, "knolo_infer_kv_pages{state=\"total\"}"),
            f64::from(CPU_KV_PAGE_POOL)
        );
        assert_eq!(
            series(&metrics.body, "knolo_infer_kv_pages{state=\"free\"}"),
            f64::from(CPU_KV_PAGE_POOL)
        );
        assert_eq!(
            series(&metrics.body, "knolo_infer_kv_pages{state=\"pinned\"}"),
            0.0
        );
        assert_eq!(series(&metrics.body, "knolo_infer_active_sequences"), 0.0);
        assert_eq!(series(&metrics.body, "knolo_infer_retained_sequences"), 0.0);
        let rss = worker_rss_kib(pid);
        if baseline.is_none() {
            baseline = Some(rss);
        }
    }
    let growth = worker_rss_kib(pid).saturating_sub(baseline.unwrap());
    assert!(growth <= RSS_SLACK_KIB, "worker RSS grew by {growth} KiB");
    let repeated = exchange(
        addr,
        "POST",
        "/knolo/infer/v1/complete",
        Some(
            r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"hi","role":"user"}],"model":"micro","requestId":"soak-0"}"#,
        ),
    );
    assert_eq!(repeated.status, 400, "{}", repeated.body);
    assert_eq!(repeated.body["error"]["code"], "RECEIPT_PERSIST_FAILED");
    supervisor.shutdown();
}

fn config_for(dir: &Path) -> ServeConfig {
    ServeConfig {
        alias: "micro".into(),
        work_dir: dir.to_path_buf(),
        lock_path: dir.join("knolo.infer.lock.json"),
        weights_dir: None,
        bind: "127.0.0.1:0".parse().unwrap(),
        worker_bin: PathBuf::from(env!("CARGO_BIN_EXE_knolo-infer-worker")),
        home: dir.join("home"),
        pause_before_forward: false,
    }
}

fn plant_accepted(home: &Path, request_id: &str) -> infer_engine::Journal {
    fs::create_dir_all(home.join("journals")).unwrap();
    let mut journal = infer_engine::Journal::create(home, request_id).unwrap();
    let payload = infer_contracts::digest_value(
        "infer-execution-trace",
        &infer_contracts::CborValue::Text("accepted".into()),
    )
    .unwrap();
    journal.append("accepted", payload).unwrap();
    journal
}

fn plant_stale_lock(home: &Path, pid: u32, starttime: u64) {
    let dir = home.join("daemon.lock");
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("owner"),
        format!("pid {pid}\nstarttime {starttime}\n"),
    )
    .unwrap();
}

fn receipt_files(home: &Path) -> usize {
    let root = home.join("receipts").join("sha256");
    let Ok(entries) = fs::read_dir(root) else {
        return 0;
    };
    entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().join("receipt.cbor").is_file())
        .count()
}

#[test]
fn a_live_daemon_lock_refuses_a_second_supervisor() {
    let dir = scratch();
    let home = dir.join("home");
    let supervisor = start(&dir, false);
    plant_accepted(&home, "still-open");
    match Supervisor::start(config_for(&dir)) {
        Ok(other) => {
            other.shutdown();
            panic!("second supervisor took a live home");
        }
        Err(err) => {
            assert_eq!(err.code, ErrorCode::ContractInvalid);
            assert_eq!(err.message, "daemon lock is held");
        }
    }
    assert!(home
        .join("journals")
        .join("still-open")
        .join("000000.accepted.cbor")
        .is_file());
    assert!(!home
        .join("journals")
        .join("still-open")
        .join("000001.failed.cbor")
        .exists());
    let mode = fs::metadata(home.join("daemon.lock"))
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o700);
    let (expected, _) = alone(&dir, "hi", &greedy(2), ServiceClass::Standard);
    let reply = exchange(
        supervisor.address(),
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("hi", r#""maxOutputTokens":2"#, "")),
    );
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(tokens_of(&reply.body), expected);
    supervisor.shutdown();
    assert!(!home.join("daemon.lock").exists());
}

#[test]
fn a_stale_daemon_lock_seals_an_open_journal_and_serves_again() {
    let dir = scratch();
    let home = dir.join("home");
    fs::create_dir_all(&home).unwrap();
    plant_accepted(&home, "lost-job");
    let mut done = plant_accepted(&home, "done-job");
    let cancelled = infer_contracts::digest_value(
        "infer-execution-trace",
        &infer_contracts::CborValue::Text("cancelled".into()),
    )
    .unwrap();
    done.append("cancelled", cancelled).unwrap();
    fs::create_dir_all(home.join("journals").join("partial-job")).unwrap();
    fs::write(
        home.join("journals")
            .join("partial-job")
            .join("sampler-plan.cbor"),
        b"side",
    )
    .unwrap();
    let mut child = std::process::Command::new("/bin/true").spawn().unwrap();
    let dead = child.id();
    child.wait().unwrap();
    plant_stale_lock(&home, dead, 1);

    let supervisor = start(&dir, false);
    let lost = home.join("journals").join("lost-job");
    assert!(lost.join("000001.failed.cbor").is_file());
    assert!(!lost.join("000002.failed.cbor").exists());
    infer_engine::verify_journal(&home, "lost-job").unwrap();
    infer_engine::verify_journal(&home, "done-job").unwrap();
    assert!(!home
        .join("journals")
        .join("done-job")
        .join("000002.failed.cbor")
        .exists());
    assert!(!home.join("journals").join("partial-job").exists());
    assert_eq!(receipt_files(&home), 0);
    let metrics = exchange_raw(supervisor.address(), "GET", "/metrics", None);
    assert_eq!(
        series(&metrics.body, "knolo_infer_worker_restarts_total"),
        0.0
    );
    let (expected, finish) = alone(&dir, "at", &greedy(2), ServiceClass::Standard);
    let reply = exchange(
        supervisor.address(),
        "POST",
        "/knolo/infer/v1/complete",
        Some(
            r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"at","role":"user"}],"model":"micro","requestId":"after-restart"}"#,
        ),
    );
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(tokens_of(&reply.body), expected);
    assert_eq!(reply.body["output"]["finishReason"], finish);
    let repeated = exchange(
        supervisor.address(),
        "POST",
        "/knolo/infer/v1/complete",
        Some(
            r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"at","role":"user"}],"model":"micro","requestId":"lost-job"}"#,
        ),
    );
    assert_eq!(repeated.status, 400, "{}", repeated.body);
    assert_eq!(repeated.body["error"]["code"], "RECEIPT_PERSIST_FAILED");
    let partial = exchange(
        supervisor.address(),
        "POST",
        "/knolo/infer/v1/complete",
        Some(
            r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"at","role":"user"}],"model":"micro","requestId":"partial-job"}"#,
        ),
    );
    assert_eq!(partial.status, 200, "{}", partial.body);
    assert_eq!(tokens_of(&partial.body), expected);
    supervisor.shutdown();
    assert!(!home.join("daemon.lock").exists());

    let again = start(&dir, false);
    assert!(lost.join("000001.failed.cbor").is_file());
    assert!(!lost.join("000002.failed.cbor").exists());
    let reply = exchange(
        again.address(),
        "POST",
        "/knolo/infer/v1/complete",
        Some(
            r#"{"generation":{"maxOutputTokens":2},"messages":[{"content":"at","role":"user"}],"model":"micro","requestId":"after-second-start"}"#,
        ),
    );
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(tokens_of(&reply.body), expected);
    again.shutdown();
}

#[test]
fn a_corrupt_journal_refuses_the_daemon_and_releases_the_lock() {
    let dir = scratch();
    let home = dir.join("home");
    fs::create_dir_all(home.join("journals").join("bad-job")).unwrap();
    fs::write(
        home.join("journals")
            .join("bad-job")
            .join("000000.accepted.cbor"),
        b"not-cbor",
    )
    .unwrap();
    plant_stale_lock(&home, std::process::id(), 0);
    match Supervisor::start(config_for(&dir)) {
        Ok(supervisor) => {
            supervisor.shutdown();
            panic!("corrupt journal started a supervisor");
        }
        Err(err) => assert_eq!(err.code, ErrorCode::CanonicalCborInvalid),
    }
    assert!(!home.join("daemon.lock").exists());
    assert!(home
        .join("journals")
        .join("bad-job")
        .join("000000.accepted.cbor")
        .is_file());
    assert!(!home
        .join("journals")
        .join("bad-job")
        .join("000001.failed.cbor")
        .exists());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_worker_names_slot0_and_matches_the_oracle() {
    let dir = scratch();
    let supervisor = start(&dir, false);
    let generation = greedy(2);
    let (expected, finish) = alone(&dir, "hi", &generation, ServiceClass::Standard);
    let reply = exchange(
        supervisor.address(),
        "POST",
        "/knolo/infer/v1/complete",
        Some(&completion("hi", r#""maxOutputTokens":2"#, "")),
    );
    assert_eq!(reply.status, 200, "{}", reply.body);
    assert_eq!(tokens_of(&reply.body), expected);
    assert_eq!(reply.body["output"]["finishReason"], finish);
    assert_eq!(reply.body["receipt"]["assurance"], "compatibility");
    let root = reply.body["receipt"]["receiptRoot"].as_str().unwrap();
    let stored = fs::read(
        dir.join("home")
            .join("receipts")
            .join("sha256")
            .join(root.strip_prefix("sha256-").unwrap())
            .join("receipt.cbor"),
    )
    .unwrap();
    let receipt = infer_contracts::InferenceReceiptV1::from_cbor(
        &infer_contracts::decode_canonical(&stored).unwrap(),
    )
    .unwrap();
    let slot = infer_engine::require_cuda_slot0().unwrap();
    let bundle = infer_engine::cuda_kernel_bundle(
        infer_native::CANDLE_CPU_VERSION,
        &slot.toolkit_version,
        &slot.architecture,
    )
    .unwrap();
    assert_eq!(bundle.build_mode, "cuda");
    assert_eq!(bundle.cuda_toolkit_version, slot.toolkit_version);
    assert_eq!(bundle.cuda_architectures.len(), 1);
    assert_eq!(bundle.cuda_architectures[0], slot.architecture);
    assert!(bundle.compiler_flags.is_empty());
    assert!(bundle.jit.is_none());
    assert_eq!(receipt.hardware.device_slot, "slot-0");
    assert_eq!(receipt.engine.backend, "native");
    assert_eq!(
        receipt.engine.backend_version,
        infer_native::CANDLE_CPU_VERSION
    );
    assert_eq!(receipt.assurance, "compatibility");
    assert_eq!(receipt.engine.kernel_bundle_root, bundle.root().unwrap());
    assert_eq!(
        receipt.engine.kernel_plan_root,
        infer_engine::cuda_kernel_plan_root().unwrap()
    );
    assert_eq!(receipt.placement.kv_block_size, BLOCK_SIZE);
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    assert_eq!(
        receipt.placement.placement_root,
        infer_engine::cuda_placement(&source)
            .unwrap()
            .root()
            .unwrap()
    );
    let binary = infer_artifact::hash_regular_file(
        Path::new(env!("CARGO_BIN_EXE_knolo-infer-worker")),
        ErrorCode::WorkerStartFailed,
    )
    .unwrap();
    let engine = infer_engine::cuda_engine_build(
        binary.sha256,
        infer_native::CANDLE_CPU_VERSION,
        bundle.root().unwrap(),
    )
    .unwrap();
    assert_eq!(engine.tensor_backend, "candle-cuda");
    assert_eq!(engine.feature_set, ["cuda".to_string()]);
    assert_eq!(receipt.engine.engine_build_root, engine.root().unwrap());
    let metrics = exchange_raw(supervisor.address(), "GET", "/metrics", None);
    assert_eq!(
        series(&metrics.body, "knolo_infer_kv_pages{state=\"total\"}"),
        8.0
    );
    supervisor.shutdown();
}
