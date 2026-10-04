//! Catalog pull against a local HTTP fixture. These tests do not contact Hugging Face.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use infer_artifact::{
    checked_in_catalog, encode_gguf, hash_regular_file, list_installed, load_catalog,
    new_lockfile, pull_catalog_row, refresh_catalog, remove_installed, write_lockfile, DigestHex,
    ErrorCode, FetchPolicy, GgufMetadata, GgufTensorDraft, GgufTensorType, GgufValue, InferFailure,
    ModelPin, PullOutcome, PullRequest, LLAMA_CHAT_TEMPLATE,
};

static TEMP: AtomicU64 = AtomicU64::new(0);

fn scratch(label: &str) -> PathBuf {
    let n = TEMP.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "knolo-catalog-{}-{n}-{label}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn digest_of(bytes: &[u8]) -> (String, u64) {
    let dir = scratch("digest");
    let path = dir.join("blob");
    fs::write(&path, bytes).unwrap();
    let hashed = hash_regular_file(&path, ErrorCode::ModelArtifactMissing).unwrap();
    let _ = fs::remove_dir_all(&dir);
    (hashed.sha256.to_string(), hashed.size)
}

fn flip_digest(digest: &str) -> String {
    let mut chars: Vec<char> = digest.chars().collect();
    let last = chars.last_mut().unwrap();
    *last = if *last == '0' { '1' } else { '0' };
    chars.into_iter().collect()
}

fn json_escape(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other if other.is_control() => out.push_str(&format!("\\u{:04x}", other as u32)),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

struct CatalogSpec<'a> {
    id: &'a str,
    weight_sha: &'a str,
    weight_size: u64,
    tok_sha: &'a str,
    tok_size: u64,
    status: &'a str,
    adapter: &'a str,
    gated: bool,
    min_ram: u64,
}

impl<'a> CatalogSpec<'a> {
    fn llama(
        id: &'a str,
        weight_sha: &'a str,
        weight_size: u64,
        tok_sha: &'a str,
        tok_size: u64,
    ) -> Self {
        Self {
            id,
            weight_sha,
            weight_size,
            tok_sha,
            tok_size,
            status: "runnable",
            adapter: "knolo.llama.v1",
            gated: false,
            min_ram: 1,
        }
    }
}

fn catalog(spec: &CatalogSpec<'_>) -> String {
    let gated = if spec.gated { "true" } else { "false" };
    let template = if spec.status == "runnable" {
        format!(
            ",\n      \"chatTemplate\": {}",
            json_escape(LLAMA_CHAT_TEMPLATE)
        )
    } else {
        String::new()
    };
    let tokenizer = if spec.status == "runnable" {
        format!(
            r#",
      "tokenizer": {{
        "filename": "tokenizer.json",
        "repo": "fixture/tiny",
        "revision": "rev1",
        "sha256": "{sha}",
        "sizeBytes": {size}
      }}"#,
            sha = spec.tok_sha,
            size = spec.tok_size,
        )
    } else {
        String::new()
    };
    format!(
        r#"{{
  "kind": "knolo.infer.library",
  "version": 1,
  "models": [{{
    "adapter": "{adapter}",
    "architecture": "llama",
    "contextReservation": 32,
    "displayName": "Tiny Llama",
    "filename": "tiny.gguf",
    "gated": {gated},
    "id": "{id}",
    "licenseId": "synthetic",
    "licenseUrl": "http://127.0.0.1/license",
    "minRamBytes": {min_ram},
    "quant": "f32",
    "repo": "fixture/tiny",
    "revision": "rev1",
    "sha256": "{weight_sha}",
    "sizeBytes": {weight_size},
    "status": "{status}",
    "tags": ["instruct"]{template}{tokenizer}
  }}]
}}
"#,
        adapter = spec.adapter,
        id = spec.id,
        min_ram = spec.min_ram,
        weight_sha = spec.weight_sha,
        weight_size = spec.weight_size,
        status = spec.status,
    )
}

struct Fixture {
    port: u16,
    hits: Arc<AtomicUsize>,
}

fn serve(files: HashMap<String, Vec<u8>>, redirect: bool, interrupt: bool) -> Fixture {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let hits = Arc::new(AtomicUsize::new(0));
    let hits_thread = Arc::clone(&hits);
    let interrupt_left = Arc::new(AtomicBool::new(interrupt));
    std::thread::spawn(move || {
        while let Ok((mut stream, _)) = listener.accept() {
            hits_thread.fetch_add(1, Ordering::SeqCst);
            let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
            let _ = answer(&mut stream, &files, redirect, &interrupt_left);
        }
    });
    Fixture { port, hits }
}

fn answer(
    stream: &mut TcpStream,
    files: &HashMap<String, Vec<u8>>,
    redirect: bool,
    interrupt_left: &AtomicBool,
) -> std::io::Result<()> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    while !buf.windows(4).any(|window| window == b"\r\n\r\n") && buf.len() < 16 * 1024 {
        let read = stream.read(&mut tmp)?;
        if read == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..read]);
    }
    let text = String::from_utf8_lossy(&buf);
    let request = text.lines().next().unwrap_or("");
    let mut parts = request.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");
    if redirect {
        return write_all(
            stream,
            "HTTP/1.0 302 Found",
            "Location: http://example.com/weight\r\n",
            b"",
        );
    }
    let name = path.rsplit('/').next().unwrap_or("");
    let Some(bytes) = files.get(name) else {
        return write_all(stream, "HTTP/1.0 404 Not Found", "", b"");
    };
    if method.eq_ignore_ascii_case("HEAD") {
        return write_body(stream, "HTTP/1.0 200 OK", "Accept-Ranges: bytes\r\n", b"");
    }
    let start = range_start(&text);
    if name.ends_with(".gguf") && start.is_none() && interrupt_left.swap(false, Ordering::SeqCst) {
        let head = format!(
            "HTTP/1.0 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            bytes.len()
        );
        stream.write_all(head.as_bytes())?;
        stream.write_all(&bytes[..64.min(bytes.len())])?;
        stream.flush()?;
        let _ = stream.shutdown(Shutdown::Write);
        return Ok(());
    }
    if let Some(start) = start {
        if start > bytes.len() {
            return write_body(stream, "HTTP/1.0 416 Range Not Satisfiable", "", b"");
        }
        let end = bytes.len().saturating_sub(1);
        let extra = format!(
            "Accept-Ranges: bytes\r\nContent-Range: bytes {start}-{end}/{}\r\n",
            bytes.len()
        );
        return write_body(stream, "HTTP/1.0 206 Partial Content", &extra, &bytes[start..]);
    }
    write_body(stream, "HTTP/1.0 200 OK", "Accept-Ranges: bytes\r\n", bytes)
}

fn range_start(headers: &str) -> Option<usize> {
    for line in headers.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("range") {
            let rest = value.trim().strip_prefix("bytes=")?;
            let start = rest.split(['-', ',']).next().unwrap_or("");
            return start.parse().ok();
        }
    }
    None
}

fn write_all(stream: &mut TcpStream, status: &str, extra: &str, body: &[u8]) -> std::io::Result<()> {
    write_body(stream, status, extra, body)
}

fn write_body(stream: &mut TcpStream, status: &str, extra: &str, body: &[u8]) -> std::io::Result<()> {
    let head = format!(
        "{status}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    if !body.is_empty() {
        stream.write_all(body)?;
    }
    stream.flush()
}

fn policy_for(port: u16) -> FetchPolicy {
    FetchPolicy {
        allow_huggingface: false,
        extra_hosts: vec!["127.0.0.1".to_string()],
        origin_override: Some(format!("http://127.0.0.1:{port}")),
    }
}

fn build_root() -> DigestHex {
    DigestHex::parse("sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
        .unwrap()
}

fn pull_text(
    home: &Path,
    text: &str,
    policy: &FetchPolicy,
    accept: bool,
    ram: Option<u64>,
    token: Option<&str>,
) -> Result<PullOutcome, InferFailure> {
    let catalog = load_catalog(text)?;
    let row = &catalog.models[0];
    let root = build_root();
    pull_catalog_row(&PullRequest {
        home,
        lock_path: &home.join("knolo.infer.lock.json"),
        row,
        policy,
        accept_license: accept,
        ram_bytes: ram,
        token,
        build_root: &root,
    })
}

fn sample_files() -> (Vec<u8>, Vec<u8>) {
    let gguf = encode_gguf(
        &[GgufMetadata {
            key: "general.architecture".into(),
            value: GgufValue::String("llama".into()),
        }],
        &[GgufTensorDraft {
            name: "token_embd.weight".into(),
            tensor_type: GgufTensorType::F32,
            shape: vec![1],
            bytes: 1.0f32.to_le_bytes().to_vec(),
        }],
    )
    .unwrap();
    let tokenizer = br#"{"kind":"knolo.llama.tokens.v1","tokens":["a"],"version":1}"#.to_vec();
    assert!(gguf.len() > 64, "interrupt needs a short prefix");
    (gguf, tokenizer)
}

fn file_map(gguf: Vec<u8>, tokenizer: Vec<u8>) -> HashMap<String, Vec<u8>> {
    let mut files = HashMap::new();
    files.insert("tiny.gguf".to_string(), gguf);
    files.insert("tokenizer.json".to_string(), tokenizer);
    files
}

#[test]
fn checked_in_catalog_parses_and_is_not_rewritten() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../catalog/library.json");
    let before = fs::read(&path).unwrap();
    let catalog = checked_in_catalog().unwrap();
    let after = fs::read(&path).unwrap();
    assert_eq!(before, after);
    let ids: Vec<_> = catalog.models.iter().map(|row| row.id.as_str()).collect();
    assert_eq!(
        ids,
        [
            "llama-3.2-1b-instruct",
            "llama-3.2-1b-uncensored",
            "llama-3.2-3b-instruct",
            "qwen3-4b-instruct",
            "qwen3-8b-abliterated",
        ]
    );
    for row in &catalog.models {
        assert_eq!(row.sha256.len(), "sha256-".len() + 64, "{}", row.id);
        assert!(row.sha256.starts_with("sha256-"));
    }
    assert_eq!(catalog.models[0].status, "runnable");
    assert_eq!(catalog.models[3].status, "not_supported");
    assert!(catalog.models[3].tokenizer.is_none());
    assert_eq!(catalog.models[1].chat_template.as_deref(), Some(LLAMA_CHAT_TEMPLATE));
}

#[test]
fn catalog_pull_pins_a_local_fixture() {
    let (gguf, tokenizer) = sample_files();
    let (weight_sha, weight_size) = digest_of(&gguf);
    let (tok_sha, tok_size) = digest_of(&tokenizer);
    let fixture = serve(file_map(gguf, tokenizer), false, false);
    let home = scratch("pin");
    let text = catalog(&CatalogSpec::llama(
        "tiny-llama",
        &weight_sha,
        weight_size,
        &tok_sha,
        tok_size,
    ));
    let outcome = pull_text(&home, &text, &policy_for(fixture.port), true, Some(8 << 30), None)
        .unwrap_or_else(|err| panic!("{err}"));
    assert_eq!(outcome.alias, "tiny-llama");
    assert_eq!(outcome.path, "models/tiny-llama/model.kmodel");
    assert!(outcome.model_image_root.starts_with("sha256-"));
    assert!(home.join("models/tiny-llama/model.kmodel").is_file());
    assert!(home.join("models/tiny-llama/tiny.gguf").is_file());
    let lock = fs::read_to_string(home.join("knolo.infer.lock.json")).unwrap();
    assert!(lock.contains("tiny-llama"));
    assert!(fixture.hits.load(Ordering::SeqCst) > 0);
    let _ = fs::remove_dir_all(&home);
}

#[test]
fn bad_hash_deletes_staging_and_leaves_the_old_pin() {
    let (gguf, tokenizer) = sample_files();
    let (weight_sha, weight_size) = digest_of(&gguf);
    let (tok_sha, tok_size) = digest_of(&tokenizer);
    let fixture = serve(file_map(gguf, tokenizer), false, false);
    let home = scratch("bad-hash");
    let policy = policy_for(fixture.port);
    let good = catalog(&CatalogSpec::llama(
        "tiny-llama",
        &weight_sha,
        weight_size,
        &tok_sha,
        tok_size,
    ));
    pull_text(&home, &good, &policy, true, Some(8 << 30), None).unwrap();
    let pinned = fs::read(home.join("models/tiny-llama/model.kmodel")).unwrap();
    let flipped = flip_digest(&weight_sha);
    let bad = catalog(&CatalogSpec::llama(
        "tiny-llama",
        &flipped,
        weight_size,
        &tok_sha,
        tok_size,
    ));
    let err = pull_text(&home, &bad, &policy, true, Some(8 << 30), None).unwrap_err();
    assert_eq!(err.code, ErrorCode::ModelDigestMismatch);
    assert!(err.message.contains("download digest does not match the catalog row"));
    assert!(!home.join("staging/tiny-llama/tiny.gguf").exists());
    assert_eq!(
        fs::read(home.join("models/tiny-llama/model.kmodel")).unwrap(),
        pinned
    );
    let _ = fs::remove_dir_all(&home);
}

#[test]
fn redirect_off_the_allowlist_downloads_nothing() {
    let fixture = serve(HashMap::new(), true, false);
    let home = scratch("redirect");
    let (gguf, tokenizer) = sample_files();
    let (weight_sha, weight_size) = digest_of(&gguf);
    let (tok_sha, tok_size) = digest_of(&tokenizer);
    let text = catalog(&CatalogSpec::llama(
        "tiny-llama",
        &weight_sha,
        weight_size,
        &tok_sha,
        tok_size,
    ));
    let err = pull_text(&home, &text, &policy_for(fixture.port), true, Some(8 << 30), None)
        .unwrap_err();
    assert!(
        err.message.contains("download host is not allowlisted: example.com"),
        "{err}"
    );
    assert!(!home.join("models").exists());
    assert_eq!(fixture.hits.load(Ordering::SeqCst), 1);
    let _ = fs::remove_dir_all(&home);
}

#[test]
fn interrupted_download_resumes() {
    let (gguf, tokenizer) = sample_files();
    let (weight_sha, weight_size) = digest_of(&gguf);
    let (tok_sha, tok_size) = digest_of(&tokenizer);
    let fixture = serve(file_map(gguf, tokenizer), false, true);
    let home = scratch("resume");
    let policy = policy_for(fixture.port);
    let text = catalog(&CatalogSpec::llama(
        "tiny-llama",
        &weight_sha,
        weight_size,
        &tok_sha,
        tok_size,
    ));
    let err = pull_text(&home, &text, &policy, true, Some(8 << 30), None).unwrap_err();
    assert!(err.message.contains("download interrupted"), "{err}");
    let partial = fs::metadata(home.join("staging/tiny-llama/tiny.gguf")).unwrap().len();
    assert!(partial > 0 && partial < weight_size, "{partial} of {weight_size}");
    let outcome = pull_text(&home, &text, &policy, true, Some(8 << 30), None)
        .unwrap_or_else(|err| panic!("{err}"));
    assert_eq!(outcome.alias, "tiny-llama");
    assert!(home.join("models/tiny-llama/model.kmodel").is_file());
    let _ = fs::remove_dir_all(&home);
}

#[test]
fn declined_license_leaves_nothing_pinned() {
    let fixture = serve(HashMap::new(), false, false);
    let home = scratch("license");
    let text = catalog(&CatalogSpec::llama(
        "tiny-llama",
        "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        4,
        "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        4,
    ));
    let err = pull_text(&home, &text, &policy_for(fixture.port), false, Some(8 << 30), None)
        .unwrap_err();
    assert!(err.message.contains("license synthetic was not accepted"), "{err}");
    assert_eq!(fixture.hits.load(Ordering::SeqCst), 0);
    assert!(!home.join("knolo.infer.lock.json").exists());
    assert!(!home.join("models").exists());
    let _ = fs::remove_dir_all(&home);
}

#[test]
fn unsupported_architecture_downloads_nothing() {
    let fixture = serve(HashMap::new(), false, false);
    let home = scratch("qwen");
    let mut spec = CatalogSpec::llama(
        "tiny-qwen",
        "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        4,
        "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        4,
    );
    spec.status = "not_supported";
    spec.adapter = "knolo.qwen.v1";
    let text = catalog(&spec);
    let err = pull_text(&home, &text, &policy_for(fixture.port), true, Some(8 << 30), None)
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::UnsupportedArchitecture);
    assert!(err.message.contains("is not supported yet"), "{err}");
    assert_eq!(fixture.hits.load(Ordering::SeqCst), 0);
    assert!(!home.join("models").exists());
    let _ = fs::remove_dir_all(&home);
}

#[test]
fn gated_row_without_a_token_downloads_nothing() {
    let fixture = serve(HashMap::new(), false, false);
    let home = scratch("gated");
    let mut spec = CatalogSpec::llama(
        "tiny-llama",
        "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        4,
        "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        4,
    );
    spec.gated = true;
    let text = catalog(&spec);
    let err = pull_text(&home, &text, &policy_for(fixture.port), true, None, None).unwrap_err();
    assert!(err.message.contains("this model is gated; set HF_TOKEN"), "{err}");
    assert_eq!(fixture.hits.load(Ordering::SeqCst), 0);
    let _ = fs::remove_dir_all(&home);
}

#[test]
fn refresh_reports_a_changed_pointer_and_does_not_write_the_catalog() {
    let pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nsize 4\n";
    let mut files = HashMap::new();
    files.insert("tiny.gguf".to_string(), pointer.to_vec());
    let fixture = serve(files, false, false);
    let catalog_path = scratch("refresh-file").join("library.json");
    let text = catalog(&CatalogSpec::llama(
        "tiny-llama",
        "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        4,
        "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        4,
    ));
    fs::write(&catalog_path, &text).unwrap();
    let before = fs::read(&catalog_path).unwrap();
    let checked = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../catalog/library.json");
    let checked_before = fs::read(&checked).unwrap();
    let catalog = load_catalog(&text).unwrap();
    let changes = refresh_catalog(&catalog, &policy_for(fixture.port), None).unwrap();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].id, "tiny-llama");
    assert!(changes[0].remote_sha256.ends_with("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
    assert_eq!(fs::read(&catalog_path).unwrap(), before);
    assert_eq!(fs::read(&checked).unwrap(), checked_before);
}

#[test]
fn rm_refuses_a_path_outside_the_home_model_directory() {
    let home = scratch("rm");
    let outside = scratch("outside");
    fs::create_dir_all(home.join("models/kept")).unwrap();
    fs::write(home.join("models/kept/model.kmodel"), b"kept").unwrap();
    fs::write(outside.join("model.kmodel"), b"keep").unwrap();
    std::os::unix::fs::symlink(&outside, home.join("models/outside")).unwrap();
    let digest = "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let mut lock = new_lockfile(&build_root());
    lock.models.insert(
        "outside".into(),
        ModelPin {
            model_image_root: digest.into(),
            artifact_root: digest.into(),
            model_image_path: "models/outside/model.kmodel".into(),
        },
    );
    lock.models.insert(
        "kept".into(),
        ModelPin {
            model_image_root: digest.into(),
            artifact_root: digest.into(),
            model_image_path: "models/kept/model.kmodel".into(),
        },
    );
    let lock_path = home.join("knolo.infer.lock.json");
    write_lockfile(&lock_path, &lock).unwrap();
    let err = remove_installed(&home, &lock_path, "outside").unwrap_err();
    assert!(
        err.message.contains("rm refuses a path outside the home model directory"),
        "{err}"
    );
    assert_eq!(fs::read(outside.join("model.kmodel")).unwrap(), b"keep");
    remove_installed(&home, &lock_path, "kept").unwrap();
    assert!(!home.join("models/kept").exists());
    let installed = list_installed(&lock_path).unwrap();
    assert!(installed.iter().all(|item| item.alias != "kept"));
    assert!(installed.iter().any(|item| item.alias == "outside"));
    let _ = fs::remove_dir_all(&home);
    let _ = fs::remove_dir_all(&outside);
}
