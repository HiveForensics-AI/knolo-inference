//! Redacted completion trace.
//!
//! Each line is one JSON object. The field set is fixed, so a prompt, an
//! output, a token id, an alias, or a path cannot be represented. The lines
//! are not part of the receipt.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure};
use serde_json::{Map, Value};

use crate::protocol::class_name;
use infer_engine::ServiceClass;

pub(crate) struct Trace {
    dir: PathBuf,
    rejected: Mutex<()>,
}

pub(crate) struct RequestTrace {
    path: PathBuf,
    request_id: String,
    admitted: bool,
    closed: bool,
}

impl Trace {
    pub(crate) fn open(home: &Path) -> Result<Self, InferFailure> {
        let dir = home.join("traces").join("by-id");
        fs::create_dir_all(&dir).map_err(|err| {
            fail(
                ErrorCode::ReceiptPersistFailed,
                format!("trace directory: {err}"),
            )
        })?;
        Ok(Self {
            dir,
            rejected: Mutex::new(()),
        })
    }

    /// A completion that never opened a request file.
    pub(crate) fn reject(&self, code: ErrorCode) {
        let _guard = self.rejected.lock().unwrap_or_else(|err| err.into_inner());
        let path = self
            .dir
            .parent()
            .unwrap_or(&self.dir)
            .join("rejected.jsonl");
        write_line(
            &path,
            &object(&[
                ("code", Value::String(code.as_str().to_string())),
                ("outcome", Value::String("rejected".to_string())),
                ("stage", Value::String("api".to_string())),
            ]),
        );
    }

    pub(crate) fn begin(
        &self,
        request_id: &str,
        class: ServiceClass,
        stream: bool,
    ) -> RequestTrace {
        let path = self.dir.join(format!("{request_id}.jsonl"));
        let trace = RequestTrace {
            path,
            request_id: request_id.to_string(),
            admitted: false,
            closed: false,
        };
        trace.write(&object(&[
            ("class", Value::String(class_name(class).to_string())),
            ("requestId", Value::String(trace.request_id.clone())),
            ("stage", Value::String("api".to_string())),
            ("stream", Value::Bool(stream)),
        ]));
        trace
    }
}

impl RequestTrace {
    pub(crate) fn prompt(&mut self, prompt_root: &str, prompt_tokens: u32) {
        let Some(root) = digest(prompt_root) else {
            self.prompt_rejected(ErrorCode::DigestInvalid);
            return;
        };
        self.write(&object(&[
            ("promptRoot", Value::String(root)),
            ("promptTokens", Value::from(prompt_tokens)),
            ("requestId", Value::String(self.request_id.clone())),
            ("stage", Value::String("prompt".to_string())),
        ]));
    }

    pub(crate) fn prompt_rejected(&mut self, code: ErrorCode) {
        self.write(&stage_outcome(
            "prompt",
            &self.request_id,
            "rejected",
            Some(code),
        ));
    }

    pub(crate) fn admitted(&mut self) {
        if self.admitted || self.closed {
            return;
        }
        self.admitted = true;
        self.write(&stage_outcome(
            "admission",
            &self.request_id,
            "admitted",
            None,
        ));
    }

    pub(crate) fn admission_rejected(&mut self, code: ErrorCode) {
        if self.admitted || self.closed {
            return;
        }
        self.admitted = true;
        self.write(&stage_outcome(
            "admission",
            &self.request_id,
            "rejected",
            Some(code),
        ));
    }

    pub(crate) fn prefill(&mut self, chunk: u32, tokens: u32) {
        if self.closed {
            return;
        }
        self.write(&object(&[
            ("chunk", Value::from(chunk)),
            ("requestId", Value::String(self.request_id.clone())),
            ("stage", Value::String("prefill".to_string())),
            ("tokens", Value::from(tokens)),
        ]));
    }

    pub(crate) fn decode(&mut self, index: u32) {
        if self.closed {
            return;
        }
        self.write(&object(&[
            ("index", Value::from(index)),
            ("requestId", Value::String(self.request_id.clone())),
            ("stage", Value::String("decode".to_string())),
        ]));
    }

    pub(crate) fn finish(
        &mut self,
        outcome: &str,
        code: Option<ErrorCode>,
        receipt_root: Option<&str>,
    ) {
        if self.closed {
            return;
        }
        self.closed = true;
        let (outcome, code, receipt) = terminal(outcome, code, receipt_root);
        let mut fields = vec![
            ("outcome", Value::String(outcome.to_string())),
            ("requestId", Value::String(self.request_id.clone())),
            ("stage", Value::String("finalize".to_string())),
        ];
        if let Some(code) = code {
            fields.push(("code", Value::String(code.as_str().to_string())));
        }
        if let Some(receipt) = receipt {
            fields.push(("receiptRoot", Value::String(receipt)));
        }
        self.write(&object(&fields));
    }

    fn write(&self, fields: &Map<String, Value>) {
        write_line(&self.path, fields);
    }
}

impl Drop for RequestTrace {
    fn drop(&mut self) {
        if !self.closed {
            self.finish("error", Some(ErrorCode::ContractInvalid), None);
        }
    }
}

fn terminal(
    outcome: &str,
    code: Option<ErrorCode>,
    receipt_root: Option<&str>,
) -> (&'static str, Option<ErrorCode>, Option<String>) {
    let outcome = match outcome {
        "stop" => "stop",
        "length" => "length",
        "cancelled" => "cancelled",
        "rejected" => "rejected",
        "error" => "error",
        _ => "error",
    };
    let receipt = receipt_root.and_then(digest);
    if matches!(outcome, "stop" | "length") {
        return match receipt {
            Some(root) => (outcome, None, Some(root)),
            None => ("error", Some(ErrorCode::DigestInvalid), None),
        };
    }
    if outcome == "cancelled" {
        return ("cancelled", None, None);
    }
    (
        outcome,
        Some(code.unwrap_or(ErrorCode::ContractInvalid)),
        None,
    )
}

fn stage_outcome(
    stage: &str,
    request_id: &str,
    outcome: &str,
    code: Option<ErrorCode>,
) -> Map<String, Value> {
    let mut fields = vec![
        ("outcome", Value::String(outcome.to_string())),
        ("requestId", Value::String(request_id.to_string())),
        ("stage", Value::String(stage.to_string())),
    ];
    if let Some(code) = code {
        fields.push(("code", Value::String(code.as_str().to_string())));
    }
    object(&fields)
}

fn object(fields: &[(&str, Value)]) -> Map<String, Value> {
    let mut map = Map::new();
    for (key, value) in fields {
        map.insert((*key).to_string(), value.clone());
    }
    map
}

fn digest(value: &str) -> Option<String> {
    DigestHex::parse(value)
        .ok()
        .map(|parsed| parsed.to_string())
}

fn write_line(path: &Path, fields: &Map<String, Value>) {
    let mut line = serde_json::to_vec(fields).unwrap_or_default();
    if line.is_empty() {
        return;
    }
    line.push(b'\n');
    let mut file = match OpenOptions::new().create(true).append(true).open(path) {
        Ok(file) => file,
        Err(_) => return,
    };
    let _ = file.write_all(&line);
    let _ = file.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "knolo-infer-trace-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn lines_keep_roots_and_drop_content() {
        let home = scratch();
        let trace = Trace::open(&home).unwrap();
        trace.reject(ErrorCode::ContractInvalid);
        let mut request = trace.begin("job-1", ServiceClass::Standard, false);
        request.prompt("NEEDLE-PROMPT", 4);
        request.prompt(
            "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            4,
        );
        request.admitted();
        request.prefill(0, 4);
        request.decode(0);
        request.finish("length", None, Some("not-a-digest"));
        request.finish(
            "length",
            None,
            Some("sha256-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        );
        drop(request);

        let rejected = fs::read_to_string(home.join("traces/rejected.jsonl")).unwrap();
        assert_eq!(
            rejected,
            "{\"code\":\"CONTRACT_INVALID\",\"outcome\":\"rejected\",\"stage\":\"api\"}\n"
        );
        assert!(!rejected.contains("NEEDLE"));

        let body = fs::read_to_string(home.join("traces/by-id/job-1.jsonl")).unwrap();
        assert!(!body.contains("NEEDLE"));
        assert!(!body.contains("not-a-digest"));
        let lines: Vec<Value> = body
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines[1]["code"], "DIGEST_INVALID");
        assert_eq!(lines[1]["stage"], "prompt");
        assert_eq!(lines.last().unwrap()["stage"], "finalize");
        assert_eq!(lines.last().unwrap()["outcome"], "error");
        assert_eq!(lines.last().unwrap()["code"], "DIGEST_INVALID");
        assert!(lines.last().unwrap().get("receiptRoot").is_none());
    }
}
