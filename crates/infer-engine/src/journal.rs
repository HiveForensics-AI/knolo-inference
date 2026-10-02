//! One directory per request. Event files are created once and never rewritten.
//!
//! A serve process that dies after `accepted` leaves that chain unsealed.
//! [`recover_open_journals`] appends `failed` before the next process listens.

use std::fs;
use std::path::{Path, PathBuf};

use infer_artifact::write_atomic;
use infer_contracts::{
    decode_canonical, digest_value, fail, CborValue, DigestHex, ErrorCode, InferFailure,
    InferenceEventV1, SamplerPlanV1,
};

pub struct Journal {
    dir: PathBuf,
    request_id: String,
    previous: Option<DigestHex>,
    next_index: u32,
}

impl Journal {
    pub fn create(home: &Path, request_id: &str) -> Result<Self, InferFailure> {
        let dir = home.join("journals").join(request_id);
        fs::create_dir(&dir).map_err(|err| {
            let message = if err.kind() == std::io::ErrorKind::AlreadyExists {
                "request journal already exists".to_string()
            } else {
                format!("cannot create the journal: {err}")
            };
            fail(ErrorCode::ReceiptPersistFailed, message)
        })?;
        Ok(Self {
            dir,
            request_id: request_id.to_string(),
            previous: None,
            next_index: 0,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn terminal(&self) -> Option<&DigestHex> {
        self.previous.as_ref()
    }

    pub fn append(
        &mut self,
        name: &str,
        payload_root: DigestHex,
    ) -> Result<DigestHex, InferFailure> {
        let event = InferenceEventV1 {
            request_id: self.request_id.clone(),
            attempt: 1,
            name: name.to_string(),
            previous_event_root: self.previous.clone(),
            payload_root,
            extensions: std::collections::BTreeMap::new(),
        };
        let bytes = event.to_bytes()?;
        let root = event.root()?;
        let path = self.dir.join(format!("{:06}.{name}.cbor", self.next_index));
        write_atomic(&path, &bytes, ErrorCode::ReceiptPersistFailed)?;
        self.previous = Some(root.clone());
        self.next_index += 1;
        Ok(root)
    }

    pub fn write_sampler(&self, plan: &SamplerPlanV1) -> Result<(), InferFailure> {
        let bytes = plan.to_bytes()?;
        write_atomic(
            &self.dir.join("sampler-plan.cbor"),
            &bytes,
            ErrorCode::ReceiptPersistFailed,
        )
    }
}

pub fn load_sampler_plan(home: &Path, request_id: &str) -> Result<SamplerPlanV1, InferFailure> {
    let path = home
        .join("journals")
        .join(request_id)
        .join("sampler-plan.cbor");
    let bytes = fs::read(&path).map_err(|err| {
        fail(
            ErrorCode::ReceiptRequired,
            format!("sampler plan is missing from the journal: {err}"),
        )
    })?;
    let value = decode_canonical(&bytes)?;
    let plan = SamplerPlanV1::from_cbor(&value)?;
    if plan.to_bytes()? != bytes {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "sampler plan bytes are not canonical",
        ));
    }
    Ok(plan)
}

pub fn verify_journal(home: &Path, request_id: &str) -> Result<DigestHex, InferFailure> {
    let dir = home.join("journals").join(request_id);
    let names = event_names(&dir)?;
    if names.is_empty() {
        return Err(fail(ErrorCode::ReceiptRequired, "journal has no events"));
    }
    let chain = read_chain(&dir, request_id, &names)?;
    if chain.started_accepted && chain.terminal {
        Ok(chain.previous)
    } else {
        Err(fail(
            ErrorCode::ContractInvalid,
            "journal does not start at accepted or end at a terminal event",
        ))
    }
}

/// Seal journals left open by a process that died after `accepted`.
///
/// A request directory with no event files was never accepted and is removed.
/// A chain that does not verify stays on disk and fails this call. The
/// appended event is `failed`. Its payload root is the execution-trace digest
/// of `WORKER_LOST`. A chain that already ends at `completed`, `failed`, or
/// `cancelled` is not rewritten.
pub fn recover_open_journals(home: &Path) -> Result<u32, InferFailure> {
    let root = home.join("journals");
    let meta = match fs::symlink_metadata(&root) {
        Ok(meta) => meta,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(err) => {
            return Err(fail(
                ErrorCode::ReceiptPersistFailed,
                format!("cannot read journals: {err}"),
            ))
        }
    };
    if meta.file_type().is_symlink() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "journal directory is a symlink",
        ));
    }
    if !meta.file_type().is_dir() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "journal directory is not a directory",
        ));
    }
    let mut ids = Vec::new();
    for entry in fs::read_dir(&root).map_err(|err| {
        fail(
            ErrorCode::ReceiptPersistFailed,
            format!("cannot read journals: {err}"),
        )
    })? {
        let entry = entry.map_err(|err| {
            fail(
                ErrorCode::ReceiptPersistFailed,
                format!("cannot read journals: {err}"),
            )
        })?;
        let path = entry.path();
        let meta = fs::symlink_metadata(&path).map_err(|err| {
            fail(
                ErrorCode::ReceiptPersistFailed,
                format!("cannot read a journal: {err}"),
            )
        })?;
        if meta.file_type().is_symlink() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "journal entry is a symlink",
            ));
        }
        if !meta.file_type().is_dir() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "journal entry is not a directory",
            ));
        }
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "journal name is not a request id",
            ));
        };
        if !valid_request_id(name) {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "journal name is not a request id",
            ));
        }
        ids.push(name.to_string());
    }
    ids.sort();
    let mut sealed = 0u32;
    for id in ids {
        sealed = sealed.saturating_add(recover_one(&root.join(&id), &id)?);
    }
    Ok(sealed)
}

struct Chain {
    previous: DigestHex,
    next_index: u32,
    terminal: bool,
    started_accepted: bool,
}

fn recover_one(dir: &Path, request_id: &str) -> Result<u32, InferFailure> {
    if !journal_files_are_regular(dir)? {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "journal entry is a symlink",
        ));
    }
    let names = event_names(dir)?;
    if names.is_empty() {
        fs::remove_dir_all(dir).map_err(|err| {
            fail(
                ErrorCode::ReceiptPersistFailed,
                format!("cannot remove an unaccepted journal: {err}"),
            )
        })?;
        return Ok(0);
    }
    let chain = read_chain(dir, request_id, &names)?;
    if chain.terminal && chain.started_accepted {
        return Ok(0);
    }
    if !chain.started_accepted {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "journal does not start at accepted or end at a terminal event",
        ));
    }
    let mut journal = Journal {
        dir: dir.to_path_buf(),
        request_id: request_id.to_string(),
        previous: Some(chain.previous),
        next_index: chain.next_index,
    };
    journal.append("failed", worker_lost_payload()?)?;
    Ok(1)
}

fn journal_files_are_regular(dir: &Path) -> Result<bool, InferFailure> {
    for entry in fs::read_dir(dir).map_err(|err| {
        fail(
            ErrorCode::ReceiptPersistFailed,
            format!("cannot read the journal: {err}"),
        )
    })? {
        let entry = entry.map_err(|err| {
            fail(
                ErrorCode::ReceiptPersistFailed,
                format!("cannot read the journal: {err}"),
            )
        })?;
        let meta = fs::symlink_metadata(entry.path()).map_err(|err| {
            fail(
                ErrorCode::ReceiptPersistFailed,
                format!("cannot read the journal: {err}"),
            )
        })?;
        if meta.file_type().is_symlink() {
            return Ok(false);
        }
        if !meta.file_type().is_file() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "journal directory is incomplete",
            ));
        }
        let file_name = entry.file_name();
        if file_name.to_str().is_none() {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "journal file name is not unicode",
            ));
        }
    }
    Ok(true)
}

fn event_names(dir: &Path) -> Result<Vec<String>, InferFailure> {
    let mut names = Vec::new();
    let entries = fs::read_dir(dir).map_err(|err| {
        fail(
            ErrorCode::ReceiptRequired,
            format!("journal is missing: {err}"),
        )
    })?;
    for entry in entries {
        let entry = entry.map_err(|err| {
            fail(
                ErrorCode::ReceiptPersistFailed,
                format!("cannot read the journal: {err}"),
            )
        })?;
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };
        if name.ends_with(".cbor") && name.as_bytes().get(6) == Some(&b'.') {
            names.push(name.to_string());
        }
    }
    names.sort();
    Ok(names)
}

fn read_chain(dir: &Path, request_id: &str, names: &[String]) -> Result<Chain, InferFailure> {
    let mut previous = None;
    let mut last_name = String::new();
    for (index, name) in names.iter().enumerate() {
        let expected_prefix = format!("{index:06}.");
        if !name.starts_with(&expected_prefix) {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "journal event index is not contiguous",
            ));
        }
        let bytes = fs::read(dir.join(name)).map_err(|err| {
            fail(
                ErrorCode::ReceiptPersistFailed,
                format!("cannot read a journal event: {err}"),
            )
        })?;
        let event = InferenceEventV1::from_cbor(&decode_canonical(&bytes)?)?;
        if event.to_bytes()? != bytes {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "journal event bytes are not canonical",
            ));
        }
        if event.request_id != request_id || event.previous_event_root != previous {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "journal event chain does not match",
            ));
        }
        let label = name[7..].trim_end_matches(".cbor");
        if event.name != label {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "journal file name does not match the event",
            ));
        }
        previous = Some(event.root()?);
        last_name = event.name;
    }
    let previous =
        previous.ok_or_else(|| fail(ErrorCode::ContractInvalid, "journal chain is empty"))?;
    let next_index = u32::try_from(names.len()).map_err(|_| {
        fail(
            ErrorCode::ContractInvalid,
            "journal event index is not contiguous",
        )
    })?;
    Ok(Chain {
        previous,
        next_index,
        terminal: last_name == "completed" || last_name == "failed" || last_name == "cancelled",
        started_accepted: names[0].ends_with(".accepted.cbor"),
    })
}

fn worker_lost_payload() -> Result<DigestHex, InferFailure> {
    digest_value(
        "infer-execution-trace",
        &CborValue::Text("WORKER_LOST".to_string()),
    )
}

fn valid_request_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn home() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("knolo-journal-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("journals")).expect("journals");
        dir
    }

    fn trace(text: &str) -> DigestHex {
        digest_value("infer-execution-trace", &CborValue::Text(text.to_string())).expect("digest")
    }

    fn accept(home: &Path, request_id: &str) -> Journal {
        let mut journal = Journal::create(home, request_id).expect("journal");
        journal
            .append("accepted", trace("accepted"))
            .expect("accepted");
        journal
    }

    #[test]
    fn an_open_journal_is_sealed_once_and_a_terminal_journal_stays() {
        let home = home();
        accept(&home, "lost-job");
        let mut done = accept(&home, "done-job");
        done.append("cancelled", trace("cancelled"))
            .expect("cancelled");
        fs::create_dir_all(home.join("journals").join("partial-job")).expect("partial");
        fs::write(
            home.join("journals")
                .join("partial-job")
                .join("sampler-plan.cbor"),
            b"side",
        )
        .expect("sidecar");

        assert_eq!(recover_open_journals(&home).expect("recover"), 1);
        assert_eq!(recover_open_journals(&home).expect("second"), 0);
        assert!(home
            .join("journals")
            .join("lost-job")
            .join("000001.failed.cbor")
            .is_file());
        assert!(!home
            .join("journals")
            .join("done-job")
            .join("000002.failed.cbor")
            .exists());
        assert!(!home.join("journals").join("partial-job").exists());
        let failed = fs::read(
            home.join("journals")
                .join("lost-job")
                .join("000001.failed.cbor"),
        )
        .expect("failed event");
        let event =
            InferenceEventV1::from_cbor(&decode_canonical(&failed).expect("cbor")).expect("event");
        assert_eq!(event.name, "failed");
        assert_eq!(event.payload_root, trace("WORKER_LOST"));
        verify_journal(&home, "lost-job").expect("lost verifies");
        verify_journal(&home, "done-job").expect("done verifies");
        let missing = verify_journal(&home, "missing-job").expect_err("missing");
        assert_eq!(missing.code, ErrorCode::ReceiptRequired);

        fs::create_dir_all(home.join("journals").join("nested-job").join("inner")).expect("nested");
        let err = recover_open_journals(&home).expect_err("nested directory");
        assert_eq!(err.code, ErrorCode::ContractInvalid);
        assert!(home
            .join("journals")
            .join("lost-job")
            .join("000001.failed.cbor")
            .is_file());
        assert!(!home
            .join("journals")
            .join("lost-job")
            .join("000002.failed.cbor")
            .exists());
    }

    #[test]
    fn a_prefill_without_completed_is_sealed_and_a_symlink_is_rejected() {
        let home = home();
        let mut journal = accept(&home, "mid-job");
        journal
            .append("prefill", trace("prefill"))
            .expect("prefill");
        assert_eq!(
            verify_journal(&home, "mid-job").expect_err("open").code,
            ErrorCode::ContractInvalid
        );
        assert_eq!(recover_open_journals(&home).expect("recover"), 1);
        assert!(home
            .join("journals")
            .join("mid-job")
            .join("000002.failed.cbor")
            .is_file());
        verify_journal(&home, "mid-job").expect("sealed");

        std::os::unix::fs::symlink("elsewhere", home.join("journals").join("link-job"))
            .expect("symlink");
        let err = recover_open_journals(&home).expect_err("symlink");
        assert_eq!(err.code, ErrorCode::ContractInvalid);
        assert!(home
            .join("journals")
            .join("mid-job")
            .join("000002.failed.cbor")
            .is_file());
    }
}
