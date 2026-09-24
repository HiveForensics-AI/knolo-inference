//! One `knolo-infer serve` process owns a home directory.
//!
//! `{home}/daemon.lock/` holds `owner`, a pid and the process start time from
//! `/proc`. The directory stays behind when the process is killed. The next
//! start takes it when that pid is gone or the start time no longer matches,
//! and leaves it alone when the recorded process is still that owner.
//! `knolo.infer.lock.json` is the model pin. This directory is not that file.

use std::fs;
use std::io::ErrorKind;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use infer_artifact::write_atomic;
use infer_contracts::{fail, ErrorCode, InferFailure};
use infer_engine::recover_open_journals;

const LOCK_DIR: &str = "daemon.lock";
const OWNER_FILE: &str = "owner";

static GRAVE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub(crate) struct DaemonLock {
    home: PathBuf,
    held: AtomicBool,
}

impl DaemonLock {
    /// Take the home, then seal journals left open by the previous process.
    /// A live owner fails with `CONTRACT_INVALID` and does not seal.
    pub(crate) fn acquire(home: &Path) -> Result<Self, InferFailure> {
        self_starttime()?;
        let lock = Self {
            home: home.to_path_buf(),
            held: AtomicBool::new(false),
        };
        lock.claim()?;
        lock.held.store(true, Ordering::SeqCst);
        if let Err(err) = recover_open_journals(&lock.home) {
            lock.release();
            return Err(err);
        }
        Ok(lock)
    }

    pub(crate) fn release(&self) {
        if !self.held.swap(false, Ordering::SeqCst) {
            return;
        }
        let dir = self.home.join(LOCK_DIR);
        if owner_is_us(&dir) {
            remove_path(&dir);
        }
    }

    fn claim(&self) -> Result<(), InferFailure> {
        let dir = self.home.join(LOCK_DIR);
        for _ in 0..8 {
            match fs::create_dir(&dir) {
                Ok(()) => {
                    if let Err(err) = install_owner(&dir) {
                        remove_path(&dir);
                        return Err(err);
                    }
                    return Ok(());
                }
                Err(err) if err.kind() == ErrorKind::AlreadyExists => match owner_state(&dir)? {
                    OwnerState::Live => {
                        return Err(fail(ErrorCode::ContractInvalid, "daemon lock is held"));
                    }
                    OwnerState::Stale => {
                        let grave = self.home.join(format!(
                            "daemon.lock.stale-{}-{}",
                            std::process::id(),
                            GRAVE.fetch_add(1, Ordering::Relaxed)
                        ));
                        if fs::rename(&dir, &grave).is_ok() {
                            remove_path(&grave);
                        }
                    }
                },
                Err(err) => {
                    return Err(fail(
                        ErrorCode::ReceiptPersistFailed,
                        format!("daemon lock: {err}"),
                    ))
                }
            }
        }
        Err(fail(ErrorCode::ContractInvalid, "daemon lock is held"))
    }
}

impl Drop for DaemonLock {
    fn drop(&mut self) {
        self.release();
    }
}

enum OwnerState {
    Live,
    Stale,
}

fn install_owner(dir: &Path) -> Result<(), InferFailure> {
    let mode = fs::Permissions::from_mode(0o700);
    fs::set_permissions(dir, mode).map_err(|err| {
        fail(
            ErrorCode::ReceiptPersistFailed,
            format!("daemon lock: {err}"),
        )
    })?;
    let starttime = self_starttime()?;
    let body = format!("pid {}\nstarttime {starttime}\n", std::process::id());
    write_atomic(
        &dir.join(OWNER_FILE),
        body.as_bytes(),
        ErrorCode::ReceiptPersistFailed,
    )?;
    fs::set_permissions(dir.join(OWNER_FILE), fs::Permissions::from_mode(0o600)).map_err(
        |err| {
            fail(
                ErrorCode::ReceiptPersistFailed,
                format!("daemon lock: {err}"),
            )
        },
    )?;
    Ok(())
}

fn owner_state(dir: &Path) -> Result<OwnerState, InferFailure> {
    let meta = match fs::symlink_metadata(dir) {
        Ok(meta) => meta,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(OwnerState::Stale),
        Err(err) => {
            return Err(fail(
                ErrorCode::ReceiptPersistFailed,
                format!("daemon lock: {err}"),
            ))
        }
    };
    if !meta.file_type().is_dir() {
        return Ok(OwnerState::Stale);
    }
    let bytes = match fs::read(dir.join(OWNER_FILE)) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(OwnerState::Stale),
        Err(err) => {
            return Err(fail(
                ErrorCode::ReceiptPersistFailed,
                format!("daemon lock: {err}"),
            ))
        }
    };
    let Some((pid, starttime)) = parse_owner(&bytes) else {
        return Ok(OwnerState::Stale);
    };
    Ok(match process_starttime(pid) {
        Some(actual) if actual == starttime => OwnerState::Live,
        _ => OwnerState::Stale,
    })
}

fn owner_is_us(dir: &Path) -> bool {
    let Ok(bytes) = fs::read(dir.join(OWNER_FILE)) else {
        return false;
    };
    let Some((pid, starttime)) = parse_owner(&bytes) else {
        return false;
    };
    pid == std::process::id() && process_starttime(pid) == Some(starttime)
}

fn parse_owner(bytes: &[u8]) -> Option<(u32, u64)> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut lines = text.lines();
    let pid = lines.next()?.strip_prefix("pid ")?.parse().ok()?;
    let starttime = lines.next()?.strip_prefix("starttime ")?.parse().ok()?;
    if lines.next().is_some() {
        return None;
    }
    Some((pid, starttime))
}

fn self_starttime() -> Result<u64, InferFailure> {
    process_starttime(std::process::id()).ok_or_else(|| {
        fail(
            ErrorCode::ContractInvalid,
            "process start time is unavailable",
        )
    })
}

fn process_starttime(pid: u32) -> Option<u64> {
    let text = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let end = text.rfind(')')?;
    let fields: Vec<&str> = text[end + 1..].split_whitespace().collect();
    // Field 22 is starttime. `stat` after the comm's closing parenthesis
    // begins at field 3, so starttime is index 19.
    fields.get(19)?.parse().ok()
}

fn remove_path(path: &Path) {
    let Ok(meta) = fs::symlink_metadata(path) else {
        return;
    };
    if meta.file_type().is_symlink() || meta.file_type().is_file() {
        let _ = fs::remove_file(path);
        return;
    }
    if meta.file_type().is_dir() {
        let _ = fs::remove_dir_all(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_home() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("knolo-daemon-lock-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("home");
        dir
    }

    #[test]
    fn a_live_lock_is_held_and_a_dead_pid_is_stale() {
        let home = temp_home();
        let first = DaemonLock::acquire(&home).expect("acquire");
        let owner = fs::read_to_string(home.join("daemon.lock").join("owner")).expect("owner");
        assert!(owner.starts_with(&format!("pid {}\n", std::process::id())));
        let mode = fs::metadata(home.join("daemon.lock"))
            .expect("lock dir")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o700);
        let file_mode = fs::metadata(home.join("daemon.lock").join("owner"))
            .expect("owner file")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(file_mode, 0o600);
        let err = DaemonLock::acquire(&home).expect_err("second");
        assert_eq!(err.code, ErrorCode::ContractInvalid);
        assert_eq!(err.message, "daemon lock is held");
        first.release();
        assert!(!home.join("daemon.lock").exists());

        fs::create_dir_all(home.join("daemon.lock")).expect("stale dir");
        fs::write(
            home.join("daemon.lock").join("owner"),
            format!("pid {}\nstarttime 0\n", std::process::id()),
        )
        .expect("stale owner");
        let taken = DaemonLock::acquire(&home).expect("stale pid is replaced");
        let owner = fs::read_to_string(home.join("daemon.lock").join("owner")).expect("owner");
        assert!(!owner.contains("starttime 0\n"));
        taken.release();

        let mut child = std::process::Command::new("/bin/true")
            .spawn()
            .expect("true");
        let pid = child.id();
        child.wait().expect("wait");
        fs::create_dir_all(home.join("daemon.lock")).expect("dead dir");
        fs::write(
            home.join("daemon.lock").join("owner"),
            format!("pid {pid}\nstarttime 1\n"),
        )
        .expect("dead owner");
        let taken = DaemonLock::acquire(&home).expect("dead pid");
        taken.release();
        assert!(!home.join("daemon.lock").exists());
    }

    #[test]
    fn self_starttime_is_stable() {
        let pid = std::process::id();
        let start = process_starttime(pid).expect("starttime");
        assert!(start > 0);
        assert_eq!(process_starttime(pid), Some(start));
        let parsed = parse_owner(format!("pid {pid}\nstarttime {start}\n").as_bytes());
        assert_eq!(parsed, Some((pid, start)));
        assert!(parse_owner(b"pid 1\nstarttime 2\nextra\n").is_none());
    }
}
