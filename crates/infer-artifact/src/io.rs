//! Bounded reads and same-directory atomic writes.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

use infer_contracts::{fail, DigestHex, ErrorCode, InferFailure};

const READ_LIMIT: u64 = 32 * 1024 * 1024;

pub struct HashedFile {
    pub size: u64,
    pub sha256: DigestHex,
}

pub fn read_utf8_limited(path: &Path, code: ErrorCode) -> Result<String, InferFailure> {
    let meta = fs::metadata(path).map_err(|err| io_err(code, "read", err))?;
    if !meta.is_file() {
        return Err(fail(code, "path is not a regular file"));
    }
    if meta.len() > READ_LIMIT {
        return Err(fail(code, "text file exceeds 32 MiB"));
    }
    let bytes = fs::read(path).map_err(|err| io_err(code, "read", err))?;
    String::from_utf8(bytes).map_err(|_| fail(code, "file is not UTF-8"))
}

pub fn read_bytes_limited(
    path: &Path,
    limit: u64,
    code: ErrorCode,
) -> Result<Vec<u8>, InferFailure> {
    let meta = fs::metadata(path).map_err(|err| io_err(code, "read", err))?;
    if !meta.is_file() {
        return Err(fail(code, "path is not a regular file"));
    }
    if meta.len() > limit {
        return Err(fail(code, "file exceeds its size limit"));
    }
    fs::read(path).map_err(|err| io_err(code, "read", err))
}

pub fn hash_regular_file(path: &Path, code: ErrorCode) -> Result<HashedFile, InferFailure> {
    let meta = fs::metadata(path).map_err(|err| io_err(code, "hash", err))?;
    if !meta.file_type().is_file() {
        return Err(fail(code, "path is not a regular file"));
    }
    let mut file = File::open(path).map_err(|err| io_err(code, "hash", err))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|err| io_err(code, "hash", err))?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .ok_or_else(|| fail(code, "file size overflows u64"))?;
        hasher.update(&buffer[..read]);
    }
    if total != meta.len() {
        return Err(fail(code, "file changed while it was hashed"));
    }
    let digest = hasher.finalize();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&digest);
    Ok(HashedFile {
        size: total,
        sha256: prefixed(bytes),
    })
}

pub fn hash_current_executable() -> Result<DigestHex, InferFailure> {
    let path = std::env::current_exe().map_err(|err| {
        fail(
            ErrorCode::ContractInvalid,
            format!("cannot resolve knolo-infer; pass --build-root ({err})"),
        )
    })?;
    Ok(hash_regular_file(&path, ErrorCode::ContractInvalid)?.sha256)
}

pub fn write_atomic(path: &Path, bytes: &[u8], code: ErrorCode) -> Result<(), InferFailure> {
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    if !parent.is_dir() {
        return Err(fail(code, "output directory does not exist"));
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| fail(code, "output path is not valid UTF-8"))?;
    let tmp = parent.join(format!(".{name}.{}.tmp", std::process::id()));
    let _ = fs::remove_file(&tmp);
    let write_result = (|| -> Result<(), InferFailure> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|err| io_err(code, "write", err))?;
        file.write_all(bytes)
            .map_err(|err| io_err(code, "write", err))?;
        file.sync_all().map_err(|err| io_err(code, "write", err))?;
        drop(file);
        fs::rename(&tmp, path).map_err(|err| io_err(code, "rename", err))?;
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    write_result
}

pub fn prefixed(bytes: [u8; 32]) -> DigestHex {
    let mut hex = String::from("sha256-");
    for byte in bytes {
        hex.push_str(&format!("{byte:02x}"));
    }
    DigestHex::parse(&hex).expect("sha256 hex is a digest")
}

fn io_err(code: ErrorCode, context: &str, err: std::io::Error) -> InferFailure {
    fail(code, format!("{context}: {err}"))
}
