//! Relative POSIX paths. The rules match `ArtifactFileV1`: no absolute
//! paths, no `.` or `..` segments, and no symlink escape from the base.

use std::fs;
use std::path::{Component, Path, PathBuf};

use infer_contracts::{fail, ErrorCode, InferFailure};

pub fn check_relative_posix(path: &str, code: ErrorCode) -> Result<(), InferFailure> {
    if path.is_empty()
        || path.len() > 512
        || path.starts_with('/')
        || path.ends_with('/')
        || path.contains('\\')
        || path.contains("//")
    {
        return Err(fail(code, "path is not relative POSIX"));
    }
    for segment in path.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(fail(code, "path is not relative POSIX"));
        }
        if !segment
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-')
        {
            return Err(fail(code, "path is not relative POSIX"));
        }
    }
    Ok(())
}

pub fn resolve_inside(
    base: &Path,
    relative: &str,
    syntax: ErrorCode,
    missing: ErrorCode,
) -> Result<PathBuf, InferFailure> {
    check_relative_posix(relative, syntax)?;
    let base = fs::canonicalize(base).map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            fail(missing, "base directory is missing")
        } else {
            fail(syntax, format!("base directory: {err}"))
        }
    })?;
    let mut joined = base.clone();
    for segment in relative.split('/') {
        joined.push(segment);
    }
    let canon = fs::canonicalize(&joined).map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            fail(missing, format!("missing file {relative}"))
        } else {
            fail(syntax, format!("{relative}: {err}"))
        }
    })?;
    if canon == base || !canon.starts_with(&base) {
        return Err(fail(
            syntax,
            format!("path escapes the base directory: {relative}"),
        ));
    }
    let meta =
        fs::symlink_metadata(&canon).map_err(|err| fail(syntax, format!("{relative}: {err}")))?;
    if !meta.file_type().is_file() {
        return Err(fail(syntax, format!("{relative} is not a regular file")));
    }
    Ok(canon)
}

pub fn portable_model_path(path: &Path) -> Result<String, InferFailure> {
    let text = path.to_str().ok_or_else(|| {
        fail(
            ErrorCode::ContractInvalid,
            "model image path is not valid UTF-8",
        )
    })?;
    let stripped = text.strip_prefix("./").unwrap_or(text);
    if Path::new(stripped).is_absolute() {
        let cwd = std::env::current_dir().map_err(|err| {
            fail(
                ErrorCode::ContractInvalid,
                format!("working directory: {err}"),
            )
        })?;
        let relative = Path::new(stripped).strip_prefix(&cwd).map_err(|_| {
            fail(
                ErrorCode::ContractInvalid,
                "model image path must be relative to the working directory",
            )
        })?;
        let mut parts = Vec::new();
        for component in relative.components() {
            match component {
                Component::Normal(part) => {
                    let part = part.to_str().ok_or_else(|| {
                        fail(
                            ErrorCode::ContractInvalid,
                            "model image path is not valid UTF-8",
                        )
                    })?;
                    parts.push(part);
                }
                _ => {
                    return Err(fail(
                        ErrorCode::ContractInvalid,
                        "model image path must be relative POSIX",
                    ))
                }
            }
        }
        let joined = parts.join("/");
        check_relative_posix(&joined, ErrorCode::ContractInvalid)?;
        return Ok(joined);
    }
    check_relative_posix(stripped, ErrorCode::ContractInvalid)?;
    Ok(stripped.to_string())
}
