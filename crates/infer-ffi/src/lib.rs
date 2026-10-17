//! C ABI for an in-process caller. A React Native module will call these
//! symbols later. This crate does not listen on a socket, spawn
//! `knolo-infer-worker`, or read `~/.knolo`.
//!
//! `knolo_infer_tiny_greedy` stages the llama-tiny generator in a temp
//! directory because `load_verified_llama` checks the image and weight files
//! on disk. The prompt is the checked-in `three` case: token ids 1, 4, 7.

use std::ffi::{c_char, CStr};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use infer_contracts::{fail, ErrorCode, InferFailure};
use infer_engine::{
    greedy_generate, llama_cpu_placement, llama_kv_layout, load_verified_llama, write_llama_model,
    ArchitectureAdapter, LlamaAdapter, ReferenceF32Backend, SingleBlockKv,
};

const TINY_PROMPT: &[u32] = &[1, 4, 7];
const OK: i32 = 0;
const BAD_ARGUMENT: i32 = 1;
const ENGINE: i32 = 2;

static ERROR_TEXT: Mutex<[u8; 512]> = Mutex::new([0; 512]);
static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Crate version. The pointer is valid for the life of the process.
#[no_mangle]
pub extern "C" fn knolo_infer_version() -> *const c_char {
    static VERSION: &CStr = c"0.1.0";
    VERSION.as_ptr()
}

/// Last error from this process. The pointer stays valid until the next
/// `knolo_infer_tiny_greedy` call. An empty string means the last call succeeded.
#[no_mangle]
pub extern "C" fn knolo_infer_last_error() -> *const c_char {
    let guard = ERROR_TEXT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.as_ptr() as *const c_char
}

/// Write greedy token ids for the llama-tiny prompt `1, 4, 7`.
///
/// Returns 0 when `out_tokens` holds `*out_len` new ids. Returns 1 when a
/// pointer is null or `out_cap` is too small; a short buffer sets `*out_len`
/// to the number of ids that were produced and does not write them. Returns 2
/// when the engine refuses the run.
///
/// # Safety
/// `out_tokens` must be writable for `out_cap` elements when it is non-null.
/// `out_len` must be writable when it is non-null. A null pointer is refused
/// and is not dereferenced.
#[no_mangle]
pub unsafe extern "C" fn knolo_infer_tiny_greedy(
    max_tokens: u32,
    out_tokens: *mut u32,
    out_cap: u32,
    out_len: *mut u32,
) -> i32 {
    if out_tokens.is_null() || out_len.is_null() {
        set_error("token buffer is null");
        return BAD_ARGUMENT;
    }
    let tokens = match tiny_tokens(max_tokens) {
        Ok(tokens) => tokens,
        Err(err) => {
            set_error(&err.message);
            return ENGINE;
        }
    };
    if tokens.len() > out_cap as usize {
        *out_len = tokens.len() as u32;
        set_error("token buffer is too small");
        return BAD_ARGUMENT;
    }
    std::ptr::copy_nonoverlapping(tokens.as_ptr(), out_tokens, tokens.len());
    *out_len = tokens.len() as u32;
    clear_error();
    OK
}

fn tiny_tokens(max_tokens: u32) -> Result<Vec<u32>, InferFailure> {
    let scratch = Scratch::new()?;
    write_llama_model(&scratch.path)?;
    let source = load_verified_llama(&scratch.path.join("llama.kmodel"), &scratch.path)?;
    let placement = llama_cpu_placement(&source)?;
    let mut model = LlamaAdapter.build(&source, &placement, &ReferenceF32Backend)?;
    let mut kv = SingleBlockKv::new(llama_kv_layout())?;
    let output = greedy_generate(model.as_mut(), &mut kv, 1, TINY_PROMPT, max_tokens)?;
    Ok(output.tokens)
}

struct Scratch {
    path: PathBuf,
}

impl Scratch {
    fn new() -> Result<Self, InferFailure> {
        let n = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("knolo-infer-ffi-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).map_err(|err| {
            fail(
                ErrorCode::ModelImageInvalid,
                format!("cannot create the llama-tiny directory: {err}"),
            )
        })?;
        Ok(Self { path })
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn set_error(message: &str) {
    let mut guard = ERROR_TEXT
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.fill(0);
    let mut used = 0;
    for ch in message.chars() {
        if ch == '\0' {
            continue;
        }
        let mut encoded = [0u8; 4];
        let bytes = ch.encode_utf8(&mut encoded);
        if used + bytes.len() >= guard.len() {
            break;
        }
        guard[used..used + bytes.len()].copy_from_slice(bytes.as_bytes());
        used += bytes.len();
    }
}

fn clear_error() {
    set_error("");
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::{
        knolo_infer_last_error, knolo_infer_tiny_greedy, knolo_infer_version, BAD_ARGUMENT, ENGINE,
        OK, TINY_PROMPT,
    };
    use infer_engine::{
        greedy_generate, llama_cpu_placement, llama_kv_layout, load_verified_llama,
        write_llama_model, ArchitectureAdapter, LlamaAdapter, ReferenceF32Backend, SingleBlockKv,
    };

    fn oracle(new_tokens: u32) -> Vec<u32> {
        let scratch = super::Scratch::new().unwrap();
        write_llama_model(&scratch.path).unwrap();
        let source =
            load_verified_llama(&scratch.path.join("llama.kmodel"), &scratch.path).unwrap();
        let placement = llama_cpu_placement(&source).unwrap();
        let mut model = LlamaAdapter
            .build(&source, &placement, &ReferenceF32Backend)
            .unwrap();
        let mut kv = SingleBlockKv::new(llama_kv_layout()).unwrap();
        greedy_generate(model.as_mut(), &mut kv, 1, TINY_PROMPT, new_tokens)
            .unwrap()
            .tokens
    }

    #[test]
    fn version_is_the_crate_version() {
        let text = unsafe { CStr::from_ptr(knolo_infer_version()) };
        assert_eq!(text.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn tiny_greedy_matches_the_cpu_oracle() {
        assert_eq!(TINY_PROMPT, &[1, 4, 7]);
        let expected = oracle(4);
        let mut out = [0u32; 8];
        let mut len = u32::MAX;
        let code =
            unsafe { knolo_infer_tiny_greedy(4, out.as_mut_ptr(), out.len() as u32, &mut len) };
        assert_eq!(code, OK);
        assert_eq!(len as usize, expected.len());
        assert_eq!(&out[..len as usize], expected.as_slice());
        let err = unsafe { CStr::from_ptr(knolo_infer_last_error()) };
        assert_eq!(err.to_bytes(), b"");
    }

    #[test]
    fn a_short_buffer_writes_the_needed_count() {
        let mut out = [7u32; 1];
        let mut len = 0u32;
        let code = unsafe { knolo_infer_tiny_greedy(4, out.as_mut_ptr(), 1, &mut len) };
        assert_eq!(code, BAD_ARGUMENT);
        assert_eq!(len, 4);
        assert_eq!(out, [7]);
        let err = unsafe { CStr::from_ptr(knolo_infer_last_error()) };
        assert_eq!(err.to_str().unwrap(), "token buffer is too small");
    }

    #[test]
    fn a_null_buffer_is_refused() {
        let mut len = 0u32;
        let code = unsafe { knolo_infer_tiny_greedy(4, std::ptr::null_mut(), 4, &mut len) };
        assert_eq!(code, BAD_ARGUMENT);
        assert_eq!(len, 0);
        let err = unsafe { CStr::from_ptr(knolo_infer_last_error()) };
        assert_eq!(err.to_str().unwrap(), "token buffer is null");
    }

    #[test]
    fn too_many_tokens_for_the_fixture_context_is_an_engine_error() {
        let mut out = [0u32; 32];
        let mut len = 9u32;
        let code =
            unsafe { knolo_infer_tiny_greedy(32, out.as_mut_ptr(), out.len() as u32, &mut len) };
        assert_eq!(code, ENGINE);
        assert_eq!(len, 9);
        let err = unsafe { CStr::from_ptr(knolo_infer_last_error()) };
        assert!(!err.to_bytes().is_empty());
    }
}
