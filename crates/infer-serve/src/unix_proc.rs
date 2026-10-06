//! The serve command blocks SIGINT and SIGTERM before it starts threads, then
//! waits on this thread. The worker does not use a parent-death signal: on
//! Linux that signal fires when the spawning thread exits, and the worker is
//! often spawned by a finished HTTP handler. Socket EOF is the lifetime.

use std::io::Error;

use infer_contracts::{fail, ErrorCode, InferFailure};

pub fn block_termination_signals() -> Result<(), InferFailure> {
    unsafe {
        let mut set = std::mem::zeroed::<libc::sigset_t>();
        if libc::sigemptyset(&mut set) != 0
            || libc::sigaddset(&mut set, libc::SIGINT) != 0
            || libc::sigaddset(&mut set, libc::SIGTERM) != 0
            || libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut()) != 0
        {
            return Err(signal_error());
        }
    }
    Ok(())
}

pub fn wait_for_termination_signal() -> Result<(), InferFailure> {
    unsafe {
        let mut set = std::mem::zeroed::<libc::sigset_t>();
        if libc::sigemptyset(&mut set) != 0
            || libc::sigaddset(&mut set, libc::SIGINT) != 0
            || libc::sigaddset(&mut set, libc::SIGTERM) != 0
        {
            return Err(signal_error());
        }
        let mut received = 0;
        if libc::sigwait(&set, &mut received) != 0 {
            return Err(signal_error());
        }
    }
    Ok(())
}

fn signal_error() -> InferFailure {
    fail(
        ErrorCode::WorkerStartFailed,
        format!(
            "termination signal could not be masked: {}",
            Error::last_os_error()
        ),
    )
}
