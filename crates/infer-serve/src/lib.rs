//! Supervisor and its worker process.
//!
//! The supervisor binds loopback HTTP, compiles the prompt, journals the
//! request, and speaks length-prefixed CBOR to `knolo-infer-worker`. The
//! worker owns the weights, the KV pool, and the continuous scheduler.
//! Without the `cuda` feature the worker is the reference oracle. With that
//! feature it places `knolo.micro.v1` on `slot-0`. `knolo-infer run` is unchanged.

mod daemon_lock;
mod frame;
mod http;
mod metrics;
mod model;
mod openai;
mod protocol;
mod receipt;
mod supervisor;
mod trace;
mod unix_proc;
mod worker;

pub use model::{build_sampler, GenerationRequest};
pub use supervisor::{ServeConfig, Supervisor};
pub use unix_proc::{block_termination_signals, wait_for_termination_signal};
pub use worker::worker_main;

#[cfg(test)]
mod corruption_campaign;
