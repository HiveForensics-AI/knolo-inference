//! Receipt for one pinned run.
//!
//! The default build records `candle-cpu` on device `cpu`. The `cuda` feature
//! records `candle-cuda` on device `slot-0`.
//!
//! `accepted` is fsynced before the forward pass. Prefill, token, and
//! completed events are fsynced before the receipt is published. Prompt text
//! stays in the caller's terminal output; the receipt stores roots. The
//! journal itself lives in `infer-engine` so `knolo-infer-worker` does not
//! link Candle.

mod run;
mod verify;

pub use infer_engine::{
    cpu_kernel_bundle, cpu_kernel_plan_root, default_home, host_engine_build, load_sampler_plan,
    probe_machine, verify_journal, Journal,
};
pub use run::{replay_pinned, run_pinned, ReplayOptions, RunOptions, RunOutput};
pub use verify::{
    request_id_of, verify_receipt_bytes, verify_receipt_journal, verify_receipt_model,
};
