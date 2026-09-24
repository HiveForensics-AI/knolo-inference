//! Candle execution of `knolo.micro.v1`.
//!
//! The reference oracle in `infer-engine` remains the numerical authority.
//! The `cuda` feature places the same graph on device ordinal 0. This crate
//! does not start a server.

mod candle_micro;

#[cfg(feature = "cuda")]
pub use candle_micro::CandleCudaBackend;
pub use candle_micro::{
    cpu_adapter_by_id, CandleCpuBackend, CandleMicroAdapter, CANDLE_CPU_VERSION,
};
