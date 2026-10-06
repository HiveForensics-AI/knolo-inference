//! Engine build for the host binary.
//!
//! The CPU bundle names no CUDA kernels. The CUDA bundle is selected by
//! `knolo-infer run` and by the serve worker when those binaries are built
//! with the `cuda` feature.

use std::collections::BTreeMap;
use std::process::Command;

use infer_contracts::{
    digest_bytes, DigestHex, EngineBuildDescriptorV1, InferFailure, KernelBundleDescriptorV1,
};

pub fn cargo_lock_root() -> Result<DigestHex, InferFailure> {
    digest_bytes("infer-engine-build", include_bytes!("../../../Cargo.lock"))
}

pub fn cpu_kernel_plan_root() -> Result<DigestHex, InferFailure> {
    digest_bytes(
        "infer-kernel-bundle",
        b"candle-cpu:rmsnorm,rope,attention,swiglu,gemv",
    )
}

pub fn cpu_kernel_bundle(candle_version: &str) -> Result<KernelBundleDescriptorV1, InferFailure> {
    let bundle = KernelBundleDescriptorV1 {
        cuda_architectures: Vec::new(),
        cuda_toolkit_version: "none".into(),
        source_root: digest_bytes("infer-kernel-bundle", b"no-cuda-kernels")?,
        compiler_flags: Vec::new(),
        build_mode: "cpu".into(),
        code_object_root: digest_bytes(
            "infer-kernel-bundle",
            format!("candle-cpu-{candle_version}").as_bytes(),
        )?,
        jit: None,
        extensions: BTreeMap::new(),
    };
    bundle.validate()?;
    Ok(bundle)
}

pub fn cuda_kernel_plan_root() -> Result<DigestHex, InferFailure> {
    digest_bytes(
        "infer-kernel-bundle",
        b"candle-cuda:rmsnorm,rope,attention,swiglu,gemv",
    )
}

pub fn cuda_kernel_bundle(
    candle_version: &str,
    toolkit_version: &str,
    architecture: &str,
) -> Result<KernelBundleDescriptorV1, InferFailure> {
    let bundle = KernelBundleDescriptorV1 {
        cuda_architectures: vec![architecture.to_string()],
        cuda_toolkit_version: toolkit_version.to_string(),
        source_root: digest_bytes("infer-kernel-bundle", b"candle-cuda-kernels")?,
        compiler_flags: Vec::new(),
        build_mode: "cuda".into(),
        code_object_root: digest_bytes(
            "infer-kernel-bundle",
            format!("candle-cuda-{candle_version}").as_bytes(),
        )?,
        jit: None,
        extensions: BTreeMap::new(),
    };
    bundle.validate()?;
    Ok(bundle)
}

pub fn reference_kernel_plan_root() -> Result<DigestHex, InferFailure> {
    digest_bytes(
        "infer-kernel-bundle",
        b"reference-f32:rmsnorm,rope,attention,swiglu,gemv",
    )
}

pub fn reference_kernel_bundle() -> Result<KernelBundleDescriptorV1, InferFailure> {
    let bundle = KernelBundleDescriptorV1 {
        cuda_architectures: Vec::new(),
        cuda_toolkit_version: "none".into(),
        source_root: digest_bytes("infer-kernel-bundle", b"no-cuda-kernels")?,
        compiler_flags: Vec::new(),
        build_mode: "cpu".into(),
        code_object_root: digest_bytes("infer-kernel-bundle", b"reference-f32")?,
        jit: None,
        extensions: BTreeMap::new(),
    };
    bundle.validate()?;
    Ok(bundle)
}

pub fn host_engine_build(
    binary_sha256: DigestHex,
    candle_version: &str,
    kernel_bundle_root: DigestHex,
) -> Result<EngineBuildDescriptorV1, InferFailure> {
    engine_build(
        binary_sha256,
        "candle-cpu",
        candle_version,
        kernel_bundle_root,
        Vec::new(),
    )
}

pub fn cuda_engine_build(
    binary_sha256: DigestHex,
    candle_version: &str,
    kernel_bundle_root: DigestHex,
) -> Result<EngineBuildDescriptorV1, InferFailure> {
    engine_build(
        binary_sha256,
        "candle-cuda",
        candle_version,
        kernel_bundle_root,
        vec!["cuda".into()],
    )
}

pub fn reference_engine_build(
    binary_sha256: DigestHex,
    kernel_bundle_root: DigestHex,
) -> Result<EngineBuildDescriptorV1, InferFailure> {
    engine_build(
        binary_sha256,
        "reference-f32",
        "1",
        kernel_bundle_root,
        Vec::new(),
    )
}

fn engine_build(
    binary_sha256: DigestHex,
    tensor_backend: &str,
    tensor_backend_version: &str,
    kernel_bundle_root: DigestHex,
    feature_set: Vec<String>,
) -> Result<EngineBuildDescriptorV1, InferFailure> {
    let build = EngineBuildDescriptorV1 {
        binary_sha256,
        source_commit: source_commit(),
        cargo_lock_root: cargo_lock_root()?,
        rustc_version: rustc_version(),
        target_triple: target_triple(),
        build_profile: if cfg!(debug_assertions) {
            "debug".into()
        } else {
            "release".into()
        },
        feature_set,
        tensor_backend: tensor_backend.into(),
        tensor_backend_version: tensor_backend_version.into(),
        kernel_bundle_root,
        extensions: BTreeMap::new(),
    };
    build.validate()?;
    Ok(build)
}

fn source_commit() -> String {
    let output = Command::new("git").args(["rev-parse", "HEAD"]).output();
    let Ok(output) = output else {
        return "unknown".into();
    };
    if !output.status.success() {
        return "unknown".into();
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let commit = text.trim();
    if commit.len() == 40
        && commit
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        commit.to_string()
    } else {
        "unknown".into()
    }
}

fn rustc_version() -> String {
    let output = Command::new("rustc").arg("--version").output();
    let Ok(output) = output else {
        return "unknown".into();
    };
    let text = String::from_utf8_lossy(&output.stdout);
    let version = text.split_whitespace().nth(1).unwrap_or("unknown");
    if version.len() <= 32 && !version.is_empty() {
        version.to_string()
    } else {
        "unknown".into()
    }
}

fn target_triple() -> String {
    let output = Command::new("rustc").arg("-vV").output();
    if let Ok(output) = output {
        let text = String::from_utf8_lossy(&output.stdout);
        for line in text.lines() {
            if let Some(host) = line.strip_prefix("host: ") {
                let host = host.trim();
                if !host.is_empty() && host.len() <= 64 {
                    return host.to_string();
                }
            }
        }
    }
    match (std::env::consts::ARCH, std::env::consts::OS) {
        ("x86_64", "linux") => "x86_64-unknown-linux-gnu".into(),
        ("aarch64", "linux") => "aarch64-unknown-linux-gnu".into(),
        _ => "unknown".into(),
    }
}
