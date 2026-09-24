//! Host probe for the receipt. `probe_machine` records a visible GPU and does
//! not select it. `require_cuda_slot0` is the check `knolo-infer run` and the
//! serve worker use before they place on `slot-0`. `runtimeVersion` is the
//! `nvcc` release when that compiler is on `PATH`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;

use infer_contracts::{fail, DigestHex, ErrorCode, GpuProbeV1, HardwareProbeV1, InferFailure};

/// Toolkit release and compute capability for GPU `slot-0`.
pub struct CudaSlot0 {
    pub toolkit_version: String,
    pub architecture: String,
}

/// Fail before a CUDA run is accepted when `slot-0` or `nvcc` is missing.
pub fn require_cuda_slot0() -> Result<CudaSlot0, InferFailure> {
    let toolkit_version = cuda_toolkit_version();
    if toolkit_version == "none" {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "nvcc did not report a CUDA toolkit version",
        ));
    }
    let Some(gpu) = probe_gpus().into_iter().find(|gpu| gpu.device == "slot-0") else {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "cuda device slot-0 is not visible",
        ));
    };
    if !capability_token(&gpu.compute_capability) {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "slot-0 compute capability is not major.minor",
        ));
    }
    Ok(CudaSlot0 {
        toolkit_version,
        architecture: gpu.compute_capability,
    })
}

fn capability_token(value: &str) -> bool {
    let Some((major, minor)) = value.split_once('.') else {
        return false;
    };
    !major.is_empty()
        && !minor.is_empty()
        && major.len() <= 3
        && minor.len() <= 3
        && major.bytes().all(|byte| byte.is_ascii_digit())
        && minor.bytes().all(|byte| byte.is_ascii_digit())
}

pub fn probe_machine(kernel_bundle_root: &DigestHex) -> Result<HardwareProbeV1, InferFailure> {
    let (logical, physical, model) = cpu_info();
    let (ram_total, ram_available) = mem_info()?;
    let mut probe = HardwareProbeV1 {
        cpu_architecture: match std::env::consts::ARCH {
            "x86_64" => "x86_64".into(),
            "aarch64" => "aarch64".into(),
            _ => "unknown".into(),
        },
        cpu_model_class: model,
        physical_cores: physical,
        logical_cores: logical.max(physical),
        ram_total_bytes: ram_total,
        ram_available_bytes: ram_available.min(ram_total),
        numa_node_count: numa_nodes(),
        gpus: probe_gpus(),
        storage_class: "unknown".into(),
        storage_available_bytes: storage_available(),
        kernel_bundles: vec![kernel_bundle_root.clone()],
        extensions: BTreeMap::new(),
    };
    if let Err(err) = probe.validate() {
        if probe.gpus.is_empty() {
            return Err(err);
        }
        probe.gpus.clear();
        probe.validate()?;
    }
    Ok(probe)
}

fn cpu_info() -> (u32, u32, String) {
    let text = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let mut logical = 0u32;
    let mut model = String::new();
    let mut pairs = BTreeMap::<(String, String), ()>::new();
    let mut physical = String::new();
    let mut core = String::new();
    let mut saw_pair = false;
    for line in text.lines() {
        if line.is_empty() {
            if !physical.is_empty() && !core.is_empty() {
                pairs.insert((physical.clone(), core.clone()), ());
                saw_pair = true;
            }
            physical.clear();
            core.clear();
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        if key == "processor" {
            logical = logical.saturating_add(1);
        } else if key == "model name" && model.is_empty() {
            model = clip_model(value);
        } else if key == "physical id" {
            physical = value.to_string();
        } else if key == "core id" {
            core = value.to_string();
        }
    }
    if !physical.is_empty() && !core.is_empty() {
        pairs.insert((physical, core), ());
        saw_pair = true;
    }
    if logical == 0 {
        logical = std::thread::available_parallelism()
            .map(|count| count.get() as u32)
            .unwrap_or(1);
    }
    let physical_cores = if saw_pair && !pairs.is_empty() {
        pairs.len() as u32
    } else {
        logical
    };
    if model.is_empty() {
        model = "unknown".into();
    }
    (logical, physical_cores.min(logical), model)
}

fn clip_model(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_control() {
            continue;
        }
        let next = out.len() + ch.len_utf8();
        if next > 64 {
            break;
        }
        out.push(ch);
    }
    if out.is_empty() {
        "unknown".into()
    } else {
        out
    }
}

fn mem_info() -> Result<(u64, u64), InferFailure> {
    let text = fs::read_to_string("/proc/meminfo").map_err(|err| {
        fail(
            ErrorCode::ContractInvalid,
            format!("cannot read memory info: {err}"),
        )
    })?;
    let mut total = None;
    let mut available = None;
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let kb: u64 = rest
            .split_whitespace()
            .next()
            .unwrap_or("0")
            .parse()
            .unwrap_or(0);
        let bytes = kb.saturating_mul(1024);
        if key == "MemTotal" {
            total = Some(bytes);
        } else if key == "MemAvailable" {
            available = Some(bytes);
        }
    }
    match (total, available) {
        (Some(total), Some(available)) if total > 0 => Ok((total, available.min(total))),
        _ => Err(fail(
            ErrorCode::ContractInvalid,
            "memory info did not report total and available bytes",
        )),
    }
}

fn numa_nodes() -> Option<u32> {
    let entries = fs::read_dir("/sys/devices/system/node").ok()?;
    let count = entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.file_name().to_str().is_some_and(|name| {
                name.starts_with("node") && name[4..].bytes().all(|b| b.is_ascii_digit())
            })
        })
        .count() as u32;
    if (1..=64).contains(&count) {
        Some(count)
    } else {
        None
    }
}

fn storage_available() -> u64 {
    let Ok(output) = Command::new("df")
        .args(["-B1", "--output=avail", "."])
        .output()
    else {
        return 0;
    };
    if !output.status.success() {
        return 0;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .nth(1)
        .and_then(|line| line.trim().parse().ok())
        .unwrap_or(0)
}

fn probe_gpus() -> Vec<GpuProbeV1> {
    let output = Command::new("timeout")
        .args([
            "2",
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let output = match output {
        Ok(output) if output.status.success() => output,
        _ => return Vec::new(),
    };
    let text = String::from_utf8_lossy(&output.stdout);
    let runtime = cuda_toolkit_version();
    let mut gpus = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if index > 15 {
            break;
        }
        let Some(gpu) = parse_gpu(index, line, &runtime) else {
            continue;
        };
        gpus.push(gpu);
    }
    gpus
}

fn cuda_toolkit_version() -> String {
    let output = Command::new("nvcc").arg("--version").output();
    let Ok(output) = output else {
        return "none".into();
    };
    if !output.status.success() {
        return "none".into();
    }
    let text = String::from_utf8_lossy(&output.stdout);
    parse_nvcc_release(&text).unwrap_or_else(|| "none".into())
}

/// `V12.2.140` from `nvcc --version` becomes `12.2.140`.
fn parse_nvcc_release(text: &str) -> Option<String> {
    for token in text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '.')) {
        let Some(rest) = token.strip_prefix('V') else {
            continue;
        };
        if rest.len() > 16
            || !rest.contains('.')
            || !rest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || byte == b'.')
            || rest.starts_with('.')
            || rest.ends_with('.')
        {
            continue;
        }
        return Some(rest.to_string());
    }
    None
}

fn parse_gpu(index: usize, line: &str, runtime_version: &str) -> Option<GpuProbeV1> {
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() < 5 {
        return None;
    }
    let free_mib: u64 = parts[parts.len() - 1].parse().ok()?;
    let total_mib: u64 = parts[parts.len() - 2].parse().ok()?;
    let driver = parts[parts.len() - 3].to_string();
    let capability = parts[parts.len() - 4].to_string();
    let name = clip_model(&parts[..parts.len() - 4].join(", "));
    if looks_like_uuid(&name) || looks_like_uuid(&driver) {
        return None;
    }
    let total = total_mib.saturating_mul(1024 * 1024);
    let free = free_mib.saturating_mul(1024 * 1024).min(total);
    let major: u32 = capability.split('.').next()?.parse().ok()?;
    let precisions = if major >= 7 {
        vec!["f16".into(), "f32".into()]
    } else {
        vec!["f32".into()]
    };
    let gpu = GpuProbeV1 {
        device: format!("slot-{index}"),
        vendor: "nvidia".into(),
        model: name,
        vram_total_bytes: total,
        vram_available_bytes: free,
        compute_capability: capability,
        driver_version: driver,
        runtime_version: runtime_version.into(),
        precisions,
    };
    Some(gpu)
}

fn looks_like_uuid(value: &str) -> bool {
    let parts: Vec<_> = value.split('-').collect();
    parts.len() == 5
        && parts[0].len() == 8
        && parts[1].len() == 4
        && parts[2].len() == 4
        && parts[3].len() == 4
        && parts[4].len() == 12
        && parts
            .iter()
            .all(|part| part.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

pub fn default_home() -> Result<std::path::PathBuf, InferFailure> {
    let home = std::env::var_os("HOME").ok_or_else(|| {
        fail(
            ErrorCode::ReceiptPersistFailed,
            "HOME is unset; pass --home",
        )
    })?;
    Ok(Path::new(&home).join(".knolo").join("infer"))
}

#[cfg(test)]
mod tests {
    use super::parse_nvcc_release;

    #[test]
    fn nvcc_release_keeps_the_version_token() {
        let text = "Cuda compilation tools, release 12.2, V12.2.140\n";
        assert_eq!(parse_nvcc_release(text).as_deref(), Some("12.2.140"));
        assert_eq!(parse_nvcc_release("no compiler"), None);
    }
}
