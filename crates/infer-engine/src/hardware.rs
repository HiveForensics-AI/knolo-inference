//! Host probe for the receipt. `probe_machine` records a visible accelerator
//! and does not select it. NVIDIA cards from `nvidia-smi` come first, then
//! AMD from `amd-smi` or `rocm-smi`, then Huawei Ascend from `npu-smi`.
//! `require_cuda_slot0` is the check `knolo-infer run` and the serve worker
//! use before they place on `slot-0`. That slot has to be NVIDIA. `runtimeVersion`
//! for NVIDIA is the `nvcc` release when that compiler is on `PATH`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use infer_contracts::{
    fail, DigestHex, ErrorCode, GpuProbeV1, HardwareProbeV1, InferFailure, PlacementPlanV1,
};
use serde_json::Value;

/// Toolkit release and compute capability for GPU `slot-0`.
pub struct CudaSlot0 {
    pub toolkit_version: String,
    pub architecture: String,
}

/// First accelerator the probe listed, for the library memory check.
pub struct AcceleratorMemory {
    pub vendor: String,
    pub total_bytes: u64,
}

/// Fail before a CUDA run is accepted when `slot-0` is missing, is not NVIDIA,
/// or `nvcc` did not answer. The vendor check comes first so an AMD or Huawei
/// machine hears that its kernel is absent, including when `nvcc` is not installed.
pub fn require_cuda_slot0() -> Result<CudaSlot0, InferFailure> {
    let gpus = probe_gpus();
    let gpu = require_nvidia_slot0(&gpus)?;
    let toolkit_version = cuda_toolkit_version();
    if toolkit_version == "none" {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "nvcc did not report a CUDA toolkit version",
        ));
    }
    if !capability_token(&gpu.compute_capability) {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "slot-0 compute capability is not major.minor",
        ));
    }
    Ok(CudaSlot0 {
        toolkit_version,
        architecture: gpu.compute_capability.clone(),
    })
}

pub fn first_accelerator_memory() -> Option<AcceleratorMemory> {
    let gpu = probe_gpus().into_iter().next()?;
    if gpu.vram_total_bytes == 0 {
        return None;
    }
    Some(AcceleratorMemory {
        vendor: gpu.vendor,
        total_bytes: gpu.vram_total_bytes,
    })
}

/// Refuse when `placement.expected_total_bytes` is above the probed free
/// count. `available` is that count. `None` means the probe reported nothing,
/// and this does not substitute a guessed free-byte count. No forward runs.
pub fn admit_placement_memory(
    placement: &PlacementPlanV1,
    available: Option<u64>,
) -> Result<(), InferFailure> {
    let Some(free) = available else {
        return Err(fail(
            ErrorCode::InsufficientMemory,
            "available memory was not probed",
        ));
    };
    if placement.expected_total_bytes > free {
        return Err(fail(
            ErrorCode::InsufficientMemory,
            format!(
                "placement reserves {} bytes and available memory is {} bytes",
                placement.expected_total_bytes, free
            ),
        ));
    }
    Ok(())
}

/// Free bytes for the placement's device. `cpu` uses `MemAvailable`. A slot
/// uses that GPU's probed available bytes. `None` when the probe did not
/// report the device.
pub fn probed_available_bytes(placement: &PlacementPlanV1) -> Option<u64> {
    let device = placement.devices.first()?;
    if device == "cpu" {
        return mem_info().ok().map(|(_, available)| available);
    }
    probe_gpus()
        .into_iter()
        .find(|gpu| gpu.device == *device)
        .map(|gpu| gpu.vram_available_bytes)
}

/// Same check as [`admit_placement_memory`] against the live probe.
pub fn admit_probed_placement(placement: &PlacementPlanV1) -> Result<(), InferFailure> {
    admit_placement_memory(placement, probed_available_bytes(placement))
}

fn require_nvidia_slot0(gpus: &[GpuProbeV1]) -> Result<&GpuProbeV1, InferFailure> {
    let Some(gpu) = gpus.iter().find(|gpu| gpu.device == "slot-0") else {
        return Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "cuda device slot-0 is not visible",
        ));
    };
    match gpu.vendor.as_str() {
        "nvidia" => Ok(gpu),
        "amd" => Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "slot-0 is amd; the ROCm kernel is not compiled in",
        )),
        "huawei" => Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            "slot-0 is huawei; the Ascend kernel is not compiled in",
        )),
        other => Err(fail(
            ErrorCode::PlacementUnsatisfiable,
            format!("slot-0 is {other}; the kernel is not compiled in"),
        )),
    }
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
    assemble_gpus(probe_nvidia(), probe_amd(), probe_huawei())
}

fn assemble_gpus(
    nvidia: Vec<GpuProbeV1>,
    amd: Vec<GpuProbeV1>,
    huawei: Vec<GpuProbeV1>,
) -> Vec<GpuProbeV1> {
    let mut gpus = nvidia;
    gpus.extend(amd);
    gpus.extend(huawei);
    gpus.truncate(16);
    for (index, gpu) in gpus.iter_mut().enumerate() {
        gpu.device = format!("slot-{index}");
    }
    gpus
}

fn probe_nvidia() -> Vec<GpuProbeV1> {
    let Some(text) = tool_stdout(&[
        "2",
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]) else {
        return Vec::new();
    };
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

fn probe_amd() -> Vec<GpuProbeV1> {
    if let Some(text) = tool_stdout(&["2", "amd-smi", "static", "--json"]) {
        let parsed = parse_amd_smi_json(&text);
        if !parsed.is_empty() {
            return parsed;
        }
    }
    tool_stdout(&["2", "rocm-smi", "--showmeminfo", "vram"])
        .map(|text| parse_rocm_smi(&text))
        .unwrap_or_default()
}

fn probe_huawei() -> Vec<GpuProbeV1> {
    tool_stdout(&["2", "npu-smi", "info"])
        .map(|text| parse_npu_smi(&text))
        .unwrap_or_default()
}

fn tool_stdout(args: &[&str]) -> Option<String> {
    let output = Command::new("timeout").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
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

fn architecture_token(value: &str) -> bool {
    (1..=32).contains(&value.len()) && value.bytes().all(|byte| byte.is_ascii_alphanumeric())
}

/// Driver and runtime tokens are the contract charset. A `+` build suffix is
/// `none` so one bad version cannot drop the rest of the probe.
fn sanitize_token(value: &str) -> String {
    let trimmed = value.trim();
    if (1..=32).contains(&trimmed.len())
        && trimmed.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || byte == b'.' || byte == b'_' || byte == b'-'
        })
    {
        trimmed.to_string()
    } else {
        "none".into()
    }
}

fn f32_device(
    index: usize,
    vendor: &str,
    model: String,
    memory: (u64, u64),
    capability: &str,
    driver: &str,
    runtime: &str,
) -> GpuProbeV1 {
    let (total, available) = memory;
    GpuProbeV1 {
        device: format!("slot-{index}"),
        vendor: vendor.into(),
        model,
        vram_total_bytes: total,
        vram_available_bytes: available.min(total),
        compute_capability: capability.into(),
        driver_version: driver.into(),
        runtime_version: runtime.into(),
        precisions: vec!["f32".into()],
    }
}

fn parse_amd_smi_json(text: &str) -> Vec<GpuProbeV1> {
    let Ok(value) = serde_json::from_str::<Value>(text) else {
        return Vec::new();
    };
    let mut gpus = Vec::new();
    for record in amd_records(&value) {
        let Some(gpu) = amd_record(record, gpus.len()) else {
            continue;
        };
        gpus.push(gpu);
    }
    gpus
}

fn amd_records(value: &Value) -> Vec<&Value> {
    if let Some(items) = value.as_array() {
        return items.iter().collect();
    }
    let Some(obj) = value.as_object() else {
        return Vec::new();
    };
    for key in ["gpus", "gpu"] {
        let Some(child) = obj.get(key) else {
            continue;
        };
        if let Some(items) = child.as_array() {
            return items.iter().collect();
        }
        if child.get("vram").is_some() || child.get("asic").is_some() {
            return vec![child];
        }
    }
    if obj.contains_key("vram") || obj.contains_key("asic") {
        vec![value]
    } else {
        Vec::new()
    }
}

fn amd_record(record: &Value, index: usize) -> Option<GpuProbeV1> {
    let market = record
        .pointer("/asic/market_name")
        .or_else(|| record.get("market_name"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty() && !name.eq_ignore_ascii_case("N/A"))?;
    if looks_like_uuid(market) {
        return None;
    }
    let model = clip_model(market);
    if model == "unknown" || looks_like_uuid(&model) {
        return None;
    }
    let gfx = record
        .pointer("/asic/target_graphics_version")
        .or_else(|| record.get("target_graphics_version"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|token| architecture_token(token))
        .unwrap_or("unknown");
    let driver = record
        .pointer("/driver/version")
        .and_then(Value::as_str)
        .map(sanitize_token)
        .unwrap_or_else(|| "none".into());
    let total = amd_vram_bytes(record)?;
    if total == 0 {
        return None;
    }
    // `amd-smi static` has no used-memory field. Available stays equal to total.
    Some(f32_device(
        index,
        "amd",
        model,
        (total, total),
        gfx,
        &driver,
        "none",
    ))
}

fn amd_vram_bytes(record: &Value) -> Option<u64> {
    let size = record.pointer("/vram/size")?;
    // These tools print MB for 1024*1024, the same unit `nvidia-smi` nounits uses.
    size_to_bytes(size)
}

fn size_to_bytes(size: &Value) -> Option<u64> {
    if let Some(obj) = size.as_object() {
        let value = json_u64(obj.get("value"))?;
        let unit = obj.get("unit").and_then(Value::as_str).unwrap_or("MB");
        return apply_unit(value, unit);
    }
    if let Some(text) = size.as_str() {
        let mut parts = text.split_whitespace();
        let value = parts.next()?.parse::<u64>().ok()?;
        let unit = parts.next().unwrap_or("MB");
        if parts.next().is_some() {
            return None;
        }
        return apply_unit(value, unit);
    }
    apply_unit(json_u64(Some(size))?, "MB")
}

fn apply_unit(value: u64, unit: &str) -> Option<u64> {
    let unit = unit.trim();
    let scale =
        if unit.is_empty() || unit.eq_ignore_ascii_case("MB") || unit.eq_ignore_ascii_case("MiB") {
            1024 * 1024
        } else if unit.eq_ignore_ascii_case("B") || unit.eq_ignore_ascii_case("bytes") {
            1
        } else if unit.eq_ignore_ascii_case("GB") || unit.eq_ignore_ascii_case("GiB") {
            1024 * 1024 * 1024
        } else {
            return None;
        };
    Some(value.saturating_mul(scale))
}

fn json_u64(value: Option<&Value>) -> Option<u64> {
    let value = value?;
    if let Some(number) = value.as_u64() {
        return Some(number);
    }
    if let Some(number) = value.as_i64() {
        return (number >= 0).then_some(number as u64);
    }
    if let Some(number) = value.as_f64() {
        if number.is_finite() && number >= 0.0 && number.fract() == 0.0 && number <= u64::MAX as f64
        {
            return Some(number as u64);
        }
        return None;
    }
    value.as_str()?.trim().parse().ok()
}

fn parse_rocm_smi(text: &str) -> Vec<GpuProbeV1> {
    let mut rows: BTreeMap<usize, (u64, u64)> = BTreeMap::new();
    for line in text.lines() {
        let Some(index) = rocm_gpu_index(line) else {
            continue;
        };
        let Some(bytes) = trailing_u64(line) else {
            continue;
        };
        let entry = rows.entry(index).or_insert((0, 0));
        if line.contains("VRAM Total Used Memory") {
            entry.1 = bytes;
        } else if line.contains("VRAM Total Memory") {
            entry.0 = bytes;
        }
    }
    let mut gpus = Vec::new();
    for (_index, (total, used)) in rows {
        if total == 0 {
            continue;
        }
        gpus.push(f32_device(
            gpus.len(),
            "amd",
            "AMD GPU".into(),
            (total, total.saturating_sub(used)),
            "unknown",
            "none",
            "none",
        ));
    }
    gpus
}

fn rocm_gpu_index(line: &str) -> Option<usize> {
    let rest = line.split("GPU[").nth(1)?;
    let (number, _) = rest.split_once(']')?;
    if number.is_empty() || !number.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    number.parse().ok()
}

fn trailing_u64(line: &str) -> Option<u64> {
    line.rsplit(':')
        .next()?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

fn parse_npu_smi(text: &str) -> Vec<GpuProbeV1> {
    let runtime = npu_runtime(text);
    let mut pending_name: Option<String> = None;
    let mut gpus = Vec::new();
    for line in text.lines() {
        let Some(first) = npu_first_cell(line) else {
            continue;
        };
        if let Some(name) = npu_name(first) {
            pending_name = Some(name);
            continue;
        }
        // The name row's hugepage pair is `0 / 0`. HBM is the last pair on the chip row.
        if !npu_chip(first) || !line.contains(" / ") {
            continue;
        }
        let model = pending_name.take().unwrap_or_else(|| "Ascend".into());
        let Some((used, total)) = last_used_total(line) else {
            continue;
        };
        if total == 0 || looks_like_uuid(&model) {
            continue;
        }
        let total_bytes = total.saturating_mul(1024 * 1024);
        let used_bytes = used.saturating_mul(1024 * 1024).min(total_bytes);
        gpus.push(f32_device(
            gpus.len(),
            "huawei",
            model,
            (total_bytes, total_bytes.saturating_sub(used_bytes)),
            "unknown",
            "none",
            &runtime,
        ));
    }
    gpus
}

fn npu_runtime(text: &str) -> String {
    for line in text.lines() {
        let Some((_, rest)) = line.split_once("Version:") else {
            continue;
        };
        let token = rest
            .trim()
            .trim_matches('|')
            .split_whitespace()
            .next()
            .unwrap_or("");
        return sanitize_token(token);
    }
    "none".into()
}

fn npu_first_cell(line: &str) -> Option<&str> {
    line.split('|').map(str::trim).find(|cell| !cell.is_empty())
}

fn npu_name(cell: &str) -> Option<String> {
    let tokens: Vec<&str> = cell.split_whitespace().collect();
    if tokens.len() < 2 || !tokens[0].bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    if tokens[1..].iter().any(|token| token.contains(':')) {
        return None;
    }
    let model = clip_model(&tokens[1..].join(" "));
    if looks_like_uuid(&model) {
        return None;
    }
    if model == "unknown" {
        Some("Ascend".into())
    } else {
        Some(model)
    }
}

fn npu_chip(cell: &str) -> bool {
    let mut tokens = cell.split_whitespace();
    let Some(token) = tokens.next() else {
        return false;
    };
    tokens.next().is_none() && token.bytes().all(|byte| byte.is_ascii_digit())
}

fn last_used_total(line: &str) -> Option<(u64, u64)> {
    let mut last = None;
    let mut rest = line;
    while let Some(at) = rest.find(" / ") {
        let used = rest[..at]
            .trim_end()
            .rsplit(|ch: char| ch.is_whitespace() || ch == '|')
            .next()
            .unwrap_or("");
        let total = rest[at + 3..]
            .trim_start()
            .split(|ch: char| ch.is_whitespace() || ch == '|')
            .next()
            .unwrap_or("");
        if let (Ok(used), Ok(total)) = (used.parse::<u64>(), total.parse::<u64>()) {
            last = Some((used, total));
        }
        rest = &rest[at + 3..];
    }
    last
}

pub fn default_home() -> Result<std::path::PathBuf, InferFailure> {
    if let Some(value) = std::env::var_os("KNOLO_INFER_HOME") {
        if !value.is_empty() {
            return Ok(PathBuf::from(value));
        }
    }
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
    use std::collections::BTreeMap;

    use infer_contracts::{ErrorCode, GpuProbeV1, HardwareProbeV1};

    use super::{
        assemble_gpus, parse_amd_smi_json, parse_gpu, parse_npu_smi, parse_nvcc_release,
        parse_rocm_smi, require_nvidia_slot0, tool_stdout,
    };

    fn probe_with(gpus: Vec<GpuProbeV1>) -> HardwareProbeV1 {
        HardwareProbeV1 {
            cpu_architecture: "x86_64".into(),
            cpu_model_class: "test".into(),
            physical_cores: 1,
            logical_cores: 1,
            ram_total_bytes: 1024,
            ram_available_bytes: 1024,
            numa_node_count: None,
            gpus,
            storage_class: "unknown".into(),
            storage_available_bytes: 0,
            kernel_bundles: Vec::new(),
            extensions: BTreeMap::new(),
        }
    }

    fn nvidia_line() -> GpuProbeV1 {
        parse_gpu(
            0,
            "NVIDIA GeForce RTX 2060 with Max-Q Design, 7.5, 535.309.01, 6144, 5800",
            "12.2.140",
        )
        .expect("nvidia line")
    }

    #[test]
    fn nvcc_release_keeps_the_version_token() {
        let text = "Cuda compilation tools, release 12.2, V12.2.140\n";
        assert_eq!(parse_nvcc_release(text).as_deref(), Some("12.2.140"));
        assert_eq!(parse_nvcc_release("no compiler"), None);
    }

    #[test]
    fn rocm_smi_reads_vram_bytes() {
        let text = "\
GPU[0] : VRAM Total Memory (B): 4278190080
GPU[0] : VRAM Total Used Memory (B): 108814336
not a memory line
";
        let gpus = parse_rocm_smi(text);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].vendor, "amd");
        assert_eq!(gpus[0].model, "AMD GPU");
        assert_eq!(gpus[0].compute_capability, "unknown");
        assert_eq!(gpus[0].vram_total_bytes, 4_278_190_080);
        assert_eq!(gpus[0].vram_available_bytes, 4_278_190_080 - 108_814_336);
        probe_with(gpus).validate().expect("rocm probe");
    }

    #[test]
    fn amd_smi_json_reads_the_market_name() {
        let text = r#"[
          {"asic":{"market_name":"N/A"},"vram":{"size":{"value":1,"unit":"MB"}}},
          {
            "asic":{"market_name":"AMD Radeon RX 7900 XTX","target_graphics_version":"gfx1100"},
            "vram":{"size":{"value":24576,"unit":"MB"}},
            "driver":{"version":"6.2.0"}
          }
        ]"#;
        let gpus = parse_amd_smi_json(text);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].vendor, "amd");
        assert_eq!(gpus[0].model, "AMD Radeon RX 7900 XTX");
        assert_eq!(gpus[0].compute_capability, "gfx1100");
        assert_eq!(gpus[0].driver_version, "6.2.0");
        assert_eq!(gpus[0].vram_total_bytes, 24_576 * 1024 * 1024);
        assert_eq!(gpus[0].vram_available_bytes, gpus[0].vram_total_bytes);
        probe_with(gpus).validate().expect("amd probe");

        let plus = r#"{"asic":{"market_name":"AMD Instinct MI300A","target_graphics_version":"gfx942"},"vram":{"size":{"value":196592,"unit":"MB"}},"driver":{"version":"6.9.0-rc5+"}}"#;
        let plus_gpus = parse_amd_smi_json(plus);
        assert_eq!(plus_gpus[0].driver_version, "none");
        assert_eq!(plus_gpus[0].compute_capability, "gfx942");
        assert_eq!(plus_gpus[0].vram_total_bytes, 196_592 * 1024 * 1024);

        let wrapped = r#"{"gpus":[{"asic":{"market_name":"AMD Radeon RX 7900 XTX","target_graphics_version":"not a token"},"vram":{"size":"8192 MB"}}]}"#;
        let wrapped_gpus = parse_amd_smi_json(wrapped);
        assert_eq!(wrapped_gpus[0].compute_capability, "unknown");
        assert_eq!(wrapped_gpus[0].vram_total_bytes, 8192 * 1024 * 1024);
    }

    #[test]
    fn npu_smi_reads_the_hbm_pair() {
        let text = "\
| npu-smi 24.1.0                   Version: 24.1.0                                               |
| NPU   Name                | Health        | Hugepages-Usage(page)|
| Chip                      | Bus-Id        | HBM-Usage(MB)        |
| 0     910B3               | OK            | 0    / 0             |
| 0                         | 0000:C1:00.0  | 0    0 / 2940    2983 / 65536 |
garbage
";
        let gpus = parse_npu_smi(text);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].vendor, "huawei");
        assert_eq!(gpus[0].model, "910B3");
        assert_eq!(gpus[0].compute_capability, "unknown");
        assert_eq!(gpus[0].runtime_version, "24.1.0");
        assert_eq!(gpus[0].driver_version, "none");
        let total = 65_536u64 * 1024 * 1024;
        let used = 2_983u64 * 1024 * 1024;
        assert_eq!(gpus[0].vram_total_bytes, total);
        assert_eq!(gpus[0].vram_available_bytes, total - used);
        probe_with(gpus).validate().expect("npu probe");
    }

    #[test]
    fn nvidia_stays_slot_0_when_amd_is_also_visible() {
        let amd = parse_rocm_smi("GPU[0] : VRAM Total Memory (B): 4278190080\n");
        let gpus = assemble_gpus(vec![nvidia_line()], amd, Vec::new());
        assert_eq!(gpus[0].device, "slot-0");
        assert_eq!(gpus[0].vendor, "nvidia");
        assert_eq!(gpus[1].device, "slot-1");
        assert_eq!(gpus[1].vendor, "amd");
        probe_with(gpus).validate().expect("mixed probe");
    }

    #[test]
    fn cuda_names_the_vendor_when_slot_0_is_not_nvidia() {
        let amd = parse_rocm_smi("GPU[0] : VRAM Total Memory (B): 4278190080\n");
        let amd_err = require_nvidia_slot0(&amd).expect_err("amd");
        assert_eq!(amd_err.code, ErrorCode::PlacementUnsatisfiable);
        assert_eq!(
            amd_err.message,
            "slot-0 is amd; the ROCm kernel is not compiled in"
        );

        let npu = parse_npu_smi(
            "| Version: 24.1.0 |\n| 0 910B3 | OK | 0 / 0 |\n| 0 | 0000:C1:00.0 | 2983 / 65536 |\n",
        );
        let npu_err = require_nvidia_slot0(&npu).expect_err("huawei");
        assert_eq!(
            npu_err.message,
            "slot-0 is huawei; the Ascend kernel is not compiled in"
        );

        let missing = require_nvidia_slot0(&[]).expect_err("missing");
        assert_eq!(missing.message, "cuda device slot-0 is not visible");

        let mut other = amd[0].clone();
        other.vendor = "other".into();
        let other_err = require_nvidia_slot0(&[other]).expect_err("other");
        assert_eq!(
            other_err.message,
            "slot-0 is other; the kernel is not compiled in"
        );
    }

    #[test]
    fn a_bad_line_does_not_drop_nvidia() {
        assert!(parse_gpu(1, "not,a,gpu", "12.2.140").is_none());
        let gpus = assemble_gpus(
            vec![nvidia_line()],
            parse_amd_smi_json("not json"),
            parse_npu_smi("not a table\n| nope |\n"),
        );
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].device, "slot-0");
        assert_eq!(gpus[0].vendor, "nvidia");
        probe_with(gpus).validate().expect("nvidia probe");
    }

    #[test]
    fn an_architecture_token_validates_and_a_hyphen_does_not() {
        let mut gpu = parse_amd_smi_json(
            r#"{"asic":{"market_name":"AMD Radeon RX 7900 XTX","target_graphics_version":"gfx1100"},"vram":{"size":{"value":24576,"unit":"MB"}},"driver":{"version":"6.2.0"}}"#,
        )
        .pop()
        .expect("gpu");
        probe_with(vec![gpu.clone()]).validate().expect("gfx1100");
        gpu.compute_capability = "unknown".into();
        probe_with(vec![gpu.clone()]).validate().expect("unknown");
        gpu.compute_capability = "gfx-1100".into();
        let err = probe_with(vec![gpu]).validate().expect_err("hyphen");
        assert!(
            err.message
                .contains("compute capability must be major.minor or an architecture token"),
            "{err}"
        );
    }

    #[test]
    fn a_missing_probe_tool_adds_nothing() {
        assert!(tool_stdout(&["2", "knolo-no-such-accelerator-tool"]).is_none());
    }
}
