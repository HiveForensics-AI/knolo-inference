//! Temp-prefix checks for `scripts/install.sh`.
//! The subprocess PATH is only the fake directory. `/bin/bash` is invoked
//! by absolute path so the shell itself is not looked up on that PATH.

use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;

fn real_bin(name: &str) -> PathBuf {
    let output = Command::new("/bin/bash")
        .args(["-lc", &format!("command -v {name}")])
        .output()
        .unwrap_or_else(|_| panic!("cannot resolve {name}"));
    assert!(
        output.status.success(),
        "missing {name}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    PathBuf::from(String::from_utf8(output.stdout).unwrap().trim().to_string())
}

fn sha256(path: &Path) -> String {
    let output = Command::new(real_bin("python3"))
        .arg("-c")
        .arg(
            "import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())",
        )
        .arg(path)
        .output()
        .expect("python3");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().to_string()
}

fn write_archive(
    dir: &Path,
    name: &str,
    infer_body: &str,
    worker_body: &str,
) -> (String, String, String) {
    let stage = dir.join(format!("stage-{}", name.replace(['.', '-'], "_")));
    fs::create_dir_all(&stage).unwrap();
    fs::write(stage.join("knolo-infer"), infer_body).unwrap();
    fs::write(stage.join("knolo-infer-worker"), worker_body).unwrap();
    let archive = dir.join(name);
    let status = Command::new("tar")
        .args([
            "-czf",
            archive.to_str().unwrap(),
            "-C",
            stage.to_str().unwrap(),
            "knolo-infer",
            "knolo-infer-worker",
        ])
        .status()
        .unwrap();
    assert!(status.success());
    (
        sha256(&archive),
        sha256(&stage.join("knolo-infer")),
        sha256(&stage.join("knolo-infer-worker")),
    )
}

fn both_sums(
    cpu_archive: &str,
    cpu_infer: &str,
    cpu_worker: &str,
    cuda_archive: &str,
    cuda_infer: &str,
    cuda_worker: &str,
) -> String {
    format!(
        "{cpu_archive}  knolo-infer-linux-x86_64.tar.gz\n\
         {cpu_infer}  cpu/knolo-infer\n\
         {cpu_worker}  cpu/knolo-infer-worker\n\
         {cuda_archive}  knolo-infer-linux-x86_64-cuda.tar.gz\n\
         {cuda_infer}  cuda/knolo-infer\n\
         {cuda_worker}  cuda/knolo-infer-worker\n"
    )
}

fn fake_path(dir: &Path, tools: &[&str]) -> PathBuf {
    let bin = dir.join("bin");
    fs::create_dir_all(&bin).unwrap();
    for name in [
        "python3", "tar", "cp", "chmod", "mv", "mkdir", "mktemp", "uname", "rm", "gzip",
    ] {
        symlink(real_bin(name), bin.join(name)).unwrap();
    }
    for tool in tools {
        let path = bin.join(tool);
        fs::write(&path, "#!/bin/sh\nexit 0\n").unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();
    }
    bin
}

fn run_install(path: &Path, dist: &Path, prefix: &Path) -> std::process::Output {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/install.sh");
    Command::new("/bin/bash")
        .arg(script)
        .env_clear()
        .env("PATH", path)
        .env("HOME", prefix)
        .env("KNOLO_INFER_DIST_DIR", dist)
        .env("KNOLO_INFER_PREFIX", prefix)
        .output()
        .expect("installer")
}

#[test]
fn installer_selects_cuda_when_present_and_falls_back_to_cpu() {
    let root = std::env::temp_dir().join(format!("knolo-install-sh-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();

    let cuda_dist = root.join("cuda-dist");
    fs::create_dir_all(&cuda_dist).unwrap();
    let (cuda_archive, cuda_infer, cuda_worker) = write_archive(
        &cuda_dist,
        "knolo-infer-linux-x86_64-cuda.tar.gz",
        "cuda-infer\n",
        "cuda-worker\n",
    );
    let (cpu_archive, cpu_infer, cpu_worker) = write_archive(
        &cuda_dist,
        "knolo-infer-linux-x86_64.tar.gz",
        "cpu-infer\n",
        "cpu-worker\n",
    );
    fs::write(
        cuda_dist.join("SHA256SUMS"),
        both_sums(
            &cpu_archive,
            &cpu_infer,
            &cpu_worker,
            &cuda_archive,
            &cuda_infer,
            &cuda_worker,
        ),
    )
    .unwrap();
    let cuda_prefix = root.join("cuda-prefix");
    let cuda_out = run_install(
        &fake_path(&root.join("cuda-tools"), &["nvidia-smi"]),
        &cuda_dist,
        &cuda_prefix,
    );
    let cuda_err = String::from_utf8_lossy(&cuda_out.stderr);
    assert!(
        cuda_out.status.success(),
        "cuda install failed: {cuda_err}\n{}",
        String::from_utf8_lossy(&cuda_out.stdout)
    );
    assert_eq!(
        fs::read_to_string(cuda_prefix.join("bin/knolo-infer")).unwrap(),
        "cuda-infer\n"
    );
    assert!(
        !cuda_err.contains("CUDA archive is not published"),
        "{cuda_err}"
    );

    fs::remove_file(cuda_dist.join("knolo-infer-linux-x86_64-cuda.tar.gz")).unwrap();
    let cpu_prefix = root.join("cpu-prefix");
    let cpu_out = run_install(
        &fake_path(&root.join("cpu-tools"), &["nvidia-smi"]),
        &cuda_dist,
        &cpu_prefix,
    );
    let cpu_err = String::from_utf8_lossy(&cpu_out.stderr);
    assert!(
        cpu_out.status.success(),
        "cpu fallback failed: {cpu_err}\n{}",
        String::from_utf8_lossy(&cpu_out.stdout)
    );
    assert!(
        cpu_err
            .contains("knolo-infer: CUDA archive is not published yet; installing the CPU build"),
        "{cpu_err}"
    );
    assert_eq!(
        fs::read_to_string(cpu_prefix.join("bin/knolo-infer")).unwrap(),
        "cpu-infer\n"
    );

    let rocm_prefix = root.join("rocm-prefix");
    let rocm_out = run_install(
        &fake_path(&root.join("rocm-tools"), &["rocm-smi"]),
        &cuda_dist,
        &rocm_prefix,
    );
    let rocm_err = String::from_utf8_lossy(&rocm_out.stderr);
    assert!(
        rocm_out.status.success(),
        "rocm install failed: {rocm_err}\n{}",
        String::from_utf8_lossy(&rocm_out.stdout)
    );
    assert!(
        rocm_err
            .contains("knolo-infer: ROCm archive is not published yet; installing the CPU build"),
        "{rocm_err}"
    );
    assert!(!rocm_err.contains("CUDA archive"), "{rocm_err}");
    assert_eq!(
        fs::read_to_string(rocm_prefix.join("bin/knolo-infer")).unwrap(),
        "cpu-infer\n"
    );

    let _ = fs::remove_dir_all(&root);
}
