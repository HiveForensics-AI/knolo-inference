//! `knolo-infer` compiles model images and runs the micro-model.
//!
//! Weight download is refused. The `cuda` feature places `run` and the serve
//! worker on `slot-0`.

mod run_cmd;

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use infer_artifact::{
    compile_manifest, hash_current_executable, new_lockfile, pin_alias, portable_model_path,
    read_lockfile, unsupported_pull, verify_image, verify_weights, write_lockfile,
    write_model_image, DigestHex, ErrorCode, InferFailure, Verification,
};
use infer_receipt::default_home;
use infer_serve::{
    block_termination_signals, wait_for_termination_signal, ServeConfig, Supervisor,
};

fn main() -> ExitCode {
    match run(std::env::args().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("knolo-infer: {err}");
            ExitCode::from(1)
        }
    }
}

fn run(args: impl Iterator<Item = String>) -> Result<(), InferFailure> {
    let flags = Flags::parse(args)?;
    if flags.help || flags.positionals.is_empty() {
        print!("{HELP}");
        return Ok(());
    }
    if flags.positionals[0].as_str() != "serve" {
        reject_serve_flags(&flags)?;
    }
    match flags.positionals[0].as_str() {
        "model" => {
            reject_execution_flags(&flags)?;
            model_command(&flags)
        }
        "pin" => {
            reject_execution_flags(&flags)?;
            pin_command(&flags)
        }
        "pull" => {
            reject_execution_flags(&flags)?;
            Err(unsupported_pull())
        }
        "run" => run_cmd::run_command(&flags),
        "receipt" => run_cmd::receipt_command(&flags),
        "replay" => run_cmd::replay_command(&flags),
        "serve" => serve_command(&flags),
        other => Err(usage(format!("unknown command {other}"))),
    }
}

fn reject_serve_flags(flags: &Flags) -> Result<(), InferFailure> {
    if flags.bind.is_some() || flags.worker.is_some() {
        return Err(usage("--bind and --worker are only valid for serve"));
    }
    Ok(())
}

fn serve_command(flags: &Flags) -> Result<(), InferFailure> {
    if flags.positionals.len() != 1 {
        return Err(usage("serve takes no positional arguments"));
    }
    if flags.prompt.is_some()
        || flags.mode.is_some()
        || flags.receipt.is_some()
        || flags.max_tokens.is_some()
        || flags.temperature_micros.is_some()
        || flags.seed.is_some()
        || flags.stream.is_some()
        || flags.out.is_some()
        || flags.build_root.is_some()
        || flags.json
    {
        return Err(usage(
            "serve accepts --model, --lock, --weights, --home, --bind, and --worker",
        ));
    }
    let alias = flags
        .model
        .clone()
        .ok_or_else(|| usage("serve requires --model"))?;
    let bind = match &flags.bind {
        Some(value) => parse_bind(value)?,
        None => "127.0.0.1:6767".parse().expect("loopback bind"),
    };
    let worker_bin = match &flags.worker {
        Some(path) => path.clone(),
        None => std::env::current_exe()
            .map_err(|err| {
                InferFailure::new(
                    ErrorCode::WorkerStartFailed,
                    format!("current executable: {err}"),
                )
            })?
            .with_file_name("knolo-infer-worker"),
    };
    let work_dir = std::env::current_dir().map_err(|err| {
        InferFailure::new(
            ErrorCode::ModelArtifactMissing,
            format!("working directory: {err}"),
        )
    })?;
    block_termination_signals()?;
    let home = match &flags.home {
        Some(home) => home.clone(),
        None => default_home()?,
    };
    let supervisor = Supervisor::start(ServeConfig {
        alias,
        work_dir,
        lock_path: flags
            .lock
            .clone()
            .unwrap_or_else(|| PathBuf::from("knolo.infer.lock.json")),
        weights_dir: flags.weights.clone(),
        bind,
        worker_bin,
        home,
        pause_before_forward: false,
    })?;
    println!("listening http://{}", supervisor.address());
    wait_for_termination_signal()?;
    supervisor.drain_for_shutdown();
    supervisor.shutdown();
    Ok(())
}

fn parse_bind(value: &str) -> Result<SocketAddr, InferFailure> {
    let addr: SocketAddr = value
        .parse()
        .map_err(|_| usage("bind address must be 127.0.0.1:port"))?;
    if addr.ip() != IpAddr::V4(Ipv4Addr::LOCALHOST) {
        return Err(usage("bind address must be 127.0.0.1"));
    }
    Ok(addr)
}

fn reject_execution_flags(flags: &Flags) -> Result<(), InferFailure> {
    if flags.model.is_some()
        || flags.prompt.is_some()
        || flags.mode.is_some()
        || flags.receipt.is_some()
        || flags.home.is_some()
        || flags.max_tokens.is_some()
        || flags.temperature_micros.is_some()
        || flags.seed.is_some()
        || flags.stream.is_some()
    {
        return Err(usage(
            "run flags are only valid for run, receipt, and replay",
        ));
    }
    Ok(())
}

fn model_command(flags: &Flags) -> Result<(), InferFailure> {
    let action = flags
        .positionals
        .get(1)
        .ok_or_else(|| usage("model requires build, inspect, or verify"))?;
    match action.as_str() {
        "build" => {
            let manifest = flags
                .positionals
                .get(2)
                .ok_or_else(|| usage("model build requires a manifest"))?;
            if flags.positionals.len() != 3 {
                return Err(usage("model build takes one manifest"));
            }
            let out = flags
                .out
                .as_ref()
                .ok_or_else(|| usage("model build requires --out"))?;
            reject_unused(flags, false, false)?;
            let compiled = compile_manifest(Path::new(manifest))?;
            write_model_image(out, &compiled.bytes)?;
            if flags.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "out": out,
                        "modelImageRoot": compiled.image_root.as_str(),
                        "artifactRoot": compiled.artifact_root.as_str(),
                        "runtimeRoot": compiled.runtime_root.as_str(),
                    }))
                    .expect("summary json")
                );
            } else {
                println!("wrote {}", out.display());
                println!("model image   {}", compiled.image_root);
                println!("artifact      {}", compiled.artifact_root);
                println!("runtime       {}", compiled.runtime_root);
            }
            Ok(())
        }
        "inspect" => {
            let file = one_file(flags, "inspect")?;
            reject_unused(flags, false, false)?;
            let bytes = std::fs::read(&file).map_err(|err| {
                InferFailure::new(
                    ErrorCode::ModelImageInvalid,
                    format!("read model image: {err}"),
                )
            })?;
            let verification = verify_image(&bytes)?;
            emit(&verification, flags.json);
            Ok(())
        }
        "verify" => {
            let file = one_file(flags, "verify")?;
            reject_unused(flags, true, false)?;
            let bytes = std::fs::read(&file).map_err(|err| {
                InferFailure::new(
                    ErrorCode::ModelImageInvalid,
                    format!("read model image: {err}"),
                )
            })?;
            let mut verification = verify_image(&bytes)?;
            if let Some(weights) = &flags.weights {
                verify_weights(&verification.image, weights)?;
                verification.weights_checked = true;
            }
            emit(&verification, flags.json);
            Ok(())
        }
        other => Err(usage(format!("unknown model command {other}"))),
    }
}

fn pin_command(flags: &Flags) -> Result<(), InferFailure> {
    if flags.positionals.len() != 3 {
        return Err(usage("pin requires an alias and a model image"));
    }
    reject_unused(flags, true, true)?;
    let alias = &flags.positionals[1];
    let kmodel = PathBuf::from(&flags.positionals[2]);
    let stored = portable_model_path(&kmodel)?;
    let bytes = std::fs::read(&kmodel).map_err(|err| {
        InferFailure::new(
            ErrorCode::ModelImageInvalid,
            format!("read model image: {err}"),
        )
    })?;
    let mut verification = verify_image(&bytes)?;
    if let Some(weights) = &flags.weights {
        verify_weights(&verification.image, weights)?;
        verification.weights_checked = true;
    }
    let lock_path = flags
        .lock
        .clone()
        .unwrap_or_else(|| PathBuf::from("knolo.infer.lock.json"));
    let mut lock = match read_lockfile(&lock_path)? {
        Some(mut existing) => {
            if let Some(root) = &flags.build_root {
                existing.engine.build_root = DigestHex::parse(root)?.to_string();
                existing.engine.channel = "native".into();
            }
            existing
        }
        None => {
            let root = match &flags.build_root {
                Some(root) => DigestHex::parse(root)?,
                None => hash_current_executable()?,
            };
            new_lockfile(&root)
        }
    };
    pin_alias(&mut lock, alias, &stored, &verification.image)?;
    write_lockfile(&lock_path, &lock)?;
    let pin = lock.models.get(alias).expect("alias was inserted");
    if flags.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "alias": alias,
                "modelImageRoot": pin.model_image_root,
                "artifactRoot": pin.artifact_root,
                "modelImagePath": pin.model_image_path,
                "engine": {
                    "channel": lock.engine.channel,
                    "buildRoot": lock.engine.build_root,
                }
            }))
            .expect("pin json")
        );
    } else {
        println!("pinned {alias}");
        println!("model image   {}", pin.model_image_root);
        println!("artifact      {}", pin.artifact_root);
        println!("path          {}", pin.model_image_path);
        println!("lock          {}", lock_path.display());
    }
    Ok(())
}

fn emit(verification: &Verification, json: bool) {
    if json {
        let files: Vec<_> = verification
            .image
            .files
            .iter()
            .map(|file| {
                serde_json::json!({
                    "path": file.path,
                    "sizeBytes": file.size_bytes,
                    "sha256": file.sha256.as_str(),
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "kind": "knolo.infer.model-image",
                "version": 1,
                "name": verification.image.name,
                "variant": verification.image.variant,
                "adapter": verification.image.architecture.adapter,
                "family": verification.image.architecture.family,
                "format": verification.image.format,
                "modelImageRoot": verification.image_root.as_str(),
                "artifactRoot": verification.artifact_root.as_str(),
                "runtimeRoot": verification.runtime_root.as_str(),
                "tokenizerRoot": verification.image.tokenizer.root.as_str(),
                "templateRoot": verification.image.template.root.as_str(),
                "fileCount": verification.image.files.len(),
                "tensorCount": verification.image.tensor_inventory.len(),
                "files": files,
                "weightsChecked": verification.weights_checked,
            }))
            .expect("inspect json")
        );
    } else {
        let weights = if verification.weights_checked {
            "checked"
        } else {
            "not opened"
        };
        println!("name            {}", verification.image.name);
        println!("variant         {}", verification.image.variant);
        println!(
            "adapter         {}",
            verification.image.architecture.adapter
        );
        println!("format          {}", verification.image.format);
        println!("model image     {}", verification.image_root);
        println!("artifact        {}", verification.artifact_root);
        println!("runtime         {}", verification.runtime_root);
        println!("files           {}", verification.image.files.len());
        println!(
            "tensors         {}",
            verification.image.tensor_inventory.len()
        );
        println!("weights         {weights}");
    }
}

fn one_file(flags: &Flags, command: &str) -> Result<PathBuf, InferFailure> {
    if flags.positionals.len() != 3 {
        return Err(usage(format!("model {command} takes one model image")));
    }
    Ok(PathBuf::from(&flags.positionals[2]))
}

fn reject_unused(flags: &Flags, weights: bool, lock: bool) -> Result<(), InferFailure> {
    let command = flags.positionals.first().map(String::as_str);
    let action = flags.positionals.get(1).map(String::as_str);
    if flags.out.is_some() && command.is_some_and(|cmd| cmd != "model" || action != Some("build")) {
        return Err(usage("--out is only valid for model build"));
    }
    if !weights && flags.weights.is_some() {
        return Err(usage("--weights is only valid for model verify and pin"));
    }
    if !lock && (flags.lock.is_some() || flags.build_root.is_some()) {
        return Err(usage("--lock and --build-root are only valid for pin"));
    }
    Ok(())
}

fn usage(message: impl Into<String>) -> InferFailure {
    InferFailure::new(ErrorCode::ContractInvalid, message)
}

struct Flags {
    json: bool,
    out: Option<PathBuf>,
    weights: Option<PathBuf>,
    lock: Option<PathBuf>,
    build_root: Option<String>,
    model: Option<String>,
    prompt: Option<String>,
    mode: Option<String>,
    receipt: Option<PathBuf>,
    home: Option<PathBuf>,
    max_tokens: Option<u32>,
    temperature_micros: Option<u32>,
    seed: Option<u64>,
    stream: Option<u64>,
    bind: Option<String>,
    worker: Option<PathBuf>,
    help: bool,
    positionals: Vec<String>,
}

impl Flags {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, InferFailure> {
        let mut flags = Self {
            json: false,
            out: None,
            weights: None,
            lock: None,
            build_root: None,
            model: None,
            prompt: None,
            mode: None,
            receipt: None,
            home: None,
            max_tokens: None,
            temperature_micros: None,
            seed: None,
            stream: None,
            bind: None,
            worker: None,
            help: false,
            positionals: Vec::new(),
        };
        let mut args = args.peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--json" => flags.json = true,
                "--help" | "-h" => flags.help = true,
                "--out" => flags.out = Some(PathBuf::from(required(&mut args, "--out")?)),
                "--weights" => {
                    flags.weights = Some(PathBuf::from(required(&mut args, "--weights")?))
                }
                "--lock" => flags.lock = Some(PathBuf::from(required(&mut args, "--lock")?)),
                "--build-root" => flags.build_root = Some(required(&mut args, "--build-root")?),
                "--model" => flags.model = Some(required(&mut args, "--model")?),
                "--prompt" => flags.prompt = Some(required(&mut args, "--prompt")?),
                "--mode" => flags.mode = Some(required(&mut args, "--mode")?),
                "--receipt" => {
                    flags.receipt = Some(PathBuf::from(required(&mut args, "--receipt")?))
                }
                "--home" => flags.home = Some(PathBuf::from(required(&mut args, "--home")?)),
                "--max-tokens" => {
                    flags.max_tokens = Some(parse_u32(&required(&mut args, "--max-tokens")?)?)
                }
                "--temperature-micros" => {
                    flags.temperature_micros =
                        Some(parse_u32(&required(&mut args, "--temperature-micros")?)?)
                }
                "--seed" => flags.seed = Some(parse_u64(&required(&mut args, "--seed")?)?),
                "--stream" => flags.stream = Some(parse_u64(&required(&mut args, "--stream")?)?),
                "--bind" => flags.bind = Some(required(&mut args, "--bind")?),
                "--worker" => flags.worker = Some(PathBuf::from(required(&mut args, "--worker")?)),
                other if other.starts_with('-') => {
                    return Err(usage(format!("unknown flag {other}")));
                }
                other => flags.positionals.push(other.to_string()),
            }
        }
        Ok(flags)
    }
}

fn parse_u32(value: &str) -> Result<u32, InferFailure> {
    value
        .parse()
        .map_err(|_| usage(format!("expected an integer, got {value}")))
}

fn parse_u64(value: &str) -> Result<u64, InferFailure> {
    value
        .parse()
        .map_err(|_| usage(format!("expected an integer, got {value}")))
}

fn required(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<String, InferFailure> {
    match args.next() {
        Some(value) if !value.starts_with('-') => Ok(value),
        _ => Err(usage(format!("{flag} needs a value"))),
    }
}

const HELP: &str = "\
knolo-infer — compile model images and run the CPU reference

Usage:
  knolo-infer model build <manifest.json|yaml> --out <file.kmodel> [--json]
  knolo-infer model inspect <file.kmodel> [--json]
  knolo-infer model verify <file.kmodel> [--weights <dir>] [--json]
  knolo-infer pin <alias> <file.kmodel> [--lock <knolo.infer.lock.json>] [--weights <dir>] [--build-root <sha256-...>] [--json]
  knolo-infer pull [<alias>]
  knolo-infer run --model <alias> --prompt <text> --mode pinned --receipt <file.cbor> [--lock <file>] [--home <dir>] [--max-tokens <n>] [--temperature-micros <n> --seed <n>] [--json]
  knolo-infer receipt verify <file.cbor> [--home <dir>] [--model <alias> --lock <file> --weights <dir>] [--json]
  knolo-infer replay <file.cbor> --model <alias> --prompt <text> [--lock <file>] [--home <dir>] [--out <file>] [--json]
  knolo-infer serve --model <alias> [--lock <file>] [--weights <dir>] [--home <dir>] [--bind 127.0.0.1:6767] [--worker <knolo-infer-worker>]

model verify without --weights checks the model image and does not open weight files.
pin records modelImageRoot, artifactRoot, and modelImagePath. It does not download weights.
pull is refused until download staging exists.
run executes knolo.micro.v1. Without the cuda feature the device is cpu. With that feature the device is slot-0. throughput mode is refused. The receipt stores roots, not prompt text.
serve listens on 127.0.0.1 and runs completions through knolo-infer-worker. Without the cuda feature the worker is the reference oracle. With that feature the worker places on slot-0. It journals each request before the forward and does not change run.
SIGINT and SIGTERM stop new completions, wait up to 30 seconds for admitted work, and then shut down.
";
