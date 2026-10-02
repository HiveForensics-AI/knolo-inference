//! `knolo-infer` compiles model images and runs `knolo.micro.v1` and `knolo.llama.v1`.
//!
//! `pull` copies a pinned local artifact and checks the digest. The `cuda`
//! feature places `run` and the serve worker on `slot-0`.

mod run_cmd;

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use infer_artifact::{
    compile_manifest, hash_current_executable, load_ed25519_public, load_ed25519_seed,
    new_lockfile, pin_alias, portable_model_path, pull_alias, read_lockfile, verify_image,
    verify_image_signatures, verify_weights, write_lockfile, write_model_image, DigestHex,
    ErrorCode, InferFailure, Verification,
};
use infer_receipt::default_home;
use infer_receipt::plan_pinned;
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
            reject_release_flags(&flags, false)?;
            pin_command(&flags)
        }
        "pull" => {
            reject_execution_flags(&flags)?;
            reject_release_flags(&flags, false)?;
            pull_command(&flags)
        }
        "plan" => plan_command(&flags),
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
        || flags.intent.is_some()
        || flags.public_key.is_some()
        || flags.knowledge_image.is_some()
        || !flags.query_receipts.is_empty()
        || !flags.reflex_receipts.is_empty()
    {
        return Err(usage(
            "serve accepts --model, --lock, --weights, --home, --bind, --worker, and --sign-key",
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
        signing_key: match &flags.sign_key {
            Some(path) => Some(load_ed25519_seed(path)?),
            None => None,
        },
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

fn plan_command(flags: &Flags) -> Result<(), InferFailure> {
    if flags.positionals.len() != 2 {
        return Err(usage("plan requires an alias"));
    }
    if flags.prompt.is_some()
        || flags.mode.is_some()
        || flags.receipt.is_some()
        || flags.home.is_some()
        || flags.max_tokens.is_some()
        || flags.temperature_micros.is_some()
        || flags.seed.is_some()
        || flags.stream.is_some()
        || flags.out.is_some()
        || flags.build_root.is_some()
        || flags.model.is_some()
        || flags.sign_key.is_some()
        || flags.public_key.is_some()
        || flags.knowledge_image.is_some()
        || !flags.query_receipts.is_empty()
        || !flags.reflex_receipts.is_empty()
    {
        return Err(usage(
            "plan accepts --intent, --lock, --weights, and --json",
        ));
    }
    let intent = flags
        .intent
        .clone()
        .unwrap_or_else(|| "interactive".to_string());
    match intent.as_str() {
        "interactive" => {}
        "throughput" => {
            return Err(InferFailure::new(
                ErrorCode::BackendNotAllowed,
                "throughput execution mode is not enabled",
            ));
        }
        _ => return Err(usage("intent must be interactive")),
    }
    let alias = &flags.positionals[1];
    let lock = flags
        .lock
        .clone()
        .unwrap_or_else(|| PathBuf::from("knolo.infer.lock.json"));
    let summary = plan_pinned(alias, &lock, flags.weights.as_deref())?;
    if flags.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "device": summary.device_slot,
                "expectedTotalBytes": summary.expected_total_bytes,
                "intent": intent,
                "kvBlockSize": summary.kv_block_size,
                "placementRoot": summary.placement_root.as_str(),
            }))
            .expect("plan json")
        );
    } else {
        println!("placement       {}", summary.placement_root);
        println!("device          {}", summary.device_slot);
        println!("intent          {intent}");
    }
    Ok(())
}

fn pull_command(flags: &Flags) -> Result<(), InferFailure> {
    if flags.positionals.len() != 2 {
        return Err(usage("pull requires an alias"));
    }
    if flags.out.is_some() || flags.build_root.is_some() || flags.model.is_some() {
        return Err(usage("pull accepts --lock, --weights, and --json"));
    }
    let alias = &flags.positionals[1];
    let lock = flags
        .lock
        .clone()
        .unwrap_or_else(|| PathBuf::from("knolo.infer.lock.json"));
    let report = pull_alias(&lock, alias, flags.weights.as_deref())?;
    if flags.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "alias": alias,
                "artifactRoot": report.artifact_root.as_str(),
                "modelImagePath": report.model_image_path,
                "modelImageRoot": report.image_root.as_str(),
            }))
            .expect("pull json")
        );
    } else {
        println!("pulled {alias}");
        println!("model image   {}", report.image_root);
        println!("artifact      {}", report.artifact_root);
        println!("path          {}", report.model_image_path);
    }
    Ok(())
}

fn model_command(flags: &Flags) -> Result<(), InferFailure> {
    let action = flags
        .positionals
        .get(1)
        .ok_or_else(|| usage("model requires build, inspect, or verify"))?;
    reject_release_flags(flags, action == "verify")?;
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
            if let Some(path) = &flags.public_key {
                let public = load_ed25519_public(path)?;
                verify_image_signatures(
                    &verification.image_root,
                    &verification.image.signatures,
                    &public,
                )?;
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

fn reject_release_flags(flags: &Flags, allow_public: bool) -> Result<(), InferFailure> {
    if flags.intent.is_some()
        || flags.sign_key.is_some()
        || flags.knowledge_image.is_some()
        || !flags.query_receipts.is_empty()
        || !flags.reflex_receipts.is_empty()
        || (!allow_public && flags.public_key.is_some())
    {
        return Err(usage(
            "that command does not take plan, evidence, or signature flags",
        ));
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
    intent: Option<String>,
    knowledge_image: Option<String>,
    query_receipts: Vec<String>,
    reflex_receipts: Vec<String>,
    sign_key: Option<PathBuf>,
    public_key: Option<PathBuf>,
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
            intent: None,
            knowledge_image: None,
            query_receipts: Vec::new(),
            reflex_receipts: Vec::new(),
            sign_key: None,
            public_key: None,
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
                "--intent" => flags.intent = Some(required(&mut args, "--intent")?),
                "--knowledge-image" => {
                    flags.knowledge_image = Some(required(&mut args, "--knowledge-image")?)
                }
                "--query-receipt" => {
                    flags
                        .query_receipts
                        .push(required(&mut args, "--query-receipt")?);
                }
                "--reflex-receipt" => {
                    flags
                        .reflex_receipts
                        .push(required(&mut args, "--reflex-receipt")?);
                }
                "--sign-key" => {
                    flags.sign_key = Some(PathBuf::from(required(&mut args, "--sign-key")?))
                }
                "--public-key" => {
                    flags.public_key = Some(PathBuf::from(required(&mut args, "--public-key")?))
                }
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
  knolo-infer pull <alias> [--lock <file>] [--weights <dir>] [--json]
  knolo-infer plan <alias> [--intent interactive] [--lock <file>] [--weights <dir>] [--json]
  knolo-infer run --model <alias> --prompt <text> --mode pinned --receipt <file.cbor> [--lock <file>] [--home <dir>] [--max-tokens <n>] [--temperature-micros <n> --seed <n>] [--knowledge-image <sha256-...>] [--query-receipt <sha256-...>] [--reflex-receipt <sha256-...>] [--sign-key <file>] [--json]
  knolo-infer receipt verify <file.cbor> [--home <dir>] [--model <alias> --lock <file> --weights <dir>] [--public-key <file>] [--knowledge-image <sha256-...>] [--query-receipt <sha256-...>] [--reflex-receipt <sha256-...>] [--json]
  knolo-infer replay <file.cbor> --model <alias> --prompt <text> [--lock <file>] [--home <dir>] [--out <file>] [--json]
  knolo-infer serve --model <alias> [--lock <file>] [--weights <dir>] [--home <dir>] [--bind 127.0.0.1:6767] [--worker <knolo-infer-worker>] [--sign-key <file>]

model verify without --weights checks the model image and does not open weight files.
pin records modelImageRoot, artifactRoot, and modelImagePath.
pull stages the pinned local image and its weight files after the digests match. It does not open a network connection.
plan prints a placement for the pinned model. It does not allocate pages or run a forward. throughput is refused.
run executes the pinned adapter. Without the cuda feature the device is cpu. With that feature the device is slot-0. throughput mode is refused. The receipt stores roots, not prompt text. --sign-key adds one ed25519 signature over the receipt id.
serve listens on 127.0.0.1 and runs completions through knolo-infer-worker. Without the cuda feature the worker is the reference oracle. With that feature the worker places on slot-0. A pinned completion receipt is same_build_replayable. exact_replay_verified stays on replay.
SIGINT and SIGTERM stop new completions, wait up to 30 seconds for admitted work, and then shut down.
";
