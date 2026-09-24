//! `run`, `receipt verify`, and `replay`.
//!
//! The default build places the model on `cpu`. The `cuda` feature places it
//! on `slot-0`.

use std::path::{Path, PathBuf};

use infer_artifact::{read_lockfile, write_atomic};
use infer_contracts::{fail, ErrorCode, InferFailure};
use infer_engine::load_verified_micro;
use infer_receipt::{
    default_home, replay_pinned, run_pinned, verify_receipt_bytes, verify_receipt_journal,
    verify_receipt_model, ReplayOptions, RunOptions,
};

use crate::Flags;

pub fn run_command(flags: &Flags) -> Result<(), InferFailure> {
    if flags.positionals.len() != 1 {
        return Err(usage("run takes flags, not extra arguments"));
    }
    reject_unused_run(flags)?;
    let alias = flags
        .model
        .clone()
        .ok_or_else(|| usage("run requires --model"))?;
    let prompt = flags
        .prompt
        .clone()
        .ok_or_else(|| usage("run requires --prompt"))?;
    let mode = flags
        .mode
        .clone()
        .ok_or_else(|| usage("run requires --mode"))?;
    let receipt_path = flags
        .receipt
        .clone()
        .ok_or_else(|| usage("run requires --receipt"))?;
    let home = home_dir(flags)?;
    let lock = flags
        .lock
        .clone()
        .unwrap_or_else(|| PathBuf::from("knolo.infer.lock.json"));
    let output = run_pinned(&RunOptions {
        alias,
        prompt,
        mode,
        lock_path: lock,
        home,
        weights_dir: flags.weights.clone(),
        max_output_tokens: flags.max_tokens,
        temperature_micros: flags.temperature_micros,
        seed: flags.seed,
        stream: flags.stream,
    })?;
    write_atomic(
        &receipt_path,
        &output.receipt_bytes,
        ErrorCode::ReceiptPersistFailed,
    )?;
    if flags.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "assurance": output.receipt.assurance,
                "finishReason": output.receipt.output.finish_reason,
                "journal": output.journal_dir,
                "outputText": output.output_text,
                "outputTokens": output.output_tokens,
                "promptTokenCount": output.receipt.prompt.prompt_token_count,
                "deviceSlot": output.receipt.hardware.device_slot,
                "receiptId": output.receipt.receipt_id.as_str(),
                "requestId": output.request_id,
            }))
            .expect("run json")
        );
    } else {
        println!("{}", output.output_text);
        println!("finish          {}", output.receipt.output.finish_reason);
        println!("tokens          {}", output.output_tokens.len());
        println!("receipt         {}", output.receipt.receipt_id);
        println!("assurance       {}", output.receipt.assurance);
        println!("device          {}", output.receipt.hardware.device_slot);
        println!("journal         {}", output.journal_dir.display());
    }
    Ok(())
}

pub fn receipt_command(flags: &Flags) -> Result<(), InferFailure> {
    let action = flags
        .positionals
        .get(1)
        .ok_or_else(|| usage("receipt requires verify"))?;
    if action != "verify" {
        return Err(usage("unknown receipt command"));
    }
    if flags.positionals.len() != 3 {
        return Err(usage("receipt verify takes one receipt"));
    }
    reject_unused_verify(flags)?;
    let path = PathBuf::from(&flags.positionals[2]);
    let bytes = std::fs::read(&path).map_err(|err| {
        fail(
            ErrorCode::ReceiptRequired,
            format!("cannot read receipt: {err}"),
        )
    })?;
    let receipt = verify_receipt_bytes(&bytes)?;
    if let Some(home) = &flags.home {
        verify_receipt_journal(home, &receipt)?;
    }
    if flags.model.is_some() || flags.weights.is_some() || flags.lock.is_some() {
        let alias = flags
            .model
            .clone()
            .ok_or_else(|| usage("receipt verify against a model requires --model"))?;
        let lock_path = flags
            .lock
            .clone()
            .unwrap_or_else(|| PathBuf::from("knolo.infer.lock.json"));
        let source = open_for_verify(&alias, &lock_path, flags.weights.as_deref())?;
        verify_receipt_model(&receipt, &source)?;
    }
    if flags.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "assurance": receipt.assurance,
                "receiptId": receipt.receipt_id.as_str(),
                "verified": true,
            }))
            .expect("verify json")
        );
    } else {
        println!("verified {}", receipt.receipt_id);
        println!("assurance {}", receipt.assurance);
    }
    Ok(())
}

pub fn replay_command(flags: &Flags) -> Result<(), InferFailure> {
    if flags.positionals.len() != 2 {
        return Err(usage("replay takes one receipt"));
    }
    reject_unused_replay(flags)?;
    let receipt_path = PathBuf::from(&flags.positionals[1]);
    let bytes = std::fs::read(&receipt_path).map_err(|err| {
        fail(
            ErrorCode::ReceiptRequired,
            format!("cannot read receipt: {err}"),
        )
    })?;
    let alias = flags
        .model
        .clone()
        .ok_or_else(|| usage("replay requires --model"))?;
    let prompt = flags
        .prompt
        .clone()
        .ok_or_else(|| usage("replay requires --prompt"))?;
    let home = home_dir(flags)?;
    let lock = flags
        .lock
        .clone()
        .unwrap_or_else(|| PathBuf::from("knolo.infer.lock.json"));
    let check = replay_pinned(&ReplayOptions {
        receipt_bytes: bytes,
        alias,
        prompt,
        lock_path: lock,
        home,
        weights_dir: flags.weights.clone(),
    })?;
    let bytes = check.to_bytes()?;
    let out = flags.out.clone().unwrap_or_else(|| {
        let mut path = receipt_path;
        path.set_extension("replay.cbor");
        path
    });
    write_atomic(&out, &bytes, ErrorCode::ReceiptPersistFailed)?;
    if flags.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "assurance": check.assurance,
                "matched": check.matched,
                "out": out,
                "replayRoot": check.root().expect("replay root").as_str(),
            }))
            .expect("replay json")
        );
    } else {
        println!("replay          {}", check.assurance);
        println!("matched         {}", check.matched);
        println!("wrote           {}", out.display());
    }
    Ok(())
}

fn open_for_verify(
    alias: &str,
    lock_path: &Path,
    weights: Option<&Path>,
) -> Result<infer_engine::VerifiedWeightSource, InferFailure> {
    let lock = read_lockfile(lock_path)?
        .ok_or_else(|| fail(ErrorCode::ModelArtifactMissing, "infer lockfile is missing"))?;
    let pin = lock
        .models
        .get(alias)
        .ok_or_else(|| fail(ErrorCode::ModelArtifactMissing, "alias is not pinned"))?;
    let cwd = std::env::current_dir().map_err(|err| {
        fail(
            ErrorCode::ModelArtifactMissing,
            format!("cannot resolve the working directory: {err}"),
        )
    })?;
    let kmodel = cwd.join(&pin.model_image_path);
    let weights = match weights {
        Some(dir) => dir.to_path_buf(),
        None => kmodel.parent().unwrap_or(Path::new(".")).to_path_buf(),
    };
    let source = load_verified_micro(&kmodel, &weights)?;
    if source.image_root.as_str() != pin.model_image_root {
        return Err(fail(
            ErrorCode::ModelDigestMismatch,
            "pinned model image root does not match the file",
        ));
    }
    Ok(source)
}

fn home_dir(flags: &Flags) -> Result<PathBuf, InferFailure> {
    if let Some(home) = &flags.home {
        Ok(home.clone())
    } else {
        default_home()
    }
}

fn reject_unused_run(flags: &Flags) -> Result<(), InferFailure> {
    if flags.out.is_some() || flags.build_root.is_some() {
        return Err(usage("run does not take --out or --build-root"));
    }
    Ok(())
}

fn reject_unused_verify(flags: &Flags) -> Result<(), InferFailure> {
    if flags.out.is_some()
        || flags.prompt.is_some()
        || flags.mode.is_some()
        || flags.receipt.is_some()
        || flags.build_root.is_some()
        || flags.max_tokens.is_some()
        || flags.temperature_micros.is_some()
        || flags.seed.is_some()
        || flags.stream.is_some()
    {
        return Err(usage("receipt verify received a run flag"));
    }
    Ok(())
}

fn reject_unused_replay(flags: &Flags) -> Result<(), InferFailure> {
    if flags.mode.is_some()
        || flags.receipt.is_some()
        || flags.build_root.is_some()
        || flags.max_tokens.is_some()
        || flags.temperature_micros.is_some()
        || flags.seed.is_some()
        || flags.stream.is_some()
    {
        return Err(usage("replay reads the sampler plan from the journal"));
    }
    Ok(())
}

fn usage(message: &str) -> InferFailure {
    fail(ErrorCode::ContractInvalid, message)
}
