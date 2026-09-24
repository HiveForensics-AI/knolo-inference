use std::fs;
use std::process::Command;

use infer_artifact::{encode_safetensors, TensorBytes};
use infer_engine::write_synthetic_model;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_knolo-infer")
}

#[test]
fn help_lists_serve_and_serve_requires_a_model() {
    let help = Command::new(bin()).arg("--help").output().unwrap();
    assert!(help.status.success());
    let text = String::from_utf8_lossy(&help.stdout);
    assert!(text.contains("knolo-infer serve"));
    let serve = Command::new(bin()).arg("serve").output().unwrap();
    assert!(!serve.status.success());
    let err = String::from_utf8_lossy(&serve.stderr);
    assert!(err.contains("serve requires --model"), "{err}");
}

fn fixture() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-cli-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let mut payload = Vec::new();
    payload.extend_from_slice(&1.0f32.to_le_bytes());
    payload.extend_from_slice(&2.0f32.to_le_bytes());
    fs::write(
        dir.join("weights.safetensors"),
        encode_safetensors(&[TensorBytes {
            name: "w".into(),
            dtype: "f32".into(),
            shape: vec![2],
            bytes: payload,
        }])
        .unwrap(),
    )
    .unwrap();
    fs::write(dir.join("tokenizer.json"), b"{\"tokens\":[\"a\"]}").unwrap();
    fs::write(dir.join("template.jinja"), b"{{ messages }}").unwrap();
    fs::write(
        dir.join("manifest.json"),
        r#"{
          "kind": "knolo.infer.model-image",
          "version": 1,
          "name": "knolo/micro",
          "variant": "f32",
          "architecture": {"family": "micro", "adapter": "knolo.micro.v1"},
          "weights": {"format": "safetensors", "files": [{"path": "weights.safetensors", "sizeBytes": 0}]},
          "tokenizer": {"embedded": "tokenizer.json"},
          "template": {"embedded": "template.jinja"},
          "specialTokens": {"bos": 1, "eos": 2, "additional": {}},
          "generationDefaults": {
            "temperatureMicros": 0, "topPMillionths": 1000000, "minPMillionths": 0,
            "repetitionPenaltyMicros": 1000000, "presencePenaltyMicros": 0,
            "frequencyPenaltyMicros": 0, "topK": 1, "maxOutputTokens": 8
          },
          "capabilities": ["text-generation"],
          "license": {"id": "synthetic", "acceptanceRequired": false},
          "sources": [],
          "tensorInventory": [{"name": "w", "shape": [2], "dtype": "f32"}],
          "precisions": ["f32"],
          "requirements": {"minimumRamBytes": 1048576, "minimumVramBytes": 0}
        }"#,
    )
    .unwrap();
    dir
}

#[test]
fn cli_builds_verifies_pins_and_refuses_pull() {
    let dir = fixture();
    let built = Command::new(bin())
        .current_dir(&dir)
        .args([
            "model",
            "build",
            "manifest.json",
            "--out",
            "micro.kmodel",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let inspect = Command::new(bin())
        .current_dir(&dir)
        .args(["model", "inspect", "micro.kmodel", "--json"])
        .output()
        .unwrap();
    assert!(
        inspect.status.success(),
        "{}",
        String::from_utf8_lossy(&inspect.stderr)
    );
    let inspect_json: serde_json::Value = serde_json::from_slice(&inspect.stdout).unwrap();
    assert_eq!(inspect_json["weightsChecked"], false);
    assert!(inspect_json["modelImageRoot"]
        .as_str()
        .unwrap()
        .starts_with("sha256-"));

    let verified = Command::new(bin())
        .current_dir(&dir)
        .args([
            "model",
            "verify",
            "micro.kmodel",
            "--weights",
            ".",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );
    let verified_json: serde_json::Value = serde_json::from_slice(&verified.stdout).unwrap();
    assert_eq!(verified_json["weightsChecked"], true);

    let pull = Command::new(bin())
        .current_dir(&dir)
        .args(["pull", "daily"])
        .output()
        .unwrap();
    assert!(!pull.status.success());
    let pull_err = String::from_utf8_lossy(&pull.stderr);
    assert!(pull_err.contains("unsupported"), "{pull_err}");

    let pin = Command::new(bin())
        .current_dir(&dir)
        .args([
            "pin",
            "daily",
            "micro.kmodel",
            "--weights",
            ".",
            "--build-root",
            inspect_json["modelImageRoot"].as_str().unwrap(),
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        pin.status.success(),
        "{}",
        String::from_utf8_lossy(&pin.stderr)
    );
    assert!(dir.join("knolo.infer.lock.json").is_file());

    let mut weights = fs::read(dir.join("weights.safetensors")).unwrap();
    let last = weights.len() - 1;
    weights[last] ^= 0xff;
    fs::write(dir.join("weights.safetensors"), weights).unwrap();
    let mismatch = Command::new(bin())
        .current_dir(&dir)
        .args(["model", "verify", "micro.kmodel", "--weights", "."])
        .output()
        .unwrap();
    assert!(!mismatch.status.success());
    assert!(String::from_utf8_lossy(&mismatch.stderr).contains("MODEL_DIGEST_MISMATCH"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_runs_verifies_and_replays_on_cpu() {
    let dir = std::env::temp_dir().join(format!("knolo-infer-run-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    write_synthetic_model(&dir).unwrap();
    let pin = Command::new(bin())
        .current_dir(&dir)
        .args([
            "pin",
            "micro",
            "micro.kmodel",
            "--lock",
            "knolo.infer.lock.json",
        ])
        .output()
        .unwrap();
    assert!(
        pin.status.success(),
        "{}",
        String::from_utf8_lossy(&pin.stderr)
    );
    let home = dir.join("home");
    let run = Command::new(bin())
        .current_dir(&dir)
        .args([
            "run",
            "--model",
            "micro",
            "--prompt",
            "hi",
            "--mode",
            "pinned",
            "--receipt",
            "receipt.cbor",
            "--lock",
            "knolo.infer.lock.json",
            "--home",
            home.to_str().unwrap(),
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}{}",
        String::from_utf8_lossy(&run.stderr),
        String::from_utf8_lossy(&run.stdout)
    );
    let summary: serde_json::Value = serde_json::from_slice(&run.stdout).unwrap();
    assert_eq!(summary["assurance"], "same_build_replayable");
    #[cfg(not(feature = "cuda"))]
    assert_eq!(summary["deviceSlot"], "cpu");
    #[cfg(feature = "cuda")]
    assert_eq!(summary["deviceSlot"], "slot-0");
    assert!(summary["receiptId"]
        .as_str()
        .unwrap()
        .starts_with("sha256-"));
    assert!(!summary["outputText"].as_str().unwrap().is_empty());
    assert!(dir.join("receipt.cbor").is_file());

    let verified = Command::new(bin())
        .current_dir(&dir)
        .args([
            "receipt",
            "verify",
            "receipt.cbor",
            "--model",
            "micro",
            "--lock",
            "knolo.infer.lock.json",
            "--weights",
            ".",
            "--home",
            home.to_str().unwrap(),
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );

    let replay = Command::new(bin())
        .current_dir(&dir)
        .args([
            "replay",
            "receipt.cbor",
            "--model",
            "micro",
            "--prompt",
            "hi",
            "--lock",
            "knolo.infer.lock.json",
            "--home",
            home.to_str().unwrap(),
            "--out",
            "replay.cbor",
        ])
        .output()
        .unwrap();
    assert!(
        replay.status.success(),
        "{}",
        String::from_utf8_lossy(&replay.stderr)
    );
    assert!(String::from_utf8_lossy(&replay.stdout).contains("exact_replay_verified"));

    let wrong = Command::new(bin())
        .current_dir(&dir)
        .args([
            "replay",
            "receipt.cbor",
            "--model",
            "micro",
            "--prompt",
            "a",
            "--lock",
            "knolo.infer.lock.json",
            "--home",
            home.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!wrong.status.success());
    assert!(String::from_utf8_lossy(&wrong.stderr).contains("REPLAY_ENVIRONMENT_MISMATCH"));

    let mut weights = fs::read(dir.join("weights.safetensors")).unwrap();
    weights[8] ^= 0xff;
    fs::write(dir.join("weights.safetensors"), &weights).unwrap();
    let flipped = Command::new(bin())
        .current_dir(&dir)
        .args([
            "receipt",
            "verify",
            "receipt.cbor",
            "--model",
            "micro",
            "--lock",
            "knolo.infer.lock.json",
            "--weights",
            ".",
        ])
        .output()
        .unwrap();
    assert!(!flipped.status.success());
    assert!(String::from_utf8_lossy(&flipped.stderr).contains("MODEL_DIGEST_MISMATCH"));

    let throughput = Command::new(bin())
        .current_dir(&dir)
        .args([
            "run",
            "--model",
            "micro",
            "--prompt",
            "hi",
            "--mode",
            "throughput",
            "--receipt",
            "other.cbor",
            "--lock",
            "knolo.infer.lock.json",
            "--home",
            home.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!throughput.status.success());
    assert!(String::from_utf8_lossy(&throughput.stderr).contains("BACKEND_NOT_ALLOWED"));
    let _ = fs::remove_dir_all(&dir);
}
