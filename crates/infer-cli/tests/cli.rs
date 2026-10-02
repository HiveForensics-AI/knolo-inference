use std::fs;
use std::process::Command;

use infer_artifact::{encode_safetensors, TensorBytes};
use infer_engine::{write_llama_model, write_synthetic_model};

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
    assert!(pull_err.contains("MODEL_ARTIFACT_MISSING"), "{pull_err}");

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

    let pulled = Command::new(bin())
        .current_dir(&dir)
        .args(["pull", "daily", "--json"])
        .output()
        .unwrap();
    assert!(
        pulled.status.success(),
        "{}",
        String::from_utf8_lossy(&pulled.stderr)
    );
    let pulled_json: serde_json::Value = serde_json::from_slice(&pulled.stdout).unwrap();
    assert_eq!(
        pulled_json["modelImageRoot"],
        verified_json["modelImageRoot"]
    );

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

#[test]
fn cli_plans_a_pinned_micro_model() {
    let dir = std::env::temp_dir().join(format!("knolo-infer-plan-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    write_synthetic_model(&dir).unwrap();
    let pin = Command::new(bin())
        .current_dir(&dir)
        .args(["pin", "daily", "micro.kmodel"])
        .output()
        .unwrap();
    assert!(
        pin.status.success(),
        "{}",
        String::from_utf8_lossy(&pin.stderr)
    );
    let planned = Command::new(bin())
        .current_dir(&dir)
        .args(["plan", "daily", "--intent", "interactive", "--json"])
        .output()
        .unwrap();
    assert!(
        planned.status.success(),
        "{}",
        String::from_utf8_lossy(&planned.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&planned.stdout).unwrap();
    assert_eq!(plan["intent"], "interactive");
    assert_eq!(plan["kvBlockSize"], 16);
    assert!(plan["placementRoot"]
        .as_str()
        .unwrap()
        .starts_with("sha256-"));
    let device = plan["device"].as_str().unwrap();
    assert!(device == "cpu" || device == "slot-0", "{device}");
    let refused = Command::new(bin())
        .current_dir(&dir)
        .args(["plan", "daily", "--intent", "throughput"])
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("BACKEND_NOT_ALLOWED"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_binds_evidence_and_checks_the_signature() {
    let dir = std::env::temp_dir().join(format!("knolo-infer-sign-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    write_synthetic_model(&dir).unwrap();
    let pin = Command::new(bin())
        .current_dir(&dir)
        .args(["pin", "micro", "micro.kmodel"])
        .output()
        .unwrap();
    assert!(
        pin.status.success(),
        "{}",
        String::from_utf8_lossy(&pin.stderr)
    );
    let seed = [7u8; 32];
    let public = infer_artifact::ed25519_public(&seed);
    fs::write(dir.join("seed.bin"), seed).unwrap();
    fs::write(dir.join("public.bin"), public).unwrap();
    let root = "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let home = dir.join("home");
    let ran = Command::new(bin())
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
            "--home",
            home.to_str().unwrap(),
            "--knowledge-image",
            root,
            "--query-receipt",
            root,
            "--reflex-receipt",
            root,
            "--sign-key",
            "seed.bin",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        ran.status.success(),
        "{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    let summary: serde_json::Value = serde_json::from_slice(&ran.stdout).unwrap();
    assert_eq!(summary["knowledgeImageRoot"], root);
    assert_eq!(summary["signed"], true);
    assert_eq!(summary["assurance"], "same_build_replayable");
    let verified = Command::new(bin())
        .current_dir(&dir)
        .args([
            "receipt",
            "verify",
            "receipt.cbor",
            "--public-key",
            "public.bin",
            "--knowledge-image",
            root,
            "--query-receipt",
            root,
            "--reflex-receipt",
            root,
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );
    let checked: serde_json::Value = serde_json::from_slice(&verified.stdout).unwrap();
    assert_eq!(checked["signatureVerified"], true);
    let mut bad = fs::read(dir.join("public.bin")).unwrap();
    bad[0] ^= 0xff;
    fs::write(dir.join("public.bin"), bad).unwrap();
    let rejected = Command::new(bin())
        .current_dir(&dir)
        .args([
            "receipt",
            "verify",
            "receipt.cbor",
            "--public-key",
            "public.bin",
        ])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("receipt signature"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cli_runs_the_llama_daily_release() {
    let dir = std::env::temp_dir().join(format!("knolo-infer-daily-{}", std::process::id()));
    let copy = std::env::temp_dir().join(format!("knolo-infer-daily-copy-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&copy);
    fs::create_dir_all(&dir).unwrap();
    write_llama_model(&dir).unwrap();

    let verified = Command::new(bin())
        .current_dir(&dir)
        .args([
            "model",
            "verify",
            "llama.kmodel",
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
    assert_eq!(verified_json["adapter"], "knolo.llama.v1");
    assert_eq!(verified_json["format"], "safetensors");
    assert_eq!(verified_json["weightsChecked"], true);

    let pin = Command::new(bin())
        .current_dir(&dir)
        .args(["pin", "daily", "llama.kmodel", "--weights", ".", "--json"])
        .output()
        .unwrap();
    assert!(
        pin.status.success(),
        "{}",
        String::from_utf8_lossy(&pin.stderr)
    );
    let pin_json: serde_json::Value = serde_json::from_slice(&pin.stdout).unwrap();
    assert_eq!(pin_json["alias"], "daily");
    assert_eq!(pin_json["modelImageRoot"], verified_json["modelImageRoot"]);
    assert_eq!(pin_json["artifactRoot"], verified_json["artifactRoot"]);

    let pulled = Command::new(bin())
        .current_dir(&dir)
        .args(["pull", "daily", "--json"])
        .output()
        .unwrap();
    assert!(
        pulled.status.success(),
        "{}",
        String::from_utf8_lossy(&pulled.stderr)
    );
    let pulled_json: serde_json::Value = serde_json::from_slice(&pulled.stdout).unwrap();
    assert_eq!(pulled_json["modelImageRoot"], pin_json["modelImageRoot"]);
    assert_eq!(pulled_json["artifactRoot"], pin_json["artifactRoot"]);

    let planned = Command::new(bin())
        .current_dir(&dir)
        .args(["plan", "daily", "--intent", "interactive", "--json"])
        .output()
        .unwrap();
    assert!(
        planned.status.success(),
        "{}",
        String::from_utf8_lossy(&planned.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&planned.stdout).unwrap();
    assert_eq!(plan["intent"], "interactive");
    assert_eq!(plan["kvBlockSize"], 16);
    assert!(plan["placementRoot"]
        .as_str()
        .unwrap()
        .starts_with("sha256-"));
    let device = plan["device"].as_str().unwrap();
    #[cfg(not(feature = "cuda"))]
    assert_eq!(device, "cpu");
    #[cfg(feature = "cuda")]
    assert_eq!(device, "slot-0");
    let refused = Command::new(bin())
        .current_dir(&dir)
        .args(["plan", "daily", "--intent", "throughput"])
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("BACKEND_NOT_ALLOWED"));

    let seed = [9u8; 32];
    let public = infer_artifact::ed25519_public(&seed);
    fs::write(dir.join("seed.bin"), seed).unwrap();
    fs::write(dir.join("public.bin"), public).unwrap();
    let root = "sha256-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let home = dir.join("home");
    let ran = Command::new(bin())
        .current_dir(&dir)
        .args([
            "run",
            "--model",
            "daily",
            "--prompt",
            "hi",
            "--mode",
            "pinned",
            "--receipt",
            "receipt.cbor",
            "--home",
            home.to_str().unwrap(),
            "--knowledge-image",
            root,
            "--query-receipt",
            root,
            "--reflex-receipt",
            root,
            "--sign-key",
            "seed.bin",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        ran.status.success(),
        "{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    let summary: serde_json::Value = serde_json::from_slice(&ran.stdout).unwrap();
    assert_eq!(summary["assurance"], "same_build_replayable");
    assert_eq!(summary["knowledgeImageRoot"], root);
    assert_eq!(summary["signed"], true);
    assert_eq!(summary["deviceSlot"], device);
    assert!(!summary["outputText"].as_str().unwrap().is_empty());
    let finish = summary["finishReason"].as_str().unwrap();
    assert!(finish == "stop" || finish == "length", "{finish}");
    assert!(!summary["outputTokens"].as_array().unwrap().is_empty());

    fs::create_dir_all(&copy).unwrap();
    for name in [
        "llama.kmodel",
        "weights.safetensors",
        "knolo.infer.lock.json",
        "receipt.cbor",
        "public.bin",
    ] {
        fs::copy(dir.join(name), copy.join(name)).unwrap();
    }
    let checked = Command::new(bin())
        .current_dir(&copy)
        .args([
            "receipt",
            "verify",
            "receipt.cbor",
            "--model",
            "daily",
            "--weights",
            ".",
            "--public-key",
            "public.bin",
            "--knowledge-image",
            root,
            "--query-receipt",
            root,
            "--reflex-receipt",
            root,
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stderr)
    );
    let checked_json: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(checked_json["verified"], true);
    assert_eq!(checked_json["signatureVerified"], true);
    assert_eq!(checked_json["assurance"], "same_build_replayable");
    assert_eq!(checked_json["artifactRoot"], pin_json["artifactRoot"]);
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&copy);
}
