use std::fs;
use std::path::PathBuf;

use infer_contracts::{ChatMessageV1, ErrorCode};
use infer_engine::{load_verified_micro, write_synthetic_model, VOCAB};
use infer_prompt::{compile_model_prompt, render_chat_template};
use serde_json::json;

#[test]
fn sandbox_rejects_a_template_that_leaves_the_process() {
    let err = render_chat_template("{% include \"weights.safetensors\" %}", &[]).unwrap_err();
    assert_eq!(err.code, ErrorCode::TemplateInvalid);
    let err = render_chat_template("{{ message.content | upper }}", &[]).unwrap_err();
    assert_eq!(err.code, ErrorCode::TemplateInvalid);
}

#[test]
fn micro_prompt_roots_are_stable_and_context_is_closed() {
    let dir = std::env::temp_dir().join(format!("knolo-prompt-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();

    let hi = compile_model_prompt(
        &source.image,
        vec![ChatMessageV1 {
            role: "user".into(),
            content: "hi".into(),
        }],
        VOCAB as u32,
        16,
        4,
    )
    .unwrap();
    let again = compile_model_prompt(
        &source.image,
        vec![ChatMessageV1 {
            role: "user".into(),
            content: "hi".into(),
        }],
        VOCAB as u32,
        16,
        4,
    )
    .unwrap();
    assert_eq!(
        hi.plan.token_id_root().unwrap(),
        again.plan.token_id_root().unwrap()
    );
    assert_eq!(hi.plan.token_ids[0], 1);
    assert!(hi.plan.rendered_text.starts_with("user: hi\n"));
    assert_eq!(hi.plan.truncation.strategy, "none");

    let too_long = compile_model_prompt(
        &source.image,
        vec![ChatMessageV1 {
            role: "user".into(),
            content: "a".repeat(20),
        }],
        VOCAB as u32,
        16,
        4,
    )
    .unwrap_err();
    assert_eq!(too_long.code, ErrorCode::ContextLimitExceeded);

    let outside = compile_model_prompt(
        &source.image,
        vec![ChatMessageV1 {
            role: "user".into(),
            content: "hello".into(),
        }],
        VOCAB as u32,
        64,
        4,
    )
    .unwrap_err();
    assert_eq!(outside.code, ErrorCode::TokenizerInvalid);
    assert!(!outside.message.contains("hello"));

    let pair = compile_model_prompt(
        &source.image,
        vec![
            ChatMessageV1 {
                role: "user".into(),
                content: "a".into(),
            },
            ChatMessageV1 {
                role: "assistant".into(),
                content: "hi".into(),
            },
        ],
        VOCAB as u32,
        64,
        4,
    )
    .unwrap();

    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = repo.join("conformance/prompt/expected.json");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let document = serde_json::to_string_pretty(&json!({
        "cases": [
            {
                "messages": [{"content": "hi", "role": "user"}],
                "maxContextTokens": 16,
                "name": "hi",
                "renderedText": hi.plan.rendered_text,
                "reservedGenerationTokens": 4,
                "tokenIdRoot": hi.plan.token_id_root().unwrap().as_str(),
                "tokenIds": hi.plan.token_ids,
            },
            {
                "messages": [
                    {"content": "a", "role": "user"},
                    {"content": "hi", "role": "assistant"}
                ],
                "maxContextTokens": 64,
                "name": "pair",
                "renderedText": pair.plan.rendered_text,
                "reservedGenerationTokens": 4,
                "tokenIdRoot": pair.plan.token_id_root().unwrap().as_str(),
                "tokenIds": pair.plan.token_ids,
            }
        ],
        "template": String::from_utf8(source.image.template.bytes.clone()).unwrap(),
        "tokenizerJson": String::from_utf8(source.image.tokenizer.bytes.clone()).unwrap(),
    }))
    .unwrap();
    let document = format!("{document}\n");
    fs::write(&path, &document).unwrap();
    assert!(document.contains(hi.plan.token_id_root().unwrap().as_str()));
}
