//! `huggingface.tokenizers.v1` encodes a Llama 3 turn with the in-tree template.

use infer_contracts::ChatMessageV1;
use infer_prompt::{encode_text, parse_tokenizer, render_chat_template};

const TEMPLATE: &str = "<|begin_of_text|>{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% endfor %}<|start_header_id|>assistant<|end_header_id|>\n\n";

fn added(id: u32, content: &str, special: bool) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "content": content,
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": special
    })
}

fn fixture_json() -> String {
    let inner = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            added(1, "<|begin_of_text|>", true),
            added(2, "<|start_header_id|>", true),
            added(3, "<|end_header_id|>", true),
            added(4, "<|eot_id|>", true),
            added(7, "hello", false),
            added(8, "\n\n", false)
        ],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "unk_token": "<unk>",
            "vocab": {
                "<unk>": 0,
                "<|begin_of_text|>": 1,
                "<|start_header_id|>": 2,
                "<|end_header_id|>": 3,
                "<|eot_id|>": 4,
                "user": 5,
                "assistant": 6,
                "hello": 7,
                "\n\n": 8
            }
        }
    });
    serde_json::to_string(&serde_json::json!({
        "kind": "huggingface.tokenizers.v1",
        "version": 1,
        "tokenizer": inner
    }))
    .unwrap()
}

#[test]
fn huggingface_tokenizer_encodes_one_llama3_turn() {
    let parsed = parse_tokenizer(fixture_json().as_bytes(), 9).unwrap();
    let rendered = render_chat_template(
        TEMPLATE,
        &[ChatMessageV1 {
            role: "user".into(),
            content: "hello".into(),
        }],
    )
    .unwrap();
    let ids = encode_text(&parsed, &rendered).unwrap();
    assert_eq!(
        ids,
        vec![1, 2, 5, 3, 8, 7, 4, 2, 6, 3, 8],
        "{rendered:?} {ids:?}"
    );
}

#[test]
fn huggingface_wrapper_rejects_null_outside_the_tokenizer() {
    let err = parse_tokenizer(
        br#"{"kind":"huggingface.tokenizers.v1","version":null,"tokenizer":{}}"#,
        1,
    )
    .unwrap_err();
    assert_eq!(err.message, "JSON null is not allowed");
}

#[test]
fn list_tokenizer_still_rejects_null() {
    let err = parse_tokenizer(
        br#"{"kind":"knolo.llama.tokens.v1","version":1,"tokens":[null]}"#,
        1,
    )
    .unwrap_err();
    assert_eq!(err.message, "JSON null is not allowed");
}
