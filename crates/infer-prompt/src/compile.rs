//! Messages, template, and tokenizer become one `PromptPlanV1`.
//! Evidence-bearing prompts fail when they do not fit. Nothing is dropped.

use std::collections::BTreeMap;

use infer_contracts::{
    fail, ChatMessageV1, DigestHex, ErrorCode, EvidenceBindingV1, InferFailure, ModelImageV1,
    PromptInputV1, PromptPlanV1, TruncationV1,
};

use crate::template::render_chat_template;
use crate::tokenizer::{encode_text, expect_special, parse_tokenizer};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledPrompt {
    pub plan: PromptPlanV1,
    pub input: PromptInputV1,
}

pub fn compile_model_prompt(
    image: &ModelImageV1,
    messages: Vec<ChatMessageV1>,
    vocab: u32,
    max_context_tokens: u32,
    reserved_generation_tokens: u32,
) -> Result<CompiledPrompt, InferFailure> {
    let input = PromptInputV1 {
        messages,
        tools_root: None,
        evidence: None,
        system_policy: None,
        extensions: BTreeMap::new(),
    };
    compile_with_evidence(
        image,
        input,
        vocab,
        max_context_tokens,
        reserved_generation_tokens,
    )
}

pub fn compile_with_evidence(
    image: &ModelImageV1,
    input: PromptInputV1,
    vocab: u32,
    max_context_tokens: u32,
    reserved_generation_tokens: u32,
) -> Result<CompiledPrompt, InferFailure> {
    input.validate()?;
    let tokenizer_root = hash_embedded("infer-tokenizer", &image.tokenizer.bytes)?;
    if tokenizer_root != image.tokenizer.root {
        return Err(fail(
            ErrorCode::TokenizerInvalid,
            "tokenizer root does not match the model image",
        ));
    }
    let template_root = hash_embedded("infer-template", &image.template.bytes)?;
    if template_root != image.template.root {
        return Err(fail(
            ErrorCode::TemplateInvalid,
            "template root does not match the model image",
        ));
    }
    let template = std::str::from_utf8(&image.template.bytes)
        .map_err(|_| fail(ErrorCode::TemplateInvalid, "template bytes are not UTF-8"))?;
    let rendered = render_chat_template(template, &input.messages)?;
    let tokenizer = parse_tokenizer(&image.tokenizer.bytes, vocab)?;
    expect_special(&tokenizer, image.special_tokens.bos, "<bos>")?;
    expect_special(&tokenizer, image.special_tokens.eos, "<eos>")?;
    expect_special(&tokenizer, image.special_tokens.pad, "<pad>")?;
    expect_special(&tokenizer, image.special_tokens.unk, "<unk>")?;
    for id in image.special_tokens.additional.values() {
        if *id >= vocab {
            return Err(fail(
                ErrorCode::TokenizerInvalid,
                "special token id is outside the vocabulary",
            ));
        }
    }
    let mut token_ids = Vec::new();
    if let Some(bos) = image.special_tokens.bos {
        token_ids.push(bos);
    }
    token_ids.extend(encode_text(&tokenizer, &rendered)?);
    if token_ids.is_empty() {
        return Err(fail(
            ErrorCode::PromptCompilationFailed,
            "compiled prompt has no tokens",
        ));
    }
    let original = u32::try_from(token_ids.len()).map_err(|_| {
        fail(
            ErrorCode::PromptCompilationFailed,
            "compiled prompt is too large",
        )
    })?;
    if original > max_context_tokens
        || u64::from(original) + u64::from(reserved_generation_tokens)
            > u64::from(max_context_tokens)
    {
        return Err(fail(
            ErrorCode::ContextLimitExceeded,
            "prompt plus reserved generation tokens exceed the context",
        ));
    }
    let mut preserved = Vec::new();
    if input
        .messages
        .iter()
        .any(|message| message.role == "system")
    {
        preserved.push("system".to_string());
    }
    if input.messages.iter().any(|message| message.role == "tool") {
        preserved.push("tool".to_string());
    }
    if input.evidence.is_some() {
        preserved.push("evidence".to_string());
    }
    preserved.sort();
    preserved.dedup();
    let truncation = TruncationV1 {
        strategy: "none".into(),
        original_token_count: original,
        final_token_count: original,
        removed_ids: Vec::new(),
        preserved_sections: preserved,
        max_context_tokens,
        reserved_generation_tokens,
    };
    let evidence_root = match &input.evidence {
        Some(evidence) => Some(evidence_root(evidence)?),
        None => None,
    };
    let plan = PromptPlanV1 {
        messages_root: input.messages_root()?,
        tools_root: input.tools_root.clone(),
        evidence_root,
        template_root,
        tokenizer_root,
        special_token_plan_root: image.special_tokens_root()?,
        rendered_text: rendered,
        token_ids,
        truncation,
        extensions: BTreeMap::new(),
    };
    plan.validate()?;
    Ok(CompiledPrompt { plan, input })
}

fn hash_embedded(domain: &str, bytes: &[u8]) -> Result<DigestHex, InferFailure> {
    infer_contracts::digest_value(domain, &infer_contracts::CborValue::Bytes(bytes.to_vec()))
}

fn evidence_root(evidence: &EvidenceBindingV1) -> Result<DigestHex, InferFailure> {
    evidence.root()
}
