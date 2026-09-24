use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    bounded_text, cbor_digest, cbor_text, cbor_u32, expect_kind_version, message_text, one_of,
    sorted_unique, text_array, u32_array, Fields,
};

use super::common::{put_extensions, string_field_array, Builder};

pub const PROMPT_INPUT_KIND: &str = "knolo.infer.prompt-input";
pub const PROMPT_PLAN_KIND: &str = "knolo.infer.prompt-plan";
pub const EVIDENCE_KIND: &str = "knolo.infer.evidence-binding";

/// `H(infer-prompt-tokens, token id array)`.
pub fn prompt_token_root(token_ids: &[u32]) -> Result<DigestHex, InferFailure> {
    digest_value(
        "infer-prompt-tokens",
        &CborValue::Array(token_ids.iter().copied().map(cbor_u32).collect()),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessageV1 {
    pub role: String,
    pub content: String,
}

impl ChatMessageV1 {
    fn validate(&self) -> Result<(), InferFailure> {
        one_of("role", &self.role, &["assistant", "system", "tool", "user"])?;
        message_text("content", &self.content, 1_048_576)?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put("content", cbor_text(&self.content));
        b.put("role", cbor_text(&self.role));
        Ok(b.finish())
    }

    fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            content: fields.text("content")?,
            role: fields.text("role")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceBindingV1 {
    pub knowledge_image_root: Option<DigestHex>,
    pub knowledge_commit_root: Option<DigestHex>,
    pub query_receipt_ids: Vec<DigestHex>,
    pub reflex_receipt_ids: Vec<DigestHex>,
    pub context_root: DigestHex,
    pub ordered_evidence_ids: Vec<String>,
    pub extensions: BTreeMap<String, CborValue>,
}

impl EvidenceBindingV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        if self.query_receipt_ids.len() > 256
            || self.reflex_receipt_ids.len() > 256
            || self.ordered_evidence_ids.len() > 4096
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "evidence list is too large",
            ));
        }
        for id in &self.ordered_evidence_ids {
            bounded_text("orderedEvidenceIds", id, 128)?;
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(EVIDENCE_KIND);
        b.put("contextRoot", cbor_digest(&self.context_root));
        put_extensions(&mut b, &self.extensions)?;
        b.put_opt(
            "knowledgeCommitRoot",
            self.knowledge_commit_root.as_ref().map(cbor_digest),
        );
        b.put_opt(
            "knowledgeImageRoot",
            self.knowledge_image_root.as_ref().map(cbor_digest),
        );
        b.put(
            "orderedEvidenceIds",
            string_field_array(&self.ordered_evidence_ids),
        );
        b.put("queryReceiptIds", digest_list(&self.query_receipt_ids));
        b.put("reflexReceiptIds", digest_list(&self.reflex_receipt_ids));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, EVIDENCE_KIND)?;
        let out = Self {
            context_root: fields.digest("contextRoot")?,
            extensions: fields.extensions()?,
            knowledge_commit_root: fields.opt_digest("knowledgeCommitRoot")?,
            knowledge_image_root: fields.opt_digest("knowledgeImageRoot")?,
            ordered_evidence_ids: text_array(
                &fields.array("orderedEvidenceIds")?,
                "orderedEvidenceIds",
            )?,
            query_receipt_ids: crate::fields::digest_array(
                &fields.array("queryReceiptIds")?,
                "queryReceiptIds",
            )?,
            reflex_receipt_ids: crate::fields::digest_array(
                &fields.array("reflexReceiptIds")?,
                "reflexReceiptIds",
            )?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-evidence", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptInputV1 {
    pub messages: Vec<ChatMessageV1>,
    pub tools_root: Option<DigestHex>,
    pub evidence: Option<EvidenceBindingV1>,
    pub system_policy: Option<String>,
    pub extensions: BTreeMap<String, CborValue>,
}

impl PromptInputV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        if self.messages.is_empty() || self.messages.len() > 256 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "messages are empty or too large",
            ));
        }
        for message in &self.messages {
            message.validate()?;
        }
        if let Some(policy) = &self.system_policy {
            message_text("systemPolicy", policy, 64 * 1024)?;
        }
        if let Some(evidence) = &self.evidence {
            evidence.validate()?;
        }
        Ok(())
    }

    pub fn messages_cbor(&self) -> Result<CborValue, InferFailure> {
        let mut items = Vec::with_capacity(self.messages.len());
        for message in &self.messages {
            items.push(message.to_cbor()?);
        }
        Ok(CborValue::Array(items))
    }

    pub fn messages_root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-prompt-input", &self.messages_cbor()?)
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(PROMPT_INPUT_KIND);
        b.put_opt(
            "evidence",
            self.evidence.as_ref().map(|e| e.to_cbor()).transpose()?,
        );
        put_extensions(&mut b, &self.extensions)?;
        b.put("messages", self.messages_cbor()?);
        b.put_opt("systemPolicy", self.system_policy.clone().map(cbor_text));
        b.put_opt("toolsRoot", self.tools_root.as_ref().map(cbor_digest));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, PROMPT_INPUT_KIND)?;
        let message_values = fields.array("messages")?;
        let mut messages = Vec::with_capacity(message_values.len());
        for item in &message_values {
            messages.push(ChatMessageV1::from_cbor(item)?);
        }
        let out = Self {
            evidence: match fields.optional("evidence")? {
                None => None,
                Some(value) => Some(EvidenceBindingV1::from_cbor(&value)?),
            },
            extensions: fields.extensions()?,
            messages,
            system_policy: fields.opt_text("systemPolicy")?,
            tools_root: fields.opt_digest("toolsRoot")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-prompt-input", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncationV1 {
    pub strategy: String,
    pub original_token_count: u32,
    pub final_token_count: u32,
    pub removed_ids: Vec<String>,
    pub preserved_sections: Vec<String>,
    pub max_context_tokens: u32,
    pub reserved_generation_tokens: u32,
}

impl TruncationV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        one_of("strategy", &self.strategy, &["drop-oldest-user", "none"])?;
        if self.max_context_tokens == 0 || self.max_context_tokens > 1_048_576 {
            return Err(fail(ErrorCode::ContractInvalid, "max context is invalid"));
        }
        if self.final_token_count > self.original_token_count
            || self.final_token_count > self.max_context_tokens
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "truncation counts are inconsistent",
            ));
        }
        let reserved =
            u64::from(self.final_token_count) + u64::from(self.reserved_generation_tokens);
        if reserved > u64::from(self.max_context_tokens) {
            return Err(fail(
                ErrorCode::ContextLimitExceeded,
                "reserved generation tokens exceed context",
            ));
        }
        sorted_unique("preservedSections", &self.preserved_sections)?;
        for section in &self.preserved_sections {
            one_of(
                "preservedSections",
                section,
                &["evidence", "system", "tool"],
            )?;
        }
        if self.removed_ids.len() > 4096 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "removed ids are too large",
            ));
        }
        for id in &self.removed_ids {
            bounded_text("removedIds", id, 128)?;
        }
        if self.strategy == "none"
            && (self.original_token_count != self.final_token_count || !self.removed_ids.is_empty())
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "strategy none cannot drop tokens",
            ));
        }
        if self.strategy == "drop-oldest-user"
            && self.original_token_count == self.final_token_count
            && self.removed_ids.is_empty()
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "drop strategy did not record a removal",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::bare();
        b.put("finalTokenCount", cbor_u32(self.final_token_count));
        b.put("maxContextTokens", cbor_u32(self.max_context_tokens));
        b.put("originalTokenCount", cbor_u32(self.original_token_count));
        b.put(
            "preservedSections",
            string_field_array(&self.preserved_sections),
        );
        b.put("removedIds", string_field_array(&self.removed_ids));
        b.put(
            "reservedGenerationTokens",
            cbor_u32(self.reserved_generation_tokens),
        );
        b.put("strategy", cbor_text(&self.strategy));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        let out = Self {
            final_token_count: fields.u32("finalTokenCount")?,
            max_context_tokens: fields.u32("maxContextTokens")?,
            original_token_count: fields.u32("originalTokenCount")?,
            preserved_sections: text_array(
                &fields.array("preservedSections")?,
                "preservedSections",
            )?,
            removed_ids: text_array(&fields.array("removedIds")?, "removedIds")?,
            reserved_generation_tokens: fields.u32("reservedGenerationTokens")?,
            strategy: fields.text("strategy")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-truncation", &self.to_cbor()?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptPlanV1 {
    pub messages_root: DigestHex,
    pub tools_root: Option<DigestHex>,
    pub evidence_root: Option<DigestHex>,
    pub template_root: DigestHex,
    pub tokenizer_root: DigestHex,
    pub special_token_plan_root: DigestHex,
    pub rendered_text: String,
    pub token_ids: Vec<u32>,
    pub truncation: TruncationV1,
    pub extensions: BTreeMap<String, CborValue>,
}

impl PromptPlanV1 {
    pub fn rendered_text_root(&self) -> Result<DigestHex, InferFailure> {
        digest_value(
            "infer-rendered-text",
            &CborValue::Bytes(self.rendered_text.as_bytes().to_vec()),
        )
    }

    pub fn token_id_root(&self) -> Result<DigestHex, InferFailure> {
        prompt_token_root(&self.token_ids)
    }

    pub fn validate(&self) -> Result<(), InferFailure> {
        message_text("renderedText", &self.rendered_text, 1_048_576)?;
        self.truncation.validate()?;
        if self.token_ids.len() != self.truncation.final_token_count as usize {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "token id count does not match truncation",
            ));
        }
        if self.token_ids.len() > 1_048_576 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "token id list is too large",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(PROMPT_PLAN_KIND);
        b.put_opt("evidenceRoot", self.evidence_root.as_ref().map(cbor_digest));
        put_extensions(&mut b, &self.extensions)?;
        b.put("messagesRoot", cbor_digest(&self.messages_root));
        b.put("renderedText", cbor_text(&self.rendered_text));
        b.put(
            "specialTokenPlanRoot",
            cbor_digest(&self.special_token_plan_root),
        );
        b.put("templateRoot", cbor_digest(&self.template_root));
        b.put(
            "tokenIds",
            CborValue::Array(self.token_ids.iter().copied().map(cbor_u32).collect()),
        );
        b.put("tokenizerRoot", cbor_digest(&self.tokenizer_root));
        b.put_opt("toolsRoot", self.tools_root.as_ref().map(cbor_digest));
        b.put("truncation", self.truncation.to_cbor()?);
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, PROMPT_PLAN_KIND)?;
        let out = Self {
            evidence_root: fields.opt_digest("evidenceRoot")?,
            extensions: fields.extensions()?,
            messages_root: fields.digest("messagesRoot")?,
            rendered_text: fields.text("renderedText")?,
            special_token_plan_root: fields.digest("specialTokenPlanRoot")?,
            template_root: fields.digest("templateRoot")?,
            token_ids: u32_array(&fields.array("tokenIds")?, "tokenIds")?,
            tokenizer_root: fields.digest("tokenizerRoot")?,
            tools_root: fields.opt_digest("toolsRoot")?,
            truncation: TruncationV1::from_cbor(&fields.require("truncation")?)?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-prompt-plan", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

fn digest_list(items: &[DigestHex]) -> CborValue {
    CborValue::Array(items.iter().map(cbor_digest).collect())
}
