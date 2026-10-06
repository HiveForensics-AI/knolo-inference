//! Phase 5 gate reports for one cold micro fixture.
//!
//! The chain links a Knowledge Image to an agent effect. The evidence check
//! binds that chain to the output. The install check names every pinned
//! artifact. The notice check records the engine inventory. None of them opens
//! a product repository. The layouts are specified in
//! `spec/KIP-INFER-0048-receipt-chain.md`,
//! `spec/KIP-INFER-0049-evidence-output.md`,
//! `spec/KIP-INFER-0050-hub-install.md`, and
//! `spec/KIP-INFER-0051-supply-notice.md`.

use std::collections::BTreeMap;

use crate::cbor::CborValue;
use crate::digest::{digest_value, DigestHex};
use crate::error::{fail, ErrorCode, InferFailure};
use crate::fields::{
    bounded_text, cbor_digest, cbor_text, cbor_u32, expect_kind_version, one_of, Fields,
};

use super::common::{put_extensions, Builder};
use super::lifecycle::cold_single;
use super::product::{LOCAL_WEIGHT_SOURCE, MICRO_CONTEXT};

pub const CHAIN_KIND: &str = "knolo.infer.chain-report";
pub const EVIDENCE_OUTPUT_KIND: &str = "knolo.infer.evidence-output-report";
pub const INSTALL_KIND: &str = "knolo.infer.install-report";
pub const NOTICE_KIND: &str = "knolo.infer.notice-report";

pub const CHAIN_LINK_COUNT: u32 = 5;
pub const INSTALL_ARTIFACT_COUNT: u32 = 4;

const ASSURANCES: &[&str] = &["compatibility", "same_build_replayable"];
const FINISHES: &[&str] = &["length", "stop"];
const FEATURE_SETS: &[&str] = &["cpu", "cuda"];

fn verified(value: &str) -> Result<(), InferFailure> {
    one_of("validationResult", value, &["verified"])
}

fn recorded(value: &str) -> Result<(), InferFailure> {
    one_of("validationResult", value, &["recorded"])
}

fn execution_mode(value: &str) -> Result<(), InferFailure> {
    one_of("executionMode", value, &["isolated-replay", "pinned"])
}

fn micro_context(noun: &str, prompt: u32, output: u32) -> Result<(), InferFailure> {
    if prompt > MICRO_CONTEXT
        || output > MICRO_CONTEXT
        || prompt.saturating_add(output) > MICRO_CONTEXT
    {
        return Err(fail(
            ErrorCode::ContextLimitExceeded,
            format!("the {noun} is the micro fixture"),
        ));
    }
    Ok(())
}

fn empty_extensions(
    extensions: &BTreeMap<String, CborValue>,
    message: &str,
) -> Result<(), InferFailure> {
    if extensions.is_empty() {
        Ok(())
    } else {
        Err(fail(ErrorCode::ContractInvalid, message))
    }
}

fn distinct_links(links: &[&DigestHex]) -> Result<(), InferFailure> {
    for (index, link) in links.iter().enumerate() {
        if links[..index].contains(link) {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the receipt chain repeats a link",
            ));
        }
    }
    Ok(())
}

/// Ordered demo from a Knowledge Image through the agent effect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainReportV1 {
    pub model_runtime_root: DigestHex,
    pub artifact_root: DigestHex,
    pub engine_build_root: DigestHex,
    pub kernel_bundle_root: DigestHex,
    pub placement_root: DigestHex,
    pub prompt_token_root: DigestHex,
    pub knowledge_image_root: DigestHex,
    pub knowledge_commit_root: DigestHex,
    pub query_receipt_root: DigestHex,
    pub query_receipt_count: u32,
    pub reflex_receipt_root: DigestHex,
    pub reflex_receipt_count: u32,
    pub receipt_root: DigestHex,
    pub effect_root: DigestHex,
    pub output_token_root: DigestHex,
    pub output_text_root: DigestHex,
    pub prompt_token_count: u32,
    pub output_token_count: u32,
    pub finish_reason: String,
    pub assurance: String,
    pub link_count: u32,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl ChainReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        verified(&self.validation_result)?;
        execution_mode(&self.execution_mode)?;
        cold_single(
            "chain",
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
        )?;
        empty_extensions(&self.extensions, "the receipt chain extensions are empty")?;
        one_of("assurance", &self.assurance, ASSURANCES)?;
        one_of("finishReason", &self.finish_reason, FINISHES)?;
        if self.prompt_token_count == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the receipt chain has no prompt",
            ));
        }
        micro_context(
            "receipt chain",
            self.prompt_token_count,
            self.output_token_count,
        )?;
        if self.finish_reason == "stop" && self.output_token_count == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a stop chain has no output tokens",
            ));
        }
        if self.knowledge_image_root == self.knowledge_commit_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the knowledge commit repeats the image",
            ));
        }
        if self.model_runtime_root == self.artifact_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the artifact repeats the model runtime",
            ));
        }
        if self.query_receipt_count == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the receipt chain has no query receipt",
            ));
        }
        if self.reflex_receipt_count == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the receipt chain has no reflex receipt",
            ));
        }
        if self.query_receipt_count > 256 || self.reflex_receipt_count > 256 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the receipt chain evidence list is too large",
            ));
        }
        if self.link_count != CHAIN_LINK_COUNT {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the receipt chain has five links",
            ));
        }
        distinct_links(&[
            &self.knowledge_image_root,
            &self.query_receipt_root,
            &self.reflex_receipt_root,
            &self.receipt_root,
            &self.effect_root,
        ])
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(CHAIN_KIND);
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("assurance", cbor_text(&self.assurance));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("effectRoot", cbor_digest(&self.effect_root));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("finishReason", cbor_text(&self.finish_reason));
        b.put("kernelBundleRoot", cbor_digest(&self.kernel_bundle_root));
        b.put(
            "knowledgeCommitRoot",
            cbor_digest(&self.knowledge_commit_root),
        );
        b.put(
            "knowledgeImageRoot",
            cbor_digest(&self.knowledge_image_root),
        );
        b.put("linkCount", cbor_u32(self.link_count));
        b.put("modelRuntimeRoot", cbor_digest(&self.model_runtime_root));
        b.put("outputTextRoot", cbor_digest(&self.output_text_root));
        b.put("outputTokenCount", cbor_u32(self.output_token_count));
        b.put("outputTokenRoot", cbor_digest(&self.output_token_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("promptTokenCount", cbor_u32(self.prompt_token_count));
        b.put("promptTokenRoot", cbor_digest(&self.prompt_token_root));
        b.put("queryReceiptCount", cbor_u32(self.query_receipt_count));
        b.put("queryReceiptRoot", cbor_digest(&self.query_receipt_root));
        b.put("receiptRoot", cbor_digest(&self.receipt_root));
        b.put("reflexReceiptCount", cbor_u32(self.reflex_receipt_count));
        b.put("reflexReceiptRoot", cbor_digest(&self.reflex_receipt_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, CHAIN_KIND)?;
        let out = Self {
            artifact_root: fields.digest("artifactRoot")?,
            assurance: fields.text("assurance")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            effect_root: fields.digest("effectRoot")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            finish_reason: fields.text("finishReason")?,
            kernel_bundle_root: fields.digest("kernelBundleRoot")?,
            knowledge_commit_root: fields.digest("knowledgeCommitRoot")?,
            knowledge_image_root: fields.digest("knowledgeImageRoot")?,
            link_count: fields.u32("linkCount")?,
            model_runtime_root: fields.digest("modelRuntimeRoot")?,
            output_text_root: fields.digest("outputTextRoot")?,
            output_token_count: fields.u32("outputTokenCount")?,
            output_token_root: fields.digest("outputTokenRoot")?,
            placement_root: fields.digest("placementRoot")?,
            prompt_token_count: fields.u32("promptTokenCount")?,
            prompt_token_root: fields.digest("promptTokenRoot")?,
            query_receipt_count: fields.u32("queryReceiptCount")?,
            query_receipt_root: fields.digest("queryReceiptRoot")?,
            receipt_root: fields.digest("receiptRoot")?,
            reflex_receipt_count: fields.u32("reflexReceiptCount")?,
            reflex_receipt_root: fields.digest("reflexReceiptRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-chain", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Output roots bound to the same evidence as the receipt chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceOutputReportV1 {
    pub evidence_root: DigestHex,
    pub knowledge_image_root: DigestHex,
    pub query_receipt_root: DigestHex,
    pub reflex_receipt_root: DigestHex,
    pub receipt_root: DigestHex,
    pub chain_root: DigestHex,
    pub output_token_root: DigestHex,
    pub output_text_root: DigestHex,
    pub prompt_token_count: u32,
    pub output_token_count: u32,
    pub finish_reason: String,
    pub assurance: String,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl EvidenceOutputReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        verified(&self.validation_result)?;
        execution_mode(&self.execution_mode)?;
        cold_single(
            "evidence",
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
        )?;
        empty_extensions(&self.extensions, "the evidence output extensions are empty")?;
        one_of("assurance", &self.assurance, ASSURANCES)?;
        one_of("finishReason", &self.finish_reason, FINISHES)?;
        if self.prompt_token_count == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the evidence output has no prompt",
            ));
        }
        micro_context(
            "evidence output",
            self.prompt_token_count,
            self.output_token_count,
        )?;
        if self.finish_reason == "stop" && self.output_token_count == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "a stop output has no output tokens",
            ));
        }
        if self.chain_root == self.receipt_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the chain repeats the receipt",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(EVIDENCE_OUTPUT_KIND);
        b.put("assurance", cbor_text(&self.assurance));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("chainRoot", cbor_digest(&self.chain_root));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("evidenceRoot", cbor_digest(&self.evidence_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("finishReason", cbor_text(&self.finish_reason));
        b.put(
            "knowledgeImageRoot",
            cbor_digest(&self.knowledge_image_root),
        );
        b.put("outputTextRoot", cbor_digest(&self.output_text_root));
        b.put("outputTokenCount", cbor_u32(self.output_token_count));
        b.put("outputTokenRoot", cbor_digest(&self.output_token_root));
        b.put("promptTokenCount", cbor_u32(self.prompt_token_count));
        b.put("queryReceiptRoot", cbor_digest(&self.query_receipt_root));
        b.put("receiptRoot", cbor_digest(&self.receipt_root));
        b.put("reflexReceiptRoot", cbor_digest(&self.reflex_receipt_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, EVIDENCE_OUTPUT_KIND)?;
        let out = Self {
            assurance: fields.text("assurance")?,
            cache_policy: fields.text("cachePolicy")?,
            chain_root: fields.digest("chainRoot")?,
            concurrency: fields.u32("concurrency")?,
            evidence_root: fields.digest("evidenceRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            finish_reason: fields.text("finishReason")?,
            knowledge_image_root: fields.digest("knowledgeImageRoot")?,
            output_text_root: fields.digest("outputTextRoot")?,
            output_token_count: fields.u32("outputTokenCount")?,
            output_token_root: fields.digest("outputTokenRoot")?,
            prompt_token_count: fields.u32("promptTokenCount")?,
            query_receipt_root: fields.digest("queryReceiptRoot")?,
            receipt_root: fields.digest("receiptRoot")?,
            reflex_receipt_root: fields.digest("reflexReceiptRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-evidence-output", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// Hub installation check for the four pinned micro-fixture artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallReportV1 {
    pub model_image_root: DigestHex,
    pub artifact_root: DigestHex,
    pub tokenizer_root: DigestHex,
    pub template_root: DigestHex,
    pub publisher: String,
    pub license_id: String,
    pub source_provider: String,
    pub artifact_count: u32,
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl InstallReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        verified(&self.validation_result)?;
        execution_mode(&self.execution_mode)?;
        cold_single(
            "install",
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
        )?;
        empty_extensions(&self.extensions, "the hub install extensions are empty")?;
        if self.source_provider != LOCAL_WEIGHT_SOURCE {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the hub install does not download weights",
            ));
        }
        if self.artifact_count != INSTALL_ARTIFACT_COUNT {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the hub install verifies four artifacts",
            ));
        }
        if self.model_image_root == self.artifact_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the weight artifact repeats the model image",
            ));
        }
        if self.tokenizer_root == self.template_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the template repeats the tokenizer",
            ));
        }
        bounded_text("publisher", &self.publisher, 128)?;
        bounded_text("licenseId", &self.license_id, 128)?;
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(INSTALL_KIND);
        b.put("artifactCount", cbor_u32(self.artifact_count));
        b.put("artifactRoot", cbor_digest(&self.artifact_root));
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("licenseId", cbor_text(&self.license_id));
        b.put("modelImageRoot", cbor_digest(&self.model_image_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("publisher", cbor_text(&self.publisher));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("sourceProvider", cbor_text(&self.source_provider));
        b.put("templateRoot", cbor_digest(&self.template_root));
        b.put("tokenizerRoot", cbor_digest(&self.tokenizer_root));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, INSTALL_KIND)?;
        let out = Self {
            artifact_count: fields.u32("artifactCount")?,
            artifact_root: fields.digest("artifactRoot")?,
            cache_policy: fields.text("cachePolicy")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            license_id: fields.text("licenseId")?,
            model_image_root: fields.digest("modelImageRoot")?,
            placement_root: fields.digest("placementRoot")?,
            publisher: fields.text("publisher")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            source_provider: fields.text("sourceProvider")?,
            template_root: fields.digest("templateRoot")?,
            tokenizer_root: fields.digest("tokenizerRoot")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-install", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}

/// NOTICE and SBOM identity for the engine build. The files are not written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoticeReportV1 {
    pub engine_build_root: DigestHex,
    pub placement_root: DigestHex,
    pub notice_root: DigestHex,
    pub sbom_root: DigestHex,
    pub component_count: u32,
    pub feature_set: String,
    pub candle_named: bool,
    pub execution_mode: String,
    pub cache_policy: String,
    pub concurrency: u32,
    pub run_count: u32,
    pub warm_state: String,
    pub request_count: u32,
    pub validation_result: String,
    pub extensions: BTreeMap<String, CborValue>,
}

impl NoticeReportV1 {
    pub fn validate(&self) -> Result<(), InferFailure> {
        recorded(&self.validation_result)?;
        execution_mode(&self.execution_mode)?;
        cold_single(
            "notice",
            &self.cache_policy,
            self.concurrency,
            self.run_count,
            &self.warm_state,
            self.request_count,
        )?;
        empty_extensions(&self.extensions, "the notice extensions are empty")?;
        one_of("featureSet", &self.feature_set, FEATURE_SETS)?;
        if self.notice_root == self.sbom_root {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the sbom repeats the notice",
            ));
        }
        if self.component_count == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the notice names no component",
            ));
        }
        if self.component_count > 256 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the notice inventory is too large",
            ));
        }
        if self.feature_set == "cpu" && self.candle_named {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the cpu notice names candle",
            ));
        }
        if self.feature_set == "cuda" && !self.candle_named {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "the cuda notice omits candle",
            ));
        }
        Ok(())
    }

    pub fn to_cbor(&self) -> Result<CborValue, InferFailure> {
        self.validate()?;
        let mut b = Builder::typed(NOTICE_KIND);
        b.put("cachePolicy", cbor_text(&self.cache_policy));
        b.put("candleNamed", CborValue::Bool(self.candle_named));
        b.put("componentCount", cbor_u32(self.component_count));
        b.put("concurrency", cbor_u32(self.concurrency));
        b.put("engineBuildRoot", cbor_digest(&self.engine_build_root));
        b.put("executionMode", cbor_text(&self.execution_mode));
        put_extensions(&mut b, &self.extensions)?;
        b.put("featureSet", cbor_text(&self.feature_set));
        b.put("noticeRoot", cbor_digest(&self.notice_root));
        b.put("placementRoot", cbor_digest(&self.placement_root));
        b.put("requestCount", cbor_u32(self.request_count));
        b.put("runCount", cbor_u32(self.run_count));
        b.put("sbomRoot", cbor_digest(&self.sbom_root));
        b.put("validationResult", cbor_text(&self.validation_result));
        b.put("warmState", cbor_text(&self.warm_state));
        Ok(b.finish())
    }

    pub fn from_cbor(value: &CborValue) -> Result<Self, InferFailure> {
        let mut fields = Fields::parse(value)?;
        expect_kind_version(&mut fields, NOTICE_KIND)?;
        let out = Self {
            cache_policy: fields.text("cachePolicy")?,
            candle_named: fields.bool("candleNamed")?,
            component_count: fields.u32("componentCount")?,
            concurrency: fields.u32("concurrency")?,
            engine_build_root: fields.digest("engineBuildRoot")?,
            execution_mode: fields.text("executionMode")?,
            extensions: fields.extensions()?,
            feature_set: fields.text("featureSet")?,
            notice_root: fields.digest("noticeRoot")?,
            placement_root: fields.digest("placementRoot")?,
            request_count: fields.u32("requestCount")?,
            run_count: fields.u32("runCount")?,
            sbom_root: fields.digest("sbomRoot")?,
            validation_result: fields.text("validationResult")?,
            warm_state: fields.text("warmState")?,
        };
        fields.finish()?;
        out.validate()?;
        Ok(out)
    }

    pub fn root(&self) -> Result<DigestHex, InferFailure> {
        digest_value("infer-notice", &self.to_cbor()?)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, InferFailure> {
        Ok(self.to_cbor()?.to_bytes())
    }
}
