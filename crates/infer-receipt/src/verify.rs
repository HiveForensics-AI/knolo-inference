//! A receipt verifies from its own bytes. The journal is checked when the
//! caller still has the home directory.

use std::path::Path;

use infer_contracts::{
    decode_contract, fail, CborValue, Contract, DigestHex, ErrorCode, InferFailure,
    InferenceReceiptV1,
};
use infer_engine::VerifiedWeightSource;

use infer_engine::{load_sampler_plan, verify_journal};

pub fn verify_receipt_bytes(bytes: &[u8]) -> Result<InferenceReceiptV1, InferFailure> {
    let receipt = match decode_contract(bytes)? {
        Contract::InferenceReceipt(receipt) => receipt,
        _ => {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "document is not an inference receipt",
            ))
        }
    };
    if receipt.to_bytes()? != bytes {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "receipt bytes are not canonical",
        ));
    }
    Ok(receipt)
}

pub fn verify_receipt_model(
    receipt: &InferenceReceiptV1,
    source: &VerifiedWeightSource,
) -> Result<(), InferFailure> {
    if receipt.model.model_image_root != source.image_root
        || receipt.model.artifact_root != source.artifact_root
        || receipt.model.model_runtime_root != source.runtime_root
    {
        return Err(fail(
            ErrorCode::ModelDigestMismatch,
            "receipt model roots do not match the image",
        ));
    }
    if receipt.model.tokenizer_root != source.image.tokenizer.root {
        return Err(fail(
            ErrorCode::TokenizerInvalid,
            "receipt tokenizer root does not match the image",
        ));
    }
    if receipt.model.template_root != source.image.template.root {
        return Err(fail(
            ErrorCode::TemplateInvalid,
            "receipt template root does not match the image",
        ));
    }
    if receipt.model.architecture_adapter_id != source.image.architecture.adapter {
        return Err(fail(
            ErrorCode::UnsupportedArchitecture,
            "receipt adapter does not match the image",
        ));
    }
    Ok(())
}

pub fn request_id_of(receipt: &InferenceReceiptV1) -> Result<String, InferFailure> {
    match receipt.extensions.get("knolo.request-id") {
        Some(CborValue::Text(id)) if !id.is_empty() => Ok(id.clone()),
        _ => Err(fail(
            ErrorCode::ReceiptRequired,
            "receipt has no request id",
        )),
    }
}

pub fn verify_receipt_journal(
    home: &Path,
    receipt: &InferenceReceiptV1,
) -> Result<(), InferFailure> {
    let request_id = request_id_of(receipt)?;
    let terminal = verify_journal(home, &request_id)?;
    if receipt.execution.event_trace_root != terminal {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "receipt event trace does not match the journal",
        ));
    }
    let plan = load_sampler_plan(home, &request_id)?;
    if plan.root()? != receipt.sampler.sampler_plan_root {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "journal sampler plan does not match the receipt",
        ));
    }
    let _ = DigestHex::parse(receipt.receipt_id.as_str())?;
    Ok(())
}
