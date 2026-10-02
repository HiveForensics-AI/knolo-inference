use std::fmt;

/// Stable failure codes. Serving codes are reserved now so later phases
/// do not rename the contract surface. Milestone 1 emits the canonical
/// CBOR, contract, and digest codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    CanonicalCborInvalid,
    ContractInvalid,
    DigestInvalid,
    ModelImageInvalid,
    ModelImageSignatureInvalid,
    ModelArtifactMissing,
    ModelDigestMismatch,
    TokenizerInvalid,
    TemplateInvalid,
    UnsupportedArchitecture,
    UnsupportedQuantization,
    UnsupportedKernel,
    PlacementUnsatisfiable,
    InsufficientMemory,
    ContextLimitExceeded,
    PromptCompilationFailed,
    ReceiptRequired,
    ReceiptPersistFailed,
    WorkerStartFailed,
    WorkerLost,
    CudaOom,
    CudaFault,
    RequestCancelled,
    RequestTimeout,
    ServiceDraining,
    ServiceUnloaded,
    BackendNotAllowed,
    ReplayEnvironmentMismatch,
    ReplayOutputMismatch,
}

impl ErrorCode {
    pub const ALL: &'static [ErrorCode] = &[
        Self::CanonicalCborInvalid,
        Self::ContractInvalid,
        Self::DigestInvalid,
        Self::ModelImageInvalid,
        Self::ModelImageSignatureInvalid,
        Self::ModelArtifactMissing,
        Self::ModelDigestMismatch,
        Self::TokenizerInvalid,
        Self::TemplateInvalid,
        Self::UnsupportedArchitecture,
        Self::UnsupportedQuantization,
        Self::UnsupportedKernel,
        Self::PlacementUnsatisfiable,
        Self::InsufficientMemory,
        Self::ContextLimitExceeded,
        Self::PromptCompilationFailed,
        Self::ReceiptRequired,
        Self::ReceiptPersistFailed,
        Self::WorkerStartFailed,
        Self::WorkerLost,
        Self::CudaOom,
        Self::CudaFault,
        Self::RequestCancelled,
        Self::RequestTimeout,
        Self::ServiceDraining,
        Self::ServiceUnloaded,
        Self::BackendNotAllowed,
        Self::ReplayEnvironmentMismatch,
        Self::ReplayOutputMismatch,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::CanonicalCborInvalid => "CANONICAL_CBOR_INVALID",
            Self::ContractInvalid => "CONTRACT_INVALID",
            Self::DigestInvalid => "DIGEST_INVALID",
            Self::ModelImageInvalid => "MODEL_IMAGE_INVALID",
            Self::ModelImageSignatureInvalid => "MODEL_IMAGE_SIGNATURE_INVALID",
            Self::ModelArtifactMissing => "MODEL_ARTIFACT_MISSING",
            Self::ModelDigestMismatch => "MODEL_DIGEST_MISMATCH",
            Self::TokenizerInvalid => "TOKENIZER_INVALID",
            Self::TemplateInvalid => "TEMPLATE_INVALID",
            Self::UnsupportedArchitecture => "UNSUPPORTED_ARCHITECTURE",
            Self::UnsupportedQuantization => "UNSUPPORTED_QUANTIZATION",
            Self::UnsupportedKernel => "UNSUPPORTED_KERNEL",
            Self::PlacementUnsatisfiable => "PLACEMENT_UNSATISFIABLE",
            Self::InsufficientMemory => "INSUFFICIENT_MEMORY",
            Self::ContextLimitExceeded => "CONTEXT_LIMIT_EXCEEDED",
            Self::PromptCompilationFailed => "PROMPT_COMPILATION_FAILED",
            Self::ReceiptRequired => "RECEIPT_REQUIRED",
            Self::ReceiptPersistFailed => "RECEIPT_PERSIST_FAILED",
            Self::WorkerStartFailed => "WORKER_START_FAILED",
            Self::WorkerLost => "WORKER_LOST",
            Self::CudaOom => "CUDA_OOM",
            Self::CudaFault => "CUDA_FAULT",
            Self::RequestCancelled => "REQUEST_CANCELLED",
            Self::RequestTimeout => "REQUEST_TIMEOUT",
            Self::ServiceDraining => "SERVICE_DRAINING",
            Self::ServiceUnloaded => "SERVICE_UNLOADED",
            Self::BackendNotAllowed => "BACKEND_NOT_ALLOWED",
            Self::ReplayEnvironmentMismatch => "REPLAY_ENVIRONMENT_MISMATCH",
            Self::ReplayOutputMismatch => "REPLAY_OUTPUT_MISMATCH",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|code| code.as_str() == value)
    }

    pub fn retryable(self) -> bool {
        matches!(
            self,
            Self::InsufficientMemory
                | Self::ReceiptPersistFailed
                | Self::WorkerStartFailed
                | Self::WorkerLost
                | Self::CudaOom
                | Self::RequestTimeout
                | Self::ServiceDraining
        )
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferFailure {
    pub code: ErrorCode,
    pub message: String,
    pub retryable: bool,
    pub request_id: Option<String>,
    pub attempt: Option<u32>,
    pub partial_receipt_root: Option<String>,
}

impl InferFailure {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            retryable: code.retryable(),
            code,
            message: message.into(),
            request_id: None,
            attempt: None,
            partial_receipt_root: None,
        }
    }
}

impl fmt::Display for InferFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for InferFailure {}

pub fn fail(code: ErrorCode, message: impl Into<String>) -> InferFailure {
    InferFailure::new(code, message)
}
