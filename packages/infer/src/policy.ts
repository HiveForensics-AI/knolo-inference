export interface ReceiptCheck {
  modelRuntimeRoot: string;
  knowledgeImageRoot?: string;
  assurance: string;
}

export interface ReceiptDecision {
  accepted: boolean;
  reason: string;
}

/// Accept a receipt when the model root, the knowledge image, and the
/// assurance match what the caller required. This helper does not import
/// Agents and does not open a Knowledge Image.
export function acceptReceipt(receipt: ReceiptCheck, required: ReceiptCheck): ReceiptDecision {
  if (receipt.modelRuntimeRoot !== required.modelRuntimeRoot) {
    return { accepted: false, reason: "model root" };
  }
  if (
    required.knowledgeImageRoot !== undefined &&
    receipt.knowledgeImageRoot !== required.knowledgeImageRoot
  ) {
    return { accepted: false, reason: "knowledge image" };
  }
  if (receipt.assurance !== required.assurance) {
    return { accepted: false, reason: "assurance" };
  }
  return { accepted: true, reason: "accepted" };
}
