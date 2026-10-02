import assert from "node:assert/strict";
import { test } from "node:test";

import { acceptReceipt } from "../dist/index.js";

const model = "sha256-" + "ab".repeat(32);
const knowledge = "sha256-" + "cd".repeat(32);

test("acceptReceipt accepts a matching model, knowledge image, and assurance", () => {
  const decision = acceptReceipt(
    { modelRuntimeRoot: model, knowledgeImageRoot: knowledge, assurance: "same_build_replayable" },
    { modelRuntimeRoot: model, knowledgeImageRoot: knowledge, assurance: "same_build_replayable" },
  );
  assert.deepEqual(decision, { accepted: true, reason: "accepted" });
});

test("acceptReceipt refuses a model root, a knowledge image, or an assurance mismatch", () => {
  assert.equal(
    acceptReceipt(
      { modelRuntimeRoot: "other", knowledgeImageRoot: knowledge, assurance: "same_build_replayable" },
      { modelRuntimeRoot: model, knowledgeImageRoot: knowledge, assurance: "same_build_replayable" },
    ).reason,
    "model root",
  );
  assert.equal(
    acceptReceipt(
      { modelRuntimeRoot: model, assurance: "same_build_replayable" },
      { modelRuntimeRoot: model, knowledgeImageRoot: knowledge, assurance: "same_build_replayable" },
    ).reason,
    "knowledge image",
  );
  assert.equal(
    acceptReceipt(
      { modelRuntimeRoot: model, knowledgeImageRoot: knowledge, assurance: "compatibility" },
      { modelRuntimeRoot: model, knowledgeImageRoot: knowledge, assurance: "same_build_replayable" },
    ).reason,
    "assurance",
  );
});

test("acceptReceipt ignores knowledge when the caller does not require it", () => {
  const decision = acceptReceipt(
    { modelRuntimeRoot: model, assurance: "same_build_replayable" },
    { modelRuntimeRoot: model, assurance: "same_build_replayable" },
  );
  assert.equal(decision.accepted, true);
});
