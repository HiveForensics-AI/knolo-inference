import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { InferFailure, compilePromptText, renderChatTemplate } from "../dist/index.js";

const expectedPath = fileURLToPath(new URL("../../../conformance/prompt/expected.json", import.meta.url));
const expected = JSON.parse(readFileSync(expectedPath, "utf8"));

test("typescript prompt tokens match the rust token-id root", () => {
  assert.ok(expected.cases.length >= 1);
  for (const item of expected.cases) {
    const compiled = compilePromptText(expected.template, expected.tokenizerJson, item.messages);
    assert.equal(compiled.renderedText, item.renderedText, item.name);
    assert.deepEqual(compiled.tokenIds, item.tokenIds, item.name);
    assert.equal(compiled.tokenIdRoot, item.tokenIdRoot, item.name);
  }
});

test("a template that reaches outside the sandbox fails", () => {
  assert.throws(
    () => renderChatTemplate('{% include "weights.safetensors" %}', []),
    (error) => error instanceof InferFailure && error.code === "TEMPLATE_INVALID",
  );
});
