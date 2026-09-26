import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { canonicalCbor, decodeCanonicalCbor, decodeContract, hashDomain, InferFailure } from "../dist/index.js";

const vectorsPath = fileURLToPath(new URL("../../../conformance/contracts/vectors.json", import.meta.url));
const vectors = JSON.parse(readFileSync(vectorsPath, "utf8")).vectors;

function fromHex(hex) {
  return Uint8Array.from(Buffer.from(hex, "hex"));
}

function toHex(bytes) {
  return Buffer.from(bytes).toString("hex");
}

test("rust and typescript agree on canonical bytes, digests, and rejections", () => {
  assert.ok(vectors.length >= 17);
  for (const vector of vectors) {
    const bytes = fromHex(vector.hex);
    if (vector.ok) {
      const value = decodeCanonicalCbor(bytes);
      assert.equal(toHex(canonicalCbor(value)), vector.hex, vector.name);
      const payload = fromHex(vector.payloadHex);
      assert.equal(hashDomain(vector.domain, payload), vector.digest, vector.name);
      if (value instanceof Map && String(value.get("kind") ?? "").startsWith("knolo.infer.")) {
        const contract = decodeContract(bytes);
        assert.equal(contract.kind, value.get("kind"), vector.name);
      }
    } else {
      assert.throws(
        () => {
          if (vector.error === "CANONICAL_CBOR_INVALID") decodeCanonicalCbor(bytes);
          else decodeContract(bytes);
        },
        (error) => error instanceof InferFailure && error.code === vector.error,
        vector.name,
      );
    }
  }
});

test("canonical CBOR matches @knolo/core on the shared subset", async (t) => {
  const corePath = fileURLToPath(
    new URL("../../../../knolo-core/packages/core/dist/index.js", import.meta.url),
  );
  if (!existsSync(corePath)) {
    t.skip("knolo-core is not checked out beside this repository");
    return;
  }
  const core = await import(corePath);
  const cases = [
    ["null", null, null],
    ["true", true, true],
    ["false", false, false],
    ["zero", 0, 0n],
    ["one", 1, 1n],
    ["twenty-three", 23, 23n],
    ["twenty-four", 24, 24n],
    ["two-fifty-five", 255, 255n],
    ["two-fifty-six", 256, 256n],
    ["sixty-five-k", 65536, 65536n],
    ["two-to-32", 2 ** 32, 4294967296n],
    ["neg-one", -1, -1n],
    ["neg-24", -24, -24n],
    ["neg-25", -25, -25n],
    ["text", "knolo", "knolo"],
    ["bytes", new Uint8Array([0, 255]), { bytes: true }],
    ["array", [1, "a"], [1n, "a"]],
    ["map", { b: 1, a: 2 }, new Map([["b", 1n], ["a", 2n]])],
  ];
  for (const [name, coreValue, inferValue] of cases) {
    const coreBytes = core.canonicalCbor(name === "bytes" ? new Uint8Array([0, 255]) : coreValue);
    const ours =
      name === "bytes"
        ? canonicalCbor(new (await import("../dist/index.js")).CborBytes(new Uint8Array([0, 255])))
        : canonicalCbor(inferValue);
    assert.equal(toHex(ours), toHex(coreBytes), name);
    const coreDigest = core.digestDomain("object", coreBytes);
    const hash = createHash("sha256");
    hash.update(Buffer.from("knolo:object:v1\0", "utf8"));
    hash.update(coreBytes);
    assert.equal(`sha256-${hash.digest("hex")}`, coreDigest, name);
  }
});
