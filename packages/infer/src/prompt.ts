import { InferFailure, canonicalCbor } from "./cbor.js";
import { digestDomain } from "./digest.js";

const FOR_OPEN = "{% for message in messages %}";
const FOR_CLOSE = "{% endfor %}";
const ROLE = "{{ message.role }}";
const CONTENT = "{{ message.content }}";
const MAX_RENDERED_BYTES = 1_048_576;

export interface ChatMessage {
  role: string;
  content: string;
}

export interface CompiledPrompt {
  renderedText: string;
  tokenIds: number[];
  tokenIdRoot: string;
}

export function renderChatTemplate(template: string, messages: ChatMessage[]): string {
  if (template.length > MAX_RENDERED_BYTES) fail("TEMPLATE_INVALID", "template exceeds the output cap");
  if (messages.length > 256) fail("TEMPLATE_INVALID", "template loop exceeds 256 messages");
  if (count(template, FOR_OPEN) !== 1 || count(template, FOR_CLOSE) !== 1) {
    fail("TEMPLATE_INVALID", "template must contain one message loop");
  }
  const openAt = template.indexOf(FOR_OPEN);
  const prefix = template.slice(0, openAt);
  const afterOpen = template.slice(openAt + FOR_OPEN.length);
  const closeAt = afterOpen.indexOf(FOR_CLOSE);
  if (closeAt < 0) fail("TEMPLATE_INVALID", "template loop is not closed");
  const body = afterOpen.slice(0, closeAt);
  const suffix = afterOpen.slice(closeAt + FOR_CLOSE.length);
  if (hasTag(prefix) || hasTag(suffix) || body.includes("{%")) {
    fail("TEMPLATE_INVALID", "template tag is not in the allowlist");
  }
  let rendered = prefix;
  for (const message of messages) {
    rendered += expandBody(body, message);
    if (rendered.length > MAX_RENDERED_BYTES) fail("TEMPLATE_INVALID", "rendered prompt exceeds the output cap");
  }
  rendered += suffix;
  if (rendered.length > MAX_RENDERED_BYTES) fail("TEMPLATE_INVALID", "rendered prompt exceeds the output cap");
  return rendered;
}

export function parseTokenizer(tokenizerJson: string): string[] {
  const value = JSON.parse(tokenizerJson) as { kind?: string; version?: number; tokens?: unknown };
  if (value.kind !== "knolo.micro.tokens.v1" || value.version !== 1 || !Array.isArray(value.tokens)) {
    fail("TOKENIZER_INVALID", "tokenizer JSON is not knolo.micro.tokens.v1");
  }
  const tokens = value.tokens.map((token) => {
    if (typeof token !== "string" || token.length === 0 || token.length > 64) {
      fail("TOKENIZER_INVALID", "a tokenizer entry is empty or longer than 64 characters");
    }
    return token;
  });
  if (new Set(tokens).size !== tokens.length) fail("TOKENIZER_INVALID", "tokenizer entries must be unique");
  return tokens;
}

export function encodeText(tokens: string[], text: string): number[] {
  const order = tokens.map((_, index) => index).sort((left, right) => {
    const byLength = tokens[right].length - tokens[left].length;
    return byLength !== 0 ? byLength : left - right;
  });
  const ids: number[] = [];
  let rest = text;
  let offset = 0;
  while (rest.length > 0) {
    let matched: { index: number; length: number } | undefined;
    for (const index of order) {
      const token = tokens[index];
      if (rest.startsWith(token)) {
        matched = { index, length: token.length };
        break;
      }
    }
    if (!matched) fail("TOKENIZER_INVALID", `tokenizer has no match at byte ${offset}`);
    ids.push(matched.index);
    rest = rest.slice(matched.length);
    offset += matched.length;
  }
  return ids;
}

export function compilePromptText(
  template: string,
  tokenizerJson: string,
  messages: ChatMessage[],
  bosId = 1,
): CompiledPrompt {
  const renderedText = renderChatTemplate(template, messages);
  const tokens = parseTokenizer(tokenizerJson);
  const tokenIds = [bosId, ...encodeText(tokens, renderedText)];
  return { renderedText, tokenIds, tokenIdRoot: tokenIdRoot(tokenIds) };
}

export function tokenIdRoot(ids: number[]): string {
  return digestDomain(
    "infer-prompt-tokens",
    canonicalCbor(ids.map((id) => BigInt(id))),
  );
}

function expandBody(body: string, message: ChatMessage): string {
  let out = "";
  let rest = body;
  while (true) {
    const index = rest.indexOf("{{");
    if (index < 0) break;
    out += rest.slice(0, index);
    if (rest.startsWith(ROLE, index)) {
      out += message.role;
      rest = rest.slice(index + ROLE.length);
    } else if (rest.startsWith(CONTENT, index)) {
      out += message.content;
      rest = rest.slice(index + CONTENT.length);
    } else {
      fail("TEMPLATE_INVALID", "template expression is not in the allowlist");
    }
  }
  if (rest.includes("{%") || rest.includes("{{") || rest.includes("}}")) {
    fail("TEMPLATE_INVALID", "template expression is not in the allowlist");
  }
  return out + rest;
}

function hasTag(text: string): boolean {
  return text.includes("{%") || text.includes("{{") || text.includes("{#");
}

function count(text: string, needle: string): number {
  let found = 0;
  let from = 0;
  while (from <= text.length) {
    const index = text.indexOf(needle, from);
    if (index < 0) break;
    found += 1;
    from = index + needle.length;
  }
  return found;
}

function fail(code: string, message: string): never {
  throw new InferFailure(code, message);
}
