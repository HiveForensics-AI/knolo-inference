export const MAX_DOCUMENT_BYTES = 32 * 1024 * 1024;
export const MAX_DEPTH = 32;
export const MAX_ITEMS = 1_048_576;
const MIN_INT = -(1n << 64n);
const MAX_INT = (1n << 64n) - 1n;

export class CborBytes {
  constructor(readonly data: Uint8Array) {}
}

export type CborValue =
  | null
  | boolean
  | bigint
  | string
  | CborBytes
  | CborValue[]
  | Map<string, CborValue>;

export class InferFailure extends Error {
  constructor(
    readonly code: string,
    message: string,
  ) {
    super(`${code}: ${message}`);
    this.name = "InferFailure";
  }
}

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder("utf-8", { fatal: true });

export function canonicalCbor(value: CborValue): Uint8Array {
  const out: number[] = [];
  encode(value, out);
  return Uint8Array.from(out);
}

export function decodeCanonicalCbor(bytes: Uint8Array): CborValue {
  if (bytes.length === 0) throw new InferFailure("CANONICAL_CBOR_INVALID", "empty CBOR input");
  if (bytes.length > MAX_DOCUMENT_BYTES) {
    throw new InferFailure("CANONICAL_CBOR_INVALID", "CBOR input exceeds the size limit");
  }
  const decoder = new Decoder(bytes);
  const value = decoder.read(0);
  if (decoder.offset !== bytes.length) {
    throw new InferFailure("CANONICAL_CBOR_INVALID", "trailing bytes after CBOR value");
  }
  const recoded = canonicalCbor(value);
  if (!bytesEqual(recoded, bytes)) {
    throw new InferFailure("CANONICAL_CBOR_INVALID", "CBOR value is not canonical");
  }
  return value;
}

function encode(value: CborValue, out: number[]): void {
  if (value === null) {
    out.push(0xf6);
    return;
  }
  if (typeof value === "boolean") {
    out.push(value ? 0xf5 : 0xf4);
    return;
  }
  if (typeof value === "bigint") {
    if (value < MIN_INT || value > MAX_INT) throw new InferFailure("CANONICAL_CBOR_INVALID", "integer out of range");
    if (value >= 0n) encodeLength(0, value, out);
    else encodeLength(1, -1n - value, out);
    return;
  }
  if (typeof value === "string") {
    const bytes = textEncoder.encode(value);
    encodeLength(3, BigInt(bytes.length), out);
    out.push(...bytes);
    return;
  }
  if (value instanceof CborBytes) {
    encodeLength(2, BigInt(value.data.length), out);
    out.push(...value.data);
    return;
  }
  if (Array.isArray(value)) {
    encodeLength(4, BigInt(value.length), out);
    for (const item of value) encode(item, out);
    return;
  }
  const entries = [...value.entries()].sort((left, right) => compareUtf8(left[0], right[0]));
  encodeLength(5, BigInt(entries.length), out);
  for (const [key, item] of entries) {
    encode(key, out);
    encode(item, out);
  }
}

function encodeLength(major: number, length: bigint, out: number[]): void {
  const head = major << 5;
  if (length < 24n) out.push(head | Number(length));
  else if (length <= 0xffn) out.push(head | 24, Number(length));
  else if (length <= 0xffffn) {
    out.push(head | 25, Number(length >> 8n), Number(length & 0xffn));
  } else if (length <= 0xffffffffn) {
    out.push(
      head | 26,
      Number((length >> 24n) & 0xffn),
      Number((length >> 16n) & 0xffn),
      Number((length >> 8n) & 0xffn),
      Number(length & 0xffn),
    );
  } else {
    out.push(head | 27);
    for (let shift = 56n; shift >= 0n; shift -= 8n) out.push(Number((length >> shift) & 0xffn));
  }
}

class Decoder {
  offset = 0;
  constructor(private readonly bytes: Uint8Array) {}

  read(depth: number): CborValue {
    if (depth > MAX_DEPTH) throw new InferFailure("CANONICAL_CBOR_INVALID", "CBOR nesting exceeds the depth limit");
    const initial = this.readByte();
    const major = initial >> 5;
    const ai = initial & 31;
    if (major === 0) return this.readArgument(ai);
    if (major === 1) return -1n - this.readArgument(ai);
    if (major === 2) return new CborBytes(this.readBytes(this.readArgument(ai)));
    if (major === 3) {
      try {
        return textDecoder.decode(this.readBytes(this.readArgument(ai)));
      } catch {
        throw new InferFailure("CANONICAL_CBOR_INVALID", "CBOR text is not UTF-8");
      }
    }
    if (major === 4) {
      const length = this.collectionLength(ai);
      const items: CborValue[] = [];
      for (let i = 0; i < length; i++) items.push(this.read(depth + 1));
      return items;
    }
    if (major === 5) {
      const length = this.collectionLength(ai);
      const map = new Map<string, CborValue>();
      let previous: string | undefined;
      for (let i = 0; i < length; i++) {
        const key = this.read(depth + 1);
        if (typeof key !== "string") throw new InferFailure("CANONICAL_CBOR_INVALID", "CBOR map key must be text");
        if (previous !== undefined && compareUtf8(key, previous) <= 0) {
          throw new InferFailure("CANONICAL_CBOR_INVALID", "CBOR map keys must be strictly increasing");
        }
        previous = key;
        map.set(key, this.read(depth + 1));
      }
      return map;
    }
    if (major === 7 && ai === 20) return false;
    if (major === 7 && ai === 21) return true;
    if (major === 7 && ai === 22) return null;
    if (ai === 31) throw new InferFailure("CANONICAL_CBOR_INVALID", "indefinite-length CBOR is rejected");
    throw new InferFailure("CANONICAL_CBOR_INVALID", "CBOR floats, tags, and extra simple values are rejected");
  }

  private collectionLength(ai: number): number {
    const length = this.readArgument(ai);
    if (length > BigInt(MAX_ITEMS) || length > BigInt(this.bytes.length - this.offset)) {
      throw new InferFailure("CANONICAL_CBOR_INVALID", "CBOR collection exceeds its limit");
    }
    return Number(length);
  }

  private readArgument(ai: number): bigint {
    let value: bigint;
    let minimal = 0n;
    if (ai < 24) value = BigInt(ai);
    else if (ai === 24) {
      value = BigInt(this.readByte());
      minimal = 24n;
    } else if (ai === 25) {
      value = this.readBe(2);
      minimal = 0x100n;
    } else if (ai === 26) {
      value = this.readBe(4);
      minimal = 0x10000n;
    } else if (ai === 27) {
      value = this.readBe(8);
      minimal = 0x100000000n;
    } else if (ai === 31) {
      throw new InferFailure("CANONICAL_CBOR_INVALID", "indefinite-length CBOR is rejected");
    } else {
      throw new InferFailure("CANONICAL_CBOR_INVALID", "reserved CBOR additional information");
    }
    if (ai >= 24 && value < minimal) {
      throw new InferFailure("CANONICAL_CBOR_INVALID", "CBOR integer is not shortest form");
    }
    if (value > MAX_INT) throw new InferFailure("CANONICAL_CBOR_INVALID", "CBOR integer exceeds the supported range");
    return value;
  }

  private readBe(width: number): bigint {
    const slice = this.readBytes(BigInt(width));
    let value = 0n;
    for (const byte of slice) value = (value << 8n) | BigInt(byte);
    return value;
  }

  private readBytes(length: bigint): Uint8Array {
    const n = Number(length);
    const end = this.offset + n;
    if (end > this.bytes.length) throw new InferFailure("CANONICAL_CBOR_INVALID", "truncated CBOR value");
    const slice = this.bytes.slice(this.offset, end);
    this.offset = end;
    return slice;
  }

  private readByte(): number {
    if (this.offset >= this.bytes.length) throw new InferFailure("CANONICAL_CBOR_INVALID", "truncated CBOR value");
    return this.bytes[this.offset++];
  }
}

function compareUtf8(left: string, right: string): number {
  const a = textEncoder.encode(left);
  const b = textEncoder.encode(right);
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return a[i] - b[i];
  return a.length - b.length;
}

function bytesEqual(left: Uint8Array, right: Uint8Array): boolean {
  if (left.length !== right.length) return false;
  for (let i = 0; i < left.length; i++) if (left[i] !== right[i]) return false;
  return true;
}
