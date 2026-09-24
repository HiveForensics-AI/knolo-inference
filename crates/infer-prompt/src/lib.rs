//! Prompt compiler for milestone 1.
//!
//! The template grammar is a closed subset: one message loop, two field
//! lookups, and literal text. It has no loader, filters, environment, or
//! recursion. Rust and TypeScript implement that same grammar so the token-id
//! root does not depend on a template engine's whitespace.

mod compile;
mod template;
mod tokenizer;

pub use compile::{compile_model_prompt, CompiledPrompt};
pub use template::{render_chat_template, MAX_RENDERED_BYTES};
pub use tokenizer::{decode_tokens, encode_text, parse_tokenizer, MicroTokenizer};
