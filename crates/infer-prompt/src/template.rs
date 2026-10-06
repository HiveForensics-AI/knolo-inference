//! One-pass renderer. Substituted role and content bytes are not scanned again.

use infer_contracts::{fail, ChatMessageV1, ErrorCode, InferFailure};

pub const MAX_RENDERED_BYTES: usize = 1_048_576;
const MAX_MESSAGES: usize = 256;
const FOR_OPEN: &str = "{% for message in messages %}";
const FOR_CLOSE: &str = "{% endfor %}";
const ROLE: &str = "{{ message.role }}";
const CONTENT: &str = "{{ message.content }}";

pub fn render_chat_template(
    template: &str,
    messages: &[ChatMessageV1],
) -> Result<String, InferFailure> {
    if template.len() > MAX_RENDERED_BYTES {
        return Err(invalid("template exceeds the output cap"));
    }
    if messages.len() > MAX_MESSAGES {
        return Err(invalid("template loop exceeds 256 messages"));
    }
    if template.matches(FOR_OPEN).count() != 1 || template.matches(FOR_CLOSE).count() != 1 {
        return Err(invalid("template must contain one message loop"));
    }
    let Some((prefix, after_open)) = template.split_once(FOR_OPEN) else {
        return Err(invalid("template must contain one message loop"));
    };
    let Some((body, suffix)) = after_open.split_once(FOR_CLOSE) else {
        return Err(invalid("template loop is not closed"));
    };
    if has_tag(prefix) || has_tag(suffix) || body.contains("{%") {
        return Err(invalid("template tag is not in the allowlist"));
    }
    let mut rendered = String::new();
    push_capped(&mut rendered, prefix)?;
    for message in messages {
        push_capped(&mut rendered, &expand_body(body, message)?)?;
    }
    push_capped(&mut rendered, suffix)?;
    Ok(rendered)
}

fn expand_body(body: &str, message: &ChatMessageV1) -> Result<String, InferFailure> {
    let mut out = String::new();
    let mut rest = body;
    while let Some(index) = rest.find("{{") {
        out.push_str(&rest[..index]);
        if let Some(after) = rest[index..].strip_prefix(ROLE) {
            out.push_str(&message.role);
            rest = after;
        } else if let Some(after) = rest[index..].strip_prefix(CONTENT) {
            out.push_str(&message.content);
            rest = after;
        } else {
            return Err(invalid("template expression is not in the allowlist"));
        }
    }
    if rest.contains("{%") || rest.contains("{{") || rest.contains("}}") {
        return Err(invalid("template expression is not in the allowlist"));
    }
    out.push_str(rest);
    Ok(out)
}

fn has_tag(text: &str) -> bool {
    text.contains("{%") || text.contains("{{") || text.contains("{#")
}

fn push_capped(out: &mut String, piece: &str) -> Result<(), InferFailure> {
    if out.len().saturating_add(piece.len()) > MAX_RENDERED_BYTES {
        return Err(invalid("rendered prompt exceeds the output cap"));
    }
    out.push_str(piece);
    Ok(())
}

fn invalid(message: &str) -> InferFailure {
    fail(ErrorCode::TemplateInvalid, message)
}
