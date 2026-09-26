//! One HTTP/1.1 request per connection.
//!
//! The body is bounded. Chunked transfer is refused. The caller decides the
//! route and writes one JSON response.

use std::io::{Read, Write};
use std::time::Duration;

use infer_contracts::{fail, ErrorCode, InferFailure};

pub const MAX_BODY: usize = 64 * 1024;
const MAX_HEAD: usize = 8 * 1024;
const READ_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

pub fn read_request(stream: &mut impl Read) -> Result<HttpRequest, InferFailure> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    let split = loop {
        if buf.len() > MAX_HEAD {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "HTTP headers exceed 8 KiB",
            ));
        }
        let n = stream.read(&mut chunk).map_err(io_http)?;
        if n == 0 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "HTTP request ended before the headers",
            ));
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(index) = find_separator(&buf) {
            break index;
        }
    };
    let head = &buf[..split];
    let mut rest = buf[split + 4..].to_vec();
    let text = std::str::from_utf8(head)
        .map_err(|_| fail(ErrorCode::ContractInvalid, "HTTP headers are not UTF-8"))?;
    let mut lines = text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| fail(ErrorCode::ContractInvalid, "HTTP request line is missing"))?;
    let mut parts = request_line.split(' ');
    let method = parts
        .next()
        .ok_or_else(|| fail(ErrorCode::ContractInvalid, "HTTP request line is invalid"))?
        .to_string();
    let path = parts
        .next()
        .ok_or_else(|| fail(ErrorCode::ContractInvalid, "HTTP request line is invalid"))?
        .to_string();
    let version = parts
        .next()
        .ok_or_else(|| fail(ErrorCode::ContractInvalid, "HTTP request line is invalid"))?;
    if parts.next().is_some() || (version != "HTTP/1.1" && version != "HTTP/1.0") {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "HTTP request line is invalid",
        ));
    }
    if path.contains('?') || path.contains('#') {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "HTTP query strings are not accepted",
        ));
    }
    let mut content_length = None;
    let mut content_type = None;
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| fail(ErrorCode::ContractInvalid, "HTTP header is missing a colon"))?;
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "HTTP header name is invalid",
            ));
        }
        let name = name.to_ascii_lowercase();
        let value = value.trim();
        if name == "transfer-encoding" {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "HTTP chunked bodies are not accepted",
            ));
        }
        if name == "content-length" {
            if content_length.is_some() {
                return Err(fail(
                    ErrorCode::ContractInvalid,
                    "HTTP content length is duplicated",
                ));
            }
            let parsed: u64 = value
                .parse()
                .map_err(|_| fail(ErrorCode::ContractInvalid, "HTTP content length is invalid"))?;
            content_length = Some(parsed);
        }
        if name == "content-type" {
            content_type = Some(value.to_ascii_lowercase());
        }
    }
    if let Some(kind) = content_type {
        if kind != "application/json" && !kind.starts_with("application/json;") {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "HTTP content type must be application/json",
            ));
        }
    }
    let length = match content_length {
        Some(length) => usize::try_from(length).unwrap_or(usize::MAX),
        None if method == "GET" => 0,
        None => {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "HTTP content length is required",
            ))
        }
    };
    if length > MAX_BODY {
        return Err(fail(ErrorCode::ContractInvalid, "HTTP body exceeds 64 KiB"));
    }
    if method == "GET" && length != 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "HTTP GET does not accept a body",
        ));
    }
    while rest.len() < length {
        let n = stream.read(&mut chunk).map_err(io_http)?;
        if n == 0 {
            return Err(fail(ErrorCode::ContractInvalid, "HTTP body ended early"));
        }
        rest.extend_from_slice(&chunk[..n]);
    }
    rest.truncate(length);
    Ok(HttpRequest {
        method,
        path,
        body: rest,
    })
}

pub fn write_response(
    stream: &mut impl Write,
    status: u16,
    body: &[u8],
) -> Result<(), InferFailure> {
    write_response_extra(stream, status, body, &[])
}

pub fn write_text(
    stream: &mut impl Write,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<(), InferFailure> {
    if content_type
        .bytes()
        .any(|byte| byte == b'\r' || byte == b'\n')
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "HTTP content type is invalid",
        ));
    }
    let reason = reason(status);
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).map_err(io_http)?;
    stream.write_all(body).map_err(io_http)?;
    stream.flush().map_err(io_http)?;
    Ok(())
}

pub fn write_response_extra(
    stream: &mut impl Write,
    status: u16,
    body: &[u8],
    extra: &[(&str, &str)],
) -> Result<(), InferFailure> {
    let reason = reason(status);
    let mut header = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (name, value) in extra {
        if !header_token(name) || value.bytes().any(|byte| byte == b'\r' || byte == b'\n') {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "HTTP extra header is invalid",
            ));
        }
        header.push_str(name);
        header.push_str(": ");
        header.push_str(value);
        header.push_str("\r\n");
    }
    header.push_str("\r\n");
    stream.write_all(header.as_bytes()).map_err(io_http)?;
    stream.write_all(body).map_err(io_http)?;
    stream.flush().map_err(io_http)?;
    Ok(())
}

fn header_token(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

pub fn write_sse_headers(
    stream: &mut std::net::TcpStream,
    request_id: &str,
    receipt: &str,
) -> Result<(), InferFailure> {
    if request_id
        .bytes()
        .any(|byte| byte == b'\r' || byte == b'\n')
        || receipt.bytes().any(|byte| byte == b'\r' || byte == b'\n')
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "event stream header is invalid",
        ));
    }
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .map_err(io_http)?;
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nX-Knolo-Receipt: {receipt}\r\nX-Knolo-Request-Id: {request_id}\r\n\r\n"
    );
    stream.write_all(header.as_bytes()).map_err(io_http)?;
    stream.flush().map_err(io_http)?;
    Ok(())
}

pub fn write_sse(
    stream: &mut impl Write,
    event: Option<&str>,
    data: &str,
) -> Result<(), InferFailure> {
    if event.is_some_and(|name| {
        name.is_empty() || name.bytes().any(|byte| byte == b'\n' || byte == b'\r')
    }) || data.bytes().any(|byte| byte == b'\n' || byte == b'\r')
    {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "event data contains a newline",
        ));
    }
    let frame = match event {
        Some(name) => format!("event: {name}\ndata: {data}\n\n"),
        None => format!("data: {data}\n\n"),
    };
    stream.write_all(frame.as_bytes()).map_err(io_http)?;
    stream.flush().map_err(io_http)?;
    Ok(())
}

pub fn prepare_stream(stream: &mut std::net::TcpStream) -> Result<(), InferFailure> {
    stream
        .set_read_timeout(Some(READ_TIMEOUT))
        .map_err(io_http)?;
    stream.set_nodelay(true).map_err(io_http)?;
    Ok(())
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Error",
    }
}

fn find_separator(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|window| window == b"\r\n\r\n")
}

fn io_http(err: std::io::Error) -> InferFailure {
    fail(
        ErrorCode::ContractInvalid,
        format!("HTTP connection failed: {err}"),
    )
}

pub fn status_for(code: ErrorCode) -> u16 {
    match code {
        ErrorCode::WorkerLost
        | ErrorCode::WorkerStartFailed
        | ErrorCode::ServiceDraining
        | ErrorCode::ServiceUnloaded => 503,
        ErrorCode::RequestTimeout => 504,
        ErrorCode::ReceiptRequired => 404,
        _ => {
            // A too-large body is still ContractInvalid. Callers that already
            // know the status pass it directly.
            400
        }
    }
}

pub fn status_for_failure(err: &InferFailure) -> u16 {
    if err.message.contains("exceeds 64 KiB") {
        413
    } else {
        status_for(err.code)
    }
}
