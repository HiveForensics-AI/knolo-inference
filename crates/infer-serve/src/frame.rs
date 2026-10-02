//! Length-prefixed canonical CBOR frames.
//!
//! The length is a big-endian u32. Zero and anything above 1 MiB are refused
//! before the body is decoded.

use std::io::{Read, Write};

use infer_contracts::{decode_canonical, fail, CborValue, ErrorCode, InferFailure};

pub const MAX_FRAME: usize = 1024 * 1024;

#[derive(Debug)]
pub struct FrameDecoder {
    buf: Vec<u8>,
    pos: usize,
}

impl FrameDecoder {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            pos: 0,
        }
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<(), InferFailure> {
        let next = self
            .buf
            .len()
            .saturating_sub(self.pos)
            .saturating_add(bytes.len());
        if next > MAX_FRAME + 4 {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "ipc frame buffer exceeded the size limit",
            ));
        }
        if self.pos > 4096 {
            self.buf.drain(..self.pos);
            self.pos = 0;
        }
        self.buf.extend_from_slice(bytes);
        Ok(())
    }

    pub fn has_unread(&self) -> bool {
        self.buf.len() > self.pos
    }

    pub fn pop(&mut self) -> Result<Option<CborValue>, InferFailure> {
        let available = self.buf.len() - self.pos;
        if available < 4 {
            return Ok(None);
        }
        let len_bytes: [u8; 4] = self.buf[self.pos..self.pos + 4]
            .try_into()
            .expect("four length bytes");
        let len = u32::from_be_bytes(len_bytes) as usize;
        if len == 0 || len > MAX_FRAME {
            return Err(fail(
                ErrorCode::ContractInvalid,
                "ipc frame length is outside 1..=1048576",
            ));
        }
        if available < 4 + len {
            return Ok(None);
        }
        let start = self.pos + 4;
        let body = self.buf[start..start + len].to_vec();
        self.pos = start + len;
        decode_canonical(&body).map(Some)
    }
}

impl Default for FrameDecoder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn write_frame(mut writer: impl Write, value: &CborValue) -> Result<(), InferFailure> {
    let body = value.to_bytes();
    if body.is_empty() || body.len() > MAX_FRAME {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "ipc frame length is outside 1..=1048576",
        ));
    }
    let len = u32::try_from(body.len()).expect("frame length fits u32");
    writer.write_all(&len.to_be_bytes()).map_err(io_frame)?;
    writer.write_all(&body).map_err(io_frame)?;
    writer.flush().map_err(io_frame)?;
    Ok(())
}

pub fn read_frame(reader: &mut impl Read) -> Result<CborValue, InferFailure> {
    let mut decoder = FrameDecoder::new();
    let mut buf = [0u8; 8192];
    loop {
        if let Some(value) = decoder.pop()? {
            return Ok(value);
        }
        let n = reader.read(&mut buf).map_err(io_frame)?;
        if n == 0 {
            return Err(fail(ErrorCode::WorkerLost, "worker closed the ipc socket"));
        }
        decoder.push(&buf[..n])?;
    }
}

fn io_frame(err: std::io::Error) -> InferFailure {
    fail(
        ErrorCode::WorkerLost,
        format!("ipc frame could not be transferred: {err}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_and_coalesced_frames_round_trip() {
        let first = CborValue::Text("ready".into());
        let second = CborValue::Integer(7);
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &first).unwrap();
        write_frame(&mut bytes, &second).unwrap();
        let mut decoder = FrameDecoder::new();
        decoder.push(&bytes[..3]).unwrap();
        assert!(decoder.pop().unwrap().is_none());
        decoder.push(&bytes[3..5]).unwrap();
        assert!(decoder.pop().unwrap().is_none());
        decoder.push(&bytes[5..]).unwrap();
        assert_eq!(decoder.pop().unwrap(), Some(first));
        assert_eq!(decoder.pop().unwrap(), Some(second));
        assert!(decoder.pop().unwrap().is_none());
    }

    #[test]
    fn a_zero_length_is_refused() {
        let mut decoder = FrameDecoder::new();
        decoder.push(&0u32.to_be_bytes()).unwrap();
        let err = decoder.pop().unwrap_err();
        assert_eq!(err.code, ErrorCode::ContractInvalid);
    }
}
