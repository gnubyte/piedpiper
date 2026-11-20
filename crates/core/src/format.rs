//! Container format (see `docs/FORMAT.md`).
//!
//! ```text
//! magic "PIED" | version u8 | mode u8 | orig_len varint | crc32(orig) u32le
//! mode 0 (stored): raw bytes follow
//! mode 1 (neural): tokenizer | model config | train config | n_tokens varint | payload
//! mode 2 (media):  H.264 recompression stream (crates/media)
//! mode 3 (jpeg):   Lepton JPEG recompression stream (third_party/lepton_jpeg)
//! mode 5 (audio):  PCM audio header | meta stream | residual stream (audio.rs)
//! ```

use crate::config::{ModelConfig, TrainConfig};
use crate::error::{Error, Result};
use crate::filter::Filter;
use crate::tokenizer::Tokenizer;

pub const MAGIC: [u8; 4] = *b"PIED";
pub const VERSION: u8 = 1;
pub const MODE_STORED: u8 = 0;
pub const MODE_NEURAL: u8 = 1;
pub const MODE_MEDIA: u8 = 2;
/// Deflate streams (PNG/ZIP/gzip/PDF) recompressed: varint meta length, meta stream,
/// plain stream (both complete PiedPiper streams).
pub const MODE_DEFLATE: u8 = 4;
pub const MODE_JPEG: u8 = 3;
/// 16-bit PCM WAV audio: sample layout, then the WAV header/trailer bytes and the
/// prediction residuals as two complete PiedPiper streams (see `audio.rs`).
pub const MODE_AUDIO: u8 = 5;

pub fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

pub fn get_varint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut v = 0u64;
    let mut shift = 0u32;
    loop {
        let b = *buf.get(*pos).ok_or_else(|| Error::Format("truncated header".into()))?;
        *pos += 1;
        if shift > 63 || (shift == 63 && b > 1) {
            return Err(Error::Format("varint overflow".into()));
        }
        v |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
        shift += 7;
    }
}

fn get_u32(buf: &[u8], pos: &mut usize) -> Result<u32> {
    let b = buf.get(*pos..*pos + 4).ok_or_else(|| Error::Format("truncated header".into()))?;
    *pos += 4;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn get_u64(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let lo = get_u32(buf, pos)? as u64;
    let hi = get_u32(buf, pos)? as u64;
    Ok(lo | (hi << 32))
}

fn get_f32(buf: &[u8], pos: &mut usize) -> Result<f32> {
    Ok(f32::from_bits(get_u32(buf, pos)?))
}

fn get_small(buf: &[u8], pos: &mut usize) -> Result<u32> {
    let v = get_varint(buf, pos)?;
    u32::try_from(v).map_err(|_| Error::Format("header field out of range".into()))
}

pub struct Header {
    pub mode: u8,
    pub orig_len: u64,
    pub crc: u32,
    pub tokenizer: Tokenizer,
    pub model: ModelConfig,
    pub train: TrainConfig,
    pub n_tokens: u64,
    /// Id of the prior the model was initialised from (0 = none).
    pub prior_id: u64,
    /// Minimum context length of the match model (0 = off).
    pub match_min: u8,
    /// Context length of the second, long-context match model (0 = off).
    pub match_long: u8,
    /// Reversible transform applied to the input before tokenizing.
    pub filter: Filter,
    /// Offset of the payload (raw bytes or range-coder stream).
    pub payload: usize,
}

pub fn write_prefix(out: &mut Vec<u8>, mode: u8, orig_len: u64, crc: u32) {
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(mode);
    put_varint(out, orig_len);
    out.extend_from_slice(&crc.to_le_bytes());
}

pub fn write_neural(
    out: &mut Vec<u8>,
    tok: &Tokenizer,
    mc: &ModelConfig,
    tc: &TrainConfig,
    n_tokens: u64,
    prior_id: u64,
    match_min: u8,
    match_long: u8,
    filter: &Filter,
) {
    tok.write(out);
    for v in [mc.d_model, mc.n_layer, mc.n_head, mc.d_ff, mc.window, mc.seg_len, mc.n_streams] {
        put_varint(out, v as u64);
    }
    for f in [tc.lr, tc.lr_min] {
        out.extend_from_slice(&f.to_bits().to_le_bytes());
    }
    put_varint(out, tc.warmup as u64);
    put_varint(out, tc.decay as u64);
    for f in [tc.beta1, tc.beta2, tc.eps, tc.clip, tc.init_scale] {
        out.extend_from_slice(&f.to_bits().to_le_bytes());
    }
    out.extend_from_slice(&tc.seed.to_le_bytes());
    put_varint(out, n_tokens);
    out.extend_from_slice(&prior_id.to_le_bytes());
    out.push(match_min);
    out.push(match_long);
    filter.write(out);
}

pub fn read_header(buf: &[u8]) -> Result<Header> {
    if buf.len() < 4 || buf[..4] != MAGIC {
        return Err(Error::Format("not a PiedPiper file (bad magic)".into()));
    }
    let mut pos = 4;
    let version = *buf.get(pos).ok_or_else(|| Error::Format("truncated header".into()))?;
    pos += 1;
    if version != VERSION {
        return Err(Error::Version(version));
    }
    let mode = *buf.get(pos).ok_or_else(|| Error::Format("truncated header".into()))?;
    pos += 1;
    let orig_len = get_varint(buf, &mut pos)?;
    let crc = get_u32(buf, &mut pos)?;
    let dummy_model = ModelConfig { d_model: 0, n_layer: 0, n_head: 0, d_ff: 0, window: 0, seg_len: 0, n_streams: 0 };
    let dummy_train = TrainConfig {
        lr: 0.0,
        lr_min: 0.0,
        warmup: 1,
        decay: 0,
        beta1: 0.0,
        beta2: 0.0,
        eps: 1.0,
        clip: 0.0,
        init_scale: 0.0,
        seed: 0,
    };
    match mode {
        MODE_STORED | MODE_MEDIA | MODE_DEFLATE | MODE_JPEG | MODE_AUDIO => Ok(Header {
            mode,
            orig_len,
            crc,
            tokenizer: Tokenizer::Bytes,
            model: dummy_model,
            train: dummy_train,
            n_tokens: orig_len,
            prior_id: 0,
            match_min: 0,
            match_long: 0,
            filter: Filter::None,
            payload: pos,
        }),
        MODE_NEURAL => {
            let tokenizer = Tokenizer::read(buf, &mut pos)?;
            let model = ModelConfig {
                d_model: get_small(buf, &mut pos)?,
                n_layer: get_small(buf, &mut pos)?,
                n_head: get_small(buf, &mut pos)?,
                d_ff: get_small(buf, &mut pos)?,
                window: get_small(buf, &mut pos)?,
                seg_len: get_small(buf, &mut pos)?,
                n_streams: get_small(buf, &mut pos)?,
            };
            let lr = get_f32(buf, &mut pos)?;
            let lr_min = get_f32(buf, &mut pos)?;
            let warmup = get_small(buf, &mut pos)?;
            let decay = get_small(buf, &mut pos)?;
            let beta1 = get_f32(buf, &mut pos)?;
            let beta2 = get_f32(buf, &mut pos)?;
            let eps = get_f32(buf, &mut pos)?;
            let clip = get_f32(buf, &mut pos)?;
            let init_scale = get_f32(buf, &mut pos)?;
            let seed = get_u64(buf, &mut pos)?;
            let train = TrainConfig { lr, lr_min, warmup, decay, beta1, beta2, eps, clip, init_scale, seed };
            let n_tokens = get_varint(buf, &mut pos)?;
            let prior_id = get_u64(buf, &mut pos)?;
            let match_min = *buf.get(pos).ok_or_else(|| Error::Format("truncated header".into()))?;
            pos += 1;
            let match_long = *buf.get(pos).ok_or_else(|| Error::Format("truncated header".into()))?;
            pos += 1;
            if match_min > 64 || match_long > 64 {
                return Err(Error::Format("invalid match model setting".into()));
            }
            let filter = Filter::read(buf, &mut pos)?;
            model
                .validate(tokenizer.vocab_size())
                .map_err(|e| match e {
                    Error::Options(m) => Error::Format(m),
                    other => other,
                })?;
            train.validate().map_err(|e| Error::Format(e.to_string()))?;
            if n_tokens > orig_len || (orig_len > 0 && n_tokens == 0) {
                return Err(Error::Format("inconsistent token count".into()));
            }
            if model.n_streams as u64 > n_tokens.max(1) {
                return Err(Error::Format("more streams than tokens".into()));
            }
            Ok(Header { mode, orig_len, crc, tokenizer, model, train, n_tokens, prior_id, match_min, match_long, filter, payload: pos })
        }
        m => Err(Error::Format(format!("unknown mode {m}"))),
    }
}

static CRC_TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

/// CRC-32 (IEEE 802.3, as used by zip/gzip/PNG).
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = CRC_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_known_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn varint_roundtrip() {
        for v in [0u64, 1, 127, 128, 300, 1 << 35, u64::MAX] {
            let mut b = Vec::new();
            put_varint(&mut b, v);
            let mut p = 0;
            assert_eq!(get_varint(&b, &mut p).unwrap(), v);
            assert_eq!(p, b.len());
        }
    }
}
