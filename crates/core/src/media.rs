//! Format mode 2: lossless recompression of H.264 and HEVC video (MP4/MOV, Matroska), VP9
//! and VP8 video (WebM/Matroska) and MPEG-2 video (Matroska, e.g. DVD rips).
//!
//! The neural model cannot shrink already-compressed video: the entropy coders of H.264 and
//! HEVC (CABAC) and VP8/VP9 (a boolean arithmetic coder) leave the bytes close to random. The
//! `piedpiper-media` crate instead parses each slice or tile down to its binary decisions,
//! codes them with a stronger context-mixing model, and rebuilds the original bitstream bit
//! for bit on decompression (see docs/DESIGN.md). The payload's own magic (`PPM2` for H.264,
//! `PPH3` for HEVC, `PPV9` for VP9, `PPV8` for VP8, `PPMV` for MPEG-2) says which recoder
//! wrote it.

use crate::codec::ProgressFn;
use crate::error::{Error, Result};

/// Name reported by [`crate::inspect`] for mode-2 streams holding H.264.
pub const NAME: &str = "h264-recode";
/// Name reported by [`crate::inspect`] for mode-2 streams holding HEVC.
pub const NAME_HEVC: &str = "hevc-recode";
/// Name reported by [`crate::inspect`] for mode-2 streams holding VP9.
pub const NAME_VP9: &str = "vp9-recode";
/// Name reported by [`crate::inspect`] for mode-2 streams holding VP8.
pub const NAME_VP8: &str = "vp8-recode";
/// Name reported by [`crate::inspect`] for mode-2 streams holding MPEG-2 video.
pub const NAME_MPEG2: &str = "mpeg2-recode";
/// Name reported by [`crate::inspect`] for mode-2 streams holding an H.264 transport stream.
pub const NAME_TS: &str = "h264-ts-recode";
/// Name reported by [`crate::inspect`] for mode-2 streams holding an HEVC transport stream.
pub const NAME_TS_HEVC: &str = "hevc-ts-recode";

/// The [`crate::inspect`] name of a mode-2 payload.
pub fn name(payload: &[u8]) -> &'static str {
    if payload.starts_with(b"PPV9") {
        NAME_VP9
    } else if payload.starts_with(b"PPV8") {
        NAME_VP8
    } else if payload.starts_with(b"PPMV") {
        NAME_MPEG2
    } else if payload.starts_with(b"PPTS") {
        if ts_is_hevc(payload) {
            NAME_TS_HEVC
        } else {
            NAME_TS
        }
    } else if payload.starts_with(b"PPH3") {
        NAME_HEVC
    } else {
        NAME
    }
}

/// Whether a `PPTS` payload holds HEVC: the stream after its side stream (whose length is a
/// varint after the magic) starts with `PPH3`.
fn ts_is_hevc(payload: &[u8]) -> bool {
    let mut len = 0u64;
    for (pos, shift) in (4usize..).zip((0..64).step_by(7)) {
        let Some(&b) = payload.get(pos) else { return false };
        len |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            let start = usize::try_from(len).ok().and_then(|n| n.checked_add(pos + 1));
            return start.and_then(|p| payload.get(p..p.checked_add(4)?)) == Some(&b"PPH3"[..]);
        }
    }
    false
}

#[cfg(feature = "media")]
type Packer = fn(&[u8], &piedpiper_media::Options, &mut piedpiper_media::Progress) -> std::result::Result<Option<Vec<u8>>, piedpiper_media::Cancelled>;

#[cfg(feature = "media")]
fn pack_h264(input: &[u8], o: &piedpiper_media::Options, p: &mut piedpiper_media::Progress) -> std::result::Result<Option<Vec<u8>>, piedpiper_media::Cancelled> {
    piedpiper_media::compress_with(input, o, p).map(|r| r.map(|(packed, _)| packed))
}

#[cfg(feature = "media")]
fn pack_vp9(input: &[u8], o: &piedpiper_media::Options, p: &mut piedpiper_media::Progress) -> std::result::Result<Option<Vec<u8>>, piedpiper_media::Cancelled> {
    piedpiper_media::vp9::stream::compress_with(input, o, p).map(|r| r.map(|(packed, _)| packed))
}

#[cfg(feature = "media")]
fn pack_vp8(input: &[u8], o: &piedpiper_media::Options, p: &mut piedpiper_media::Progress) -> std::result::Result<Option<Vec<u8>>, piedpiper_media::Cancelled> {
    piedpiper_media::vp8::stream::compress_with(input, o, p).map(|r| r.map(|(packed, _)| packed))
}

#[cfg(feature = "media")]
fn pack_mpeg2(input: &[u8], o: &piedpiper_media::Options, p: &mut piedpiper_media::Progress) -> std::result::Result<Option<Vec<u8>>, piedpiper_media::Cancelled> {
    piedpiper_media::mpeg2::stream::compress_with(input, o, p).map(|r| r.map(|(packed, _)| packed))
}

#[cfg(feature = "media")]
fn pack_ts(input: &[u8], o: &piedpiper_media::Options, p: &mut piedpiper_media::Progress) -> std::result::Result<Option<Vec<u8>>, piedpiper_media::Cancelled> {
    piedpiper_media::ts::compress_with(input, o, p).map(|r| r.map(|(packed, _)| packed))
}

#[cfg(feature = "media")]
fn unpack(payload: &[u8], progress: &mut piedpiper_media::Progress) -> std::result::Result<Vec<u8>, piedpiper_media::bits::Bad> {
    if payload.starts_with(piedpiper_media::vp9::stream::MAGIC) {
        piedpiper_media::vp9::stream::decompress_with(payload, 0, progress)
    } else if payload.starts_with(piedpiper_media::vp8::stream::MAGIC) {
        piedpiper_media::vp8::stream::decompress_with(payload, 0, progress)
    } else if payload.starts_with(piedpiper_media::mpeg2::stream::MAGIC) {
        piedpiper_media::mpeg2::stream::decompress_with(payload, 0, progress)
    } else if payload.starts_with(piedpiper_media::ts::MAGIC) {
        piedpiper_media::ts::decompress_with(payload, 0, progress)
    } else {
        piedpiper_media::decompress_with(payload, 0, progress)
    }
}

/// Recompresses `input` if it is a video file with an H.264, HEVC, VP9, VP8 or MPEG-2 track (or an H.264
/// or HEVC MPEG transport stream) and recompression
/// makes it smaller. Returns `Ok(None)` otherwise. The result is decompressed and compared
/// with the input before it is returned, so a parser bug can only cost compression, never
/// data.
#[cfg(feature = "media")]
pub fn compress(input: &[u8], crc: u32, progress: &mut ProgressFn<'_>) -> Result<Option<Vec<u8>>> {
    use crate::format::{self, MODE_MEDIA};
    let pack: Packer = if piedpiper_media::probe(input) {
        pack_h264
    } else if piedpiper_media::vp9::stream::probe(input) {
        pack_vp9
    } else if piedpiper_media::vp8::stream::probe(input) {
        pack_vp8
    } else if piedpiper_media::mpeg2::stream::probe(input) {
        pack_mpeg2
    } else if piedpiper_media::ts::probe(input) {
        pack_ts
    } else {
        return Ok(None);
    };
    let total = input.len() as u64;
    // Recompression and the verifying decompression take about the same time.
    let mut cancelled = false;
    let packed = {
        let mut p = |d: u64, _t: u64| {
            let ok = progress(d / 2, total);
            cancelled |= !ok;
            ok
        };
        match pack(input, &piedpiper_media::Options::default(), &mut p) {
            Err(piedpiper_media::Cancelled) => return Err(Error::Cancelled),
            Ok(None) => return Ok(None),
            Ok(Some(packed)) => packed,
        }
    };
    if packed.len() + 16 >= input.len() {
        return Ok(None);
    }
    let back = {
        let mut p = |d: u64, _t: u64| {
            let ok = progress(total / 2 + d / 2, total);
            cancelled |= !ok;
            ok
        };
        unpack(&packed, &mut p)
    };
    if cancelled {
        return Err(Error::Cancelled);
    }
    if !matches!(&back, Ok(b) if b.as_slice() == input) {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(packed.len() + 16);
    format::write_prefix(&mut out, MODE_MEDIA, total, crc);
    out.extend_from_slice(&packed);
    progress(total, total);
    Ok(Some(out))
}

#[cfg(feature = "media")]
pub fn decompress(payload: &[u8], orig_len: u64, progress: &mut ProgressFn<'_>) -> Result<Vec<u8>> {
    let mut cancelled = false;
    let out = {
        let mut p = |d: u64, t: u64| {
            let ok = progress(d, t);
            cancelled |= !ok;
            ok
        };
        unpack(payload, &mut p)
    };
    if cancelled {
        return Err(Error::Cancelled);
    }
    let out = out.map_err(|_| Error::Format("corrupt video recompression stream".into()))?;
    if out.len() as u64 != orig_len {
        return Err(Error::Format("video recompression stream has the wrong length".into()));
    }
    Ok(out)
}

/// True for already-compressed video that mode 2 cannot recode (AV1 or VP9 in MP4,
/// other Matroska/WebM, AVI, MPEG-TS, FLV) and whose bytes look incompressible. The neural model
/// saves next to nothing on such files, so the encoder stores them right away.
#[cfg(feature = "media")]
pub fn opaque(input: &[u8]) -> bool {
    piedpiper_media::is_opaque_video(input)
}

#[cfg(not(feature = "media"))]
pub fn opaque(_input: &[u8]) -> bool {
    false
}

#[cfg(not(feature = "media"))]
pub fn compress(_input: &[u8], _crc: u32, _progress: &mut ProgressFn<'_>) -> Result<Option<Vec<u8>>> {
    Ok(None)
}

#[cfg(not(feature = "media"))]
pub fn decompress(_payload: &[u8], _orig_len: u64, _progress: &mut ProgressFn<'_>) -> Result<Vec<u8>> {
    Err(Error::Format("this build cannot decompress recompressed video (feature `media` is off)".into()))
}
