//! Format mode 5: lossless 16-bit PCM audio (WAV files).
//!
//! A byte-level model handles PCM audio poorly. It cannot learn linear prediction from
//! little-endian sample bytes, so on real recordings the plain neural path comes out 8–45%
//! larger than FLAC (see docs/DESIGN.md). This mode puts a classic lossless-audio front end
//! in front of the Transformer:
//!
//! 1. Parse the RIFF/WAVE container. Everything outside the PCM samples (header chunks,
//!    trailing chunks, a dangling odd byte) is kept verbatim as `meta`.
//! 2. Decorrelate stereo with integer mid/side lifting, which is exactly invertible.
//! 3. Predict every sample of every channel with a cascade of three normalised-LMS filters
//!    fed with the channel's first differences. Stage 1 also sees the other channel.
//! 4. Zigzag the integer prediction residuals and emit two bytes each, most significant
//!    first. `FF FF` escapes to three more bytes.
//!
//! `meta` and the residual bytes are then coded as two ordinary PiedPiper streams, as in
//! mode 4. The residuals get the online-learning Transformer, the match model and, if one
//! is given, a prior trained on residuals (`piedpiper train` on WAV files).
//!
//! The predictor uses only IEEE-754 double-precision `+ - * /` in a fixed order, plus a
//! rounding step built from truncation, so encoder and decoder compute identical bits on
//! every platform.

use crate::codec::{CompressOptions, DecompressOptions, ProgressFn};
use crate::error::{Error, Result};

/// Name reported by [`crate::inspect`] for mode-5 streams.
pub const NAME: &str = "pcm-audio";

#[cfg(feature = "audio")]
mod imp {
    use super::*;
    use crate::codec::{compress_with_progress, decompress_with_progress};
    use crate::filter::FilterChoice;
    use crate::format::{self, MODE_AUDIO, MODE_NEURAL, MODE_STORED};
    use crate::fpenv::FpGuard;

    /// Version of the predictor and residual layout recorded in each stream.
    const AUDIO_VERSION: u8 = 1;
    /// Taps of the predictor cascade: stage 1 sees `K1` own and `KX` cross-channel values.
    const K1: usize = 32;
    const KX: usize = 16;
    const N1: usize = K1 + KX;
    const K2: usize = 16;
    const K3: usize = 4;
    const MU1: f64 = 0.003;
    const MU2: f64 = 0.006;
    const MU3: f64 = 0.02;
    /// Predictions are clamped to ±2^20, so every residual stays below 2^21 in magnitude
    /// and the three-byte escape always suffices.
    const PRED_LIMIT: f64 = 1_048_576.0;
    /// Shorter files are not worth a separate mode.
    const MIN_FRAMES: usize = 64;
    /// KSDATAFORMAT_SUBTYPE_PCM, the sub-format GUID of WAVE_FORMAT_EXTENSIBLE PCM files.
    const SUBTYPE_PCM: [u8; 16] = [1, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71];

    pub(super) struct Wav<'a> {
        pub header: &'a [u8],
        pub pcm: &'a [u8],
        pub trailer: &'a [u8],
        pub channels: usize,
        pub frames: usize,
    }

    fn le16(b: &[u8], i: usize) -> u16 {
        u16::from_le_bytes([b[i], b[i + 1]])
    }

    fn le32(b: &[u8], i: usize) -> u32 {
        u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
    }

    /// Finds the samples of a 16-bit PCM RIFF/WAVE file (mono or stereo).
    pub(super) fn parse(input: &[u8]) -> Option<Wav<'_>> {
        if input.len() < 12 || &input[0..4] != b"RIFF" || &input[8..12] != b"WAVE" {
            return None;
        }
        let mut channels = None;
        let mut pos = 12usize;
        while pos.checked_add(8)? <= input.len() {
            let id = &input[pos..pos + 4];
            let size = le32(input, pos + 4) as usize;
            let body = pos + 8;
            if id == b"fmt " {
                let f = input.get(body..body.checked_add(size)?)?;
                if f.len() < 16 {
                    return None;
                }
                let (tag, ch, align, bits) = (le16(f, 0), le16(f, 2) as usize, le16(f, 12) as usize, le16(f, 14));
                let pcm = match tag {
                    1 => true,
                    0xFFFE => f.len() >= 40 && f[24..40] == SUBTYPE_PCM,
                    _ => false,
                };
                if !pcm || bits != 16 || !(1..=2).contains(&ch) || align != 2 * ch {
                    return None;
                }
                channels = Some(ch);
            } else if id == b"data" {
                let ch = channels?;
                let len = size.min(input.len() - body);
                let frames = len / (2 * ch);
                if frames < MIN_FRAMES {
                    return None;
                }
                let end = body + frames * 2 * ch;
                return Some(Wav {
                    header: &input[..body],
                    pcm: &input[body..end],
                    trailer: &input[end..],
                    channels: ch,
                    frames,
                });
            }
            pos = body.checked_add(size)?.checked_add(size & 1)?;
        }
        None
    }

    fn zigzag(r: i64) -> u64 {
        ((r << 1) ^ (r >> 63)) as u64
    }

    fn unzigzag(u: u64) -> i64 {
        ((u >> 1) as i64) ^ -((u & 1) as i64)
    }

    fn put_residual(out: &mut Vec<u8>, r: i64) {
        let u = zigzag(r);
        if u < 0xFFFF {
            out.extend_from_slice(&[(u >> 8) as u8, u as u8]);
        } else {
            debug_assert!(u < 1 << 24);
            out.extend_from_slice(&[0xFF, 0xFF, (u >> 16) as u8, (u >> 8) as u8, u as u8]);
        }
    }

    fn get_residual(b: &[u8], pos: &mut usize) -> Option<i64> {
        let h = b.get(*pos..*pos + 2)?;
        *pos += 2;
        let u = ((h[0] as u64) << 8) | h[1] as u64;
        if u != 0xFFFF {
            return Some(unzigzag(u));
        }
        let e = b.get(*pos..*pos + 3)?;
        *pos += 3;
        Some(unzigzag(((e[0] as u64) << 16) | ((e[1] as u64) << 8) | e[2] as u64))
    }

    /// Integer mid/side lifting: exactly invertible, mid fits 16 bits, side 17 bits.
    fn to_mid_side(l: i64, r: i64) -> (i64, i64) {
        let s = l - r;
        (r + (s >> 1), s)
    }

    fn from_mid_side(m: i64, s: i64) -> (i64, i64) {
        let r = m - (s >> 1);
        (s + r, r)
    }

    /// Rounds half away from zero with truncation and subtraction only (both exact).
    fn round_half_away(x: f64) -> i64 {
        let t = x as i64;
        let f = x - t as f64;
        if f >= 0.5 {
            t + 1
        } else if f <= -0.5 {
            t - 1
        } else {
            t
        }
    }

    fn dot(w: &[f64], h: &[f64]) -> f64 {
        let mut acc = 0.0;
        for i in 0..w.len() {
            acc += w[i] * h[i];
        }
        acc
    }

    /// Normalised LMS step: `w += mu * err * h / (1 + |h|²)`.
    fn nlms(w: &mut [f64], h: &[f64], err: f64, mu: f64) {
        let mut pow = 1.0;
        for &x in h {
            pow += x * x;
        }
        let g = mu * err / pow;
        for i in 0..w.len() {
            w[i] += g * h[i];
        }
    }

    fn push(h: &mut [f64], v: f64) {
        h.copy_within(0..h.len() - 1, 1);
        h[0] = v;
    }

    #[derive(Clone)]
    struct Channel {
        prev: i64,
        /// Stage-1 input: own first differences, then the other channel's (most recent first).
        h1: [f64; N1],
        w1: [f64; N1],
        h2: [f64; K2],
        w2: [f64; K2],
        h3: [f64; K3],
        w3: [f64; K3],
        /// Recent first differences, most recent first (read by the other channel).
        e0: [f64; KX],
    }

    struct Prediction {
        value: i64,
        p1: f64,
        p2: f64,
        p3: f64,
    }

    /// The cascade for all channels. Encoder and decoder call `predict` / `update` in the
    /// same order (frame by frame, channel 0 first), so both see the same state.
    struct Predictor {
        ch: Vec<Channel>,
    }

    impl Predictor {
        fn new(channels: usize) -> Self {
            let c = Channel {
                prev: 0,
                h1: [0.0; N1],
                w1: [0.0; N1],
                h2: [0.0; K2],
                w2: [0.0; K2],
                h3: [0.0; K3],
                w3: [0.0; K3],
                e0: [0.0; KX],
            };
            Predictor { ch: vec![c; channels] }
        }

        /// Prediction of channel `c`'s next first difference. Channel 1 sees channel 0's
        /// difference of the current frame; channel 0 sees channel 1's of earlier frames.
        fn predict(&mut self, c: usize) -> Prediction {
            if self.ch.len() == 2 {
                let other = self.ch[1 - c].e0;
                self.ch[c].h1[K1..].copy_from_slice(&other);
            }
            let s = &self.ch[c];
            let p1 = dot(&s.w1, &s.h1);
            let p2 = dot(&s.w2, &s.h2);
            let p3 = dot(&s.w3, &s.h3);
            let p = (p1 + p2 + p3).clamp(-PRED_LIMIT, PRED_LIMIT);
            Prediction { value: round_half_away(p), p1, p2, p3 }
        }

        fn update(&mut self, c: usize, x: i64, e0: i64, p: &Prediction) {
            let s = &mut self.ch[c];
            let e0 = e0 as f64;
            let e1 = e0 - p.p1;
            let e2 = e1 - p.p2;
            let e3 = e2 - p.p3;
            nlms(&mut s.w1, &s.h1, e1, MU1);
            nlms(&mut s.w2, &s.h2, e2, MU2);
            nlms(&mut s.w3, &s.h3, e3, MU3);
            push(&mut s.h1[..K1], e0);
            push(&mut s.h2, e1);
            push(&mut s.h3, e2);
            push(&mut s.e0, e0);
            s.prev = x;
        }
    }

    /// The residual byte stream for the samples of `w`.
    pub(super) fn encode_residuals(w: &Wav) -> Vec<u8> {
        let _fp = FpGuard::new();
        let nch = w.channels;
        let mut out = Vec::with_capacity(w.pcm.len() + w.pcm.len() / 64 + 16);
        let mut pred = Predictor::new(nch);
        let mut x = [0i64; 2];
        for frame in w.pcm.chunks_exact(2 * nch) {
            let a = i16::from_le_bytes([frame[0], frame[1]]) as i64;
            if nch == 2 {
                let b = i16::from_le_bytes([frame[2], frame[3]]) as i64;
                let (m, s) = to_mid_side(a, b);
                x = [m, s];
            } else {
                x[0] = a;
            }
            for c in 0..nch {
                let p = pred.predict(c);
                let e0 = x[c] - pred.ch[c].prev;
                put_residual(&mut out, e0 - p.value);
                pred.update(c, x[c], e0, &p);
            }
        }
        out
    }

    /// Inverse of [`encode_residuals`]: appends the little-endian PCM samples to `out`.
    pub(super) fn decode_residuals(res: &[u8], nch: usize, frames: usize, out: &mut Vec<u8>) -> Result<()> {
        let _fp = FpGuard::new();
        let bad = || Error::Format("corrupt audio residual stream".into());
        let mut pred = Predictor::new(nch);
        let mut pos = 0usize;
        let mut x = [0i64; 2];
        for _ in 0..frames {
            for c in 0..nch {
                let r = get_residual(res, &mut pos).ok_or_else(bad)?;
                let p = pred.predict(c);
                let e0 = r + p.value;
                let v = pred.ch[c].prev + e0;
                // Mid (or mono) samples fit 16 bits; the side channel needs 17.
                let (lo, hi) = if c == 1 { (-65535, 65535) } else { (-32768, 32767) };
                if !(lo..=hi).contains(&v) {
                    return Err(bad());
                }
                x[c] = v;
                pred.update(c, v, e0, &p);
            }
            if nch == 2 {
                let (l, r) = from_mid_side(x[0], x[1]);
                let range = -32768..=32767;
                if !range.contains(&l) || !range.contains(&r) {
                    return Err(bad());
                }
                out.extend_from_slice(&(l as i16).to_le_bytes());
                out.extend_from_slice(&(r as i16).to_le_bytes());
            } else {
                out.extend_from_slice(&(x[0] as i16).to_le_bytes());
            }
        }
        if pos != res.len() {
            return Err(bad());
        }
        Ok(())
    }

    fn scaled(done: u64, work: u64, total: u64) -> u64 {
        (done.min(work) as u128 * total as u128 / work.max(1) as u128) as u64
    }

    pub(super) fn compress(
        input: &[u8],
        crc: u32,
        opts: &CompressOptions,
        progress: &mut ProgressFn<'_>,
    ) -> Result<Option<Vec<u8>>> {
        let Some(wav) = parse(input) else { return Ok(None) };
        let residuals = encode_residuals(&wav);
        // Check the predictor round trip before relying on it (cheap next to the network):
        // a mismatch would be a bug, and the file then takes the plain neural path.
        let mut check = Vec::with_capacity(wav.pcm.len());
        if decode_residuals(&residuals, wav.channels, wav.frames, &mut check).is_err() || check != wav.pcm {
            return Ok(None);
        }
        drop(check);
        let mut meta = Vec::with_capacity(wav.header.len() + wav.trailer.len());
        meta.extend_from_slice(wav.header);
        meta.extend_from_slice(wav.trailer);
        // Inner streams are plain neural (or stored) streams: no container modes inside.
        let mut inner = opts.clone();
        inner.audio = false;
        inner.media = false;
        inner.jpeg = false;
        inner.deflate = false;
        inner.allow_stored = true;
        inner.filter = FilterChoice::Off; // residuals are not machine code
        let mut meta_opts = inner.clone();
        meta_opts.prior = None; // a prior is for the residuals
        let total = input.len() as u64;
        let work = (meta.len() + residuals.len()) as u64;
        let mlen = meta.len() as u64;
        let m = compress_with_progress(&meta, &meta_opts, &mut |d, _| progress(scaled(d, work, total), total))?;
        let r = compress_with_progress(&residuals, &inner, &mut |d, _| progress(scaled(mlen + d, work, total), total))?;
        let mut out = Vec::with_capacity(m.len() + r.len() + 32);
        format::write_prefix(&mut out, MODE_AUDIO, total, crc);
        out.push(AUDIO_VERSION);
        out.push(wav.channels as u8);
        out.push(16);
        format::put_varint(&mut out, wav.frames as u64);
        format::put_varint(&mut out, wav.header.len() as u64);
        format::put_varint(&mut out, m.len() as u64);
        out.extend_from_slice(&m);
        out.extend_from_slice(&r);
        if opts.allow_stored && out.len() >= input.len() + 10 {
            // Noise-like audio: storing beats the neural path, which would do no better here.
            let mut s = Vec::with_capacity(input.len() + 16);
            format::write_prefix(&mut s, MODE_STORED, total, crc);
            s.extend_from_slice(input);
            return Ok(Some(s));
        }
        Ok(Some(out))
    }

    pub(super) fn decompress(
        body: &[u8],
        orig_len: u64,
        opts: &DecompressOptions,
        progress: &mut ProgressFn<'_>,
    ) -> Result<Vec<u8>> {
        let bad = |m: &str| Error::Format(m.to_string());
        let byte = |pos: usize| body.get(pos).copied().ok_or_else(|| bad("truncated audio header"));
        if byte(0)? != AUDIO_VERSION {
            return Err(bad("unsupported audio stream version"));
        }
        let (channels, bits) = (byte(1)? as usize, byte(2)?);
        if !(1..=2).contains(&channels) || bits != 16 {
            return Err(bad("unsupported audio sample layout"));
        }
        let mut pos = 3;
        let frames = format::get_varint(body, &mut pos)?;
        let header_len = format::get_varint(body, &mut pos)?;
        let meta_len = format::get_varint(body, &mut pos)?;
        let pcm_len = frames
            .checked_mul(2 * channels as u64)
            .filter(|&n| n <= orig_len)
            .ok_or_else(|| bad("inconsistent audio sizes"))?;
        let meta_len = usize::try_from(meta_len).map_err(|_| bad("inconsistent audio sizes"))?;
        let meta_end = pos.checked_add(meta_len).filter(|&e| e <= body.len()).ok_or_else(|| bad("truncated audio stream"))?;
        let (meta_s, res_s) = (&body[pos..meta_end], &body[meta_end..]);
        for s in [meta_s, res_s] {
            if s.len() > 5 && s[5] != MODE_NEURAL && s[5] != MODE_STORED {
                return Err(bad("nested container streams are not allowed"));
            }
        }
        let meta = decompress_with_progress(meta_s, opts, &mut |_, _| true)?;
        if meta.len() as u64 != orig_len - pcm_len || header_len > meta.len() as u64 {
            return Err(bad("audio metadata has the wrong length"));
        }
        let res = decompress_with_progress(res_s, opts, progress)?;
        // Every sample takes at least two residual bytes, which bounds what we allocate.
        if pcm_len > res.len() as u64 {
            return Err(bad("audio residual stream is too short"));
        }
        let header_len = header_len as usize;
        let mut out = Vec::with_capacity(meta.len() + pcm_len as usize);
        out.extend_from_slice(&meta[..header_len]);
        decode_residuals(&res, channels, frames as usize, &mut out)?;
        out.extend_from_slice(&meta[header_len..]);
        Ok(out)
    }
}

/// Recompresses `input` if it is a 16-bit PCM WAV file (mono or stereo). Returns
/// `Ok(None)` for anything else.
#[cfg(feature = "audio")]
pub fn compress(input: &[u8], crc: u32, opts: &CompressOptions, progress: &mut ProgressFn<'_>) -> Result<Option<Vec<u8>>> {
    imp::compress(input, crc, opts, progress)
}

#[cfg(feature = "audio")]
pub fn decompress(body: &[u8], orig_len: u64, opts: &DecompressOptions, progress: &mut ProgressFn<'_>) -> Result<Vec<u8>> {
    imp::decompress(body, orig_len, opts, progress)
}

/// The prediction residuals the audio mode codes for `data`, if it is a 16-bit PCM WAV
/// file. Train an audio prior on these with [`crate::train_prior`]; `piedpiper train`
/// does this automatically for WAV inputs.
#[cfg(feature = "audio")]
pub fn residuals(data: &[u8]) -> Option<Vec<u8>> {
    imp::parse(data).map(|w| imp::encode_residuals(&w))
}

#[cfg(not(feature = "audio"))]
pub fn compress(_input: &[u8], _crc: u32, _opts: &CompressOptions, _progress: &mut ProgressFn<'_>) -> Result<Option<Vec<u8>>> {
    Ok(None)
}

#[cfg(not(feature = "audio"))]
pub fn decompress(_body: &[u8], _orig_len: u64, _opts: &DecompressOptions, _progress: &mut ProgressFn<'_>) -> Result<Vec<u8>> {
    Err(Error::Format("this build cannot decompress audio streams (built without the `audio` feature)".into()))
}

#[cfg(not(feature = "audio"))]
pub fn residuals(_data: &[u8]) -> Option<Vec<u8>> {
    None
}

#[cfg(all(test, feature = "audio"))]
mod tests {
    use super::imp::{encode_residuals, parse};
    use crate::format::MODE_AUDIO;
    use crate::rng::Rng;
    use crate::{compress, decompress, decompress_with_progress, inspect, CompressOptions, DecompressOptions, ModelConfig};

    fn wav(channels: u16, samples: &[i16], extra_fmt: &[u8], trailer: &[u8]) -> Vec<u8> {
        let data_len = (samples.len() * 2) as u32;
        let fmt_len = 16 + extra_fmt.len() as u32;
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(4 + 8 + fmt_len + 8 + data_len + trailer.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&fmt_len.to_le_bytes());
        let tag: u16 = if extra_fmt.is_empty() { 1 } else { 0xFFFE };
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&22050u32.to_le_bytes());
        out.extend_from_slice(&(22050 * 2 * channels as u32).to_le_bytes());
        out.extend_from_slice(&(2 * channels).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(extra_fmt);
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for s in samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out.extend_from_slice(trailer);
        out
    }

    /// Two decaying tones plus a little noise; right channel is a scaled copy of the left
    /// with its own noise, so mid/side and cross-channel prediction both matter.
    fn music(frames: usize, stereo: bool, seed: u64) -> Vec<i16> {
        let mut rng = Rng::new(seed);
        let (mut a, mut b) = (12000i64, 0i64); // integer "magic circle" oscillators
        let (mut c, mut d) = (5000i64, 0i64);
        let mut out = Vec::new();
        for _ in 0..frames {
            a -= (b * 201) >> 12;
            b += (a * 201) >> 12;
            c -= (d * 37) >> 10;
            d += (c * 37) >> 10;
            let noise = (rng.next_u64() % 61) as i64 - 30;
            let l = (a + c + noise).clamp(-32768, 32767);
            out.push(l as i16);
            if stereo {
                let n2 = (rng.next_u64() % 41) as i64 - 20;
                out.push(((l * 3) / 4 + n2).clamp(-32768, 32767) as i16);
            }
        }
        out
    }

    fn small_model() -> CompressOptions {
        CompressOptions {
            model: Some(ModelConfig { d_model: 32, n_layer: 1, n_head: 2, d_ff: 48, window: 24, seg_len: 8, n_streams: 2 }),
            allow_stored: false,
            ..CompressOptions::level(1)
        }
    }

    fn check_inverse(data: &[u8]) {
        let w = parse(data).expect("parses");
        let res = encode_residuals(&w);
        let mut back = w.header.to_vec();
        super::imp::decode_residuals(&res, w.channels, w.frames, &mut back).unwrap();
        back.extend_from_slice(w.trailer);
        assert_eq!(back, data);
    }

    #[test]
    fn residuals_invert_exactly_on_hard_signals() {
        let mut rng = Rng::new(1);
        let noise: Vec<i16> = (0..4000).map(|_| rng.next_u64() as i16).collect();
        let square: Vec<i16> = (0..4000).map(|i| if (i / 7) % 2 == 0 { 32767 } else { -32768 }).collect();
        let silence = vec![0i16; 2000];
        for ch in [1u16, 2] {
            for sig in [&noise, &square, &silence, &music(2000, ch == 2, 3)] {
                let n = sig.len() - sig.len() % ch as usize;
                check_inverse(&wav(ch, &sig[..n], &[], b""));
            }
        }
    }

    #[test]
    fn parser_accepts_pcm_and_rejects_the_rest() {
        let s = music(1000, true, 5);
        let mut ext = vec![22, 0, 16, 0, 3, 0, 0, 0]; // cbSize, valid bits, channel mask
        ext.extend_from_slice(&[1, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71]);
        assert!(parse(&wav(2, &s, &ext, b"")).is_some(), "WAVE_FORMAT_EXTENSIBLE PCM");
        let mut float = ext.clone();
        float[8] = 3; // IEEE float sub-format
        assert!(parse(&wav(2, &s, &float, b"")).is_none());
        let mut bits24 = wav(2, &s, &[], b"");
        bits24[34] = 24;
        assert!(parse(&bits24).is_none());
        let mut three = wav(2, &s, &[], b"");
        three[22] = 3;
        assert!(parse(&three).is_none());
        assert!(parse(&wav(1, &s[..40], &[], b"")).is_none(), "too short");
        let good = wav(2, &s, &[], b"");
        for cut in [0, 11, 20, 30, 43] {
            assert!(parse(&good[..cut]).is_none());
        }
    }

    #[test]
    fn roundtrip_in_mode_5_with_metadata() {
        let mut trailer = b"LIST\x16\x00\x00\x00INFOISFT\x0a\x00\x00\x00piedpiper\x00".to_vec();
        trailer.push(0x5A); // a dangling byte after the last chunk
        for (ch, frames) in [(2u16, 3000), (1, 5000)] {
            let data = wav(ch, &music(frames, ch == 2, 7), &[], &trailer);
            let c = compress(&data, &small_model()).unwrap();
            assert_eq!(c[5], MODE_AUDIO);
            let info = inspect(&c).unwrap();
            assert_eq!(info.tokenizer, super::NAME);
            assert!(info.model.is_none());
            assert_eq!(decompress(&c).unwrap(), data);
        }
    }

    #[test]
    fn prediction_beats_the_plain_neural_path() {
        let data = wav(2, &music(6000, true, 11), &[], b"");
        let with = compress(&data, &small_model()).unwrap();
        let without = compress(&data, &CompressOptions { audio: false, ..small_model() }).unwrap();
        assert_eq!(with[5], MODE_AUDIO);
        assert_ne!(without[5], MODE_AUDIO);
        assert!(with.len() * 4 < without.len() * 3, "{} vs {}", with.len(), without.len());
        assert_eq!(decompress(&without).unwrap(), data);
    }

    #[test]
    fn thread_count_does_not_change_output() {
        let data = wav(2, &music(2500, true, 13), &[], b"");
        let a = compress(&data, &CompressOptions { threads: 1, ..small_model() }).unwrap();
        let b = compress(&data, &CompressOptions { threads: 4, ..small_model() }).unwrap();
        assert_eq!(a, b);
        let back = decompress_with_progress(&a, &DecompressOptions { threads: 3, prior: None }, &mut |_, _| true).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn noise_is_stored_and_corruption_is_an_error() {
        let mut rng = Rng::new(17);
        let noise: Vec<i16> = (0..3000).map(|_| rng.next_u64() as i16).collect();
        let data = wav(1, &noise, &[], b"");
        let c = compress(&data, &CompressOptions { allow_stored: true, ..small_model() }).unwrap();
        assert!(inspect(&c).unwrap().stored);
        assert_eq!(decompress(&c).unwrap(), data);

        let data = wav(2, &music(3000, true, 19), &[], b"");
        let c = compress(&data, &small_model()).unwrap();
        // Header fields, the inner metadata stream, and the residual payload (not its last
        // few bytes: a range coder's flush bytes can legitimately be "don't care").
        for k in [7usize, 12, 14, 20, c.len() / 2, c.len() * 3 / 4] {
            let mut bad = c.clone();
            bad[k] ^= 0x24;
            assert!(decompress(&bad).is_err(), "flip at {k}");
        }
        assert!(decompress(&c[..c.len() * 2 / 3]).is_err());
    }
}
