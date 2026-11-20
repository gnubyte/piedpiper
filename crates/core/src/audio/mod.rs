//! Format mode 5: lossless 16-bit PCM audio (WAV files).
//!
//! A byte-level model handles PCM audio poorly. It cannot learn linear prediction from
//! little-endian sample bytes, so on real recordings the plain neural path comes out 8–45%
//! larger than FLAC (see docs/DESIGN.md). This mode codes the samples the way dedicated
//! lossless audio codecs do:
//!
//! 1. Parse the RIFF/WAVE container. Everything outside the PCM samples (header chunks,
//!    trailing chunks, a dangling odd byte) is kept verbatim as `meta`.
//! 2. Decorrelate stereo with integer mid/side lifting, which is exactly invertible.
//! 3. Predict every sample of every channel with a cascade of normalised-LMS filters fed
//!    with the channel's first differences ([`lms`]). The level picks the cascade:
//!    256+32+16+8 taps at level 1, 512+256+32+16+8 at levels 2–3, 1024+256+32+16+8 above.
//! 4. Per block of 4096 frames and channel, the encoder also fits a linear predictor to the
//!    block itself ([`lpc`], as FLAC does) and uses it instead of the cascade where its
//!    residuals are cheaper: on clean synthetic tones and very short clips, which the
//!    adaptive filters cannot learn in time. Its coefficients go in the stream.
//! 5. Code the integer prediction residuals with a small context-mixing model driving a
//!    binary arithmetic coder ([`rc`]). On residuals this does as well as the Transformer,
//!    thousands of times faster.
//!
//! With a prior, the residuals instead go to the neural model as two bytes each, most
//! significant first (`FF FF` escapes to three more bytes), and every block uses the
//! cascade: a prior trained on residuals (`piedpiper train` on WAV files) can beat the
//! context-mixing coder. `meta` is always an ordinary nested PiedPiper stream.
//!
//! Version-1 streams (one fixed 32+16+4-tap cascade, residuals always coded by the network)
//! still decode.
//!
//! The predictors use only IEEE-754 double-precision `+ - * /` in a fixed order and the coder
//! only integers, so encoder and decoder compute identical bits on every platform.

use crate::codec::{CompressOptions, DecompressOptions, ProgressFn};
use crate::error::{Error, Result};

#[cfg(feature = "audio")]
mod lms;
#[cfg(feature = "audio")]
mod lpc;
#[cfg(feature = "audio")]
mod rc;

/// Name reported by [`crate::inspect`] for mode-5 streams.
pub const NAME: &str = "pcm-audio";

#[cfg(feature = "audio")]
mod imp {
    use super::lms::{Cascade, Predict, PRESETS, V1};
    use super::lpc::{self, Lpc};
    use super::rc;
    use super::*;
    use crate::codec::{compress_with_progress, decompress_with_progress};
    use crate::filter::FilterChoice;
    use crate::format::{self, MODE_AUDIO, MODE_NEURAL, MODE_STORED};
    use crate::fpenv::FpGuard;

    /// Version 1: fixed predictor, residuals coded by the network. Decoded only.
    const VERSION_1: u8 = 1;
    /// Version 2: predictor preset and residual coder recorded in the stream.
    const VERSION_2: u8 = 2;
    /// Residual coders of version 2.
    const CODER_CM: u8 = 0;
    const CODER_NEURAL: u8 = 1;
    /// Shorter files are not worth a separate mode.
    const MIN_FRAMES: usize = 64;
    /// Frames between progress reports (and per block of the round-trip check).
    const BLOCK_FRAMES: usize = 1 << 16;
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

    /// The level's predictor preset (an index into [`PRESETS`]).
    pub(super) fn preset_for_level(level: u32) -> u8 {
        match level {
            0 | 1 => 0,
            2 | 3 => 1,
            _ => 2,
        }
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

    /// The byte form of residuals that the neural model codes.
    pub(super) fn residual_bytes(res: &[i32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 * res.len() + res.len() / 64 + 16);
        for &r in res {
            put_residual(&mut out, r as i64);
        }
        out
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

    /// Prediction residuals of `w` under the cascade, frame-major (channel 0, then channel 1,
    /// of each frame). Every `BLOCK_FRAMES` frames `progress(f)` is told the frames done
    /// (returning `false` cancels) and `block(first_frame, residuals)` gets the block just
    /// finished, which it may rewrite.
    pub(super) fn residuals<P: Predict>(
        p: &mut P,
        w: &Wav,
        progress: &mut dyn FnMut(usize) -> bool,
        block: &mut dyn FnMut(usize, &mut [i32]),
    ) -> Result<Vec<i32>> {
        let _fp = FpGuard::new();
        let nch = w.channels;
        let mut out = Vec::with_capacity(w.frames * nch);
        let mut x = [0i64; 2];
        for (f, frame) in w.pcm.chunks_exact(2 * nch).enumerate() {
            if f % BLOCK_FRAMES == 0 {
                if f > 0 {
                    block(f - BLOCK_FRAMES, &mut out[(f - BLOCK_FRAMES) * nch..]);
                }
                if !progress(f) {
                    return Err(Error::Cancelled);
                }
            }
            let a = i16::from_le_bytes([frame[0], frame[1]]) as i64;
            if nch == 2 {
                let (m, s) = to_mid_side(a, i16::from_le_bytes([frame[2], frame[3]]) as i64);
                x = [m, s];
            } else {
                x[0] = a;
            }
            for (c, &v) in x.iter().enumerate().take(nch) {
                let pr = p.predict(c);
                // |v - prev| < 2^18 and |pr| ≤ 2^20, so the residual fits easily.
                out.push((v - p.prev(c) - pr) as i32);
                p.update(c, v);
            }
        }
        if w.frames > 0 {
            let last = (w.frames - 1) / BLOCK_FRAMES * BLOCK_FRAMES;
            block(last, &mut out[last * nch..]);
        }
        Ok(out)
    }

    /// Channel `c`'s values (mid or side for stereo) for frames `from..to`.
    fn channel_values(w: &Wav, c: usize, from: usize, to: usize) -> Vec<i64> {
        let nch = w.channels;
        w.pcm[from * 2 * nch..to * 2 * nch]
            .chunks_exact(2 * nch)
            .map(|f| {
                let a = i16::from_le_bytes([f[0], f[1]]) as i64;
                if nch == 1 {
                    return a;
                }
                let (m, s) = to_mid_side(a, i16::from_le_bytes([f[2], f[3]]) as i64);
                if c == 0 {
                    m
                } else {
                    s
                }
            })
            .collect()
    }

    /// For frames `f0..` (their cascade residuals in `res`, frame-major) appends one plan per
    /// block and channel to `plans`: a block predictor where it beats the cascade (`switching`
    /// only), whose residuals then replace the cascade's. The blocks are independent, so with
    /// `parallel` they are analysed on the thread pool (the results do not depend on it).
    fn choose_blocks(w: &Wav, f0: usize, res: &mut [i32], plans: &mut Vec<Option<Lpc>>, switching: bool, parallel: bool) {
        let nch = w.channels;
        let end = f0 + res.len() / nch;
        let jobs: Vec<(usize, usize)> = (f0..end).step_by(lpc::BLOCK).flat_map(|b0| (0..nch).map(move |c| (b0, c))).collect();
        if !switching {
            plans.extend(jobs.iter().map(|_| None));
            return;
        }
        let cascade_res: &[i32] = res;
        let decide = |&(b0, c): &(usize, usize)| -> Option<(Lpc, Vec<i64>)> {
            let b1 = (b0 + lpc::BLOCK).min(end);
            let from = b0.saturating_sub(lpc::MAX_ORDER);
            let x = channel_values(w, c, from, b1);
            let cascade: Vec<i64> = (b0..b1).map(|f| cascade_res[(f - f0) * nch + c] as i64).collect();
            let plan = lpc::choose(&x, b0 - from, b1 - from, &cascade)?;
            let r = lpc::block_residuals(&plan, &x, b0 - from, b1 - from);
            Some((plan, r))
        };
        #[cfg(feature = "parallel")]
        let decisions: Vec<_> = if parallel {
            use rayon::prelude::*;
            jobs.par_iter().map(decide).collect()
        } else {
            jobs.iter().map(decide).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let decisions: Vec<_> = {
            let _ = parallel;
            jobs.iter().map(decide).collect()
        };
        for (&(b0, c), decision) in jobs.iter().zip(decisions) {
            plans.push(decision.map(|(plan, r)| {
                for (k, v) in r.into_iter().enumerate() {
                    res[(b0 + k - f0) * nch + c] = v as i32;
                }
                plan
            }));
        }
    }

    /// Where reconstruction takes its block headers and residuals from.
    pub(super) trait Source {
        /// Channel `c`'s header at the start of a block (`None` if corrupt).
        fn header(&mut self, c: usize) -> Option<Option<Lpc>>;
        /// Channel `c`'s next residual (`None` if corrupt).
        fn residual(&mut self, c: usize) -> Option<i64>;
    }

    /// Residuals in memory, with one plan per block and channel if `plans` is given
    /// (otherwise every block uses the cascade).
    pub(super) struct Stored<'a> {
        pub res: std::slice::Iter<'a, i32>,
        pub plans: Option<std::slice::Iter<'a, Option<Lpc>>>,
    }

    impl Source for Stored<'_> {
        fn header(&mut self, _: usize) -> Option<Option<Lpc>> {
            match &mut self.plans {
                Some(p) => p.next().cloned(),
                None => Some(None),
            }
        }

        fn residual(&mut self, _: usize) -> Option<i64> {
            self.res.next().map(|&r| r as i64)
        }
    }

    impl Source for rc::ResidualDecoder<'_> {
        fn header(&mut self, c: usize) -> Option<Option<Lpc>> {
            rc::ResidualDecoder::header(self, c)
        }

        fn residual(&mut self, c: usize) -> Option<i64> {
            Some(self.next(c))
        }
    }

    /// Reconstruction state: the cascade (or version-1 predictor), each channel's recent
    /// values and its block predictor for the current block.
    pub(super) struct Decoding<P> {
        p: P,
        nch: usize,
        hist: Vec<lpc::History>,
        plan: Vec<Option<Lpc>>,
        frame: usize,
    }

    impl<P: Predict> Decoding<P> {
        pub(super) fn new(p: P, nch: usize) -> Self {
            Decoding { p, nch, hist: vec![lpc::History::new(); nch], plan: vec![None; nch], frame: 0 }
        }

        /// Decodes the next `frames` frames from `src` and appends their little-endian PCM to
        /// `out`, checking that every sample is in range. The cascade sees every sample, also
        /// in blocks that use a block predictor.
        pub(super) fn run(&mut self, frames: usize, src: &mut impl Source, out: &mut Vec<u8>, progress: &mut dyn FnMut(usize) -> bool) -> Result<()> {
            let _fp = FpGuard::new();
            let bad = || Error::Format("corrupt audio residual stream".into());
            let nch = self.nch;
            let mut x = [0i64; 2];
            for f in 0..frames {
                if f % BLOCK_FRAMES == 0 && !progress(f) {
                    return Err(Error::Cancelled);
                }
                if self.frame.is_multiple_of(lpc::BLOCK) {
                    for (c, plan) in self.plan.iter_mut().enumerate() {
                        *plan = src.header(c).ok_or_else(bad)?;
                    }
                }
                for (c, xc) in x.iter_mut().enumerate().take(nch) {
                    let pr = self.p.predict(c);
                    let base = match &self.plan[c] {
                        Some(l) => l.predict(self.hist[c].get()),
                        None => self.p.prev(c) + pr,
                    };
                    let v = base + src.residual(c).ok_or_else(bad)?;
                    // Mid (or mono) samples fit 16 bits; the side channel needs 17.
                    let (lo, hi) = if c == 1 { (-65535, 65535) } else { (-32768, 32767) };
                    if !(lo..=hi).contains(&v) {
                        return Err(bad());
                    }
                    *xc = v;
                    self.p.update(c, v);
                    self.hist[c].push(v);
                }
                self.frame += 1;
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
            Ok(())
        }
    }

    /// Residuals, frame-major, and one block plan per block and channel (block-major).
    pub(super) type Checked = (Vec<i32>, Vec<Option<Lpc>>);

    /// The residuals of `w` with `preset` and the block plans (block predictors only with
    /// `switching`), if they decode back to its samples (`None` would mean a bug; the file
    /// then takes the plain neural path). With `parallel`, the check runs on a second thread a
    /// block behind the encoder; otherwise it follows it. `progress(checking, frames_done)`
    /// reports both phases; returning `false` cancels.
    pub(super) fn checked_residuals(
        preset: usize,
        w: &Wav,
        parallel: bool,
        switching: bool,
        progress: &mut dyn FnMut(bool, usize) -> bool,
    ) -> Result<Option<Checked>> {
        let nch = w.channels;
        let mut pred = Cascade::new(preset, nch);
        let mut plans: Vec<Option<Lpc>> = Vec::new();
        // Decodes one block of frames with `check` and compares it with the original samples.
        let verify = |check: &mut Decoding<Cascade>, f0: usize, block: &[i32], block_plans: &[Option<Lpc>], buf: &mut Vec<u8>| -> bool {
            buf.clear();
            let n = block.len() / nch;
            let mut src = Stored { res: block.iter(), plans: Some(block_plans.iter()) };
            check.run(n, &mut src, buf, &mut |_| true).is_ok() && buf[..] == w.pcm[f0 * 2 * nch..(f0 + n) * 2 * nch]
        };
        #[cfg(feature = "parallel")]
        if parallel {
            let (tx, rx) = std::sync::mpsc::channel::<(usize, Vec<i32>, Vec<Option<Lpc>>)>();
            return std::thread::scope(|s| {
                let checker = s.spawn(|| {
                    let mut check = Decoding::new(Cascade::new(preset, nch), nch);
                    let mut buf = Vec::new();
                    let mut ok = true;
                    for (f0, block, block_plans) in rx {
                        ok = ok && verify(&mut check, f0, &block, &block_plans, &mut buf);
                    }
                    ok
                });
                let res = residuals(&mut pred, w, &mut |f| progress(false, f), &mut |f0, block| {
                    let start = plans.len();
                    choose_blocks(w, f0, block, &mut plans, switching, true);
                    // A send fails only if the checker panicked; join() reports that.
                    let _ = tx.send((f0, block.to_vec(), plans[start..].to_vec()));
                });
                drop(tx);
                let ok = checker.join().unwrap_or(false);
                let res = res?;
                Ok(ok.then_some((res, plans)))
            });
        }
        let _ = parallel;
        let res = residuals(&mut pred, w, &mut |f| progress(false, f), &mut |f0, block| choose_blocks(w, f0, block, &mut plans, switching, false))?;
        let mut check = Decoding::new(Cascade::new(preset, nch), nch);
        let mut buf = Vec::new();
        let per_chunk = BLOCK_FRAMES / lpc::BLOCK * nch;
        for (i, block) in res.chunks(BLOCK_FRAMES * nch).enumerate() {
            if !progress(true, i * BLOCK_FRAMES) {
                return Err(Error::Cancelled);
            }
            let block_plans = &plans[(i * per_chunk).min(plans.len())..((i + 1) * per_chunk).min(plans.len())];
            if !verify(&mut check, i * BLOCK_FRAMES, block, block_plans, &mut buf) {
                return Ok(None);
            }
        }
        Ok(Some((res, plans)))
    }

    pub(super) fn compress(
        input: &[u8],
        crc: u32,
        opts: &CompressOptions,
        progress: &mut ProgressFn<'_>,
    ) -> Result<Option<Vec<u8>>> {
        let Some(wav) = parse(input) else { return Ok(None) };
        let (nch, frames) = (wav.channels, wav.frames);
        let preset = preset_for_level(opts.level);
        let coder = if opts.prior.is_some() { CODER_NEURAL } else { CODER_CM };
        // Progress: prediction, the round-trip check, then coding (most of the time when the
        // network codes the residuals).
        let total = input.len() as u64;
        let (a, b) = if coder == CODER_NEURAL { (total / 20, total / 10) } else { (total * 2 / 5, total * 4 / 5) };
        let span = |lo: u64, hi: u64, done: usize, of: usize| lo + (hi - lo) * done as u64 / of.max(1) as u64;

        let parallel = cfg!(feature = "parallel") && opts.threads != 1;
        // Block predictors need block headers, which only the context-mixing coder carries.
        let switching = coder == CODER_CM;
        let checked = checked_residuals(preset as usize, &wav, parallel, switching, &mut |check, f| {
            let (lo, hi) = if check { (a, b) } else if parallel { (0, b) } else { (0, a) };
            progress(span(lo, hi, f, frames), total)
        });
        let Some((res, plans)) = checked? else { return Ok(None) };

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
        let m = compress_with_progress(&meta, &meta_opts, &mut |_, _| true)?;
        let payload = if coder == CODER_NEURAL {
            compress_with_progress(&residual_bytes(&res), &inner, &mut |d, t| progress(span(b, total, d as usize, t as usize), total))?
        } else {
            let n = res.len();
            rc::encode(&res, nch, &plans, &mut |i| progress(span(b, total, i, n), total)).ok_or(Error::Cancelled)?
        };

        let mut out = Vec::with_capacity(m.len() + payload.len() + 32);
        format::write_prefix(&mut out, MODE_AUDIO, total, crc);
        out.extend_from_slice(&[VERSION_2, nch as u8, 16, preset, coder]);
        format::put_varint(&mut out, frames as u64);
        format::put_varint(&mut out, wav.header.len() as u64);
        format::put_varint(&mut out, m.len() as u64);
        out.extend_from_slice(&m);
        out.extend_from_slice(&payload);
        if opts.allow_stored && out.len() >= input.len() + 10 {
            // Noise-like audio: storing it costs less.
            let mut s = Vec::with_capacity(input.len() + 16);
            format::write_prefix(&mut s, MODE_STORED, total, crc);
            s.extend_from_slice(input);
            return Ok(Some(s));
        }
        progress(total, total);
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
        let version = byte(0)?;
        let (channels, bits) = (byte(1)? as usize, byte(2)?);
        if version != VERSION_1 && version != VERSION_2 {
            return Err(bad("unsupported audio stream version"));
        }
        if !(1..=2).contains(&channels) || bits != 16 {
            return Err(bad("unsupported audio sample layout"));
        }
        let (preset, coder, mut pos) = if version == VERSION_2 {
            let (p, c) = (byte(3)?, byte(4)?);
            if p as usize >= PRESETS.len() || c > CODER_NEURAL {
                return Err(bad("unsupported audio predictor or coder"));
            }
            (p as usize, c, 5)
        } else {
            (0, CODER_NEURAL, 3)
        };
        let frames = format::get_varint(body, &mut pos)?;
        let header_len = format::get_varint(body, &mut pos)?;
        let meta_len = format::get_varint(body, &mut pos)?;
        let pcm_len = frames
            .checked_mul(2 * channels as u64)
            .filter(|&n| n <= orig_len)
            .ok_or_else(|| bad("inconsistent audio sizes"))?;
        let meta_len = usize::try_from(meta_len).map_err(|_| bad("inconsistent audio sizes"))?;
        let meta_end = pos.checked_add(meta_len).filter(|&e| e <= body.len()).ok_or_else(|| bad("truncated audio stream"))?;
        let (meta_s, payload) = (&body[pos..meta_end], &body[meta_end..]);
        let nested = if coder == CODER_NEURAL { &[meta_s, payload][..] } else { &[meta_s][..] };
        for s in nested {
            if s.len() > 5 && s[5] != MODE_NEURAL && s[5] != MODE_STORED {
                return Err(bad("nested container streams are not allowed"));
            }
        }
        let meta = decompress_with_progress(meta_s, opts, &mut |_, _| true)?;
        if meta.len() as u64 != orig_len - pcm_len || header_len > meta.len() as u64 {
            return Err(bad("audio metadata has the wrong length"));
        }
        let (header_len, frames) = (header_len as usize, frames as usize);
        // The context-mixing coder can describe hours of silence in a few bytes, so grow the
        // output as samples arrive rather than trusting the header for the allocation.
        let mut out = Vec::with_capacity(meta.len() + (pcm_len as usize).min(1 << 26));
        out.extend_from_slice(&meta[..header_len]);
        let total = orig_len;
        if coder == CODER_NEURAL {
            let bytes = decompress_with_progress(payload, opts, progress)?;
            // Every sample takes at least two residual bytes.
            if pcm_len > bytes.len() as u64 {
                return Err(bad("audio residual stream is too short"));
            }
            let mut rpos = 0usize;
            let mut res = Vec::with_capacity(frames * channels);
            for _ in 0..frames * channels {
                res.push(get_residual(&bytes, &mut rpos).ok_or_else(|| bad("corrupt audio residual stream"))? as i32);
            }
            if rpos != bytes.len() {
                return Err(bad("corrupt audio residual stream"));
            }
            let mut src = Stored { res: res.iter(), plans: None };
            if version == VERSION_1 {
                Decoding::new(V1::new(channels), channels).run(frames, &mut src, &mut out, &mut |_| true)?;
            } else {
                Decoding::new(Cascade::new(preset, channels), channels).run(frames, &mut src, &mut out, &mut |_| true)?;
            }
        } else {
            let mut dec = rc::ResidualDecoder::new(payload, channels);
            Decoding::new(Cascade::new(preset, channels), channels).run(frames, &mut dec, &mut out, &mut |f| {
                progress(total * f as u64 / frames.max(1) as u64, total)
            })?;
            // The coder reads four bytes ahead and pads with 0xFF past the end.
            if dec.consumed() > payload.len() + 4 {
                return Err(bad("truncated audio residual stream"));
            }
        }
        out.extend_from_slice(&meta[header_len..]);
        progress(total, total);
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

/// The prediction residuals the audio mode codes for `data` when a prior is given, if it is
/// a 16-bit PCM WAV file (with the default level's predictor). Train an audio prior on these
/// with [`crate::train_prior`]; `piedpiper train` does this automatically for WAV inputs.
#[cfg(feature = "audio")]
pub fn residuals(data: &[u8]) -> Option<Vec<u8>> {
    let w = imp::parse(data)?;
    let preset = imp::preset_for_level(crate::config::DEFAULT_LEVEL) as usize;
    let res = imp::residuals(&mut lms::Cascade::new(preset, w.channels), &w, &mut |_| true, &mut |_, _| {}).ok()?;
    Some(imp::residual_bytes(&res))
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
    use super::imp::{checked_residuals, parse, residuals, Decoding, Stored};
    use super::lms::{Cascade, Predict, PRESETS, V1};
    use crate::format::MODE_AUDIO;
    use crate::rng::Rng;
    use crate::{
        compress, decompress, decompress_with_progress, inspect, train_prior, CompressOptions, DecompressOptions, ModelConfig, Prior, TrainOptions,
    };
    use std::sync::Arc;

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

    fn check_inverse<P: Predict>(data: &[u8], make: impl Fn(usize) -> P) {
        let w = parse(data).expect("parses");
        let res = residuals(&mut make(w.channels), &w, &mut |_| true, &mut |_, _| {}).unwrap();
        let mut back = w.header.to_vec();
        let mut src = Stored { res: res.iter(), plans: None };
        Decoding::new(make(w.channels), w.channels).run(w.frames, &mut src, &mut back, &mut |_| true).unwrap();
        back.extend_from_slice(w.trailer);
        assert_eq!(back, data);
    }

    /// A clean tone from an integer oscillator, then a stretch of the music signal.
    fn tone_then_music(frames: usize, stereo: bool) -> Vec<i16> {
        let (mut a, mut b) = (15000i64, 0i64);
        let mut out = Vec::new();
        for _ in 0..frames {
            a -= (b * 301) >> 12;
            b += (a * 301) >> 12;
            out.push(a as i16);
            if stereo {
                out.push((a / 2) as i16);
            }
        }
        out.extend(music(frames, stereo, 31));
        out
    }

    #[test]
    fn block_predictors_take_over_where_they_win() {
        for stereo in [false, true] {
            let ch = if stereo { 2 } else { 1 };
            let data = wav(ch, &tone_then_music(9000, stereo), &[], b"");
            let w = parse(&data).unwrap();
            for parallel in [false, true] {
                let (res, plans) = checked_residuals(1, &w, parallel, true, &mut |_, _| true).unwrap().expect("round trip");
                assert_eq!(res.len(), w.frames * w.channels);
                assert_eq!(plans.len(), w.frames.div_ceil(4096) * w.channels);
                assert!(plans.iter().any(|p| p.is_some()), "the tone takes a block predictor");
                assert!(plans.iter().any(|p| p.is_none()), "the cascade keeps the rest");
            }
            let c = compress(&data, &small_model()).unwrap();
            assert_eq!(c[5], MODE_AUDIO);
            assert_eq!(decompress(&c).unwrap(), data);
            let (_, plain) = checked_residuals(1, &w, false, false, &mut |_, _| true).unwrap().unwrap();
            assert!(plain.iter().all(|p| p.is_none()));
        }
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
                let data = wav(ch, &sig[..n], &[], b"");
                check_inverse(&data, V1::new);
                for preset in 0..PRESETS.len() {
                    check_inverse(&data, |c| Cascade::new(preset, c));
                }
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
    fn roundtrip_in_mode_5_with_metadata_at_every_preset() {
        let mut trailer = b"LIST\x16\x00\x00\x00INFOISFT\x0a\x00\x00\x00piedpiper\x00".to_vec();
        trailer.push(0x5A); // a dangling byte after the last chunk
        for (ch, frames) in [(2u16, 3000), (1, 5000)] {
            let data = wav(ch, &music(frames, ch == 2, 7), &[], &trailer);
            let mut sizes = Vec::new();
            for level in [1, 2, 4] {
                let c = compress(&data, &CompressOptions { level, ..small_model() }).unwrap();
                assert_eq!(c[5], MODE_AUDIO);
                let info = inspect(&c).unwrap();
                assert_eq!(info.tokenizer, super::NAME);
                assert!(info.model.is_none());
                assert_eq!(decompress(&c).unwrap(), data);
                sizes.push(c.len());
            }
            assert!(sizes[0] != sizes[1] && sizes[1] != sizes[2], "levels pick different predictors: {sizes:?}");
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
    fn a_prior_sends_the_residuals_to_the_network() {
        let train = wav(1, &music(3000, false, 21), &[], b"");
        let topts = TrainOptions {
            model: Some(ModelConfig { d_model: 32, n_layer: 1, n_head: 2, d_ff: 32, window: 16, seg_len: 8, n_streams: 2 }),
            ..Default::default()
        };
        let rep = train_prior(&super::residuals(&train).unwrap(), &topts, &mut |_, _| true).unwrap();
        let prior = Arc::new(Prior::from_bytes(&rep.prior).unwrap());
        let data = wav(1, &music(2500, false, 23), &[], b"");
        let opts = CompressOptions { prior: Some(prior.clone()), allow_stored: false, ..CompressOptions::level(2) };
        let c = compress(&data, &opts).unwrap();
        assert_eq!(c[5], MODE_AUDIO);
        assert_ne!(c, compress(&data, &CompressOptions { allow_stored: false, ..CompressOptions::level(2) }).unwrap());
        let back = decompress_with_progress(&c, &DecompressOptions { threads: 0, prior: Some(prior) }, &mut |_, _| true).unwrap();
        assert_eq!(back, data);
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
        for k in [7usize, 12, 14, 16, 20, c.len() / 2, c.len() * 3 / 4] {
            let mut bad = c.clone();
            bad[k] ^= 0x24;
            assert!(decompress(&bad).is_err(), "flip at {k}");
        }
        assert!(decompress(&c[..c.len() * 2 / 3]).is_err());
    }

    #[test]
    fn cancelling_stops_compression() {
        let data = wav(2, &music(200_000, true, 29), &[], b"");
        let mut calls = 0;
        let r = crate::compress_with_progress(&data, &small_model(), &mut |_, _| {
            calls += 1;
            calls < 3
        });
        assert!(matches!(r, Err(crate::Error::Cancelled)), "{:?}", r.err());
    }
}
