//! VP9 in Matroska/WebM: the `PPV9` recompressed format and its drivers.
//!
//! Layout: the magic `PPV9`, a header stream (varint length + arithmetic-coded data) with the
//! file length, the table of VP9 frames (Matroska blocks), the segment boundaries and every
//! byte outside those frames (container, audio). Then, per segment: one byte giving the
//! model's hash-table size, a varint length and an arithmetic-coded stream.
//!
//! In a segment every block is coded as its superframe index (raw bytes) followed by its
//! frames. A frame is either recoded — its two headers as bytes, then for each tile the
//! boolean decisions, coded with the context-mixing model (which gets libvpx's own
//! probability as one input), and a tail fix-up — or stored raw. A frame is recoded only after
//! re-encoding its decisions with libvpx's boolean coder has reproduced every tile, so
//! decompression always rebuilds the input exactly.
//!
//! Parsing a frame needs state from earlier frames (probability contexts, reference sizes,
//! the previous frame's motion vectors). Both sides therefore parse every frame, raw ones
//! included, from its original bytes. Segments start at key frames, where VP9 resets that
//! state, and are coded and decoded in parallel.

use super::bool::{BoolDecoder, BoolEncoder};
use super::tile::{kind, BoolIo, FrameParse, TileIo, TileParser};
use super::{compressed_header, end_frame, split_superframe, uncompressed_header, FrameCtx, FrameHeader, State, KEY_FRAME};
use crate::bits::{Bad, R};
use crate::container::{Cancelled, Options, Progress};
use crate::h264::mb::Hint;
use crate::mkv;
use crate::model::{
    cabac_p1, code_bin, code_byte, code_flag, code_num, ArithDec, ArithEnc, ByteCtx, Coder, Model, MAX_TABLE_BITS,
    MIN_TABLE_BITS,
};

pub const MAGIC: &[u8; 4] = b"PPV9";

pub(crate) const N_FILELEN: u32 = 1;
pub(crate) const N_NSAMP: u32 = 3;
pub(crate) const N_GAP: u32 = 4;
pub(crate) const N_SIZE: u32 = 5;
pub(crate) const N_NSEG: u32 = 11;
pub(crate) const N_SEGSTART: u32 = 12;
const N_INDEX: u32 = 20;
pub(crate) const N_HDRLEN: u32 = 21;
pub(crate) const N_DROP: u32 = 22;
pub(crate) const N_FIX: u32 = 23;
pub(crate) const F_REC: u32 = 30;
pub(crate) const F_COPY: u32 = 32;
pub(crate) const N_COPY: u32 = 24;
const F_RAW: u32 = 31;
/// Container gaps at least this long that look random (compressed audio) are stored at one
/// bit per bit instead of going through the model (same rule as the H.264 driver).
const RAW_GAP_MIN: usize = 512;

/// Longest tail (re-encoded bytes dropped, original bytes added) a tile may need.
pub(crate) const MAX_FIX: usize = 32;
/// Video data per segment; segments are cut only at key frames.
pub(crate) const SEG_TARGET: usize = 3 << 20;
pub(crate) const HEADER_BITS: u32 = 20;
const MAX_WORKERS: usize = 8;
/// Largest frame parsed, in 8x8 units (8192 x 8192 pixels).
const MAX_MIS: usize = 1 << 20;

/// Why a frame was stored raw.
pub const FAIL_NAMES: [&str; 5] = ["header/unsupported", "before first key frame", "tile parse", "re-encode mismatch", "show-existing"];
pub(crate) const FAIL_HEADER: usize = 0;
pub(crate) const FAIL_NO_KEY: usize = 1;
pub(crate) const FAIL_PARSE: usize = 2;
pub(crate) const FAIL_MISMATCH: usize = 3;
pub(crate) const FAIL_SHOW: usize = 4;

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub samples: usize,
    pub frames: usize,
    pub recoded: usize,
    pub tiles: usize,
    pub bools: u64,
    /// Original size of the recoded frames.
    pub frame_bytes: usize,
    pub raw_bytes: usize,
    pub other_bytes: usize,
    pub fails: [usize; 5],
    pub out_header: usize,
    pub out_other: usize,
    pub out_frames: usize,
    pub out_raw: usize,
    /// Frames coded as a copy of an identical earlier frame of the segment, their size and
    /// output.
    pub copies: usize,
    pub copy_bytes: usize,
    pub out_copy: usize,
    /// Cost of the recoded decisions under libvpx's probabilities and under the model, in
    /// bits (with `Options::stats`).
    pub vp9_bits: f64,
    pub model_bits: f64,
    /// Per symbol kind (index = kind - 40): (decisions, libvpx bits, model bits).
    pub kind_cost: Vec<(u64, f64, f64)>,
    pub segments: usize,
    /// AC-3 audio frames recoded between the video frames (MPEG-2 rips), their size and the
    /// size of their coded runs.
    pub audio_frames: usize,
    pub audio_bytes: usize,
    pub out_audio: usize,
}

impl Stats {
    pub(crate) fn merge(&mut self, o: &Stats) {
        self.frames += o.frames;
        self.recoded += o.recoded;
        self.tiles += o.tiles;
        self.bools += o.bools;
        self.frame_bytes += o.frame_bytes;
        self.raw_bytes += o.raw_bytes;
        for (a, b) in self.fails.iter_mut().zip(o.fails) {
            *a += b;
        }
        self.out_frames += o.out_frames;
        self.out_raw += o.out_raw;
        self.copies += o.copies;
        self.copy_bytes += o.copy_bytes;
        self.out_copy += o.out_copy;
        self.vp9_bits += o.vp9_bits;
        self.model_bits += o.model_bits;
        if self.kind_cost.len() < o.kind_cost.len() {
            self.kind_cost.resize(o.kind_cost.len(), (0, 0.0, 0.0));
        }
        for (a, b) in self.kind_cost.iter_mut().zip(&o.kind_cost) {
            a.0 += b.0;
            a.1 += b.1;
            a.2 += b.2;
        }
    }

    pub(crate) fn add_cost(&mut self, r: &Rec, p: u32) {
        if self.kind_cost.is_empty() {
            self.kind_cost = vec![(0, 0.0, 0.0); 24];
        }
        let p0 = r.prob as f64 / 256.0;
        let pv = if r.bit == 1 { 1.0 - p0 } else { p0 };
        let pm = if r.bit == 1 { p as f64 / 65536.0 } else { 1.0 - p as f64 / 65536.0 };
        if self.kind_cost.len() < 40 {
            self.kind_cost.resize(40, (0, 0.0, 0.0));
        }
        // tokens are split by tree node (24 + node)
        let slot = if r.h.kind == kind::TOKEN { 24 + (r.h.a as usize & 15).min(10) } else { (r.h.kind as usize).saturating_sub(40).min(23) };
        let kc = &mut self.kind_cost[slot];
        kc.0 += 1;
        kc.1 -= pv.log2();
        kc.2 -= pm.log2();
        self.vp9_bits -= pv.log2();
        self.model_bits -= pm.log2();
    }
}

/// CABAC state whose probability is closest (in odds) to each libvpx probability, so the
/// model can take libvpx's estimate as its direct input.
pub(crate) fn state_map() -> [u8; 256] {
    let mut t = [0u8; 256];
    for (prob, slot) in t.iter_mut().enumerate().skip(1) {
        let p1 = ((256 - prob) as u64) << 8;
        let mut best = (0u8, u128::MAX, 1u128);
        for s in 0..128u8 {
            let q = cabac_p1(s) as u64;
            let x = q as u128 * (65536 - p1) as u128;
            let y = p1 as u128 * (65536 - q) as u128;
            let (num, den) = if x >= y { (x, y.max(1)) } else { (y, x.max(1)) };
            if best.1 == u128::MAX || num * best.2 < best.1 * den {
                best = (s, num, den);
            }
        }
        *slot = best.0;
    }
    t[0] = t[1];
    t
}

#[derive(Clone, Copy)]
pub(crate) struct Rec {
    pub(crate) h: Hint,
    pub(crate) prob: u8,
    pub(crate) bit: u8,
}

struct TileRec {
    recs: Vec<Rec>,
    drop: usize,
    fix: Vec<u8>,
    dropped: Vec<u8>,
}

/// Decodes original tiles, re-encodes them and records every decision.
struct VerifyIo<'a> {
    f: &'a [u8],
    ranges: Vec<(usize, usize)>,
    dec: BoolDecoder<'a>,
    enc: BoolEncoder,
    recs: Vec<Rec>,
    tiles: Vec<TileRec>,
    exact: bool,
}

impl<'a> VerifyIo<'a> {
    fn new(f: &'a [u8], ranges: Vec<(usize, usize)>) -> Self {
        VerifyIo { f, ranges, dec: BoolDecoder::new(&[]), enc: BoolEncoder::new_raw(), recs: Vec::new(), tiles: Vec::new(), exact: true }
    }
}

impl BoolIo for VerifyIo<'_> {
    #[inline]
    fn read(&mut self, prob: u8, h: &Hint) -> R<u32> {
        if self.dec.overrun() {
            return Err(Bad);
        }
        let b = self.dec.read(prob);
        self.enc.write(b, prob);
        self.recs.push(Rec { h: *h, prob, bit: b as u8 });
        Ok(b)
    }
}

impl TileIo for VerifyIo<'_> {
    fn begin(&mut self, t: usize) -> R<()> {
        let (o, l) = self.ranges[t];
        let f = self.f;
        self.dec = BoolDecoder::new(&f[o..o + l]);
        self.enc = BoolEncoder::new_raw();
        self.recs = Vec::with_capacity(l * 6);
        Ok(())
    }

    fn end(&mut self, t: usize) -> R<()> {
        let (o, l) = self.ranges[t];
        let orig = &self.f[o..o + l];
        let data = std::mem::replace(&mut self.enc, BoolEncoder::new_raw()).finish();
        let common = data.iter().zip(orig).take_while(|(a, b)| a == b).count();
        let drop = data.len() - common;
        let fix = orig[common..].to_vec();
        if drop > MAX_FIX || fix.len() > MAX_FIX {
            self.exact = false;
        }
        self.tiles.push(TileRec { recs: std::mem::take(&mut self.recs), drop, fix, dropped: data[common..].to_vec() });
        Ok(())
    }
}

/// Decodes original tiles (to keep the parse state in step for raw frames).
struct ReadIo<'a> {
    f: &'a [u8],
    ranges: Vec<(usize, usize)>,
    dec: BoolDecoder<'a>,
}

impl BoolIo for ReadIo<'_> {
    #[inline]
    fn read(&mut self, prob: u8, _h: &Hint) -> R<u32> {
        if self.dec.overrun() {
            return Err(Bad);
        }
        Ok(self.dec.read(prob))
    }
}

impl TileIo for ReadIo<'_> {
    fn begin(&mut self, t: usize) -> R<()> {
        let (o, l) = self.ranges[t];
        let f = self.f;
        self.dec = BoolDecoder::new(&f[o..o + l]);
        Ok(())
    }

    fn end(&mut self, _t: usize) -> R<()> {
        Ok(())
    }
}

/// Decodes decisions from the model and re-encodes them with libvpx's boolean coder.
struct ModelIo<'a, 'd> {
    m: &'a mut Model,
    c: &'a mut ArithDec<'d>,
    map: &'a [u8; 256],
    enc: BoolEncoder,
    tiles: Vec<Vec<u8>>,
}

impl BoolIo for ModelIo<'_, '_> {
    #[inline]
    fn read(&mut self, prob: u8, h: &Hint) -> R<u32> {
        if self.c.overrun() {
            return Err(Bad);
        }
        let b = code_bin(self.m, self.c, h, Some(self.map[prob as usize]), 0);
        self.enc.write(b, prob);
        Ok(b)
    }
}

impl TileIo for ModelIo<'_, '_> {
    fn begin(&mut self, _t: usize) -> R<()> {
        self.enc = BoolEncoder::new_raw();
        Ok(())
    }

    fn end(&mut self, _t: usize) -> R<()> {
        let mut data = std::mem::replace(&mut self.enc, BoolEncoder::new_raw()).finish();
        let drop = num(self.m, self.c, N_DROP, MAX_FIX)?;
        let nfix = num(self.m, self.c, N_FIX, MAX_FIX)?;
        if drop > data.len() {
            return Err(Bad);
        }
        let dropped = data.split_off(data.len() - drop);
        let mut fix = Vec::new();
        code_fix(self.m, self.c, &dropped, &mut fix, nfix);
        data.extend_from_slice(&fix);
        self.tiles.push(data);
        Ok(())
    }
}

/// Codes the replacement tail bytes of a tile, with the re-encoded bytes as context.
pub(crate) fn code_fix<C: Coder>(m: &mut Model, c: &mut C, dropped: &[u8], fix: &mut Vec<u8>, n: usize) {
    for j in 0..n {
        let ours = dropped.get(j).copied().unwrap_or(0) as u32;
        let want = fix.get(j).copied().unwrap_or(0);
        let mut c0 = 1u32;
        for i in (0..8).rev() {
            let bit = ((want >> i) & 1) as u32;
            let h = Hint { ctx: 0, kind: kind::FIX, a: c0 | ours << 8, b: j.min(3) as u32, c: 0, d: 0 };
            let b = code_bin(m, c, &h, None, bit);
            c0 = (c0 << 1) | b;
        }
        if j < fix.len() {
            fix[j] = c0 as u8;
        } else {
            fix.push(c0 as u8);
        }
    }
}

/// Codes frame header bytes with the previous recoded frame's header as context.
pub(crate) fn code_hdr<C: Coder>(m: &mut Model, c: &mut C, prev: &[u8], bytes: &mut [u8]) {
    let mut last = 0u32;
    for (i, byte) in bytes.iter_mut().enumerate() {
        let p = prev.get(i).copied().unwrap_or(0) as u32;
        let mut c0 = 1u32;
        for k in (0..8).rev() {
            let bit = ((*byte >> k) & 1) as u32;
            let h = Hint { ctx: 0, kind: kind::HDR, a: c0 | p << 8 | (i.min(63) as u32) << 16, b: c0 | last << 8, c: c0 | p << 8, d: 0 };
            let b = code_bin(m, c, &h, None, bit);
            c0 = (c0 << 1) | b;
        }
        *byte = c0 as u8;
        last = c0 & 0xFF;
    }
}

/// True if `b`'s byte histogram is so close to uniform that the model would save under
/// about 1% (chi-square against uniform; random data averages 255).
pub(crate) fn looks_random(b: &[u8]) -> bool {
    let mut counts = [0u32; 256];
    for &x in b {
        counts[x as usize] += 1;
    }
    let n = b.len() as u64;
    let sum_sq: u64 = counts.iter().map(|&c| c as u64 * c as u64).sum();
    let chi2 = (256 * sum_sq / n.max(1)).saturating_sub(n);
    chi2 < 255 + n / 9
}

pub(crate) fn put_gap(m: &mut Model, enc: &mut ArithEnc, bc: &mut ByteCtx, gap: &[u8]) {
    if gap.len() >= RAW_GAP_MIN {
        let raw = looks_random(gap);
        code_flag(m, enc, F_RAW, 0, raw as u32);
        if raw {
            for &b in gap {
                for i in (0..8).rev() {
                    enc.encode(((b >> i) & 1) as u32, 1 << 15);
                }
            }
            return;
        }
    }
    for &b in gap {
        code_byte(m, enc, kind::RAW, bc, b);
    }
}

pub(crate) fn get_gap(m: &mut Model, c: &mut ArithDec, bc: &mut ByteCtx, gap: &mut [u8]) {
    if gap.len() >= RAW_GAP_MIN && code_flag(m, c, F_RAW, 0, 0) == 1 {
        for b in gap.iter_mut() {
            let mut v = 0u32;
            for _ in 0..8 {
                v = (v << 1) | c.decode(1 << 15);
            }
            *b = v as u8;
        }
        return;
    }
    for b in gap.iter_mut() {
        *b = code_byte(m, c, kind::RAW, bc, 0);
    }
}

pub(crate) fn code_raw<C: Coder>(m: &mut Model, c: &mut C, rb: &mut ByteCtx, bytes: &mut [u8]) {
    for b in bytes.iter_mut() {
        *b = code_byte(m, c, kind::RAW, rb, *b);
    }
}

/// Decodes a number and checks it against `max` before narrowing it.
pub(crate) fn num(m: &mut Model, c: &mut ArithDec, id: u32, max: usize) -> R<usize> {
    let v = code_num(m, c, id, 0);
    if v > max as u64 {
        return Err(Bad);
    }
    Ok(v as usize)
}

/// Parse state of a segment.
#[derive(Default)]
struct SegState {
    st: State,
    /// False until a key frame parses, and after any frame fails to parse.
    valid: bool,
    /// Header bytes of the last recoded frame (model context).
    prev_hdr: Vec<u8>,
}

/// Parses both frame headers from `f` (which must hold at least the headers). `None` for a
/// frame that only shows an earlier one.
pub(crate) fn frame_headers(st: &mut State, f: &[u8]) -> R<Option<(FrameHeader, FrameCtx)>> {
    let mut h = uncompressed_header(st, f)?;
    if h.show_existing {
        return Ok(None);
    }
    if h.mi_rows() * h.mi_cols() > MAX_MIS {
        return Err(Bad);
    }
    let start = h.uncomp_bytes;
    let comp = f.get(start..start + h.comp_bytes).ok_or(Bad)?;
    let mut fc = st.ctxs[h.ctx_idx].clone();
    compressed_header(&mut h, &mut fc, comp)?;
    Ok(Some((h, fc)))
}

/// Parses every tile of a frame through `io` and updates the state carried to later frames.
pub(crate) fn parse_tiles<T: TileIo>(st: &mut State, h: &FrameHeader, fc: &FrameCtx, io: &mut T) -> R<()> {
    let mut fp = FrameParse::new(h);
    let cols = 1usize << h.log2_tile_cols;
    let rows = 1usize << h.log2_tile_rows;
    {
        let n = fp.mi_rows * fp.mi_cols;
        let prev = (h.use_prev_mvs && st.prev_mvs.len() == n).then_some(&st.prev_mvs[..]);
        let colo = (st.prev_grid.len() == n).then_some((&st.prev_grid[..], &st.prev_eob));
        for r in 0..rows {
            for c in 0..cols {
                let t = r * cols + c;
                io.begin(t)?;
                TileParser::new(h, fc, &mut fp, prev, colo, r, c).run(io)?;
                io.end(t)?;
            }
        }
    }
    let mvs = std::mem::take(&mut fp.frame_mvs);
    end_frame(st, h, fc, mvs);
    if h.show_frame {
        st.prev_grid = std::mem::take(&mut fp.grid);
        st.prev_eob = std::mem::take(&mut fp.eob_map);
    }
    Ok(())
}

/// Parses an original frame. The encoder calls this for every frame (recording through a
/// [`VerifyIo`]); the decoder calls it for frames stored raw, so both sides' state evolves
/// identically, failures included.
fn parse_frame<T: TileIo>(ss: &mut SegState, f: &[u8], make: impl FnOnce(Vec<(usize, usize)>) -> T) -> Result<(FrameHeader, T), usize> {
    let (h, fc) = match frame_headers(&mut ss.st, f) {
        Err(_) => {
            ss.valid = false;
            return Err(FAIL_HEADER);
        }
        Ok(None) => return Err(FAIL_SHOW),
        Ok(Some(x)) => x,
    };
    if h.frame_type == KEY_FRAME {
        ss.valid = true;
    }
    if !ss.valid {
        return Err(FAIL_NO_KEY);
    }
    let ranges = match FrameParse::tile_ranges(&h, f, h.uncomp_bytes + h.comp_bytes) {
        Ok(r) => r,
        Err(_) => {
            ss.valid = false;
            return Err(FAIL_PARSE);
        }
    };
    let mut io = make(ranges);
    if parse_tiles(&mut ss.st, &h, &fc, &mut io).is_err() {
        ss.valid = false;
        return Err(FAIL_PARSE);
    }
    Ok((h, io))
}

/// Frame sizes from a superframe index (as [`split_superframe`] reads it).
fn index_sizes(idx: &[u8], data_len: usize) -> Option<Vec<usize>> {
    let marker = *idx.last()?;
    let frames = (marker & 7) as usize + 1;
    let mag = ((marker >> 3) & 3) as usize + 1;
    if marker & 0xE0 != 0xC0 || idx.len() != 2 + mag * frames || idx[0] != marker {
        return None;
    }
    let sizes: Vec<usize> = (0..frames).map(|i| (0..mag).fold(0usize, |a, k| a | (idx[1 + i * mag + k] as usize) << (8 * k))).collect();
    (sizes.iter().sum::<usize>() == data_len).then_some(sizes)
}

fn is_keyframe(smp: &[u8]) -> bool {
    smp.len() > 4 && smp[0] & 0xFC == 0x80 && smp[1..4] == [0x49, 0x83, 0x42]
}

struct SegPlan {
    first: usize,
    end: usize,
}

fn plan_segments(input: &[u8], samples: &[(u64, u32)]) -> Vec<SegPlan> {
    let mut plans = Vec::new();
    let mut start = 0;
    let mut acc = 0usize;
    for (i, &(o, s)) in samples.iter().enumerate() {
        if i > start && acc >= SEG_TARGET && is_keyframe(&input[o as usize..o as usize + s as usize]) {
            plans.push(SegPlan { first: start, end: i });
            start = i;
            acc = 0;
        }
        acc += s as usize;
    }
    plans.push(SegPlan { first: start, end: samples.len() });
    plans
}

pub(crate) fn auto_table_bits(len: usize) -> u32 {
    if len <= 2_000_000 {
        22
    } else {
        23
    }
}

pub(crate) fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

pub(crate) fn get_varint(b: &[u8], pos: &mut usize) -> R<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let x = *b.get(*pos).ok_or(Bad)?;
        *pos += 1;
        v |= ((x & 0x7F) as u64) << shift;
        if x & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err(Bad)
}

fn worker_count(threads: usize, jobs: usize) -> usize {
    if cfg!(target_family = "wasm") {
        return 1;
    }
    let t = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()).min(MAX_WORKERS) } else { threads };
    t.min(jobs).max(1)
}

pub(crate) type Job<'a, T> = dyn Fn(usize, &mut dyn FnMut(u64) -> bool) -> T + Sync + 'a;

/// Runs `job(i, tick)` for `i` in `0..n` on up to `threads` threads and returns the results
/// in order; progress is reported from the calling thread (same scheme as the H.264 driver).
pub(crate) fn run_jobs<T: Send>(n: usize, threads: usize, total: u64, base: u64, progress: &mut Progress, job: &Job<T>) -> Vec<T> {
    let workers = worker_count(threads, n);
    if workers <= 1 {
        let mut done = base;
        let mut keep = true;
        return (0..n)
            .map(|i| {
                job(i, &mut |b| {
                    done += b;
                    keep = keep && progress(done.min(total), total);
                    keep
                })
            })
            .collect();
    }
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::Mutex;
    let next = AtomicUsize::new(0);
    let done = AtomicU64::new(base);
    let cancel = AtomicBool::new(false);
    let slots: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                sc.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= n {
                        break;
                    }
                    let r = job(i, &mut |b| {
                        done.fetch_add(b, Ordering::Relaxed);
                        !cancel.load(Ordering::Relaxed)
                    });
                    *slots[i].lock().unwrap() = Some(r);
                })
            })
            .collect();
        while !handles.iter().all(|h| h.is_finished()) {
            if !progress(done.load(Ordering::Relaxed).min(total), total) {
                cancel.store(true, Ordering::Relaxed);
            }
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
    });
    slots.into_iter().map(|m| m.into_inner().unwrap().expect("worker finished its job")).collect()
}

fn find_frames(input: &[u8]) -> Option<Vec<(u64, u32)>> {
    let (_, samples) = mkv::find_track(input, b"V_VP9")?;
    let mut end = 0u64;
    for &(o, s) in &samples {
        if o < end || o + s as u64 > input.len() as u64 {
            return None;
        }
        end = o + s as u64;
    }
    Some(samples)
}

/// True if `input` is a Matroska/WebM file with a VP9 video track (parses only the
/// container).
pub fn probe(input: &[u8]) -> bool {
    find_frames(input).is_some()
}

/// Recompresses a WebM/Matroska file whose video track is VP9. `None` when the file is not
/// one this module handles or no frame could be recoded.
pub fn compress(input: &[u8]) -> Option<(Vec<u8>, Stats)> {
    compress_with(input, &Options::default(), &mut |_, _| true).ok().flatten()
}

/// Like [`compress`], with options and progress reporting.
pub fn compress_with(input: &[u8], opts: &Options, progress: &mut Progress) -> Result<Option<(Vec<u8>, Stats)>, Cancelled> {
    let Some(samples) = find_frames(input) else { return Ok(None) };
    let total = input.len() as u64;
    let plans = plan_segments(input, &samples);
    let mut st = Stats { samples: samples.len(), segments: plans.len(), ..Default::default() };
    let mut m = Model::new(HEADER_BITS);
    let mut enc = ArithEnc::new();
    code_num(&mut m, &mut enc, N_FILELEN, input.len() as u64);
    code_num(&mut m, &mut enc, N_NSAMP, samples.len() as u64);
    let mut prev = 0u64;
    for &(o, s) in &samples {
        code_num(&mut m, &mut enc, N_GAP, o - prev);
        code_num(&mut m, &mut enc, N_SIZE, s as u64);
        prev = o + s as u64;
    }
    code_num(&mut m, &mut enc, N_NSEG, plans.len() as u64);
    for w in plans.windows(2) {
        code_num(&mut m, &mut enc, N_SEGSTART, (w[1].first - w[0].first) as u64);
    }
    st.out_header = enc.out.len();
    let mut bc = ByteCtx::default();
    let mut pos = 0usize;
    for &(o, s) in &samples {
        put_gap(&mut m, &mut enc, &mut bc, &input[pos..o as usize]);
        st.other_bytes += o as usize - pos;
        pos = (o + s as u64) as usize;
    }
    put_gap(&mut m, &mut enc, &mut bc, &input[pos..]);
    st.other_bytes += input.len() - pos;
    let header = enc.finish();
    st.out_other = header.len() - st.out_header;
    if !progress(st.other_bytes as u64, total) {
        return Err(Cancelled);
    }
    let map = state_map();
    let results = run_jobs(plans.len(), opts.threads, total, st.other_bytes as u64, progress, &|i, tick| {
        let plan = &plans[i];
        let size: usize = samples[plan.first..plan.end].iter().map(|&(_, s)| s as usize).sum();
        let bits = opts.table_bits.unwrap_or_else(|| auto_table_bits(size)).clamp(MIN_TABLE_BITS, MAX_TABLE_BITS);
        encode_segment(input, &samples[plan.first..plan.end], bits, opts.stats, &map, tick).map(|(d, s)| (bits, d, s))
    });
    let mut out = MAGIC.to_vec();
    put_varint(&mut out, header.len() as u64);
    out.extend_from_slice(&header);
    for r in results {
        let (bits, data, s) = r?;
        out.push(bits as u8);
        put_varint(&mut out, data.len() as u64);
        out.extend_from_slice(&data);
        st.merge(&s);
    }
    progress(total, total);
    if st.recoded + st.copies == 0 {
        return Ok(None);
    }
    Ok(Some((out, st)))
}

fn encode_segment(
    input: &[u8],
    samples: &[(u64, u32)],
    bits: u32,
    stats: bool,
    map: &[u8; 256],
    tick: &mut dyn FnMut(u64) -> bool,
) -> Result<(Vec<u8>, Stats), Cancelled> {
    let mut st = Stats::default();
    let mut ss = SegState::default();
    let mut m = Model::new(bits);
    let mut enc = ArithEnc::new();
    let mut rb = ByteCtx::default();
    let mut last_flag = 0u32;
    let mut last_copy = 0u32;
    // the latest frame number of each distinct frame content seen in this segment
    let mut seen: std::collections::HashMap<&[u8], usize> = std::collections::HashMap::new();
    let mut fno = 0usize;
    for &(o, s) in samples {
        if !tick(s as u64) {
            return Err(Cancelled);
        }
        let smp = &input[o as usize..o as usize + s as usize];
        let (frames, idx_len) = split_superframe(smp);
        code_num(&mut m, &mut enc, N_INDEX, idx_len as u64);
        let mut idx = smp[smp.len() - idx_len..].to_vec();
        code_raw(&mut m, &mut enc, &mut rb, &mut idx);
        for (fo, fl) in frames {
            let f = &smp[fo..fo + fl];
            st.frames += 1;
            let before = enc.out.len();
            let earlier = seen.insert(f, fno);
            fno += 1;
            code_flag(&mut m, &mut enc, F_COPY, last_copy, earlier.is_some() as u32);
            last_copy = earlier.is_some() as u32;
            if let Some(k) = earlier {
                // an identical earlier frame: code the distance, and parse the frame only to
                // keep the state in step (the decoder does the same)
                code_num(&mut m, &mut enc, N_COPY, (fno - 2 - k) as u64);
                let _ = parse_frame(&mut ss, f, |ranges| ReadIo { f, ranges, dec: BoolDecoder::new(&[]) });
                st.copies += 1;
                st.copy_bytes += fl;
                st.out_copy += enc.out.len() - before;
                continue;
            }
            match parse_frame(&mut ss, f, |ranges| VerifyIo::new(f, ranges)) {
                Ok((h, io)) if io.exact => {
                    code_flag(&mut m, &mut enc, F_REC, last_flag, 1);
                    last_flag = 1;
                    let hlen = h.uncomp_bytes + h.comp_bytes;
                    code_num(&mut m, &mut enc, N_HDRLEN, hlen as u64);
                    let mut hdr = f[..hlen].to_vec();
                    code_hdr(&mut m, &mut enc, &ss.prev_hdr, &mut hdr);
                    ss.prev_hdr = hdr;
                    for t in &io.tiles {
                        for r in &t.recs {
                            let p = m.predict(&r.h, Some(map[r.prob as usize]));
                            enc.encode(r.bit as u32, p);
                            m.update(r.bit as u32);
                            if stats {
                                st.add_cost(r, p);
                            }
                        }
                        st.bools += t.recs.len() as u64;
                        code_num(&mut m, &mut enc, N_DROP, t.drop as u64);
                        code_num(&mut m, &mut enc, N_FIX, t.fix.len() as u64);
                        let mut fix = t.fix.clone();
                        code_fix(&mut m, &mut enc, &t.dropped, &mut fix, t.fix.len());
                    }
                    st.tiles += io.tiles.len();
                    st.recoded += 1;
                    st.frame_bytes += fl;
                    st.out_frames += enc.out.len() - before;
                }
                res => {
                    st.fails[res.err().unwrap_or(FAIL_MISMATCH)] += 1;
                    code_flag(&mut m, &mut enc, F_REC, last_flag, 0);
                    last_flag = 0;
                    let mut raw = f.to_vec();
                    code_raw(&mut m, &mut enc, &mut rb, &mut raw);
                    st.raw_bytes += fl;
                    st.out_raw += enc.out.len() - before;
                }
            }
        }
    }
    Ok((enc.finish(), st))
}

/// Rebuilds the original file from [`compress`] output.
pub fn decompress(data: &[u8]) -> R<Vec<u8>> {
    decompress_with(data, 0, &mut |_, _| true)
}

pub(crate) type Region<'a> = std::sync::Mutex<Option<(&'a mut [u8], usize, usize, usize)>>;

/// Like [`decompress`], with a thread count (0 = all cores; never changes the result) and
/// progress reporting (returning `false` aborts with an error).
pub fn decompress_with(data: &[u8], threads: usize, progress: &mut Progress) -> R<Vec<u8>> {
    if data.len() < 5 || &data[..4] != MAGIC {
        return Err(Bad);
    }
    let mut pos = 4;
    let hlen = usize::try_from(get_varint(data, &mut pos)?).map_err(|_| Bad)?;
    let header = data.get(pos..pos.checked_add(hlen).ok_or(Bad)?).ok_or(Bad)?;
    pos += hlen;
    let mut c = ArithDec::new(header);
    let mut m = Model::new(HEADER_BITS);
    let file_len = code_num(&mut m, &mut c, N_FILELEN, 0);
    if file_len > 1 << 40 {
        return Err(Bad);
    }
    let file_len = usize::try_from(file_len).map_err(|_| Bad)?;
    let n = num(&mut m, &mut c, N_NSAMP, file_len)?;
    if n == 0 {
        return Err(Bad);
    }
    let mut samples = Vec::with_capacity(n);
    let mut prev = 0u64;
    for _ in 0..n {
        let gap = code_num(&mut m, &mut c, N_GAP, 0);
        let size = code_num(&mut m, &mut c, N_SIZE, 0);
        let o = prev.checked_add(gap).ok_or(Bad)?;
        prev = o.checked_add(size).ok_or(Bad)?;
        if prev > file_len as u64 || size == 0 {
            return Err(Bad);
        }
        samples.push((o as usize, size as usize));
    }
    let nseg = num(&mut m, &mut c, N_NSEG, n)?;
    if nseg == 0 {
        return Err(Bad);
    }
    let mut starts = vec![0usize];
    for _ in 1..nseg {
        let d = num(&mut m, &mut c, N_SEGSTART, n)?;
        let s = starts.last().copied().unwrap_or(0) + d;
        if d == 0 || s >= n {
            return Err(Bad);
        }
        starts.push(s);
    }
    let mut out = vec![0u8; file_len];
    let mut bc = ByteCtx::default();
    let mut p = 0usize;
    for &(o, s) in &samples {
        get_gap(&mut m, &mut c, &mut bc, &mut out[p..o]);
        p = o + s;
    }
    get_gap(&mut m, &mut c, &mut bc, &mut out[p..]);
    if c.overrun() {
        return Err(Bad);
    }
    let mut segs = Vec::with_capacity(nseg);
    for _ in 0..nseg {
        let bits = *data.get(pos).ok_or(Bad)? as u32;
        pos += 1;
        if !(MIN_TABLE_BITS..=MAX_TABLE_BITS).contains(&bits) {
            return Err(Bad);
        }
        let len = usize::try_from(get_varint(data, &mut pos)?).map_err(|_| Bad)?;
        segs.push((bits, data.get(pos..pos.checked_add(len).ok_or(Bad)?).ok_or(Bad)?));
        pos += len;
    }
    let other = (file_len - samples.iter().map(|&(_, s)| s).sum::<usize>()) as u64;
    let mut regions: Vec<Region> = Vec::with_capacity(nseg);
    let mut rest: &mut [u8] = &mut out;
    let mut consumed = 0usize;
    for k in 0..nseg {
        let (first, end) = (starts[k], if k + 1 < nseg { starts[k + 1] } else { n });
        let lo = samples[first].0;
        let hi = samples[end - 1].0 + samples[end - 1].1;
        let (_, tail) = std::mem::take(&mut rest).split_at_mut(lo - consumed);
        let (region, tail) = tail.split_at_mut(hi - lo);
        rest = tail;
        consumed = hi;
        regions.push(std::sync::Mutex::new(Some((region, lo, first, end))));
    }
    let map = state_map();
    let results = run_jobs(nseg, threads, file_len as u64, other, progress, &|k, tick| {
        let taken = regions[k].lock().unwrap().take();
        let (region, lo, first, end) = taken.expect("each region is decoded once");
        let (bits, seg) = segs[k];
        decode_segment(seg, bits, region, lo, &samples[first..end], &map, tick)
    });
    for r in results {
        r?;
    }
    Ok(out)
}

fn decode_segment(
    data: &[u8],
    bits: u32,
    out: &mut [u8],
    base: usize,
    samples: &[(usize, usize)],
    map: &[u8; 256],
    tick: &mut dyn FnMut(u64) -> bool,
) -> R<()> {
    let mut c = ArithDec::new(data);
    let mut m = Model::new(bits);
    let mut ss = SegState::default();
    let mut rb = ByteCtx::default();
    let mut last_flag = 0u32;
    let mut last_copy = 0u32;
    // (offset in `out`, length) of every frame of the segment so far
    let mut placed: Vec<(usize, usize)> = Vec::new();
    for &(o, s) in samples {
        if !tick(s as u64) {
            return Err(Bad);
        }
        let idx_len = num(&mut m, &mut c, N_INDEX, s)?;
        let mut idx = vec![0u8; idx_len];
        code_raw(&mut m, &mut c, &mut rb, &mut idx);
        let sizes = if idx_len == 0 { vec![s] } else { index_sizes(&idx, s - idx_len).ok_or(Bad)? };
        let mut w = o - base;
        for size in sizes {
            let copy = code_flag(&mut m, &mut c, F_COPY, last_copy, 0);
            last_copy = copy;
            if copy == 1 {
                let d = num(&mut m, &mut c, N_COPY, placed.len())?;
                let k = placed.len().checked_sub(d + 1).ok_or(Bad)?;
                let (po, pl) = placed[k];
                if pl != size {
                    return Err(Bad);
                }
                out.copy_within(po..po + pl, w);
                let raw: &[u8] = &out[w..w + size];
                let _ = parse_frame(&mut ss, raw, |ranges| ReadIo { f: raw, ranges, dec: BoolDecoder::new(&[]) });
                placed.push((w, size));
                w += size;
                continue;
            }
            let rec = code_flag(&mut m, &mut c, F_REC, last_flag, 0);
            last_flag = rec;
            if rec == 1 {
                let frame = decode_frame(&mut ss, &mut m, &mut c, map, size)?;
                out[w..w + size].copy_from_slice(&frame);
            } else {
                let dst = &mut out[w..w + size];
                code_raw(&mut m, &mut c, &mut rb, dst);
                let raw: &[u8] = dst;
                let _ = parse_frame(&mut ss, raw, |ranges| ReadIo { f: raw, ranges, dec: BoolDecoder::new(&[]) });
            }
            if c.overrun() {
                return Err(Bad);
            }
            placed.push((w, size));
            w += size;
        }
        out[w..w + idx_len].copy_from_slice(&idx);
    }
    Ok(())
}

/// Rebuilds a recoded frame of `size` bytes.
fn decode_frame(ss: &mut SegState, m: &mut Model, c: &mut ArithDec, map: &[u8; 256], size: usize) -> R<Vec<u8>> {
    let hlen = num(m, c, N_HDRLEN, size)?;
    let mut hdr = vec![0u8; hlen];
    code_hdr(m, c, &ss.prev_hdr, &mut hdr);
    let (h, fc) = frame_headers(&mut ss.st, &hdr)?.ok_or(Bad)?;
    if h.uncomp_bytes + h.comp_bytes != hlen {
        return Err(Bad);
    }
    if h.frame_type == KEY_FRAME {
        ss.valid = true;
    }
    if !ss.valid {
        return Err(Bad);
    }
    let mut io = ModelIo { m, c, map, enc: BoolEncoder::new_raw(), tiles: Vec::new() };
    parse_tiles(&mut ss.st, &h, &fc, &mut io)?;
    let mut frame = hdr.clone();
    let n = io.tiles.len();
    for (i, t) in io.tiles.iter().enumerate() {
        if i + 1 < n {
            frame.extend_from_slice(&u32::try_from(t.len()).map_err(|_| Bad)?.to_be_bytes());
        }
        frame.extend_from_slice(t);
    }
    ss.prev_hdr = hdr;
    if frame.len() != size {
        return Err(Bad);
    }
    Ok(frame)
}

/// Development aid: (repeated tiles, their bytes, all tile bytes) over the parsable frames.
pub fn tile_repeats(input: &[u8]) -> (usize, usize, usize) {
    let Some(samples) = find_frames(input) else { return (0, 0, 0) };
    let mut ss = SegState::default();
    let mut seen = std::collections::HashSet::new();
    let (mut n, mut bytes, mut total) = (0, 0, 0);
    for &(o, s) in &samples {
        let smp = &input[o as usize..o as usize + s as usize];
        let (frames, _) = split_superframe(smp);
        for (fo, fl) in frames {
            let f = &smp[fo..fo + fl];
            if let Ok((_, io)) = parse_frame(&mut ss, f, |ranges| ReadIo { f, ranges, dec: BoolDecoder::new(&[]) }) {
                for &(to, tl) in &io.ranges {
                    total += tl;
                    if !seen.insert(&f[to..to + tl]) {
                        n += 1;
                        bytes += tl;
                    }
                }
            }
        }
    }
    (n, bytes, total)
}

/// Development aid: per-frame outcome of the parse (no model).
pub fn diagnose(input: &[u8], limit: usize) -> Vec<String> {
    let Some(samples) = find_frames(input) else { return vec!["no VP9 track".into()] };
    let mut out = Vec::new();
    let mut ss = SegState::default();
    let mut counts = [0usize; 6];
    let mut fno = 0usize;
    for &(o, s) in &samples {
        let smp = &input[o as usize..o as usize + s as usize];
        let (frames, _) = split_superframe(smp);
        for (fo, fl) in frames {
            let f = &smp[fo..fo + fl];
            let r = parse_frame(&mut ss, f, |ranges| VerifyIo::new(f, ranges));
            let (slot, msg) = match &r {
                Ok((h, io)) => {
                    let bools: usize = io.tiles.iter().map(|t| t.recs.len()).sum();
                    let bits: f64 = io
                        .tiles
                        .iter()
                        .flat_map(|t| &t.recs)
                        .map(|r| {
                            let p0 = r.prob as f64 / 256.0;
                            -(if r.bit == 1 { 1.0 - p0 } else { p0 }).log2()
                        })
                        .sum();
                    let fixes: Vec<(usize, usize)> = io.tiles.iter().map(|t| (t.drop, t.fix.len())).collect();
                    (
                        if io.exact { 5 } else { FAIL_MISMATCH },
                        format!(
                            "frame {fno}: {} {}x{} q{} tiles {} bools {bools} ideal {:.0} B of {fl} B, fix {:?}{}",
                            if h.intra() { "key" } else if h.show_frame { "inter" } else { "hidden" },
                            h.width,
                            h.height,
                            h.base_q,
                            io.tiles.len(),
                            bits / 8.0,
                            fixes,
                            if io.exact { "" } else { " MISMATCH" }
                        ),
                    )
                }
                Err(e) => (*e, format!("frame {fno}: {} ({fl} B)", FAIL_NAMES[*e])),
            };
            counts[slot] += 1;
            if out.len() < limit && (slot != 5 || fno < 8) {
                out.push(msg);
            }
            fno += 1;
        }
    }
    out.push(format!(
        "recodable {} / {fno}; {}",
        counts[5],
        FAIL_NAMES.iter().zip(&counts).map(|(n, c)| format!("{n} {c}")).collect::<Vec<_>>().join(", ")
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_map_is_monotonic() {
        let t = state_map();
        // P(1) falls as the probability of a 0 rises.
        for p in 1..255 {
            assert!(cabac_p1(t[p + 1]) <= cabac_p1(t[p]), "prob {p}");
        }
        assert!(cabac_p1(t[128]).abs_diff(32768) < 3000);
    }

    #[test]
    fn synthetic_streams_round_trip() {
        for (seed, w, h, lc, lr) in [(7, 64, 48, 0, 0), (8, 520, 72, 1, 1), (9, 200, 136, 0, 2)] {
            let webm = crate::vp9::synth::synthetic_webm(seed, w, h, lc, lr);
            assert!(probe(&webm));
            let (packed, st) = compress(&webm).expect("recodes");
            assert_eq!((st.frames, st.recoded), (8, 7), "{w}x{h}: {:?}", st.fails);
            assert_eq!(st.fails[FAIL_SHOW], 1);
            assert_eq!(decompress(&packed).expect("decodes"), webm, "{w}x{h}");
            assert!(packed.len() < webm.len(), "{w}x{h}: {} vs {}", packed.len(), webm.len());
            // more threads, same bytes
            let opts = Options { threads: 3, ..Default::default() };
            let (again, _) = compress_with(&webm, &opts, &mut |_, _| true).unwrap().unwrap();
            assert_eq!(again, packed);
        }
    }

    #[test]
    fn damaged_frames_are_stored_raw() {
        let clean = crate::vp9::synth::synthetic_webm(11, 64, 48, 0, 0);
        let (_, samples) = mkv::find_track(&clean, b"V_VP9").unwrap();
        // Flip bits inside an inter frame's tile data (decoding garbage with the same
        // probabilities still re-encodes to the same bytes, so it may well be recoded), and
        // make a later header unsupported.
        let mut webm = clean.clone();
        let (o, s) = samples[1];
        webm[(o + s as u64 / 2) as usize] ^= 0x5A;
        let (o, _) = samples[4];
        webm[o as usize] |= 0x20; // profile 1 (4:4:4)
        let (packed, st) = compress(&webm).expect("other frames still recode");
        assert!(st.recoded >= 3 && st.recoded < 7, "{st:?}");
        assert!(st.fails[FAIL_HEADER] >= 1, "{st:?}");
        assert_eq!(decompress(&packed).unwrap(), webm);
        let mut bad = packed.clone();
        let n = bad.len();
        bad[n - 5] ^= 1;
        assert_ne!(decompress(&bad).ok(), Some(webm));
    }

    #[test]
    fn repeated_frames_are_copied() {
        use crate::vp9::synth::{frames, FrameSpec};
        let f = frames(13, 64, 48, 0, 0, &[Some(FrameSpec::key(50)), Some(FrameSpec::inter(70)), Some(FrameSpec::inter(90))]);
        // the same frames again, as a static scene repeats them
        let blocks = vec![f[0].clone(), f[1].clone(), f[2].clone(), f[1].clone(), f[2].clone(), f[0].clone(), f[1].clone()];
        let webm = mkv::write_mkv_codec(b"V_VP9", None, &blocks, 64, 48);
        let (packed, st) = compress(&webm).expect("recodes");
        assert_eq!((st.frames, st.copies), (7, 4), "{st:?}");
        assert_eq!(decompress(&packed).unwrap(), webm);
    }

    #[test]
    fn corrupt_streams_fail_cleanly() {
        let webm = crate::vp9::synth::synthetic_webm(12, 64, 48, 0, 0);
        let (packed, _) = compress(&webm).unwrap();
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        for k in 0..300 {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let mut bad = packed.clone();
            let i = (s >> 33) as usize % bad.len();
            bad[i] ^= 1 << (k % 8);
            // must return (an error or wrong bytes), never panic or hang
            let _ = decompress(&bad);
            let _ = decompress(&bad[..i]);
        }
    }

    #[test]
    fn index_sizes_reads_split_superframe() {
        let mut data = vec![1u8; 300];
        data.extend(vec![2u8; 5]);
        // two frames, 2-byte sizes: marker 0b110_01_001
        let marker = 0xC0 | (1 << 3) | 1;
        let idx = [marker, 44, 1, 5, 0, marker];
        data.extend_from_slice(&idx);
        let (frames, len) = split_superframe(&data);
        assert_eq!(len, 6);
        assert_eq!(frames, vec![(0, 300), (300, 5)]);
        assert_eq!(index_sizes(&idx, 305), Some(vec![300, 5]));
        assert_eq!(index_sizes(&idx, 306), None);
    }
}
