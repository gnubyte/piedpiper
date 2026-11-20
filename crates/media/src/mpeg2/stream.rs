//! MPEG-2 video in Matroska (`V_MPEG2`): the `PPMV` recompressed format and its drivers.
//!
//! Outer layout as `PPV9` (header stream with the frame table and container bytes, then
//! segments coded in parallel). A frame is a run of start-code units; for each unit the
//! stream holds a continuation flag, the start code value and then either the payload
//! bytes or, for a slice the parser handles, every bit of the slice coded with the model in
//! its syntax context, an end decision after each macroblock, and the slice's trailing bits.
//! Slice parsing needs only the sequence and picture headers in effect, which both sides
//! read from the units themselves (and from the track's CodecPrivate at a segment start).
//!
//! AC-3 audio frames found between the video frames (DVD rips carry several AC-3 tracks)
//! are left out of the container bytes and coded in runs of their own with [`crate::ac3`];
//! the runs follow the last segment, and a stream without AC-3 frames has none.

use super::parse::{extension, kind, picture_header, sequence_header, BitIo, Pic, PicCtx, Seq, SliceParser};
use crate::ac3;
use crate::bits::{Bad, BitReader, BitWriter, R};
use crate::container::{Cancelled, Options, Progress};
use crate::h264::mb::Hint;
use crate::mkv;
use crate::model::{code_bin, code_byte, code_flag, code_num, ArithDec, ArithEnc, ByteCtx, Coder, Model, MAX_TABLE_BITS, MIN_TABLE_BITS};
use crate::vp9::stream::{
    auto_table_bits, get_gap, get_varint, num, put_gap, put_varint, run_jobs, Rec, Region, Stats, HEADER_BITS, N_FILELEN, N_GAP,
    N_NSAMP, N_NSEG, N_SEGSTART, N_SIZE, SEG_TARGET,
};

pub const MAGIC: &[u8; 4] = b"PPMV";

const N_PREFIX: u32 = 40;
const N_UNITLEN: u32 = 41;
const N_TAIL: u32 = 42;
const F_UNIT: u32 = 43;
const F_SLICE: u32 = 44;
/// Slices whose bits after the last macroblock exceed this are stored raw.
const MAX_TAIL_BITS: usize = 1 << 13;
/// AC-3 frames are coded in runs of about this many bytes (each with its own model, in
/// parallel).
const AC3_RUN: usize = 8 << 20;

/// Codes `input[from..to]` without the AC-3 frames in it (`frames[*next..]`, in order).
#[allow(clippy::too_many_arguments)]
fn put_gap_ex(m: &mut Model, enc: &mut ArithEnc, bc: &mut ByteCtx, input: &[u8], from: usize, to: usize, frames: &[(usize, usize)], next: &mut usize) {
    let mut p = from;
    while *next < frames.len() && frames[*next].0 < to {
        let (o, s) = frames[*next];
        put_gap(m, enc, bc, &input[p..o]);
        p = o + s;
        *next += 1;
    }
    put_gap(m, enc, bc, &input[p..to]);
}

/// Decodes `out[from..to]` around the AC-3 frames that fall into it, which it copies in.
#[allow(clippy::too_many_arguments)]
fn get_gap_ex(m: &mut Model, c: &mut ArithDec, bc: &mut ByteCtx, out: &mut [u8], from: usize, to: usize, frames: &[(usize, Vec<u8>)], next: &mut usize) -> R<()> {
    let mut p = from;
    while *next < frames.len() && frames[*next].0 < to {
        let (o, ref f) = frames[*next];
        if o < p || o + f.len() > to {
            return Err(Bad);
        }
        get_gap(m, c, bc, &mut out[p..o]);
        out[o..o + f.len()].copy_from_slice(f);
        p = o + f.len();
        *next += 1;
    }
    get_gap(m, c, bc, &mut out[p..to]);
    Ok(())
}

/// `(offset, size)` byte ranges, or `(first, end)` index ranges.
type Ranges = Vec<(usize, usize)>;

/// The recodable AC-3 frames outside the video samples, and runs of them (index ranges).
fn find_ac3(input: &[u8], samples: &[(u64, u32)]) -> (Ranges, Ranges) {
    let mut cand = Vec::new();
    let mut pos = 0usize;
    for &(o, s) in samples {
        ac3::scan(input, pos, o as usize, &mut cand);
        pos = (o + s as u64) as usize;
    }
    ac3::scan(input, pos, input.len(), &mut cand);
    let mut probe = ac3::Probe::default();
    let frames: Vec<(usize, usize)> = cand.into_iter().filter(|&(o, s)| ac3::parses(&input[o..o + s], &mut probe)).collect();
    let mut runs = Vec::new();
    let (mut first, mut acc) = (0usize, 0usize);
    for (i, &(_, s)) in frames.iter().enumerate() {
        if acc >= AC3_RUN {
            runs.push((first, i));
            first = i;
            acc = 0;
        }
        acc += s;
    }
    if first < frames.len() {
        runs.push((first, frames.len()));
    }
    (frames, runs)
}

fn is_slice(code: u8) -> bool {
    (1..=0xAF).contains(&code)
}

/// (start code value, payload range) of every unit of a frame, and the length of any bytes
/// before the first start code.
fn units(f: &[u8]) -> (usize, Vec<(u8, usize, usize)>) {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 < f.len() {
        if f[i] == 0 && f[i + 1] == 0 && f[i + 2] == 1 {
            starts.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    let prefix = starts.first().copied().unwrap_or(f.len());
    let out = starts
        .iter()
        .enumerate()
        .map(|(k, &s)| {
            let end = starts.get(k + 1).copied().unwrap_or(f.len());
            (f[s + 3], s + 4, end)
        })
        .collect();
    (prefix, out)
}

/// Reads the original slice bits and records them with their contexts.
struct VerifyBits<'a> {
    r: BitReader<'a>,
    data: &'a [u8],
    len: usize,
    recs: Vec<Rec>,
}

impl BitIo for VerifyBits<'_> {
    #[inline]
    fn bit(&mut self, h: &Hint) -> R<u32> {
        let b = self.r.bit()?;
        self.recs.push(Rec { h: *h, prob: 128, bit: b as u8 });
        Ok(b)
    }

    fn more(&mut self, h: &Hint) -> R<bool> {
        let pos = self.r.pos();
        let data = self.data;
        let mut zero = true;
        for k in pos..pos + 23 {
            if k < self.len && (data[k >> 3] >> (7 - (k & 7))) & 1 == 1 {
                zero = false;
                break;
            }
        }
        self.recs.push(Rec { h: *h, prob: 128, bit: !zero as u8 });
        Ok(!zero)
    }
}

/// Decodes slice bits from the model and writes them out.
struct ModelBits<'a, 'd> {
    m: &'a mut Model,
    c: &'a mut ArithDec<'d>,
    w: BitWriter,
}

impl BitIo for ModelBits<'_, '_> {
    #[inline]
    fn bit(&mut self, h: &Hint) -> R<u32> {
        if self.c.overrun() {
            return Err(Bad);
        }
        let b = code_bin(self.m, self.c, h, Some(0), 0);
        self.w.put(b);
        Ok(b)
    }

    fn more(&mut self, h: &Hint) -> R<bool> {
        if self.c.overrun() {
            return Err(Bad);
        }
        Ok(code_bin(self.m, self.c, h, Some(0), 0) == 1)
    }
}

/// Header state of a segment.
#[derive(Clone, Copy, Default)]
struct Hdr {
    seq: Seq,
    pic: Pic,
    have_seq: bool,
}

impl Hdr {
    /// Updates the state from a non-slice unit.
    fn unit(&mut self, code: u8, p: &[u8]) {
        match code {
            0xB3 => {
                if sequence_header(p, &mut self.seq).is_ok() {
                    self.have_seq = true;
                }
            }
            0xB5 => {
                let _ = extension(p, &mut self.seq, &mut self.pic);
            }
            0x00 => {
                let _ = picture_header(p, &mut self.pic);
            }
            _ => {}
        }
    }

    /// The initial state from the track's CodecPrivate (a sequence header and extension).
    fn from_private(private: Option<&[u8]>) -> Self {
        let mut h = Hdr::default();
        if let Some(p) = private {
            let (_, us) = units(p);
            for (code, a, b) in us {
                h.unit(code, &p[a..b]);
            }
        }
        h.pic = Pic::default();
        h
    }
}

fn tail_bit<C: Coder>(m: &mut Model, c: &mut C, i: usize, bit: u32) -> u32 {
    code_bin(m, c, &Hint { ctx: 0, kind: kind::TAIL, a: (i.min(15)) as u32, b: 0, c: 0, d: 0 }, None, bit)
}

fn code_bytes<C: Coder>(m: &mut Model, c: &mut C, bc: &mut ByteCtx, k: u8, bytes: &mut [u8]) {
    for b in bytes.iter_mut() {
        *b = code_byte(m, c, k, bc, *b);
    }
}

fn find_frames(input: &[u8]) -> Option<mkv::TrackData> {
    let (private, samples) = mkv::find_track(input, b"V_MPEG2")?;
    let mut end = 0u64;
    for &(o, s) in &samples {
        if o < end || o + s as u64 > input.len() as u64 {
            return None;
        }
        end = o + s as u64;
    }
    Some((private, samples))
}

/// True if `input` is a Matroska file with an MPEG-2 video track.
pub fn probe(input: &[u8]) -> bool {
    find_frames(input).is_some()
}

pub fn compress(input: &[u8]) -> Option<(Vec<u8>, Stats)> {
    compress_with(input, &Options::default(), &mut |_, _| true).ok().flatten()
}

pub fn compress_with(input: &[u8], opts: &Options, progress: &mut Progress) -> Result<Option<(Vec<u8>, Stats)>, Cancelled> {
    let Some((private, samples)) = find_frames(input) else { return Ok(None) };
    let init = Hdr::from_private(private.as_deref());
    let total = input.len() as u64;
    // segments start at frames that begin with a sequence header (or a GOP or I-picture)
    let mut starts = vec![0usize];
    let mut acc = 0usize;
    for (i, &(o, s)) in samples.iter().enumerate() {
        let f = &input[o as usize..o as usize + s as usize];
        let gop = f.len() > 4 && f[..3] == [0, 0, 1] && (f[3] == 0xB3 || f[3] == 0xB8);
        if i > *starts.last().unwrap() && acc >= SEG_TARGET && gop {
            starts.push(i);
            acc = 0;
        }
        acc += s as usize;
    }
    let (ac3f, ac3_runs) = find_ac3(input, &samples);
    let mut st = Stats { samples: samples.len(), segments: starts.len(), ..Default::default() };
    st.audio_frames = ac3f.len();
    st.audio_bytes = ac3f.iter().map(|&(_, s)| s).sum();
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
    code_num(&mut m, &mut enc, N_NSEG, starts.len() as u64);
    for w in starts.windows(2) {
        code_num(&mut m, &mut enc, N_SEGSTART, (w[1] - w[0]) as u64);
    }
    st.out_header = enc.out.len();
    let mut bc = ByteCtx::default();
    let mut pos = 0usize;
    let mut next = 0usize;
    for &(o, s) in &samples {
        put_gap_ex(&mut m, &mut enc, &mut bc, input, pos, o as usize, &ac3f, &mut next);
        st.other_bytes += o as usize - pos;
        pos = (o + s as u64) as usize;
    }
    put_gap_ex(&mut m, &mut enc, &mut bc, input, pos, input.len(), &ac3f, &mut next);
    st.other_bytes += input.len() - pos;
    st.other_bytes -= st.audio_bytes;
    let header = enc.finish();
    st.out_other = header.len() - st.out_header;
    if !progress(st.other_bytes as u64, total) {
        return Err(Cancelled);
    }
    let nseg = starts.len();
    let results = run_jobs(nseg, opts.threads, total, st.other_bytes as u64, progress, &|i, tick| {
        let (first, end) = (starts[i], if i + 1 < nseg { starts[i + 1] } else { samples.len() });
        let size: usize = samples[first..end].iter().map(|&(_, s)| s as usize).sum();
        let bits = opts.table_bits.unwrap_or_else(|| auto_table_bits(size)).clamp(MIN_TABLE_BITS, MAX_TABLE_BITS);
        encode_segment(input, &samples[first..end], init, bits, opts.stats, tick).map(|(d, s)| (bits, d, s))
    });
    let video: u64 = samples.iter().map(|&(_, s)| s as u64).sum();
    let audio = run_jobs(ac3_runs.len(), opts.threads, total, st.other_bytes as u64 + video, progress, &|i, tick| {
        let (a, b) = ac3_runs[i];
        let size: usize = ac3f[a..b].iter().map(|&(_, s)| s).sum();
        let bits = opts.table_bits.unwrap_or_else(|| auto_table_bits(size)).clamp(MIN_TABLE_BITS, MAX_TABLE_BITS);
        let data = ac3::encode_run(input, &ac3f[a..b], bits).map_err(|_| Cancelled);
        if !tick(size as u64) {
            return Err(Cancelled);
        }
        data.map(|d| (bits, d))
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
    if !audio.is_empty() {
        let before = out.len();
        put_varint(&mut out, audio.len() as u64);
        for r in audio {
            let (bits, data) = r?;
            out.push(bits as u8);
            put_varint(&mut out, data.len() as u64);
            out.extend_from_slice(&data);
        }
        st.out_audio = out.len() - before;
    }
    progress(total, total);
    if st.recoded == 0 {
        return Ok(None);
    }
    Ok(Some((out, st)))
}

fn encode_segment(
    input: &[u8],
    samples: &[(u64, u32)],
    init: Hdr,
    bits: u32,
    stats: bool,
    tick: &mut dyn FnMut(u64) -> bool,
) -> Result<(Vec<u8>, Stats), Cancelled> {
    let mut st = Stats::default();
    let mut hdr = init;
    let mut m = Model::new(bits);
    let mut enc = ArithEnc::new();
    let mut hb = ByteCtx::default();
    let mut rb = ByteCtx::default();
    let mut prev_code = 0u32;
    let mut pctx = PicCtx::default();
    for &(o, s) in samples {
        if !tick(s as u64) {
            return Err(Cancelled);
        }
        let f = &input[o as usize..o as usize + s as usize];
        st.frames += 1;
        let (prefix, us) = units(f);
        code_num(&mut m, &mut enc, N_PREFIX, prefix as u64);
        let mut pre = f[..prefix].to_vec();
        code_bytes(&mut m, &mut enc, &mut rb, kind::RAW, &mut pre);
        for &(code, a, b) in &us {
            code_flag(&mut m, &mut enc, F_UNIT, 0, 1);
            code_byte(&mut m, &mut enc, kind::START, &mut ByteCtx::default(), code);
            let p = &f[a..b];
            let before = enc.out.len();
            if is_slice(code) {
                st.tiles += 1;
                let mut io = VerifyBits { r: BitReader::new(p), data: p, len: p.len() * 8, recs: Vec::with_capacity(p.len() * 9) };
                let parsed = SliceParser::new(&hdr.seq, &hdr.pic, &mut pctx, code).and_then(|sp| sp.run(&mut io));
                let used = io.r.pos();
                let ok = parsed.is_ok() && used <= p.len() * 8 && p.len() * 8 - used <= MAX_TAIL_BITS;
                code_flag(&mut m, &mut enc, F_SLICE, prev_code.min(2), ok as u32);
                if ok {
                    for r in &io.recs {
                        let pr = m.predict(&r.h, Some(0));
                        enc.encode(r.bit as u32, pr);
                        m.update(r.bit as u32);
                        if stats {
                            st.add_cost(r, pr);
                        }
                    }
                    st.bools += io.recs.len() as u64;
                    // the bits after the last macroblock (byte alignment and stuffing)
                    let tail = p.len() * 8 - used;
                    code_num(&mut m, &mut enc, N_TAIL, tail as u64);
                    for k in 0..tail {
                        let i = used + k;
                        tail_bit(&mut m, &mut enc, k, ((p[i >> 3] >> (7 - (i & 7))) & 1) as u32);
                    }
                    st.recoded += 1;
                    st.frame_bytes += p.len() + 4;
                    st.out_frames += enc.out.len() - before;
                    prev_code = 1;
                    continue;
                }
                st.fails[2] += 1;
                prev_code = 2;
            } else {
                hdr.unit(code, p);
                if code == 0 {
                    pctx.picture(hdr.seq.width, hdr.seq.height);
                }
                prev_code = 0;
            }
            code_num(&mut m, &mut enc, N_UNITLEN, p.len() as u64);
            let mut raw = p.to_vec();
            code_bytes(&mut m, &mut enc, if is_slice(code) { &mut rb } else { &mut hb }, if is_slice(code) { kind::RAW } else { kind::HDR }, &mut raw);
            st.raw_bytes += p.len() + 4;
            st.out_raw += enc.out.len() - before;
        }
        code_flag(&mut m, &mut enc, F_UNIT, 0, 0);
    }
    Ok((enc.finish(), st))
}

pub fn decompress(data: &[u8]) -> R<Vec<u8>> {
    decompress_with(data, 0, &mut |_, _| true)
}

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
    // AC-3 runs (only when the stream has any): decoded first, the container bytes go around them
    let mut runs = Vec::new();
    if pos < data.len() {
        let nruns = usize::try_from(get_varint(data, &mut pos)?).map_err(|_| Bad)?;
        if nruns == 0 || nruns > file_len / 64 + 1 {
            return Err(Bad);
        }
        for _ in 0..nruns {
            let bits = *data.get(pos).ok_or(Bad)? as u32;
            pos += 1;
            if !(MIN_TABLE_BITS..=MAX_TABLE_BITS).contains(&bits) {
                return Err(Bad);
            }
            let len = usize::try_from(get_varint(data, &mut pos)?).map_err(|_| Bad)?;
            runs.push((bits, data.get(pos..pos.checked_add(len).ok_or(Bad)?).ok_or(Bad)?));
            pos += len;
        }
        if pos != data.len() {
            return Err(Bad);
        }
    }
    let decoded = run_jobs(runs.len(), threads, file_len as u64, 0, progress, &|k, tick| {
        let (bits, run) = runs[k];
        let r = ac3::decode_run(run, bits, file_len);
        if let Ok(f) = &r {
            if !tick(f.iter().map(|(_, b)| b.len() as u64).sum()) {
                return Err(Bad);
            }
        }
        r
    });
    let mut ac3f: Vec<(usize, Vec<u8>)> = Vec::new();
    for r in decoded {
        ac3f.extend(r?);
    }
    if ac3f.windows(2).any(|w| w[1].0 < w[0].0 + w[0].1.len()) {
        return Err(Bad);
    }
    let mut out = vec![0u8; file_len];
    let mut bc = ByteCtx::default();
    let mut p = 0usize;
    let mut next = 0usize;
    for &(o, s) in &samples {
        get_gap_ex(&mut m, &mut c, &mut bc, &mut out, p, o, &ac3f, &mut next)?;
        p = o + s;
    }
    get_gap_ex(&mut m, &mut c, &mut bc, &mut out, p, file_len, &ac3f, &mut next)?;
    if c.overrun() || next != ac3f.len() {
        return Err(Bad);
    }
    // the container is complete now: read the track's CodecPrivate for the initial headers
    let init = Hdr::from_private(find_frames(&out).and_then(|(p, _)| p).as_deref());
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
    let results = run_jobs(nseg, threads, file_len as u64, other, progress, &|k, tick| {
        let taken = regions[k].lock().unwrap().take();
        let (region, lo, first, end) = taken.expect("each region is decoded once");
        let (bits, seg) = segs[k];
        decode_segment(seg, bits, region, lo, &samples[first..end], init, tick)
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
    init: Hdr,
    tick: &mut dyn FnMut(u64) -> bool,
) -> R<()> {
    let mut c = ArithDec::new(data);
    let mut m = Model::new(bits);
    let mut hdr = init;
    let mut hb = ByteCtx::default();
    let mut rb = ByteCtx::default();
    let mut prev_code = 0u32;
    let mut pctx = PicCtx::default();
    for &(o, size) in samples {
        if !tick(size as u64) {
            return Err(Bad);
        }
        let mut f: Vec<u8> = Vec::with_capacity(size);
        let prefix = num(&mut m, &mut c, N_PREFIX, size)?;
        let mut pre = vec![0u8; prefix];
        code_bytes(&mut m, &mut c, &mut rb, kind::RAW, &mut pre);
        f.extend_from_slice(&pre);
        while code_flag(&mut m, &mut c, F_UNIT, 0, 0) == 1 {
            if f.len() > size || c.overrun() {
                return Err(Bad);
            }
            let code = code_byte(&mut m, &mut c, kind::START, &mut ByteCtx::default(), 0);
            f.extend_from_slice(&[0, 0, 1, code]);
            if is_slice(code) {
                if code_flag(&mut m, &mut c, F_SLICE, prev_code.min(2), 0) == 1 {
                    let mut io = ModelBits { m: &mut m, c: &mut c, w: BitWriter::new() };
                    SliceParser::new(&hdr.seq, &hdr.pic, &mut pctx, code)?.run(&mut io)?;
                    let mut w = io.w;
                    let tail = num(&mut m, &mut c, N_TAIL, MAX_TAIL_BITS)?;
                    for k in 0..tail {
                        w.put(tail_bit(&mut m, &mut c, k, 0));
                    }
                    if !w.bit_len().is_multiple_of(8) {
                        return Err(Bad);
                    }
                    f.extend_from_slice(&w.finish());
                    prev_code = 1;
                    continue;
                }
                prev_code = 2;
            } else {
                prev_code = 0;
            }
            let len = num(&mut m, &mut c, N_UNITLEN, size)?;
            let mut raw = vec![0u8; len];
            let slice = is_slice(code);
            code_bytes(&mut m, &mut c, if slice { &mut rb } else { &mut hb }, if slice { kind::RAW } else { kind::HDR }, &mut raw);
            if slice {
                // the encoder tried to parse this slice, and the attempt updated the picture
                // context up to where it stopped: repeat it
                let mut io = VerifyBits { r: BitReader::new(&raw), data: &raw, len: raw.len() * 8, recs: Vec::new() };
                let _ = SliceParser::new(&hdr.seq, &hdr.pic, &mut pctx, code).and_then(|sp| sp.run(&mut io));
            } else {
                hdr.unit(code, &raw);
                if code == 0 {
                    pctx.picture(hdr.seq.width, hdr.seq.height);
                }
            }
            f.extend_from_slice(&raw);
        }
        if f.len() != size || c.overrun() {
            return Err(Bad);
        }
        let w = o - base;
        out[w..w + size].copy_from_slice(&f);
    }
    Ok(())
}

/// Development aid: how the slices of the first `limit` frames parse.
pub fn diagnose(input: &[u8], limit: usize) -> Vec<String> {
    let Some((private, samples)) = find_frames(input) else { return vec!["no MPEG-2 track".into()] };
    let mut hdr = Hdr::from_private(private.as_deref());
    let mut out = vec![format!("initial seq {:?}", hdr.seq)];
    let (mut ok, mut bad, mut tails) = (0usize, 0usize, 0usize);
    let mut pctx = PicCtx::default();
    for (fno, &(o, s)) in samples.iter().enumerate().take(limit) {
        let f = &input[o as usize..o as usize + s as usize];
        let (_, us) = units(f);
        for &(code, a, b) in &us {
            let p = &f[a..b];
            if is_slice(code) {
                let mut io = VerifyBits { r: BitReader::new(p), data: p, len: p.len() * 8, recs: Vec::new() };
                let r = SliceParser::new(&hdr.seq, &hdr.pic, &mut pctx, code).and_then(|sp| sp.run(&mut io));
                let used = io.r.pos();
                match r {
                    Ok(info) if used <= p.len() * 8 => {
                        ok += 1;
                        tails += p.len() * 8 - used;
                        if out.len() < 12 {
                            out.push(format!("frame {fno} slice {code}: {} B, {} MBs, {} coefs, tail {} bits", p.len(), info.mbs, info.coefs, p.len() * 8 - used));
                        }
                    }
                    _ => {
                        bad += 1;
                        if out.len() < 40 {
                            out.push(format!("frame {fno} slice {code}: FAILED at bit {used} of {} ({:?})", p.len() * 8, hdr.pic));
                        }
                    }
                }
            } else {
                hdr.unit(code, p);
                if code == 0 {
                    pctx.picture(hdr.seq.width, hdr.seq.height);
                }
            }
        }
    }
    out.push(format!("slices ok {ok} failed {bad}, tail bits {tails}"));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_mpeg2_round_trips() {
        for (seed, w, h) in [(1, 4, 3), (2, 7, 2)] {
            let mkv = crate::mpeg2::synth::synthetic_mkv(seed, w, h, 8);
            assert!(probe(&mkv));
            let (packed, st) = compress(&mkv).expect("recodes");
            assert_eq!(st.recoded, st.tiles, "{st:?}");
            assert_eq!(st.recoded as u32, 8 * h, "{st:?}");
            assert_eq!(decompress(&packed).unwrap(), mkv);
            assert!(packed.len() < mkv.len(), "{} vs {}", packed.len(), mkv.len());
            let opts = Options { threads: 2, ..Default::default() };
            assert_eq!(compress_with(&mkv, &opts, &mut |_, _| true).unwrap().unwrap().0, packed);
        }
    }

    #[test]
    fn ac3_frames_round_trip() {
        // AC-3 frames after the video (in a real rip they sit between video blocks), with a
        // damaged one that must stay plain bytes
        let mkv = crate::mpeg2::synth::synthetic_mkv(4, 5, 3, 8);
        let frames = crate::ac3::synth::frames(11, 40);
        let mut file = mkv.clone();
        let mut total = 0;
        for (i, f) in frames.iter().enumerate() {
            file.extend((0..(i % 7) * 3).map(|k| (k * 5 + i) as u8));
            let mut f = f.clone();
            if i == 17 {
                let n = f.len() / 2;
                f[n] ^= 0x10;
            } else {
                total += f.len();
            }
            file.extend_from_slice(&f);
        }
        let (packed, st) = compress(&file).expect("recodes");
        assert_eq!(st.audio_frames, 39, "{st:?}");
        assert_eq!(st.audio_bytes, total);
        assert!(st.out_audio > 0 && st.out_audio < total);
        assert_eq!(decompress(&packed).unwrap(), file);
        let opts = Options { threads: 3, ..Default::default() };
        assert_eq!(compress_with(&file, &opts, &mut |_, _| true).unwrap().unwrap().0, packed);
        // the video-only file is coded exactly as before: no AC-3 section
        let (plain, st) = compress(&mkv).unwrap();
        assert_eq!(st.audio_frames, 0);
        assert_eq!(decompress(&plain).unwrap(), mkv);
        // a truncated AC-3 section fails cleanly
        assert!(decompress(&packed[..packed.len() - 3]).is_err());
    }

    #[test]
    fn damaged_mpeg2_falls_back_and_corrupt_input_fails_cleanly() {
        let mut f = crate::mpeg2::synth::frames(3, 4, 2, 4);
        let n = f[2].len();
        f[2][n - 3] ^= 0xFF; // garble the end of a slice
        let mkv = crate::mkv::write_mkv_codec(b"V_MPEG2", None, &f, 64, 32);
        let (packed, _) = compress(&mkv).expect("other slices recode");
        assert_eq!(decompress(&packed).unwrap(), mkv);
        let mut s = 9u64;
        for k in 0..200 {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let mut bad = packed.clone();
            let i = (s >> 33) as usize % bad.len();
            bad[i] ^= 1 << (k % 8);
            let _ = decompress(&bad);
        }
    }
}
