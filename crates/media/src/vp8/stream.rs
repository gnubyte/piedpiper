//! VP8 in Matroska/WebM: the `PPV8` recompressed format and its drivers. The layout follows
//! `PPV9` (see vp9/stream.rs and docs/FORMAT.md): a header stream with the block table and
//! container bytes, then segments that start at key frames. In a segment each frame is a
//! copy of an identical earlier frame, raw bytes, or recoded: its 3- or 10-byte tag, every
//! boolean decision of all its partitions in parse order (coded with the model), and a tail
//! fix-up per partition. Partition sizes are rebuilt, not stored.

use super::parse::{parse_frame, read_tag, PartIo, State, Tag};
use crate::bits::{Bad, R};
use crate::container::{Cancelled, Options, Progress};
use crate::h264::mb::Hint;
use crate::mkv;
use crate::model::{code_bin, code_flag, code_num, ArithDec, ArithEnc, ByteCtx, Model, MAX_TABLE_BITS, MIN_TABLE_BITS};
use crate::vp9::bool::{BoolDecoder, BoolEncoder};
use crate::vp9::stream::{
    auto_table_bits, code_fix, code_hdr, code_raw, get_gap, get_varint, num, put_gap, put_varint, run_jobs, state_map, Rec, Region,
    Stats, FAIL_HEADER, FAIL_MISMATCH, FAIL_NO_KEY, FAIL_PARSE, F_COPY, F_REC, HEADER_BITS, MAX_FIX, N_COPY, N_DROP, N_FILELEN,
    N_FIX, N_GAP, N_HDRLEN, N_NSAMP, N_NSEG, N_SEGSTART, N_SIZE, SEG_TARGET,
};

pub const MAGIC: &[u8; 4] = b"PPV8";

/// Byte ranges of the partitions of a frame: the first from its tag, the token partitions
/// from the size table that follows it (3-byte little-endian sizes; the last runs to the end).
fn partition_ranges(f: &[u8], tag: &Tag, n: usize) -> R<Vec<(usize, usize)>> {
    let first_end = tag.len.checked_add(tag.first_size).ok_or(Bad)?;
    if first_end > f.len() {
        return Err(Bad);
    }
    let mut out = vec![(tag.len, tag.first_size)];
    let table = first_end;
    let mut p = table.checked_add(3 * (n - 1)).ok_or(Bad)?;
    if p > f.len() {
        return Err(Bad);
    }
    for i in 0..n {
        let size = if i + 1 < n {
            let b = &f[table + 3 * i..table + 3 * i + 3];
            b[0] as usize | (b[1] as usize) << 8 | (b[2] as usize) << 16
        } else {
            f.len() - p
        };
        if size > f.len() - p {
            return Err(Bad);
        }
        out.push((p, size));
        p += size;
    }
    Ok(out)
}

/// A partition's tail fix-up: bytes to drop, replacement bytes, the dropped bytes.
type Fixup = (usize, Vec<u8>, Vec<u8>);

/// Decodes the original partitions, re-encodes them and records every decision.
struct VerifyIo<'a> {
    f: &'a [u8],
    tag: Tag,
    ranges: Vec<(usize, usize)>,
    dec: Vec<BoolDecoder<'a>>,
    enc: Vec<BoolEncoder>,
    recs: Vec<Rec>,
}

impl<'a> VerifyIo<'a> {
    fn new(f: &'a [u8], tag: Tag) -> Self {
        let first = &f[tag.len..(tag.len + tag.first_size).min(f.len())];
        VerifyIo { f, tag, ranges: Vec::new(), dec: vec![BoolDecoder::new(first)], enc: vec![BoolEncoder::new_raw()], recs: Vec::with_capacity(f.len() * 6) }
    }

    /// Per partition: (bytes to drop from the re-encoded end, original replacement bytes,
    /// dropped bytes), or `None` if some partition differs by more than a tail.
    fn fixups(self) -> (Vec<Rec>, Option<Vec<Fixup>>) {
        let mut out = Vec::new();
        let ranges = if self.ranges.is_empty() { vec![(self.tag.len, self.tag.first_size)] } else { self.ranges.clone() };
        for (enc, &(o, l)) in self.enc.into_iter().zip(&ranges) {
            let data = enc.finish_vp8();
            let orig = &self.f[o..o + l];
            let common = data.iter().zip(orig).take_while(|(a, b)| a == b).count();
            let drop = data.len() - common;
            let fix = orig[common..].to_vec();
            if drop > MAX_FIX || fix.len() > MAX_FIX {
                return (self.recs, None);
            }
            out.push((drop, fix, data[common..].to_vec()));
        }
        (self.recs, Some(out))
    }
}

impl PartIo for VerifyIo<'_> {
    #[inline]
    fn read(&mut self, part: usize, prob: u8, h: &Hint) -> R<u32> {
        let d = self.dec.get_mut(part).ok_or(Bad)?;
        if d.overrun() {
            return Err(Bad);
        }
        let b = d.read(prob);
        self.enc[part].write(b, prob);
        self.recs.push(Rec { h: *h, prob, bit: b as u8 });
        Ok(b)
    }

    fn token_partitions(&mut self, n: usize) -> R<()> {
        self.ranges = partition_ranges(self.f, &self.tag, n)?;
        let f = self.f;
        for &(o, l) in &self.ranges[1..] {
            self.dec.push(BoolDecoder::new(&f[o..o + l]));
            self.enc.push(BoolEncoder::new_raw());
        }
        Ok(())
    }
}

/// Decodes the original partitions (keeps the parse state in step for raw frames).
struct ReadIo<'a> {
    f: &'a [u8],
    tag: Tag,
    dec: Vec<BoolDecoder<'a>>,
}

impl<'a> ReadIo<'a> {
    fn new(f: &'a [u8], tag: Tag) -> Self {
        let first = &f[tag.len..(tag.len + tag.first_size).min(f.len())];
        ReadIo { f, tag, dec: vec![BoolDecoder::new(first)] }
    }
}

impl PartIo for ReadIo<'_> {
    #[inline]
    fn read(&mut self, part: usize, prob: u8, _h: &Hint) -> R<u32> {
        let d = self.dec.get_mut(part).ok_or(Bad)?;
        if d.overrun() {
            return Err(Bad);
        }
        Ok(d.read(prob))
    }

    fn token_partitions(&mut self, n: usize) -> R<()> {
        let f = self.f;
        for &(o, l) in &partition_ranges(f, &self.tag, n)?[1..] {
            self.dec.push(BoolDecoder::new(&f[o..o + l]));
        }
        Ok(())
    }
}

/// Decodes decisions from the model and re-encodes each partition.
struct ModelIo<'a, 'd> {
    m: &'a mut Model,
    c: &'a mut ArithDec<'d>,
    map: &'a [u8; 256],
    enc: Vec<BoolEncoder>,
}

impl PartIo for ModelIo<'_, '_> {
    #[inline]
    fn read(&mut self, part: usize, prob: u8, h: &Hint) -> R<u32> {
        if self.c.overrun() || part >= self.enc.len() {
            return Err(Bad);
        }
        let b = code_bin(self.m, self.c, h, Some(self.map[prob as usize]), 0);
        self.enc[part].write(b, prob);
        Ok(b)
    }

    fn token_partitions(&mut self, n: usize) -> R<()> {
        for _ in 0..n {
            self.enc.push(BoolEncoder::new_raw());
        }
        Ok(())
    }
}

#[derive(Default)]
struct SegState {
    st: State,
    valid: bool,
    prev_hdr: Vec<u8>,
}

/// Parses an original frame (both sides do this for raw and copied frames).
fn parse_original<'a, T: PartIo>(ss: &mut SegState, f: &'a [u8], make: impl FnOnce(&'a [u8], Tag) -> T) -> Result<(Tag, T), usize> {
    let tag = match read_tag(f) {
        Ok(t) if t.len + t.first_size <= f.len() => t,
        _ => {
            ss.valid = false;
            return Err(FAIL_HEADER);
        }
    };
    if tag.key {
        ss.valid = true;
    }
    if !ss.valid {
        return Err(FAIL_NO_KEY);
    }
    let mut io = make(f, tag);
    if parse_frame(&mut ss.st, &tag, &mut io).is_err() {
        ss.valid = false;
        return Err(FAIL_PARSE);
    }
    Ok((tag, io))
}

fn is_keyframe(smp: &[u8]) -> bool {
    smp.len() > 10 && smp[0] & 1 == 0 && smp[3..6] == [0x9d, 0x01, 0x2a]
}

fn find_frames(input: &[u8]) -> Option<Vec<(u64, u32)>> {
    let (_, samples) = mkv::find_track(input, b"V_VP8")?;
    let mut end = 0u64;
    for &(o, s) in &samples {
        if o < end || o + s as u64 > input.len() as u64 {
            return None;
        }
        end = o + s as u64;
    }
    Some(samples)
}

/// True if `input` is a Matroska/WebM file with a VP8 video track.
pub fn probe(input: &[u8]) -> bool {
    find_frames(input).is_some()
}

pub fn compress(input: &[u8]) -> Option<(Vec<u8>, Stats)> {
    compress_with(input, &Options::default(), &mut |_, _| true).ok().flatten()
}

pub fn compress_with(input: &[u8], opts: &Options, progress: &mut Progress) -> Result<Option<(Vec<u8>, Stats)>, Cancelled> {
    let Some(samples) = find_frames(input) else { return Ok(None) };
    let total = input.len() as u64;
    // segments start at key frames once they hold about SEG_TARGET bytes
    let mut starts = vec![0usize];
    let mut acc = 0usize;
    for (i, &(o, s)) in samples.iter().enumerate() {
        if i > *starts.last().unwrap() && acc >= SEG_TARGET && is_keyframe(&input[o as usize..o as usize + s as usize]) {
            starts.push(i);
            acc = 0;
        }
        acc += s as usize;
    }
    let mut st = Stats { samples: samples.len(), segments: starts.len(), ..Default::default() };
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
    let nseg = starts.len();
    let results = run_jobs(nseg, opts.threads, total, st.other_bytes as u64, progress, &|i, tick| {
        let (first, end) = (starts[i], if i + 1 < nseg { starts[i + 1] } else { samples.len() });
        let size: usize = samples[first..end].iter().map(|&(_, s)| s as usize).sum();
        let bits = opts.table_bits.unwrap_or_else(|| auto_table_bits(size)).clamp(MIN_TABLE_BITS, MAX_TABLE_BITS);
        encode_segment(input, &samples[first..end], bits, opts.stats, &map, tick).map(|(d, s)| (bits, d, s))
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
    let (mut last_flag, mut last_copy) = (0u32, 0u32);
    let mut seen: std::collections::HashMap<&[u8], usize> = std::collections::HashMap::new();
    for (fno, &(o, s)) in samples.iter().enumerate() {
        if !tick(s as u64) {
            return Err(Cancelled);
        }
        let f = &input[o as usize..o as usize + s as usize];
        st.frames += 1;
        let before = enc.out.len();
        let earlier = seen.insert(f, fno);
        code_flag(&mut m, &mut enc, F_COPY, last_copy, earlier.is_some() as u32);
        last_copy = earlier.is_some() as u32;
        if let Some(k) = earlier {
            code_num(&mut m, &mut enc, N_COPY, (fno - 1 - k) as u64);
            let _ = parse_original(&mut ss, f, ReadIo::new);
            st.copies += 1;
            st.copy_bytes += f.len();
            st.out_copy += enc.out.len() - before;
            continue;
        }
        let res = parse_original(&mut ss, f, VerifyIo::new).map(|(tag, io)| (tag, io.fixups()));
        match res {
            Ok((tag, (recs, Some(fixes)))) => {
                code_flag(&mut m, &mut enc, F_REC, last_flag, 1);
                last_flag = 1;
                code_num(&mut m, &mut enc, N_HDRLEN, tag.len as u64);
                let mut hdr = f[..tag.len].to_vec();
                code_hdr(&mut m, &mut enc, &ss.prev_hdr, &mut hdr);
                ss.prev_hdr = hdr;
                for r in &recs {
                    let p = m.predict(&r.h, Some(map[r.prob as usize]));
                    enc.encode(r.bit as u32, p);
                    m.update(r.bit as u32);
                    if stats {
                        st.add_cost(r, p);
                    }
                }
                st.bools += recs.len() as u64;
                for (drop, fix, dropped) in &fixes {
                    code_num(&mut m, &mut enc, N_DROP, *drop as u64);
                    code_num(&mut m, &mut enc, N_FIX, fix.len() as u64);
                    let mut fx = fix.clone();
                    code_fix(&mut m, &mut enc, dropped, &mut fx, fix.len());
                }
                st.tiles += fixes.len();
                st.recoded += 1;
                st.frame_bytes += f.len();
                st.out_frames += enc.out.len() - before;
            }
            res => {
                st.fails[res.err().unwrap_or(FAIL_MISMATCH)] += 1;
                code_flag(&mut m, &mut enc, F_REC, last_flag, 0);
                last_flag = 0;
                let mut raw = f.to_vec();
                code_raw(&mut m, &mut enc, &mut rb, &mut raw);
                st.raw_bytes += f.len();
                st.out_raw += enc.out.len() - before;
            }
        }
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
    let (mut last_flag, mut last_copy) = (0u32, 0u32);
    let mut placed: Vec<(usize, usize)> = Vec::new();
    for &(o, size) in samples {
        if !tick(size as u64) {
            return Err(Bad);
        }
        let w = o - base;
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
            let _ = parse_original(&mut ss, raw, ReadIo::new);
        } else if code_flag(&mut m, &mut c, F_REC, last_flag, 0) == 1 {
            last_flag = 1;
            let frame = decode_frame(&mut ss, &mut m, &mut c, map, size)?;
            out[w..w + size].copy_from_slice(&frame);
        } else {
            last_flag = 0;
            let dst = &mut out[w..w + size];
            code_raw(&mut m, &mut c, &mut rb, dst);
            let raw: &[u8] = dst;
            let _ = parse_original(&mut ss, raw, ReadIo::new);
        }
        if c.overrun() {
            return Err(Bad);
        }
        placed.push((w, size));
    }
    Ok(())
}

fn decode_frame(ss: &mut SegState, m: &mut Model, c: &mut ArithDec, map: &[u8; 256], size: usize) -> R<Vec<u8>> {
    let hlen = num(m, c, N_HDRLEN, 10)?;
    let mut hdr = vec![0u8; hlen];
    code_hdr(m, c, &ss.prev_hdr, &mut hdr);
    let tag = read_tag(&hdr)?;
    if tag.len != hlen {
        return Err(Bad);
    }
    if tag.key {
        ss.valid = true;
    }
    if !ss.valid {
        return Err(Bad);
    }
    let mut io = ModelIo { m, c, map, enc: vec![BoolEncoder::new_raw()] };
    parse_frame(&mut ss.st, &tag, &mut io)?;
    let encs = std::mem::take(&mut io.enc);
    let mut parts = Vec::with_capacity(encs.len());
    for e in encs {
        let mut data = e.finish_vp8();
        let drop = num(io.m, io.c, N_DROP, MAX_FIX)?;
        let nfix = num(io.m, io.c, N_FIX, MAX_FIX)?;
        if drop > data.len() {
            return Err(Bad);
        }
        let dropped = data.split_off(data.len() - drop);
        let mut fix = Vec::new();
        code_fix(io.m, io.c, &dropped, &mut fix, nfix);
        data.extend_from_slice(&fix);
        parts.push(data);
    }
    if parts.len() < 2 || parts[0].len() != tag.first_size {
        return Err(Bad);
    }
    let mut frame = hdr.clone();
    frame.extend_from_slice(&parts[0]);
    let n = parts.len() - 1;
    for p in &parts[1..n] {
        let l = p.len();
        if l >= 1 << 24 {
            return Err(Bad);
        }
        frame.extend_from_slice(&[l as u8, (l >> 8) as u8, (l >> 16) as u8]);
    }
    for p in &parts[1..] {
        frame.extend_from_slice(p);
    }
    ss.prev_hdr = hdr;
    if frame.len() != size {
        return Err(Bad);
    }
    Ok(frame)
}

/// Development aid: per-frame outcome of the parse (no model).
pub fn diagnose(input: &[u8], limit: usize) -> Vec<String> {
    let Some(samples) = find_frames(input) else { return vec!["no VP8 track".into()] };
    let mut out = Vec::new();
    let mut ss = SegState::default();
    let mut counts = [0usize; 6];
    for (fno, &(o, s)) in samples.iter().enumerate() {
        let f = &input[o as usize..o as usize + s as usize];
        let r = parse_original(&mut ss, f, VerifyIo::new).map(|(tag, io)| (tag, io.fixups()));
        let (slot, msg) = match r {
            Ok((tag, (recs, fixes))) => {
                let bits: f64 = recs
                    .iter()
                    .map(|r| {
                        let p0 = r.prob as f64 / 256.0;
                        -(if r.bit == 1 { 1.0 - p0 } else { p0 }).log2()
                    })
                    .sum();
                let ok = fixes.is_some();
                let fx: Vec<(usize, usize)> = fixes.unwrap_or_default().iter().map(|(d, f, _)| (*d, f.len())).collect();
                (
                    if ok { 5 } else { FAIL_MISMATCH },
                    format!(
                        "frame {fno}: {} {}x{} bools {} ideal {:.0} B of {s} B, fix {:?}{}",
                        if tag.key { "key" } else if tag.show { "inter" } else { "hidden" },
                        ss.st.width,
                        ss.st.height,
                        recs.len(),
                        bits / 8.0,
                        fx,
                        if ok { "" } else { " MISMATCH" }
                    ),
                )
            }
            Err(e) => (e, format!("frame {fno}: {} ({s} B)", crate::vp9::stream::FAIL_NAMES[e])),
        };
        counts[slot] += 1;
        if out.len() < limit && (slot != 5 || fno < 8) {
            out.push(msg);
        }
    }
    out.push(format!(
        "recodable {} / {}; {}",
        counts[5],
        samples.len(),
        crate::vp9::stream::FAIL_NAMES.iter().zip(&counts).map(|(n, c)| format!("{n} {c}")).collect::<Vec<_>>().join(", ")
    ));
    out
}

/// Synthetic VP8 streams for tests: pseudo-random decisions pushed through the parser
/// itself (header fields included) and coded with libvpx's boolean coder.
#[doc(hidden)]
pub mod synth {
    use super::super::parse::{parse_frame, PartIo, State, Tag};
    use crate::bits::R;
    use crate::h264::mb::Hint;
    use crate::vp9::bool::BoolEncoder;

    struct GenIo {
        s: u64,
        enc: Vec<BoolEncoder>,
    }

    impl PartIo for GenIo {
        fn read(&mut self, part: usize, prob: u8, h: &Hint) -> R<u32> {
            self.s = self.s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let r = (self.s >> 33) as u32;
                    // header fields stay 0 except segmentation (3-5), filter level (13), the token
            // partition count (20) and the quantizer (21); vectors stay short
            let tame = (h.kind == super::super::parse::kind::HEADER && !matches!(h.a & 0xFF, 3 | 4 | 5 | 13 | 20 | 21))
                || h.kind == super::super::parse::kind::MV_LONG;
            let bit = if tame {
                0
            } else if r & 3 != 0 {
                (prob < 128) as u32
            } else {
                ((r >> 8) & 255 >= prob as u32) as u32
            };
            self.enc[part].write(bit, prob);
            Ok(bit)
        }

        fn token_partitions(&mut self, n: usize) -> R<()> {
            for _ in 0..n {
                self.enc.push(BoolEncoder::new_raw());
            }
            Ok(())
        }
    }

    /// `frames` frames of `w` x `h`, a key frame every `key_every`.
    pub fn frames(seed: u64, w: u32, h: u32, n: usize, key_every: usize) -> Vec<Vec<u8>> {
        let mut st = State::default();
        let mut out = Vec::new();
        for i in 0..n {
            let key = i % key_every == 0;
            let tag = Tag { key, version: 0, show: true, first_size: 1, len: if key { 10 } else { 3 }, width: w, height: h };
            let mut io = GenIo { s: seed ^ (i as u64) << 32, enc: vec![BoolEncoder::new_raw()] };
            parse_frame(&mut st, &tag, &mut io).expect("random decisions parse");
            let parts: Vec<Vec<u8>> = io.enc.into_iter().map(|e| e.finish_vp8()).collect();
            let first = parts[0].len();
            let raw = (!key as u32) | 1 << 4 | (first as u32) << 5;
            let mut f = vec![raw as u8, (raw >> 8) as u8, (raw >> 16) as u8];
            if key {
                f.extend_from_slice(&[0x9d, 0x01, 0x2a, w as u8, (w >> 8) as u8, h as u8, (h >> 8) as u8]);
            }
            f.extend_from_slice(&parts[0]);
            let n = parts.len() - 1;
            for p in &parts[1..n] {
                let l = p.len();
                f.extend_from_slice(&[l as u8, (l >> 8) as u8, (l >> 16) as u8]);
            }
            for p in &parts[1..] {
                f.extend_from_slice(p);
            }
            out.push(f);
        }
        out
    }

    pub fn synthetic_webm(seed: u64, w: u32, h: u32, n: usize) -> Vec<u8> {
        let f = frames(seed, w, h, n, 5);
        crate::mkv::write_mkv_codec(b"V_VP8", None, &f, w, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_vp8_round_trips() {
        for (seed, w, h) in [(1, 48, 32), (2, 100, 70)] {
            let webm = synth::synthetic_webm(seed, w, h, 9);
            assert!(probe(&webm));
            let (packed, st) = compress(&webm).expect("recodes");
            assert_eq!((st.frames, st.recoded), (9, 9), "{w}x{h}: {:?}", st.fails);
            assert!(st.tiles > 18, "some frames use several token partitions: {}", st.tiles);
            assert_eq!(decompress(&packed).unwrap(), webm);
            assert!(packed.len() < webm.len(), "{} vs {}", packed.len(), webm.len());
            let opts = Options { threads: 2, ..Default::default() };
            assert_eq!(compress_with(&webm, &opts, &mut |_, _| true).unwrap().unwrap().0, packed);
        }
    }

    #[test]
    fn vp8_copies_and_damage() {
        let f = synth::frames(3, 48, 32, 4, 4);
        let mut blocks = vec![f[0].clone(), f[1].clone(), f[2].clone(), f[1].clone(), f[3].clone()];
        blocks[4][5] ^= 0x40; // damage inside the first partition of the last frame
        let webm = crate::mkv::write_mkv_codec(b"V_VP8", None, &blocks, 48, 32);
        let (packed, st) = compress(&webm).expect("recodes");
        assert_eq!(st.copies, 1, "{st:?}");
        assert_eq!(decompress(&packed).unwrap(), webm);
        let mut s = 7u64;
        for k in 0..200 {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let mut bad = packed.clone();
            let i = (s >> 33) as usize % bad.len();
            bad[i] ^= 1 << (k % 8);
            let _ = decompress(&bad);
        }
    }
}
