//! H.264 and HEVC video in MPEG transport streams: the `PPTS` recompressed format. Packets are
//! the 188 bytes of a plain transport stream (`.ts`) or the 192-byte source packets of BDAV
//! streams (`.m2ts`, `.mts`: a 4-byte `TP_extra_header`, then the 188-byte packet).
//!
//! A transport stream cuts the video's elementary stream into packet payloads and mixes in
//! packet headers, adaptation fields, PES headers and other programs' packets. The encoder
//! splits the file into that "skeleton" (every byte that is not video elementary-stream
//! payload) and the elementary stream itself, an Annex B byte stream. The stream's NAL units
//! become length-prefixed samples, one per access unit, that the H.264 or HEVC recoder
//! ([`crate::container`]) codes as if they came from an MP4 track; the skeleton and the zero
//! bytes around each start code are coded separately. The decoder rebuilds the elementary
//! stream and pours it back into the packets, finding the payloads by re-reading the packet
//! and PES headers in the skeleton.
//!
//! Layout: magic `PPTS`, the length of the side stream (varint), the side stream (file
//! length, video PID plus 8192 for 192-byte packets, skeleton bytes, bytes before the first
//! start code, number of NAL units and each unit's extra leading zeros, zero bytes at the end,
//! and for 192-byte packets their 4-byte headers as differences), then a `PPM3` (H.264) or
//! `PPH3` (HEVC) stream of the units.

use crate::bits::{Bad, R};
use crate::container::{compress_track, decompress_track, Cancelled, Codec, Options, Progress, Stats};
use crate::model::{code_byte, code_num, ArithDec, ArithEnc, ByteCtx, Model};
use crate::mp4::AvcTrack;
use crate::vp9::stream::{get_varint, put_varint};

pub const MAGIC: &[u8; 4] = b"PPTS";
/// A transport stream packet.
const PKT: usize = 188;
/// Model size of the side stream (log2 slots).
const SIDE_BITS: u32 = 20;
const N_LEN: u32 = 80;
const N_PID: u32 = 81;
const N_SKEL: u32 = 82;
const N_LEAD: u32 = 83;
const N_UNITS: u32 = 84;
const N_ZEROS: u32 = 85;
const N_TRAIL: u32 = 86;
const N_ATS0: u32 = 87;
const N_ATS: u32 = 88;
const K_SKEL: u8 = 30;
const K_LEAD: u8 = 31;

/// How packets sit in the file: `size` bytes each, the 188-byte transport packet at `off`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Layout {
    size: usize,
    off: usize,
}

/// Plain transport stream packets.
const TS: Layout = Layout { size: PKT, off: 0 };
/// BDAV source packets (`.m2ts`, `.mts`): a 4-byte `TP_extra_header` before each packet.
const BDAV: Layout = Layout { size: PKT + 4, off: 4 };

/// The packet layout of `input`, if its first 64 packets (at least 8) have sync bytes.
fn layout(input: &[u8]) -> Option<Layout> {
    [TS, BDAV].into_iter().find(|l| input.len() >= l.size * 8 && input.chunks_exact(l.size).take(64).all(|p| p[l.off] == 0x47))
}

/// The 188-byte transport packets of `input` with what comes before each (the BDAV header).
fn packets(input: &[u8], l: Layout) -> impl Iterator<Item = (&[u8], &[u8; PKT])> {
    input.chunks_exact(l.size).map(move |p| (&p[..l.off], p[l.off..].try_into().expect("packet size")))
}

/// Whether `input` looks like a transport stream with an H.264 or HEVC program.
pub fn probe(input: &[u8]) -> bool {
    layout(input).is_some_and(|l| video(input, l).is_some())
}

/// Offset of a packet's payload, or `None` when it has none.
fn payload_start(pkt: &[u8]) -> Option<usize> {
    let afc = (pkt[3] >> 4) & 3;
    if afc & 1 == 0 {
        return None;
    }
    let off = if afc & 2 != 0 { 5 + pkt[4] as usize } else { 4 };
    (off < PKT).then_some(off)
}

/// The PID and codec of the first H.264 (stream type 0x1B) or HEVC (0x24) stream of the first
/// program in the PAT.
fn video(input: &[u8], l: Layout) -> Option<(u16, Codec)> {
    let mut pmt = None;
    for (_, pkt) in packets(input, l).take(50_000) {
        if pkt[0] != 0x47 {
            return None;
        }
        let pid = ((pkt[1] as u16 & 0x1F) << 8) | pkt[2] as u16;
        if pkt[1] & 0x40 == 0 || (pid != 0 && Some(pid) != pmt) {
            continue;
        }
        let Some(off) = payload_start(pkt) else { continue };
        let pl = &pkt[off..];
        let sec = pl.get(1 + *pl.first()? as usize..)?;
        let len = ((*sec.get(1)? as usize & 0x0F) << 8) | *sec.get(2)? as usize;
        let end = (3 + len).checked_sub(4)?.min(sec.len());
        if pid == 0 {
            if sec[0] != 0 {
                continue;
            }
            for prog in sec.get(8..end)?.as_chunks::<4>().0 {
                if u16::from_be_bytes([prog[0], prog[1]]) != 0 {
                    pmt = Some(((prog[2] as u16 & 0x1F) << 8) | prog[3] as u16);
                    break;
                }
            }
        } else {
            if sec[0] != 2 {
                continue;
            }
            let pil = ((*sec.get(10)? as usize & 0x0F) << 8) | *sec.get(11)? as usize;
            let mut i = 12 + pil;
            while i + 5 <= end {
                let epid = ((sec[i + 1] as u16 & 0x1F) << 8) | sec[i + 2] as u16;
                match sec[i] {
                    0x1B => return Some((epid, Codec::Avc)),
                    0x24 => return Some((epid, Codec::Hevc)),
                    _ => {}
                }
                i += 5 + (((sec[i + 3] as usize & 0x0F) << 8) | sec[i + 4] as usize);
            }
            return None;
        }
    }
    None
}

/// Bytes of a video packet's payload that belong to the PES header (at a payload unit start).
fn pes_header_len(pl: &[u8]) -> Option<usize> {
    if pl.len() < 9 || pl[..3] != [0, 0, 1] || !(0xE0..=0xEF).contains(&pl[3]) {
        return None;
    }
    let n = 9 + pl[8] as usize;
    (n <= pl.len()).then_some(n)
}

/// Splits a transport stream into skeleton, video elementary stream and the 4-byte headers of
/// BDAV packets (as big-endian numbers).
fn split(input: &[u8], vpid: u16, l: Layout) -> Option<(Vec<u8>, Vec<u8>, Vec<u32>)> {
    let mut skel = Vec::with_capacity(input.len() / 8);
    let mut es = Vec::with_capacity(input.len());
    let mut ats = Vec::new();
    let full = input.len() / l.size * l.size;
    for (extra, pkt) in packets(input, l) {
        if pkt[0] != 0x47 {
            return None;
        }
        if let Ok(h) = <[u8; 4]>::try_from(extra) {
            ats.push(u32::from_be_bytes(h));
        }
        let pid = ((pkt[1] as u16 & 0x1F) << 8) | pkt[2] as u16;
        match payload_start(pkt) {
            Some(off) if pid == vpid => {
                let pl = &pkt[off..];
                let h = if pkt[1] & 0x40 != 0 { pes_header_len(pl)? } else { 0 };
                skel.extend_from_slice(&pkt[..off + h]);
                es.extend_from_slice(&pl[h..]);
            }
            _ => skel.extend_from_slice(pkt),
        }
    }
    skel.extend_from_slice(&input[full..]);
    Some((skel, es, ats))
}

/// Pours the elementary stream back into the packets: the inverse of [`split`].
fn merge(skel: &[u8], es: &[u8], ats: &[u32], vpid: u16, file_len: usize, l: Layout) -> R<Vec<u8>> {
    let mut out = Vec::with_capacity(file_len);
    let (mut s, mut e) = (0usize, 0usize);
    if l == BDAV && ats.len() != file_len / l.size {
        return Err(Bad);
    }
    for k in 0..file_len / l.size {
        if s + 4 > skel.len() {
            return Err(Bad);
        }
        if l == BDAV {
            out.extend_from_slice(&ats[k].to_be_bytes());
        }
        let hdr = &skel[s..(s + 5).min(skel.len())];
        if hdr[0] != 0x47 {
            return Err(Bad);
        }
        let pid = ((hdr[1] as u16 & 0x1F) << 8) | hdr[2] as u16;
        let afc = (hdr[3] >> 4) & 3;
        let off = if afc & 2 != 0 { 5 + *hdr.get(4).ok_or(Bad)? as usize } else { 4 };
        if pid == vpid && afc & 1 != 0 && off < PKT {
            let mut h = 0;
            if hdr[1] & 0x40 != 0 {
                let p = skel.get(s + off..).ok_or(Bad)?;
                if p.len() < 9 {
                    return Err(Bad);
                }
                h = 9 + p[8] as usize;
                if off + h > PKT {
                    return Err(Bad);
                }
            }
            out.extend_from_slice(skel.get(s..s + off + h).ok_or(Bad)?);
            s += off + h;
            let n = PKT - off - h;
            out.extend_from_slice(es.get(e..e + n).ok_or(Bad)?);
            e += n;
        } else {
            out.extend_from_slice(skel.get(s..s + PKT).ok_or(Bad)?);
            s += PKT;
        }
    }
    out.extend_from_slice(skel.get(s..).ok_or(Bad)?);
    if out.len() != file_len || e != es.len() {
        return Err(Bad);
    }
    Ok(out)
}

/// An Annex B byte stream as the bytes before the first start code, NAL units with the
/// number of zero bytes before each start code beyond the two it needs, and trailing zeros.
struct AnnexB {
    lead: Vec<u8>,
    units: Vec<(usize, std::ops::Range<usize>)>,
    trail: usize,
}

fn annexb(es: &[u8]) -> AnnexB {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= es.len() {
        if es[i] == 0 && es[i + 1] == 0 && es[i + 2] == 1 {
            starts.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    // a unit runs to the zero run in front of the next start code
    let zero_run = |end: usize, floor: usize| {
        let mut k = end;
        while k > floor && es[k - 1] == 0 {
            k -= 1;
        }
        k
    };
    let Some(&first) = starts.first() else {
        return AnnexB { lead: es.to_vec(), units: Vec::new(), trail: 0 };
    };
    let lead_end = zero_run(first, 0);
    let mut units = Vec::with_capacity(starts.len());
    let mut zeros = first - lead_end;
    for (k, &st) in starts.iter().enumerate() {
        let body = st + 3;
        let (end, next_zeros) = match starts.get(k + 1) {
            Some(&nx) => {
                let e = zero_run(nx, body);
                (e, nx - e)
            }
            None => (zero_run(es.len(), body), 0),
        };
        units.push((zeros, body..end));
        zeros = next_zeros;
    }
    let last_end = units.last().map_or(0, |u| u.1.end);
    AnnexB { lead: es[..lead_end].to_vec(), units, trail: es.len() - last_end }
}

/// Whether HEVC NAL unit `nal` starts a new access unit after a sample holding `vcl` (whether
/// it already has a slice): an access unit delimiter, or, once the sample has a slice, a
/// parameter set, a prefix SEI or reserved non-VCL unit, or the first slice segment of a picture
/// (H.265 7.4.2.4.4). Streams without delimiters split correctly too.
fn hevc_au_start(nal: &[u8], vcl: bool) -> bool {
    match (nal[0] >> 1) & 63 {
        35 => true,
        0..=31 => vcl && nal[2] & 0x80 != 0,
        32..=34 | 39 | 41..=44 | 48..=55 => vcl,
        _ => false,
    }
}

/// Compresses a transport stream with an H.264 or HEVC program; `None` when it is not one this
/// handles or the recoder gains nothing.
pub fn compress_with(input: &[u8], opts: &Options, progress: &mut Progress) -> Result<Option<(Vec<u8>, Stats)>, Cancelled> {
    let Some(l) = layout(input) else { return Ok(None) };
    let Some((vpid, codec)) = video(input, l) else { return Ok(None) };
    let Some((skel, es, ats)) = split(input, vpid, l) else { return Ok(None) };
    if merge(&skel, &es, &ats, vpid, input.len(), l).ok().as_deref() != Some(input) {
        return Ok(None);
    }
    let ab = annexb(&es);
    let min_len = if codec == Codec::Hevc { 3 } else { 1 };
    if ab.units.is_empty() || ab.units.iter().any(|u| u.1.len() < min_len || u.0 > 64) {
        return Ok(None);
    }
    // the units as length-prefixed samples, one per access unit: for H.264 a new sample at each
    // access unit delimiter, for HEVC as in [`hevc_au_start`]
    let mut v = Vec::with_capacity(es.len() + ab.units.len() * 4);
    let mut samples: Vec<(u64, u32)> = Vec::new();
    let mut open = false;
    let mut vcl = false;
    for (_, r) in &ab.units {
        let nal = &es[r.clone()];
        let start = match codec {
            Codec::Avc => nal[0] & 31 == 9,
            Codec::Hevc => hevc_au_start(nal, vcl),
        };
        if start || !open {
            samples.push((v.len() as u64, 0));
            open = true;
            vcl = false;
        }
        vcl |= codec == Codec::Hevc && (nal[0] >> 1) & 63 <= 31;
        v.extend_from_slice(&(nal.len() as u32).to_be_bytes());
        v.extend_from_slice(nal);
        let last = samples.last_mut().expect("opened");
        last.1 = (v.len() as u64 - last.0) as u32;
    }
    let track = AvcTrack { nal_len_size: 4, param_nals: Vec::new(), samples };
    if codec == Codec::Hevc && !crate::hevc::recode::quick_check(&v, &track) {
        return Ok(None); // nothing recodable: let the caller store the file quickly
    }
    let Some((packed, st)) = compress_track(&v, codec, &track, false, opts, progress)? else { return Ok(None) };
    // side stream
    let mut m = Model::new(SIDE_BITS);
    let mut enc = ArithEnc::new();
    code_num(&mut m, &mut enc, N_LEN, input.len() as u64);
    code_num(&mut m, &mut enc, N_PID, vpid as u64 | ((l == BDAV) as u64) << 13);
    code_num(&mut m, &mut enc, N_SKEL, skel.len() as u64);
    let mut bc = ByteCtx::default();
    for &b in &skel {
        code_byte(&mut m, &mut enc, K_SKEL, &mut bc, b);
    }
    code_num(&mut m, &mut enc, N_LEAD, ab.lead.len() as u64);
    let mut bc = ByteCtx::default();
    for &b in &ab.lead {
        code_byte(&mut m, &mut enc, K_LEAD, &mut bc, b);
    }
    code_num(&mut m, &mut enc, N_UNITS, ab.units.len() as u64);
    for (z, _) in &ab.units {
        code_num(&mut m, &mut enc, N_ZEROS, *z as u64);
    }
    code_num(&mut m, &mut enc, N_TRAIL, ab.trail as u64);
    // BDAV headers: the first, then each difference from the previous difference
    let (mut prev, mut step) = (0u32, 0u32);
    for (k, &a) in ats.iter().enumerate() {
        if k == 0 {
            code_num(&mut m, &mut enc, N_ATS0, a as u64);
        } else {
            let d = a.wrapping_sub(prev);
            let dd = d.wrapping_sub(step) as i32;
            code_num(&mut m, &mut enc, N_ATS, ((dd << 1) ^ (dd >> 31)) as u32 as u64);
            step = d;
        }
        prev = a;
    }
    let side = enc.finish();
    let mut out = MAGIC.to_vec();
    put_varint(&mut out, side.len() as u64);
    out.extend_from_slice(&side);
    out.extend_from_slice(&packed);
    if out.len() >= input.len() {
        return Ok(None);
    }
    Ok(Some((out, st)))
}

pub fn decompress_with(data: &[u8], threads: usize, progress: &mut Progress) -> R<Vec<u8>> {
    if data.len() < 5 || &data[..4] != MAGIC {
        return Err(Bad);
    }
    let mut pos = 4;
    let slen = usize::try_from(get_varint(data, &mut pos)?).map_err(|_| Bad)?;
    let side = data.get(pos..pos.checked_add(slen).ok_or(Bad)?).ok_or(Bad)?;
    pos += slen;
    let v = decompress_track(&data[pos..], Some(&[]), threads, progress)?;
    let mut m = Model::new(SIDE_BITS);
    let mut c = ArithDec::new(side);
    let file_len = code_num(&mut m, &mut c, N_LEN, 0);
    if file_len > 1 << 42 {
        return Err(Bad);
    }
    let file_len = file_len as usize;
    let pid = code_num(&mut m, &mut c, N_PID, 0);
    let vpid = (pid & 0x1FFF) as u16;
    let l = match pid >> 13 {
        0 => TS,
        1 => BDAV,
        _ => return Err(Bad),
    };
    let n = code_num(&mut m, &mut c, N_SKEL, 0) as usize;
    if n > file_len {
        return Err(Bad);
    }
    let mut bc = ByteCtx::default();
    let skel: Vec<u8> = (0..n).map(|_| code_byte(&mut m, &mut c, K_SKEL, &mut bc, 0)).collect();
    let lead_n = code_num(&mut m, &mut c, N_LEAD, 0) as usize;
    if lead_n > file_len {
        return Err(Bad);
    }
    let mut bc = ByteCtx::default();
    let lead: Vec<u8> = (0..lead_n).map(|_| code_byte(&mut m, &mut c, K_LEAD, &mut bc, 0)).collect();
    let units = code_num(&mut m, &mut c, N_UNITS, 0) as usize;
    if units > file_len {
        return Err(Bad);
    }
    // the elementary stream: the units are the length-prefixed records of `v`
    let mut es = Vec::with_capacity(file_len);
    es.extend_from_slice(&lead);
    let mut p = 0usize;
    for _ in 0..units {
        let z = code_num(&mut m, &mut c, N_ZEROS, 0) as usize;
        if z > 64 || c.overrun() {
            return Err(Bad);
        }
        let len = u32::from_be_bytes(v.get(p..p + 4).ok_or(Bad)?.try_into().map_err(|_| Bad)?) as usize;
        let nal = v.get(p + 4..p + 4 + len).ok_or(Bad)?;
        es.resize(es.len() + z, 0);
        es.extend_from_slice(&[0, 0, 1]);
        es.extend_from_slice(nal);
        p += 4 + len;
    }
    if p != v.len() {
        return Err(Bad);
    }
    let trail = code_num(&mut m, &mut c, N_TRAIL, 0) as usize;
    if trail > file_len || c.overrun() {
        return Err(Bad);
    }
    es.resize(es.len() + trail, 0);
    let mut ats = Vec::new();
    if l == BDAV {
        let (mut prev, mut step) = (0u32, 0u32);
        for k in 0..file_len / l.size {
            let a = if k == 0 {
                u32::try_from(code_num(&mut m, &mut c, N_ATS0, 0)).map_err(|_| Bad)?
            } else {
                let z = u32::try_from(code_num(&mut m, &mut c, N_ATS, 0)).map_err(|_| Bad)?;
                let dd = ((z >> 1) as i32) ^ -((z & 1) as i32);
                step = step.wrapping_add(dd as u32);
                prev.wrapping_add(step)
            };
            if c.overrun() {
                return Err(Bad);
            }
            ats.push(a);
            prev = a;
        }
    }
    merge(&skel, &es, &ats, vpid, file_len, l)
}

pub fn compress(input: &[u8]) -> Option<(Vec<u8>, Stats)> {
    compress_with(input, &Options::default(), &mut |_, _| true).ok().flatten()
}

pub fn decompress(data: &[u8]) -> R<Vec<u8>> {
    decompress_with(data, 0, &mut |_, _| true)
}

/// Synthetic transport streams for tests.
#[doc(hidden)]
pub mod synth {
    use super::PKT;

    /// A transport stream around an H.264 elementary stream: PAT, PMT, video PES packets
    /// (one per access unit, adaptation-field stuffing in the last packet), and packets of a
    /// second "audio" PID.
    pub fn synthetic_ts(es_units: &[Vec<u8>]) -> Vec<u8> {
        synthetic_ts_with(es_units, 0x1B, false)
    }

    /// [`synthetic_ts`] for video of PMT `stream_type` (0x1B H.264, 0x24 HEVC), as 192-byte
    /// BDAV source packets (a 4-byte arrival time stamp before each packet) when `bdav`.
    pub fn synthetic_ts_with(es_units: &[Vec<u8>], stream_type: u8, bdav: bool) -> Vec<u8> {
        let ts = synthetic_188(es_units, stream_type);
        if !bdav {
            return ts;
        }
        let mut out = Vec::with_capacity(ts.len() / PKT * (PKT + 4));
        for (k, p) in ts.as_chunks::<PKT>().0.iter().enumerate() {
            let ats = (k as u32).wrapping_mul(2_160_000 / 7) & 0x3FFF_FFFF;
            out.extend_from_slice(&ats.to_be_bytes());
            out.extend_from_slice(p);
        }
        out
    }

    fn synthetic_188(es_units: &[Vec<u8>], stream_type: u8) -> Vec<u8> {
        let mut out = Vec::new();
        let mut cc = [0u8; 3];
        let packet = |out: &mut Vec<u8>, pid: u16, start: bool, cc: &mut u8, body: &[u8]| -> usize {
            // returns the bytes of `body` used
            let room = PKT - 4;
            let n = body.len().min(room);
            out.push(0x47);
            out.push(((start as u8) << 6) | (pid >> 8) as u8);
            out.push(pid as u8);
            if n < room {
                // adaptation field with stuffing
                let af = room - n - 1;
                out.push(0x30 | (*cc & 15));
                out.push(af as u8);
                if af > 0 {
                    out.push(0);
                    out.extend(std::iter::repeat_n(0xFF, af - 1));
                }
            } else {
                out.push(0x10 | (*cc & 15));
            }
            out.extend_from_slice(&body[..n]);
            *cc = cc.wrapping_add(1);
            n
        };
        // PAT: program 1 -> PMT PID 0x100; PMT: H.264 on 0x101, audio (0x0F) on 0x102
        let pat = [0u8, 0x00, 0xB0, 0x0D, 0x00, 0x01, 0xC1, 0x00, 0x00, 0x00, 0x01, 0xE1, 0x00, 0, 0, 0, 0];
        let pmt = [
            0u8, 0x02, 0xB0, 0x17, 0x00, 0x01, 0xC1, 0x00, 0x00, 0xE1, 0x01, 0xF0, 0x00, stream_type, 0xE1, 0x01, 0xF0, 0x00,
            0x0F, 0xE1, 0x02, 0xF0, 0x00, 0, 0, 0, 0,
        ];
        let mut pat_cc = 0u8;
        let mut pmt_cc = 0u8;
        let mut pat_b = pat.to_vec();
        pat_b.resize(PKT - 4, 0xFF);
        let mut pmt_b = pmt.to_vec();
        pmt_b.resize(PKT - 4, 0xFF);
        packet(&mut out, 0, true, &mut pat_cc, &pat_b);
        packet(&mut out, 0x100, true, &mut pmt_cc, &pmt_b);
        for (i, au) in es_units.iter().enumerate() {
            let mut pes = vec![0, 0, 1, 0xE0, 0, 0, 0x80, 0x80, 5, 0x21, 0, 1, 0, 1];
            pes.extend_from_slice(au);
            let mut used = 0;
            let mut first = true;
            while used < pes.len() {
                used += packet(&mut out, 0x101, first, &mut cc[0], &pes[used..]);
                first = false;
            }
            if i % 3 == 1 {
                let audio: Vec<u8> = (0..PKT - 4).map(|k| (k * 31 + i * 7) as u8).collect();
                packet(&mut out, 0x102, true, &mut cc[1], &audio);
            }
        }
        out
    }

    /// The synthetic HEVC MP4 of [`crate::hevc::testgen`] (configuration `config`) as a
    /// transport stream: the parameter sets before each IRAP picture, access unit delimiters
    /// when `aud`, 4- and 3-byte start codes in turn; 192-byte packets when `bdav`.
    pub fn from_hevc_mp4(seed: u64, frames: usize, config: &str, aud: bool, bdav: bool) -> Vec<u8> {
        let cfg = crate::hevc::testgen::config(config).expect("known configuration");
        let mp4 = crate::hevc::testgen::synthetic_mp4(seed, frames, &cfg);
        let track = crate::mp4::find_hevc_track(&mp4).expect("synthetic track");
        let mut aus = Vec::new();
        for (k, &(o, s)) in track.samples.iter().enumerate() {
            let smp = &mp4[o as usize..o as usize + s as usize];
            let sc: &[u8] = if k % 2 == 0 { &[0, 0, 0, 1] } else { &[0, 0, 1] };
            let mut au = Vec::new();
            if aud {
                au.extend_from_slice(&[0, 0, 0, 1, 0x46, 0x01, 0x50]);
            }
            let mut nals = Vec::new();
            let mut i = 0;
            while i + 4 <= smp.len() {
                let n = u32::from_be_bytes(smp[i..i + 4].try_into().expect("4 bytes")) as usize;
                nals.push(&smp[i + 4..i + 4 + n]);
                i += 4 + n;
            }
            if nals.iter().any(|n| (16..=23).contains(&((n[0] >> 1) & 63))) {
                for p in &track.param_nals {
                    au.extend_from_slice(sc);
                    au.extend_from_slice(p);
                }
            }
            for n in nals {
                au.extend_from_slice(sc);
                au.extend_from_slice(n);
            }
            aus.push(au);
        }
        synthetic_ts_with(&aus, 0x24, bdav)
    }

    /// The synthetic H.264 MP4 of [`crate::testgen`] (CABAC slices) as a transport stream:
    /// access units with delimiters, 4- and 3-byte start codes in turn.
    pub fn from_mp4(seed: u64, frames: usize, w_mbs: u32, h_mbs: u32) -> Vec<u8> {
        from_mp4_with(seed, frames, w_mbs, h_mbs, false)
    }

    /// [`from_mp4`], in 192-byte BDAV packets when `bdav`.
    pub fn from_mp4_with(seed: u64, frames: usize, w_mbs: u32, h_mbs: u32, bdav: bool) -> Vec<u8> {
        let mp4 = crate::testgen::synthetic_mp4(seed, frames, w_mbs, h_mbs);
        let track = crate::mp4::find_avc_track(&mp4).expect("synthetic track");
        let mut aus = Vec::new();
        for (k, &(o, s)) in track.samples.iter().enumerate() {
            let smp = &mp4[o as usize..o as usize + s as usize];
            let mut au = vec![0, 0, 0, 1, 0x09, 0xF0];
            if k == 0 {
                for p in &track.param_nals {
                    au.extend_from_slice(&[0, 0, 0, 1]);
                    au.extend_from_slice(p);
                }
            }
            let mut i = 0;
            while i + 4 <= smp.len() {
                let n = u32::from_be_bytes(smp[i..i + 4].try_into().expect("4 bytes")) as usize;
                au.extend_from_slice(if k % 2 == 0 { &[0, 0, 0, 1][..] } else { &[0, 0, 1][..] });
                au.extend_from_slice(&smp[i + 4..i + 4 + n]);
                i += 4 + n;
            }
            aus.push(au);
        }
        synthetic_ts_with(&aus, 0x1B, bdav)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(ts: &[u8], l: Layout, codec: Codec) {
        assert!(probe(ts));
        assert_eq!(layout(ts), Some(l));
        assert_eq!(video(ts, l), Some((0x101, codec)));
        let (packed, st) = compress(ts).expect("recodes");
        assert_eq!(st.recoded, st.slices, "{:?}", st.fails);
        assert!(st.recoded > 0);
        assert_eq!(st.hevc, codec == Codec::Hevc);
        assert_eq!(decompress(&packed).unwrap(), ts);
        assert!(decompress(&packed[..packed.len() - 1]).is_err());
    }

    #[test]
    fn ts_round_trip() {
        round_trip(&synth::from_mp4(5, 8, 6, 4), TS, Codec::Avc);
    }

    #[test]
    fn bdav_round_trip() {
        round_trip(&synth::from_mp4_with(6, 8, 6, 4, true), BDAV, Codec::Avc);
    }

    #[test]
    fn hevc_round_trip() {
        for (k, cfg) in ["main", "wpp", "tiles"].into_iter().enumerate() {
            round_trip(&synth::from_hevc_mp4(20 + k as u64, 6, cfg, true, false), TS, Codec::Hevc);
        }
        round_trip(&synth::from_hevc_mp4(30, 6, "main", true, true), BDAV, Codec::Hevc);
    }

    #[test]
    fn hevc_access_units_without_delimiters() {
        // the samples must still be whole pictures: the same sample count as with delimiters
        let with = synth::from_hevc_mp4(40, 8, "main", true, false);
        let without = synth::from_hevc_mp4(40, 8, "main", false, false);
        let (_, a) = compress(&with).expect("recodes");
        let (packed, b) = compress(&without).expect("recodes");
        assert_eq!(a.samples, b.samples);
        assert_eq!(b.recoded, b.slices);
        assert_eq!(decompress(&packed).unwrap(), without);
    }
}
