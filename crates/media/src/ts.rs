//! H.264 video in MPEG transport streams (188-byte packets): the `PPTS` recompressed format.
//!
//! A transport stream cuts the video's elementary stream into packet payloads and mixes in
//! packet headers, adaptation fields, PES headers and other programs' packets. The encoder
//! splits the file into that "skeleton" (every byte that is not video elementary-stream
//! payload) and the elementary stream itself, an Annex B byte stream. The stream's NAL units
//! become length-prefixed samples that the H.264 recoder ([`crate::container`]) codes as if
//! they came from an MP4 track; the skeleton and the zero bytes around each start code are
//! coded separately. The decoder rebuilds the elementary stream and pours it back into the
//! packets, finding the payloads by re-reading the packet and PES headers in the skeleton.
//!
//! Layout: magic `PPTS`, the length of the side stream (varint), the side stream (file
//! length, video PID, skeleton bytes, bytes before the first start code, number of NAL units
//! and each unit's extra leading zeros, zero bytes at the end), then a `PPM3` stream of the
//! units.

use crate::bits::{Bad, R};
use crate::container::{compress_track, decompress_track, Cancelled, Codec, Options, Progress, Stats};
use crate::model::{code_byte, code_num, ArithDec, ArithEnc, ByteCtx, Model};
use crate::mp4::AvcTrack;
use crate::vp9::stream::{get_varint, put_varint};

pub const MAGIC: &[u8; 4] = b"PPTS";
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
const K_SKEL: u8 = 30;
const K_LEAD: u8 = 31;

/// Whether `input` looks like a transport stream with an H.264 program.
pub fn probe(input: &[u8]) -> bool {
    input.len() >= PKT * 8 && input.as_chunks::<PKT>().0.iter().take(64).all(|p| p[0] == 0x47) && video_pid(input).is_some()
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

/// The PID of the first H.264 stream (type 0x1B) of the first program in the PAT.
fn video_pid(input: &[u8]) -> Option<u16> {
    let mut pmt = None;
    for pkt in input.as_chunks::<PKT>().0.iter().take(50_000) {
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
                if sec[i] == 0x1B {
                    return Some(epid);
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

/// Splits a transport stream into skeleton and video elementary stream.
fn split(input: &[u8], vpid: u16) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut skel = Vec::with_capacity(input.len() / 8);
    let mut es = Vec::with_capacity(input.len());
    let full = input.len() / PKT * PKT;
    for pkt in input[..full].as_chunks::<PKT>().0 {
        if pkt[0] != 0x47 {
            return None;
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
    Some((skel, es))
}

/// Pours the elementary stream back into the packets: the inverse of [`split`].
fn merge(skel: &[u8], es: &[u8], vpid: u16, file_len: usize) -> R<Vec<u8>> {
    let mut out = Vec::with_capacity(file_len);
    let (mut s, mut e) = (0usize, 0usize);
    for _ in 0..file_len / PKT {
        if s + 4 > skel.len() {
            return Err(Bad);
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

/// Compresses a transport stream with an H.264 program; `None` when it is not one this
/// handles or the recoder gains nothing.
pub fn compress_with(input: &[u8], opts: &Options, progress: &mut Progress) -> Result<Option<(Vec<u8>, Stats)>, Cancelled> {
    if !probe(input) {
        return Ok(None);
    }
    let Some(vpid) = video_pid(input) else { return Ok(None) };
    let Some((skel, es)) = split(input, vpid) else { return Ok(None) };
    if merge(&skel, &es, vpid, input.len()).ok().as_deref() != Some(input) {
        return Ok(None);
    }
    let ab = annexb(&es);
    if ab.units.is_empty() || ab.units.iter().any(|u| u.1.is_empty() || u.0 > 64) {
        return Ok(None);
    }
    // the units as length-prefixed samples, a new sample at each access unit delimiter
    let mut v = Vec::with_capacity(es.len() + ab.units.len() * 4);
    let mut samples: Vec<(u64, u32)> = Vec::new();
    let mut open = false;
    for (_, r) in &ab.units {
        let nal = &es[r.clone()];
        if nal[0] & 31 == 9 || !open {
            samples.push((v.len() as u64, 0));
            open = true;
        }
        v.extend_from_slice(&(nal.len() as u32).to_be_bytes());
        v.extend_from_slice(nal);
        let last = samples.last_mut().expect("opened");
        last.1 = (v.len() as u64 - last.0) as u32;
    }
    let track = AvcTrack { nal_len_size: 4, param_nals: Vec::new(), samples };
    let Some((packed, st)) = compress_track(&v, Codec::Avc, &track, false, opts, progress)? else { return Ok(None) };
    // side stream
    let mut m = Model::new(SIDE_BITS);
    let mut enc = ArithEnc::new();
    code_num(&mut m, &mut enc, N_LEN, input.len() as u64);
    code_num(&mut m, &mut enc, N_PID, vpid as u64);
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
    let vpid = u16::try_from(code_num(&mut m, &mut c, N_PID, 0)).map_err(|_| Bad)?;
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
    merge(&skel, &es, vpid, file_len)
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
            0u8, 0x02, 0xB0, 0x17, 0x00, 0x01, 0xC1, 0x00, 0x00, 0xE1, 0x01, 0xF0, 0x00, 0x1B, 0xE1, 0x01, 0xF0, 0x00, 0x0F, 0xE1,
            0x02, 0xF0, 0x00, 0, 0, 0, 0,
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

    /// The synthetic H.264 MP4 of [`crate::testgen`] (CABAC slices) as a transport stream:
    /// access units with delimiters, 4- and 3-byte start codes in turn.
    pub fn from_mp4(seed: u64, frames: usize, w_mbs: u32, h_mbs: u32) -> Vec<u8> {
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
        synthetic_ts(&aus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ts_round_trip() {
        let ts = synth::from_mp4(5, 8, 6, 4);
        assert!(probe(&ts));
        assert_eq!(video_pid(&ts), Some(0x101));
        let (packed, st) = compress(&ts).expect("recodes");
        assert_eq!(st.recoded, st.slices, "{:?}", st.fails);
        assert!(st.recoded > 0);
        assert_eq!(decompress(&packed).unwrap(), ts);
        assert!(decompress(&packed[..packed.len() - 1]).is_err());
    }
}
