//! Minimal Matroska / WebM reader: finds a video track by codec ID and the byte range of
//! every frame. For H.264 (`V_MPEG4/ISO/AVC`) it returns an [`AvcTrack`] (the same
//! description the MP4 reader produces, so the recoder handles both containers alike), and
//! the same for HEVC (`V_MPEGH/ISO/HEVC`); the VP9 recoder uses [`find_track`] with `V_VP9`.
//!
//! Only what locating frames needs is parsed: EBML element headers, `Tracks`, and the
//! `SimpleBlock` / `BlockGroup` headers inside `Cluster`s. Laced blocks and tracks with
//! content encodings (compression, encryption, header stripping) are left alone; their
//! bytes simply stay with the container data.

use crate::mp4::AvcTrack;

const EBML: u32 = 0x1A45_DFA3;
const SEGMENT: u32 = 0x1853_8067;
const TRACKS: u32 = 0x1654_AE6B;
const TRACK_ENTRY: u32 = 0xAE;
const TRACK_NUMBER: u32 = 0xD7;
const TRACK_TYPE: u32 = 0x83;
const CODEC_ID: u32 = 0x86;
const CODEC_PRIVATE: u32 = 0x63A2;
const CONTENT_ENCODINGS: u32 = 0x6D80;
const CLUSTER: u32 = 0x1F43_B675;
const SIMPLE_BLOCK: u32 = 0xA3;
const BLOCK_GROUP: u32 = 0xA0;
const BLOCK: u32 = 0xA1;
/// Level-1 elements that end an unknown-size cluster.
const TOP_LEVEL: [u32; 8] =
    [CLUSTER, 0x1C53_BB6B, 0x1254_C367, 0x1043_A770, 0x1941_A469, 0x114D_9B74, 0x1549_A966, TRACKS];

/// Reads an element ID (marker bits kept). Returns (id, length).
fn read_id(b: &[u8], pos: usize) -> Option<(u32, usize)> {
    let first = *b.get(pos)?;
    let len = first.leading_zeros() as usize + 1;
    if len > 4 {
        return None;
    }
    let mut v = 0u32;
    for k in 0..len {
        v = (v << 8) | *b.get(pos + k)? as u32;
    }
    Some((v, len))
}

/// Reads an element size. Returns (size, length); `None` size means "unknown".
fn read_size(b: &[u8], pos: usize) -> Option<(Option<u64>, usize)> {
    let first = *b.get(pos)?;
    let len = first.leading_zeros() as usize + 1;
    if len > 8 {
        return None;
    }
    let mut v = (first as u64) & (0xFF >> len);
    let mut all_ones = v == (0xFF >> len) as u64;
    for k in 1..len {
        let x = *b.get(pos + k)?;
        all_ones &= x == 0xFF;
        v = (v << 8) | x as u64;
    }
    Some((if all_ones { None } else { Some(v) }, len))
}

/// One element: (id, payload start, payload end), where an unknown size extends to `limit`.
fn element(b: &[u8], pos: usize, limit: usize) -> Option<(u32, usize, usize, bool)> {
    let (id, il) = read_id(b, pos)?;
    let (size, sl) = read_size(b, pos + il)?;
    let start = pos + il + sl;
    if start > limit {
        return None;
    }
    match size {
        Some(s) => {
            let end = start.checked_add(usize::try_from(s).ok()?)?;
            (end <= limit).then_some((id, start, end, false))
        }
        None => Some((id, start, limit, true)),
    }
}

fn uint(b: &[u8]) -> u64 {
    b.iter().fold(0u64, |a, &x| a << 8 | x as u64)
}

struct Track {
    number: u64,
    private: Option<Vec<u8>>,
}

/// The first video track with codec ID `want` (and CodecPrivate data if `need_private`).
fn parse_tracks(b: &[u8], start: usize, end: usize, want: &[u8], need_private: bool) -> Option<Track> {
    let mut pos = start;
    while pos < end {
        let (id, s, e, _) = element(b, pos, end)?;
        if id == TRACK_ENTRY {
            let (mut number, mut kind, mut codec, mut private, mut encoded) = (0u64, 0u64, &[][..], None, false);
            let mut p = s;
            while p < e {
                let (cid, cs, ce, _) = element(b, p, e)?;
                match cid {
                    TRACK_NUMBER => number = uint(&b[cs..ce]),
                    TRACK_TYPE => kind = uint(&b[cs..ce]),
                    CODEC_ID => codec = &b[cs..ce],
                    CODEC_PRIVATE => private = Some(b[cs..ce].to_vec()),
                    CONTENT_ENCODINGS => encoded = true,
                    _ => {}
                }
                p = ce;
            }
            let codec = codec.split(|&x| x == 0).next().unwrap_or(&[]);
            if kind == 1 && codec == want && !encoded && number > 0 && (private.is_some() || !need_private) {
                return Some(Track { number, private });
            }
        }
        pos = e;
    }
    None
}

/// Parameter sets and NAL length size from an `AVCDecoderConfigurationRecord`.
fn parse_avcc(c: &[u8]) -> Option<(usize, Vec<Vec<u8>>)> {
    if c.len() < 7 || c[0] != 1 {
        return None;
    }
    let nls = (c[4] & 3) as usize + 1;
    let mut nals = Vec::new();
    let mut p = 5;
    for group in 0..2 {
        let count = if group == 0 { (c[p] & 31) as usize } else { *c.get(p)? as usize };
        p += 1;
        for _ in 0..count {
            let l = u16::from_be_bytes(c.get(p..p + 2)?.try_into().ok()?) as usize;
            nals.push(c.get(p + 2..p + 2 + l)?.to_vec());
            p += 2 + l;
        }
    }
    Some((nls, nals))
}

/// Frame byte range inside a (Simple)Block payload, if it belongs to `track` and is not laced.
fn block_frame(b: &[u8], start: usize, end: usize, track: u64) -> Option<(u64, u32)> {
    let (num, nl) = read_size(b, start)?;
    if num? != track {
        return None;
    }
    let flags = *b.get(start + nl + 2)?;
    if (flags >> 1) & 3 != 0 {
        return None; // laced
    }
    let data = start + nl + 3;
    (data < end).then(|| (data as u64, (end - data) as u32))
}

/// Finds the first H.264 video track of a Matroska/WebM file.
pub fn find_avc_track(file: &[u8]) -> Option<AvcTrack> {
    let (private, samples) = find_track_with(file, b"V_MPEG4/ISO/AVC", true)?;
    let (nal_len_size, param_nals) = parse_avcc(&private?)?;
    Some(AvcTrack { nal_len_size, param_nals, samples })
}

/// Finds the first HEVC (`V_MPEGH/ISO/HEVC`) video track of a Matroska/WebM file.
pub fn find_hevc_track(file: &[u8]) -> Option<AvcTrack> {
    let (private, samples) = find_track_with(file, b"V_MPEGH/ISO/HEVC", true)?;
    let (nal_len_size, param_nals) = crate::mp4::parse_hvcc(&private?)?;
    Some(AvcTrack { nal_len_size, param_nals, samples })
}

/// A track's CodecPrivate data (if any) and the `(offset, size)` of every unlaced frame.
pub type TrackData = (Option<Vec<u8>>, Vec<(u64, u32)>);

/// Finds the first video track with Matroska codec ID `codec` (for example `V_VP9`).
pub fn find_track(file: &[u8], codec: &[u8]) -> Option<TrackData> {
    find_track_with(file, codec, false)
}

fn find_track_with(file: &[u8], codec: &[u8], need_private: bool) -> Option<TrackData> {
    let (id, _, hdr_end, _) = element(file, 0, file.len())?;
    if id != EBML {
        return None;
    }
    // The segment may claim more bytes than the file has (an incomplete download or a cut
    // test clip): read what is there.
    let (id, il) = read_id(file, hdr_end)?;
    let (size, sl) = read_size(file, hdr_end + il)?;
    if id != SEGMENT {
        return None;
    }
    let seg_start = hdr_end + il + sl;
    let seg_end = size.and_then(|s| usize::try_from(s).ok()).map_or(file.len(), |s| seg_start.saturating_add(s).min(file.len()));
    // Tracks come before the clusters in every muxer we know; collect clusters as we go.
    let mut track: Option<Track> = None;
    let mut samples = Vec::new();
    let mut pos = seg_start;
    while pos < seg_end {
        let Some((id, s, e, unknown)) = element(file, pos, seg_end) else { break };
        match id {
            TRACKS => track = track.or_else(|| parse_tracks(file, s, e, codec, need_private)),
            CLUSTER => {
                let t = track.as_ref()?;
                let mut p = s;
                let mut cluster_end = e;
                while p < cluster_end {
                    let Some((cid, cs, ce, _)) = element(file, p, cluster_end) else { break };
                    if unknown && TOP_LEVEL.contains(&cid) {
                        cluster_end = p;
                        break;
                    }
                    match cid {
                        SIMPLE_BLOCK => samples.extend(block_frame(file, cs, ce, t.number)),
                        BLOCK_GROUP => {
                            let mut q = cs;
                            while q < ce {
                                let Some((gid, gs, ge, _)) = element(file, q, ce) else { break };
                                if gid == BLOCK {
                                    samples.extend(block_frame(file, gs, ge, t.number));
                                }
                                q = ge;
                            }
                        }
                        _ => {}
                    }
                    p = ce;
                }
                pos = cluster_end;
                continue;
            }
            _ => {}
        }
        pos = e;
    }
    let t = track?;
    (!samples.is_empty()).then_some((t.private, samples))
}

/// Writes a minimal Matroska file around H.264 frames (length-prefixed NAL units) and an
/// `avcC` record. For tests and for remuxing sample clips.
pub fn write_mkv(avcc: &[u8], frames: &[Vec<u8>], width: u32, height: u32) -> Vec<u8> {
    write_mkv_codec(b"V_MPEG4/ISO/AVC", Some(avcc), frames, width, height)
}

/// Writes a minimal Matroska file with one video track (`codec`, optional CodecPrivate).
pub fn write_mkv_codec(codec: &[u8], private: Option<&[u8]>, frames: &[Vec<u8>], width: u32, height: u32) -> Vec<u8> {
    fn size(v: usize) -> Vec<u8> {
        // 8-byte sizes keep the writer simple and are valid EBML.
        let mut b = vec![0x01];
        b.extend_from_slice(&(v as u64).to_be_bytes()[1..]);
        b
    }
    fn el(id: u32, body: &[u8]) -> Vec<u8> {
        let idb = id.to_be_bytes();
        let skip = idb.iter().position(|&x| x != 0).unwrap_or(3);
        let mut out = idb[skip..].to_vec();
        out.extend(size(body.len()));
        out.extend_from_slice(body);
        out
    }
    fn uint_el(id: u32, v: u64) -> Vec<u8> {
        el(id, &v.to_be_bytes())
    }
    let header = el(EBML, &[el(0x4282, b"matroska"), uint_el(0x4287, 4), uint_el(0x4285, 2)].concat());
    let video = el(0xE0, &[uint_el(0xB0, width as u64), uint_el(0xBA, height as u64)].concat());
    let mut entry = [uint_el(TRACK_NUMBER, 1), uint_el(0x73C5, 1), uint_el(TRACK_TYPE, 1), el(CODEC_ID, codec)].concat();
    if let Some(p) = private {
        entry.extend(el(CODEC_PRIVATE, p));
    }
    entry.extend(video);
    let entry = el(TRACK_ENTRY, &entry);
    let tracks = el(TRACKS, &entry);
    let info = el(0x1549_A966, &[uint_el(0x2AD7B1, 1_000_000), el(0x4D80, b"piedpiper test muxer")].concat());
    let mut clusters = Vec::new();
    for (k, chunk) in frames.chunks(8).enumerate() {
        let mut body = uint_el(0xE7, (k * 8 * 33) as u64);
        for (i, f) in chunk.iter().enumerate() {
            let mut blk = vec![0x81]; // track 1
            blk.extend_from_slice(&((i * 33) as i16).to_be_bytes());
            blk.push(if k == 0 && i == 0 { 0x80 } else { 0x00 });
            blk.extend_from_slice(f);
            body.extend(el(SIMPLE_BLOCK, &blk));
        }
        clusters.extend(el(CLUSTER, &body));
    }
    let segment = el(SEGMENT, &[info, tracks, clusters].concat());
    [header, segment].concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_back_what_the_writer_wrote() {
        let mp4 = crate::testgen::synthetic_mp4(21, 10, 8, 6);
        let t = crate::mp4::find_avc_track(&mp4).unwrap();
        let mut avcc = vec![1, 100, 0, 40, 0xFC | (t.nal_len_size as u8 - 1), 0xE0 | 1];
        avcc.extend_from_slice(&(t.param_nals[0].len() as u16).to_be_bytes());
        avcc.extend_from_slice(&t.param_nals[0]);
        avcc.push(1);
        avcc.extend_from_slice(&(t.param_nals[1].len() as u16).to_be_bytes());
        avcc.extend_from_slice(&t.param_nals[1]);
        let frames: Vec<Vec<u8>> =
            t.samples.iter().map(|&(o, s)| mp4[o as usize..o as usize + s as usize].to_vec()).collect();
        let mkv = write_mkv(&avcc, &frames, 128, 96);
        let m = find_avc_track(&mkv).expect("track");
        assert_eq!(m.nal_len_size, t.nal_len_size);
        assert_eq!(m.param_nals, t.param_nals);
        assert_eq!(m.samples.len(), frames.len());
        for (&(o, s), f) in m.samples.iter().zip(&frames) {
            assert_eq!(&mkv[o as usize..o as usize + s as usize], &f[..]);
        }
        assert!(crate::mp4::find_avc_track_iso(&mkv).is_none());
        assert!(crate::mp4::find_avc_track(&mkv).is_some());
    }

    #[test]
    fn rejects_other_files() {
        assert!(find_avc_track(b"not matroska at all").is_none());
        assert!(find_avc_track(&[0x1A, 0x45, 0xDF, 0xA3, 0x80]).is_none());
    }
}
