//! Minimal ISO-BMFF (MP4 / MOV / M4V) reader: finds the H.264 or HEVC video track, its
//! parameter sets and the byte range of every sample, from the sample table or, in
//! fragmented files, from the movie fragments (`moof` boxes).

/// The first H.264 (`avc1`/`avc3`) or HEVC (`hvc1`/`hev1`) track of a file.
pub struct AvcTrack {
    pub nal_len_size: usize,
    /// Parameter-set NAL units (header included) from the `avcC` / `hvcC` box. For HEVC this
    /// also holds the VPS and any SEI units the record carries.
    pub param_nals: Vec<Vec<u8>>,
    /// `(offset, size)` of every sample, in decoding order.
    pub samples: Vec<(u64, u32)>,
}

fn be32(b: &[u8], i: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(i..i + 4)?.try_into().ok()?))
}

fn be64(b: &[u8], i: usize) -> Option<u64> {
    Some(u64::from_be_bytes(b.get(i..i + 8)?.try_into().ok()?))
}

fn be16(b: &[u8], i: usize) -> Option<usize> {
    Some(u16::from_be_bytes(b.get(i..i + 2)?.try_into().ok()?) as usize)
}

/// Iterates over the boxes in `b[start..end]`, yielding `(type, payload_start, end)`.
fn boxes(b: &[u8], start: usize, end: usize) -> impl Iterator<Item = ([u8; 4], usize, usize)> + '_ {
    let mut pos = start;
    std::iter::from_fn(move || {
        if pos + 8 > end {
            return None;
        }
        let size32 = be32(b, pos)? as u64;
        let typ: [u8; 4] = b.get(pos + 4..pos + 8)?.try_into().ok()?;
        let (size, hdr) = match size32 {
            1 => (be64(b, pos + 8)?, 16),
            0 => ((end - pos) as u64, 8),
            s => (s, 8),
        };
        if size < hdr as u64 || pos as u64 + size > end as u64 {
            return None;
        }
        let item = (typ, pos + hdr, pos + size as usize);
        pos += size as usize;
        Some(item)
    })
}

fn child(b: &[u8], start: usize, end: usize, typ: &[u8; 4]) -> Option<(usize, usize)> {
    boxes(b, start, end).find(|(t, _, _)| t == typ).map(|(_, s, e)| (s, e))
}

/// True if the file has a video track (any codec), or is a fragmented MP4.
pub fn has_video_track(file: &[u8]) -> bool {
    if boxes(file, 0, file.len()).any(|(t, _, _)| &t == b"moof") {
        return true;
    }
    let Some((ms, me)) = child(file, 0, file.len(), b"moov") else { return false };
    boxes(file, ms, me).filter(|(t, _, _)| t == b"trak").any(|(_, ts, te)| {
        child(file, ts, te, b"mdia")
            .and_then(|(ds, de)| child(file, ds, de, b"hdlr"))
            .is_some_and(|(hs, he)| he >= hs + 12 && &file[hs + 8..hs + 12] == b"vide")
    })
}

/// Finds the first H.264 track of an MP4/MOV file (plain or fragmented), or of a
/// Matroska/WebM file (see [`crate::mkv`]). Returns `None` for anything else (no `moov`,
/// other codecs, malformed tables).
pub fn find_avc_track(file: &[u8]) -> Option<AvcTrack> {
    find_avc_track_iso(file).or_else(|| crate::mkv::find_avc_track(file))
}

/// The ISO-BMFF part of [`find_avc_track`].
pub fn find_avc_track_iso(file: &[u8]) -> Option<AvcTrack> {
    let (c, samples) = find_track_iso(file, &[b"avc1", b"avc3"], b"avcC")?;
    if c.len() < 7 {
        return None;
    }
    let nal_len_size = (c[4] & 3) as usize + 1;
    let mut param_nals = Vec::new();
    let mut p = 5;
    let nsps = (c[p] & 31) as usize;
    p += 1;
    for _ in 0..nsps {
        let l = be16(c, p)?;
        param_nals.push(c.get(p + 2..p + 2 + l)?.to_vec());
        p += 2 + l;
    }
    let npps = *c.get(p)? as usize;
    p += 1;
    for _ in 0..npps {
        let l = be16(c, p)?;
        param_nals.push(c.get(p + 2..p + 2 + l)?.to_vec());
        p += 2 + l;
    }
    Some(AvcTrack { nal_len_size, param_nals, samples })
}

/// Finds the first HEVC track of an MP4/MOV file (`hvc1`/`hev1` sample entry with an `hvcC`
/// box), or of a Matroska/WebM file (`V_MPEGH/ISO/HEVC`).
pub fn find_hevc_track(file: &[u8]) -> Option<AvcTrack> {
    find_hevc_track_iso(file).or_else(|| crate::mkv::find_hevc_track(file))
}

/// The ISO-BMFF part of [`find_hevc_track`].
pub fn find_hevc_track_iso(file: &[u8]) -> Option<AvcTrack> {
    let (c, samples) = find_track_iso(file, &[b"hvc1", b"hev1"], b"hvcC")?;
    let (nal_len_size, param_nals) = parse_hvcc(c)?;
    Some(AvcTrack { nal_len_size, param_nals, samples })
}

/// NAL length size and the NAL units (VPS, SPS, PPS, SEI) of an
/// `HEVCDecoderConfigurationRecord` (ISO/IEC 14496-15, 8.3.3).
pub fn parse_hvcc(c: &[u8]) -> Option<(usize, Vec<Vec<u8>>)> {
    if c.len() < 23 || c[0] != 1 {
        return None;
    }
    let nal_len_size = (c[21] & 3) as usize + 1;
    let mut nals = Vec::new();
    let mut p = 23;
    for _ in 0..c[22] {
        let n = be16(c, p + 1)?;
        p += 3;
        for _ in 0..n {
            let l = be16(c, p)?;
            nals.push(c.get(p + 2..p + 2 + l)?.to_vec());
            p += 2 + l;
        }
    }
    Some((nal_len_size, nals))
}

/// Payload of a track's decoder configuration box and the track's sample table.
type TrackConfig<'a> = (&'a [u8], Vec<(u64, u32)>);

/// The first video track whose sample entry type is one of `entries`: the payload of its
/// `config` child box and its samples.
fn find_track_iso<'a>(file: &'a [u8], entries: &[&[u8; 4]], config: &[u8; 4]) -> Option<TrackConfig<'a>> {
    let (ms, me) = child(file, 0, file.len(), b"moov")?;
    let mvex = child(file, ms, me, b"mvex");
    for (t, ts, te) in boxes(file, ms, me) {
        if &t != b"trak" {
            continue;
        }
        let Some((ds, de)) = child(file, ts, te, b"mdia") else { continue };
        let Some((hs, he)) = child(file, ds, de, b"hdlr") else { continue };
        if he < hs + 12 || &file[hs + 8..hs + 12] != b"vide" {
            continue;
        }
        let (ns, ne) = child(file, ds, de, b"minf")?;
        let (ss, se) = child(file, ns, ne, b"stbl")?;
        let (sds, sde) = child(file, ss, se, b"stsd")?;
        let (etyp, es, ee) = boxes(file, sds + 8, sde).next()?;
        if !entries.contains(&&etyp) {
            continue;
        }
        // VisualSampleEntry fields take 78 bytes before the child boxes.
        let (cs, ce) = child(file, es + 78, ee, config)?;
        let c = file.get(cs..ce)?;
        let samples = match mvex {
            None => sample_table(file, ss, se)?,
            Some((xs, xe)) => {
                // Fragmented: samples in the `moov` tables (usually none), then those of
                // every movie fragment.
                let (hs, _) = child(file, ts, te, b"tkhd")?;
                let id = be32(file, hs + if *file.get(hs)? == 1 { 20 } else { 12 })?;
                let (zs, _) = child(file, ss, se, b"stsz")?;
                let mut samples = if be32(file, zs + 8)? > 0 { sample_table(file, ss, se)? } else { Vec::new() };
                fragment_samples(file, &trex_sizes(file, xs, xe), id, &mut samples)?;
                samples
            }
        };
        return Some((c, samples));
    }
    None
}

/// `(track_ID, default_sample_size)` of every `trex` box in `mvex`.
fn trex_sizes(file: &[u8], xs: usize, xe: usize) -> Vec<(u32, u32)> {
    boxes(file, xs, xe)
        .filter(|(t, _, _)| t == b"trex")
        .filter_map(|(_, s, _)| Some((be32(file, s + 4)?, be32(file, s + 16)?)))
        .collect()
}

/// Appends the samples of track `id` from every movie fragment, in file order
/// (ISO/IEC 14496-12, 8.8: `tfhd` base offsets and defaults, `trun` sample runs).
fn fragment_samples(file: &[u8], trex: &[(u32, u32)], id: u32, out: &mut Vec<(u64, u32)>) -> Option<()> {
    let mut pos = 0usize;
    for (t, fs, fe) in boxes(file, 0, file.len()) {
        let moof = pos as u64;
        pos = fe;
        if &t != b"moof" {
            continue;
        }
        // Without an explicit base, a track fragment's data starts at the `moof` box (first
        // fragment, or default-base-is-moof) or where the previous one's data ended.
        let mut prev_end: Option<u64> = None;
        for (tt, ts, te) in boxes(file, fs, fe) {
            if &tt != b"traf" {
                continue;
            }
            let (hs, he) = child(file, ts, te, b"tfhd")?;
            let flags = be32(file, hs)? & 0xFF_FFFF;
            let tid = be32(file, hs + 4)?;
            let mut p = hs + 8;
            let mut base = None;
            if flags & 1 != 0 {
                base = Some(be64(file, p)?);
                p += 8;
            }
            if flags & 2 != 0 {
                p += 4; // sample description index
            }
            if flags & 8 != 0 {
                p += 4; // default sample duration
            }
            let mut def_size = trex.iter().find(|e| e.0 == tid).map_or(0, |e| e.1);
            if flags & 0x10 != 0 {
                def_size = be32(file, p)?;
                p += 4;
            }
            if flags & 0x20 != 0 {
                p += 4; // default sample flags
            }
            if p > he {
                return None;
            }
            let base = match (base, prev_end) {
                (Some(b), _) => b,
                (None, Some(end)) if flags & 0x2_0000 == 0 => end,
                _ => moof,
            };
            let mut cur = base;
            for (rt, rs, re) in boxes(file, ts, te) {
                if &rt != b"trun" {
                    continue;
                }
                let rflags = be32(file, rs)? & 0xFF_FFFF;
                let n = be32(file, rs + 4)? as usize;
                let mut q = rs + 8;
                if rflags & 1 != 0 {
                    let off = be32(file, q)? as i32 as i64;
                    cur = u64::try_from(base as i64 + off).ok()?;
                    q += 4;
                }
                if rflags & 4 != 0 {
                    q += 4; // first sample flags
                }
                let per = [0x100, 0x200, 0x400, 0x800].iter().filter(|&&f| rflags & f != 0).count() * 4;
                if n > 50_000_000 || q + n * per > re {
                    return None;
                }
                for i in 0..n {
                    let r = q + i * per + if rflags & 0x100 != 0 { 4 } else { 0 };
                    let size = if rflags & 0x200 != 0 { be32(file, r)? } else { def_size };
                    if tid == id {
                        out.push((cur, size));
                    }
                    cur += size as u64;
                }
            }
            prev_end = Some(cur);
        }
    }
    Some(())
}

fn sample_table(file: &[u8], ss: usize, se: usize) -> Option<Vec<(u64, u32)>> {
    let (zs, _) = child(file, ss, se, b"stsz")?;
    let fixed = be32(file, zs + 4)?;
    let n = be32(file, zs + 8)? as usize;
    if n > 50_000_000 {
        return None;
    }
    let mut sizes = Vec::with_capacity(n);
    for i in 0..n {
        sizes.push(if fixed != 0 { fixed } else { be32(file, zs + 12 + 4 * i)? });
    }
    let offsets: Vec<u64> = if let Some((cs, _)) = child(file, ss, se, b"stco") {
        let nc = be32(file, cs + 4)? as usize;
        (0..nc).map(|i| be32(file, cs + 8 + 4 * i).map(u64::from)).collect::<Option<_>>()?
    } else {
        let (cs, _) = child(file, ss, se, b"co64")?;
        let nc = be32(file, cs + 4)? as usize;
        (0..nc).map(|i| be64(file, cs + 8 + 8 * i)).collect::<Option<_>>()?
    };
    let (scs, _) = child(file, ss, se, b"stsc")?;
    let ne = be32(file, scs + 4)? as usize;
    let mut entries = Vec::with_capacity(ne);
    for i in 0..ne {
        let first = be32(file, scs + 8 + 12 * i)? as usize;
        let per = be32(file, scs + 12 + 12 * i)? as usize;
        entries.push((first, per));
    }
    if entries.is_empty() || entries[0].0 != 1 {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    let mut si = 0;
    let mut e = 0;
    for (ci, &off) in offsets.iter().enumerate() {
        while e + 1 < entries.len() && entries[e + 1].0 <= ci + 1 {
            e += 1;
        }
        let mut o = off;
        for _ in 0..entries[e].1 {
            if si >= n {
                break;
            }
            out.push((o, sizes[si]));
            o += sizes[si] as u64;
            si += 1;
        }
    }
    (si == n).then_some(out)
}
