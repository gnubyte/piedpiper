//! Synthetic H.264 test files.
//!
//! [`synthetic_mp4`] builds a small MP4 whose video track holds syntactically valid CABAC
//! slices (I, P and B, several slices per picture, 8x8 transforms, in-band parameter sets),
//! produced by driving the slice parser with pseudo-random bins drawn from CABAC's own
//! probabilities and encoding them with the CABAC encoder. The pictures are noise, but the
//! files exercise the recompressor end to end without shipping third-party video.
//!
//! [`synthetic_mp4_cavlc`] does the same for CAVLC streams (Baseline, or High profile with
//! CAVLC), driving the CAVLC slice parser with random but valid syntax.

use crate::bits::{escape, BitWriter};
use crate::h264::cabac::{Encoder, NUM_CTX};
use crate::h264::cavlc;
use crate::h264::mb::{kind, BinIo, Hint, Picture, SliceParser};
use crate::h264::ParamSets;
use crate::bits::R;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

fn ue(w: &mut BitWriter, v: u32) {
    let x = v as u64 + 1;
    let n = 64 - x.leading_zeros();
    w.put_bits(0, n - 1);
    w.put_bits(x as u32, n);
}

fn se(w: &mut BitWriter, v: i32) {
    ue(w, if v > 0 { (2 * v - 1) as u32 } else { (-2 * v) as u32 });
}

fn trailing(mut w: BitWriter) -> Vec<u8> {
    w.put(1);
    w.finish()
}

fn nal(hdr: u8, rbsp: &[u8]) -> Vec<u8> {
    let mut out = vec![hdr];
    escape(rbsp, &mut out);
    out
}

fn sps(w_mbs: u32, h_mbs: u32) -> Vec<u8> {
    sps_with(100, w_mbs, h_mbs)
}

/// An SPS of profile 100 (High) or 66 (Baseline).
fn sps_with(profile: u32, w_mbs: u32, h_mbs: u32) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(profile, 8);
    w.put_bits(0, 8);
    w.put_bits(40, 8);
    ue(&mut w, 0); // sps id
    if profile == 100 {
        ue(&mut w, 1); // chroma_format_idc
        ue(&mut w, 0);
        ue(&mut w, 0);
        w.put(0); // qpprime
        w.put(0); // no scaling matrices
    }
    ue(&mut w, 0); // log2_max_frame_num - 4
    ue(&mut w, 0); // poc type 0
    ue(&mut w, 2); // log2_max_poc_lsb - 4
    ue(&mut w, 2); // max refs
    w.put(0);
    ue(&mut w, w_mbs - 1);
    ue(&mut w, h_mbs - 1);
    w.put(1); // frame_mbs_only
    w.put(1); // direct_8x8_inference
    w.put(0); // no cropping
    w.put(0); // no VUI
    nal(0x67, &trailing(w))
}

fn pps() -> Vec<u8> {
    let mut w = BitWriter::new();
    ue(&mut w, 0);
    ue(&mut w, 0);
    w.put(1); // CABAC
    w.put(0);
    ue(&mut w, 0); // one slice group
    ue(&mut w, 1); // 2 refs in L0
    ue(&mut w, 0);
    w.put(0); // no weighted pred
    w.put_bits(0, 2);
    se(&mut w, 0);
    se(&mut w, 0);
    se(&mut w, 0);
    w.put(1); // deblocking control present
    w.put(0);
    w.put(0);
    w.put(1); // transform_8x8_mode
    w.put(0);
    se(&mut w, 0);
    nal(0x68, &trailing(w))
}

/// A CAVLC PPS: two L0 references by default; with `high`, two L1 references and 8x8
/// transforms.
fn pps_cavlc(high: bool) -> Vec<u8> {
    let mut w = BitWriter::new();
    ue(&mut w, 0);
    ue(&mut w, 0);
    w.put(0); // CAVLC
    w.put(0);
    ue(&mut w, 0); // one slice group
    ue(&mut w, 1); // 2 refs in L0
    ue(&mut w, high as u32);
    w.put(0); // no weighted pred
    w.put_bits(0, 2);
    se(&mut w, 0);
    se(&mut w, 0);
    se(&mut w, 0);
    w.put(1); // deblocking control present
    w.put(0);
    w.put(0);
    if high {
        w.put(1); // transform_8x8_mode
        w.put(0);
        se(&mut w, 0);
    }
    nal(0x68, &trailing(w))
}

/// Draws bins, ends the slice after `mbs` macroblocks and records the CABAC payload.
///
/// Each bin's probability is a fixed pseudo-random function of its syntax element, context
/// and hint fields, so (like real video) the bins depend on more context than CABAC's
/// context index captures.
struct GenIo<'a> {
    rng: &'a mut Rng,
    enc: Encoder,
    mbs: u32,
}

impl BinIo for GenIo<'_> {
    fn decision(&mut self, state: u8, h: &Hint) -> R<u32> {
        const LEVELS: [u32; 6] = [1500, 6000, 20000, 45000, 59000, 64000];
        let mut x = (h.kind as u64) << 56 ^ (h.ctx as u64) << 40 ^ (h.a as u64) << 20 ^ h.b as u64;
        x = (x ^ (x >> 31)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let p1 = LEVELS[((x >> 40) % 6) as usize];
        // Keep QP deltas, reference indices and coefficient magnitudes small, like real
        // streams.
        let p1 = match h.kind {
            kind::DQP => p1 / 16,
            kind::REF => p1 / 4,
            kind::ABS | kind::SIG => p1 / 2,
            _ => p1,
        };
        let bin = ((self.rng.next() & 0xFFFF) < p1) as u32;
        self.enc.decision(state, bin);
        Ok(bin)
    }

    fn bypass(&mut self, _h: &Hint) -> R<u32> {
        let bin = self.rng.next() & 1;
        self.enc.bypass(bin);
        Ok(bin)
    }

    fn terminate(&mut self, h: &Hint) -> R<u32> {
        let bin = if h.kind == kind::END {
            self.mbs -= 1;
            (self.mbs == 0) as u32
        } else {
            0 // never I_PCM
        };
        self.enc.terminate(bin);
        Ok(bin)
    }
}

fn mp4_box(typ: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut b = ((body.len() + 8) as u32).to_be_bytes().to_vec();
    b.extend_from_slice(typ);
    b.extend_from_slice(body);
    b
}

fn full_box(typ: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut v = vec![0, 0, 0, 0];
    v.extend_from_slice(body);
    mp4_box(typ, &v)
}

/// A synthetic MP4 with `frames` pictures of `w_mbs` x `h_mbs` macroblocks.
pub fn synthetic_mp4(seed: u64, frames: usize, w_mbs: u32, h_mbs: u32) -> Vec<u8> {
    let mut rng = Rng(seed ^ 0x5DEE_CE66_D1CE_4E5B);
    let sps_nal = sps(w_mbs, h_mbs);
    let pps_nal = pps();
    let mut ps = ParamSets::default();
    ps.add_nal(&sps_nal);
    ps.add_nal(&pps_nal);
    let n_mbs = w_mbs * h_mbs;
    let mut pic = Picture::default();
    let mut samples: Vec<Vec<u8>> = Vec::new();
    for f in 0..frames {
        let mut sample = Vec::new();
        let push = |sample: &mut Vec<u8>, n: &[u8]| {
            sample.extend_from_slice(&(n.len() as u32).to_be_bytes());
            sample.extend_from_slice(n);
        };
        let idr = f == 0;
        if idr {
            push(&mut sample, &sps_nal);
            push(&mut sample, &pps_nal);
        }
        if f % 4 == 1 {
            // An SEI-like NAL unit that stays raw.
            let sei: Vec<u8> = (0..20).map(|_| rng.below(255) as u8 + 1).collect();
            push(&mut sample, &nal(0x06, &sei));
        }
        let slice_type = if idr { 2 } else if f % 3 == 2 { 1 } else { 0 };
        let nal_ref_idc = if slice_type == 1 { 0 } else { 2 };
        // One or two slices per picture.
        let split = if f % 2 == 1 && n_mbs > 2 { 1 + rng.below(n_mbs - 1) } else { n_mbs };
        let mut first = 0;
        while first < n_mbs {
            let count = if first == 0 { split } else { n_mbs - first };
            let mut w = BitWriter::new();
            ue(&mut w, first);
            ue(&mut w, slice_type + 5);
            ue(&mut w, 0); // pps
            w.put_bits(f as u32 % 16, 4); // frame_num
            if idr {
                ue(&mut w, 0);
            }
            w.put_bits((2 * f as u32) % 64, 6); // poc lsb
            if slice_type == 1 {
                w.put(1); // direct_spatial_mv_pred
            }
            if slice_type != 2 {
                w.put(0); // no ref count override
                w.put(0); // no list modification (L0)
                if slice_type == 1 {
                    w.put(0);
                }
            }
            if nal_ref_idc != 0 {
                if idr {
                    w.put(0);
                    w.put(0);
                } else {
                    w.put(0);
                }
            }
            if slice_type != 2 {
                ue(&mut w, f as u32 % 3); // cabac_init_idc
            }
            se(&mut w, rng.below(9) as i32 - 4); // slice_qp_delta
            ue(&mut w, 0); // deblocking on
            se(&mut w, 0);
            se(&mut w, 0);
            let hdr_bits = w.bit_len();
            while w.bit_len() & 7 != 0 {
                w.put(1);
            }
            let mut rbsp = w.finish();
            let nal_type = if idr { 5 } else { 1 };
            let hdr = (nal_ref_idc << 5 | nal_type) as u8;
            let probe = {
                let mut p = rbsp.clone();
                p.extend_from_slice(&[0xFF; 8]);
                p
            };
            let (sh, sps_ref, pps_ref) = ps.slice_header(&probe, nal_type, nal_ref_idc).expect("synthetic header");
            assert_eq!(sh.data_bit, hdr_bits);
            let mut io = GenIo { rng: &mut rng, enc: Encoder::new(), mbs: count };
            let mut p = SliceParser::new(sps_ref, pps_ref, &sh, &mut pic).expect("parser");
            p.run(&mut io).expect("synthetic slice");
            p.commit();
            rbsp.extend_from_slice(&io.enc.w.finish());
            if f % 5 == 3 {
                rbsp.extend_from_slice(&[0, 0, 0, 0]); // cabac_zero_words
            }
            push(&mut sample, &nal(hdr, &rbsp));
            first += count;
        }
        samples.push(sample);
    }
    let _ = NUM_CTX;
    mp4_file(&mut rng, &samples, &sps_nal, &pps_nal, 100)
}

/// A synthetic MP4 with CAVLC slices of `w_mbs` x `h_mbs` macroblocks: Baseline profile (I and
/// P slices) or, with `high`, High profile with CAVLC (B slices, 8x8 transforms). Pictures
/// have one or more slices; there are skip runs, reference-count overrides, I_PCM macroblocks
/// and trailing zero bytes.
pub fn synthetic_mp4_cavlc(seed: u64, frames: usize, w_mbs: u32, h_mbs: u32, high: bool) -> Vec<u8> {
    let (mut rng, samples, sps_nal, pps_nal, profile) = cavlc_samples(seed, frames, w_mbs, h_mbs, high);
    mp4_file(&mut rng, &samples, &sps_nal, &pps_nal, profile)
}

/// Like [`synthetic_mp4_cavlc`] (Baseline), as a fragmented MP4: an empty sample table in
/// `moov`, then `moof`/`mdat` pairs of one to three pictures with some "audio" of a second
/// track, using each way ISO-BMFF has of locating fragment data (explicit base offset,
/// offset from the `moof` box, continuing after the previous track fragment).
pub fn synthetic_mp4_fragmented(seed: u64, frames: usize, w_mbs: u32, h_mbs: u32) -> Vec<u8> {
    let (mut rng, samples, sps_nal, pps_nal, profile) = cavlc_samples(seed, frames, w_mbs, h_mbs, false);
    mp4_file_fragmented(&mut rng, &samples, &sps_nal, &pps_nal, profile)
}

/// The samples of [`synthetic_mp4_cavlc`], with the generator state for the container and
/// the parameter sets.
fn cavlc_samples(seed: u64, frames: usize, w_mbs: u32, h_mbs: u32, high: bool) -> (Rng, Vec<Vec<u8>>, Vec<u8>, Vec<u8>, u8) {
    let mut rng = Rng(seed ^ 0x0CA7_1C0D_E5EE_D001);
    let mut grng = cavlc::synth::Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let profile = if high { 100 } else { 66 };
    let sps_nal = sps_with(profile, w_mbs, h_mbs);
    let pps_nal = pps_cavlc(high);
    let mut ps = ParamSets::default();
    ps.add_nal(&sps_nal);
    ps.add_nal(&pps_nal);
    let n_mbs = w_mbs * h_mbs;
    let mut cpic = cavlc::Picture::default();
    let mut samples: Vec<Vec<u8>> = Vec::new();
    for f in 0..frames {
        let mut sample = Vec::new();
        let push = |sample: &mut Vec<u8>, n: &[u8]| {
            sample.extend_from_slice(&(n.len() as u32).to_be_bytes());
            sample.extend_from_slice(n);
        };
        let idr = f == 0;
        if idr {
            push(&mut sample, &sps_nal);
            push(&mut sample, &pps_nal);
        }
        let slice_type = if idr || f % 7 == 3 {
            2
        } else if high && f % 3 == 2 {
            1
        } else {
            0
        };
        let nal_ref_idc = if slice_type == 1 { 0 } else { 2 };
        let nal_type = if idr { 5 } else { 1 };
        let mut first = 0;
        while first < n_mbs {
            let mut hw = BitWriter::new();
            ue(&mut hw, first);
            ue(&mut hw, slice_type + 5);
            ue(&mut hw, 0); // pps
            hw.put_bits(f as u32 % 16, 4); // frame_num
            if idr {
                ue(&mut hw, 0);
            }
            hw.put_bits((2 * f as u32) % 64, 6); // poc lsb
            if slice_type == 1 {
                hw.put(rng.below(2)); // direct_spatial_mv_pred
            }
            if slice_type != 2 {
                // sometimes three L0 references (ref_idx is then ue(v) instead of one bit)
                let more_refs = rng.below(3) == 0;
                hw.put(more_refs as u32);
                if more_refs {
                    ue(&mut hw, 2);
                    if slice_type == 1 {
                        ue(&mut hw, rng.below(3));
                    }
                }
                hw.put(0); // no list modification (L0)
                if slice_type == 1 {
                    hw.put(0);
                }
            }
            if nal_ref_idc != 0 {
                if idr {
                    hw.put(0);
                    hw.put(0);
                } else {
                    hw.put(0);
                }
            }
            se(&mut hw, rng.below(9) as i32 - 4); // slice_qp_delta
            ue(&mut hw, 0); // deblocking on
            se(&mut hw, 0);
            se(&mut hw, 0);
            let hdr_bits = hw.bit_len();
            let head = hw.finish();
            let mut probe = head.clone();
            probe.extend_from_slice(&[0xFF; 8]);
            let (sh, sps_ref, pps_ref) = ps.slice_header(&probe, nal_type, nal_ref_idc).expect("synthetic header");
            assert_eq!(sh.data_bit, hdr_bits);
            let mut w = BitWriter::new();
            for i in 0..hdr_bits {
                w.put(((head[i / 8] >> (7 - i % 8)) & 1) as u32);
            }
            let mut io = cavlc::synth::GenIo::new(&mut grng, &mut w);
            io.skip = [100, 300, 700][f % 3];
            io.go_on = if f % 2 == 1 { 900 } else { 995 };
            io.pcm = 20;
            let mut p = cavlc::SliceParser::new(sps_ref, pps_ref, &sh, &mut cpic).expect("parser");
            p.run(&mut io).expect("synthetic slice");
            let mbs = p.mbs() as u32;
            p.commit();
            w.put(1); // rbsp_stop_one_bit
            let mut rbsp = w.finish();
            if f % 5 == 4 {
                rbsp.extend_from_slice(&[0, 0]); // trailing zero bytes
            }
            push(&mut sample, &nal((nal_ref_idc << 5 | nal_type) as u8, &rbsp));
            first += mbs;
        }
        samples.push(sample);
    }
    (rng, samples, sps_nal, pps_nal, profile as u8)
}

/// `avcC` sample entry box (`stsd`) shared by the plain and fragmented layouts.
fn stsd_box(sps_nal: &[u8], pps_nal: &[u8], profile: u8) -> Vec<u8> {
    let mut avcc = vec![1, profile, 0, 40, 0xFF, 0xE1];
    avcc.extend_from_slice(&(sps_nal.len() as u16).to_be_bytes());
    avcc.extend_from_slice(sps_nal);
    avcc.push(1);
    avcc.extend_from_slice(&(pps_nal.len() as u16).to_be_bytes());
    avcc.extend_from_slice(pps_nal);
    let mut entry = vec![0u8; 78];
    entry[7] = 1; // data_reference_index
    entry.extend_from_slice(&mp4_box(b"avcC", &avcc));
    let mut stsd_body = 1u32.to_be_bytes().to_vec();
    stsd_body.extend_from_slice(&mp4_box(b"avc1", &entry));
    full_box(b"stsd", &stsd_body)
}

/// The fragmented MP4 file around `samples` (see [`synthetic_mp4_fragmented`]).
fn mp4_file_fragmented(rng: &mut Rng, samples: &[Vec<u8>], sps_nal: &[u8], pps_nal: &[u8], profile: u8) -> Vec<u8> {
    let ftyp = mp4_box(b"ftyp", b"iso6\0\0\x02\0iso6avc1dash");
    let stbl = mp4_box(
        b"stbl",
        &[stsd_box(sps_nal, pps_nal, profile), full_box(b"stsz", &[0; 8]), full_box(b"stsc", &[0; 4]), full_box(b"stco", &[0; 4])]
            .concat(),
    );
    let mut hdlr_body = vec![0u8; 4];
    hdlr_body.extend_from_slice(b"vide");
    hdlr_body.extend_from_slice(&[0u8; 13]);
    let mdia = mp4_box(b"mdia", &[full_box(b"hdlr", &hdlr_body), mp4_box(b"minf", &stbl)].concat());
    let mut tkhd = vec![0u8; 80];
    tkhd[8..12].copy_from_slice(&1u32.to_be_bytes()); // track_ID
    let trak = mp4_box(b"trak", &[full_box(b"tkhd", &tkhd), mdia].concat());
    let trex = |id: u32, size: u32| {
        let mut b = id.to_be_bytes().to_vec();
        b.extend_from_slice(&[0, 0, 0, 1, 0, 0, 0, 0]);
        b.extend_from_slice(&size.to_be_bytes());
        b.extend_from_slice(&[0; 4]);
        full_box(b"trex", &b)
    };
    let moov = mp4_box(b"moov", &[trak, mp4_box(b"mvex", &[trex(1, 0), trex(2, 7)].concat())].concat());
    let mut out = [ftyp, moov].concat();
    // tfhd / trun builders (flags, then the optional fields in order)
    let tfhd = |flags: u32, id: u32, base: Option<u64>| {
        let mut b = flags.to_be_bytes().to_vec();
        b.extend_from_slice(&id.to_be_bytes());
        if let Some(v) = base {
            b.extend_from_slice(&v.to_be_bytes());
        }
        mp4_box(b"tfhd", &b)
    };
    let trun = |offset: Option<i32>, sizes: &[usize]| {
        let flags = 0x200 | offset.is_some() as u32;
        let mut b = flags.to_be_bytes().to_vec();
        b.extend_from_slice(&(sizes.len() as u32).to_be_bytes());
        if let Some(o) = offset {
            b.extend_from_slice(&o.to_be_bytes());
        }
        for &s in sizes {
            b.extend_from_slice(&(s as u32).to_be_bytes());
        }
        mp4_box(b"trun", &b)
    };
    let mut i = 0;
    let mut seq = 1u32;
    while i < samples.len() {
        let n = (1 + seq as usize % 3).min(samples.len() - i);
        let group = &samples[i..i + n];
        let sizes: Vec<usize> = group.iter().map(Vec::len).collect();
        let video: Vec<u8> = group.concat();
        let audio: Vec<u8> = (0..3 + seq % 17).map(|_| rng.below(256) as u8).collect();
        let layout = seq % 3;
        // `moof` with the given data offsets (fields have fixed widths, so the size does not
        // depend on their values)
        let moof = |data: u64, moof_at: u64| -> Vec<u8> {
            let mfhd = full_box(b"mfhd", &seq.to_be_bytes());
            let trafs = match layout {
                // video first in mdat, offset from the moof box
                0 => mp4_box(b"traf", &[tfhd(0x2_0000, 1, None), trun(Some((data - moof_at) as i32), &sizes)].concat()),
                // audio first; video located by an explicit base offset
                1 => [
                    mp4_box(b"traf", &[tfhd(0x1, 1, Some(data + audio.len() as u64)), trun(None, &sizes)].concat()),
                    mp4_box(b"traf", &[tfhd(0x2_0000, 2, None), trun(Some((data - moof_at) as i32), &[audio.len()])].concat()),
                ]
                .concat(),
                // audio first (data offset from the moof box), then video continuing after it
                _ => [
                    mp4_box(b"traf", &[tfhd(0, 2, None), trun(Some((data - moof_at) as i32), &[audio.len()])].concat()),
                    mp4_box(b"traf", &[tfhd(0, 1, None), trun(None, &sizes)].concat()),
                ]
                .concat(),
            };
            mp4_box(b"moof", &[mfhd, trafs].concat())
        };
        let at = out.len() as u64;
        let len = moof(0, 0).len() as u64;
        let data = at + len + 8;
        out.extend_from_slice(&moof(data, at));
        let body = if layout == 0 { [video, audio].concat() } else { [audio, video].concat() };
        out.extend_from_slice(&mp4_box(b"mdat", &body));
        i += n;
        seq += 1;
    }
    out.extend_from_slice(&mp4_box(b"mfra", &[0u8; 24]));
    out
}

/// The MP4 file around `samples`.
fn mp4_file(rng: &mut Rng, samples: &[Vec<u8>], sps_nal: &[u8], pps_nal: &[u8], profile: u8) -> Vec<u8> {
    let ftyp = mp4_box(b"ftyp", b"isom\0\0\x02\0isomavc1");
    // Encoders often leave padding for later metadata edits.
    let free = mp4_box(b"free", &[0u8; 1500]);
    // mdat: samples with a little "audio" between them.
    let mut mdat_body = Vec::new();
    let mut rel_offsets = Vec::new();
    for (i, s) in samples.iter().enumerate() {
        rel_offsets.push(mdat_body.len());
        mdat_body.extend_from_slice(s);
        let gap: Vec<u8> = (0..(i * 7 % 23)).map(|_| rng.below(256) as u8).collect();
        mdat_body.extend_from_slice(&gap);
    }
    let mdat_start = ftyp.len() + free.len() + 8;
    let stsd = stsd_box(sps_nal, pps_nal, profile);
    let mut stsz_body = 0u32.to_be_bytes().to_vec();
    stsz_body.extend_from_slice(&(samples.len() as u32).to_be_bytes());
    for s in samples {
        stsz_body.extend_from_slice(&(s.len() as u32).to_be_bytes());
    }
    let stsz = full_box(b"stsz", &stsz_body);
    let mut stsc_body = 1u32.to_be_bytes().to_vec();
    stsc_body.extend_from_slice(&[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]); // one sample per chunk
    let stsc = full_box(b"stsc", &stsc_body);
    let mut stco_body = (samples.len() as u32).to_be_bytes().to_vec();
    for o in &rel_offsets {
        stco_body.extend_from_slice(&((mdat_start + o) as u32).to_be_bytes());
    }
    let stco = full_box(b"stco", &stco_body);
    let stbl = mp4_box(b"stbl", &[stsd, stsz, stsc, stco].concat());
    let minf = mp4_box(b"minf", &stbl);
    let mut hdlr_body = vec![0u8; 4];
    hdlr_body.extend_from_slice(b"vide");
    hdlr_body.extend_from_slice(&[0u8; 13]);
    let hdlr = full_box(b"hdlr", &hdlr_body);
    let mdia = mp4_box(b"mdia", &[hdlr, minf].concat());
    let trak = mp4_box(b"trak", &mdia);
    let moov = mp4_box(b"moov", &trak);
    [ftyp, free, mp4_box(b"mdat", &mdat_body), moov].concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crc32(data: &[u8]) -> u32 {
        let mut c = 0xFFFF_FFFFu32;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
        }
        !c
    }

    #[test]
    fn cavlc_roundtrip() {
        for (seed, frames, w, h, high) in
            [(1u64, 8usize, 4u32, 3u32, false), (2, 12, 7, 5, false), (3, 3, 1, 1, false), (4, 10, 6, 4, true), (5, 14, 11, 3, true)]
        {
            let file = synthetic_mp4_cavlc(seed, frames, w, h, high);
            let (packed, st) = crate::compress(&file).expect("recompressible");
            assert_eq!(st.recoded, st.slices, "seed {seed}: some slices were stored raw: {:?}", st.fails);
            assert!(st.slices >= frames);
            assert_eq!(crate::decompress(&packed).unwrap(), file, "seed {seed}");
        }
    }

    #[test]
    fn fragmented_mp4_roundtrip() {
        for (seed, frames, w, h) in [(31u64, 12usize, 6u32, 4u32), (32, 7, 3, 2)] {
            let file = synthetic_mp4_fragmented(seed, frames, w, h);
            let (_, samples, ..) = cavlc_samples(seed, frames, w, h, false);
            let t = crate::mp4::find_avc_track(&file).expect("fragmented track");
            assert_eq!(t.samples.len(), samples.len());
            for (&(o, s), want) in t.samples.iter().zip(&samples) {
                assert_eq!(&file[o as usize..o as usize + s as usize], &want[..], "seed {seed}");
            }
            let (packed, st) = crate::compress(&file).expect("recompressible");
            assert_eq!(st.recoded, st.slices, "seed {seed}: {:?}", st.fails);
            assert_eq!(crate::decompress(&packed).unwrap(), file, "seed {seed}");
        }
    }

    #[test]
    fn cavlc_damaged_slices_stay_exact() {
        // Flipped bits make some slices unparseable (stored raw) and others parse into
        // different syntax; either way the file must come back unchanged.
        let clean = synthetic_mp4_cavlc(9, 10, 5, 4, true);
        for k in 0..12u64 {
            let mut file = clean.clone();
            let mut x = k.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
            for _ in 0..3 {
                x ^= x >> 29;
                x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                // stay inside mdat (after ftyp + free + mdat header), away from the moov
                let pos = 1600 + (x as usize % (file.len() / 2));
                file[pos] ^= 1 << (x >> 61);
            }
            if let Some((packed, _)) = crate::compress(&file) {
                assert_eq!(crate::decompress(&packed).unwrap(), file, "damage pattern {k}");
            }
        }
    }

    #[test]
    fn cabac_damaged_slices_stay_exact() {
        // Damaged CABAC slices are stored raw while the pictures around them are rebuilt for
        // the model; both sides must give the macroblocks of such slices the same content.
        let clean = synthetic_mp4(11, 12, 6, 4);
        let mut raw_seen = false;
        for k in 0..12u64 {
            let mut file = clean.clone();
            let mut x = k.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x2545_F491_4F6C_DD1D;
            for _ in 0..2 {
                x ^= x >> 29;
                x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                let pos = 1600 + (x as usize % (file.len() / 2));
                file[pos] ^= 1 << (x >> 61);
            }
            if let Some((packed, st)) = crate::compress(&file) {
                raw_seen |= st.recoded < st.slices;
                assert_eq!(crate::decompress(&packed).unwrap(), file, "damage pattern {k}");
            }
        }
        assert!(raw_seen, "no damage pattern made a slice raw");
    }

    #[test]
    fn stream_version_follows_entropy_coding() {
        let cabac = synthetic_mp4(12, 5, 3, 2);
        let (packed, _) = crate::compress(&cabac).unwrap();
        assert_eq!(&packed[..4], crate::MAGIC4, "CABAC streams use the decoded pictures");
        let opts = crate::Options { no_pixels: true, ..Default::default() };
        let (old, _) = crate::compress_with(&cabac, &opts, &mut |_, _| true).unwrap().unwrap();
        assert_eq!(&old[..4], crate::MAGIC);
        assert_eq!(crate::decompress(&old).unwrap(), cabac);
        let cavlc = synthetic_mp4_cavlc(13, 5, 3, 2, true);
        let (packed, _) = crate::compress(&cavlc).unwrap();
        assert_eq!(&packed[..4], crate::MAGIC, "CAVLC streams keep the older layout");
    }

    #[test]
    fn synthetic_roundtrip() {
        for (seed, frames, w, h) in [(1u64, 6usize, 4u32, 3u32), (2, 9, 7, 5), (3, 3, 1, 1), (4, 12, 11, 2)] {
            let file = synthetic_mp4(seed, frames, w, h);
            let (packed, st) = crate::compress(&file).expect("recompressible");
            assert_eq!(st.recoded, st.slices, "seed {seed}: some slices were stored raw: {:?}", st.fails);
            assert!(st.slices >= frames);
            assert_eq!(crate::decompress(&packed).unwrap(), file, "seed {seed}");
        }
    }

    #[test]
    fn corrupt_streams_fail_cleanly() {
        let file = synthetic_mp4(5, 4, 3, 3);
        let (packed, _) = crate::compress(&file).unwrap();
        for cut in [5, packed.len() / 3, packed.len() - 1] {
            let r = crate::decompress(&packed[..cut]);
            assert!(r.is_err() || r.unwrap() != file);
        }
        let mut flipped = packed.clone();
        let mid = flipped.len() / 2;
        flipped[mid] ^= 0x40;
        let r = crate::decompress(&flipped);
        assert!(r.is_err() || r.unwrap() != file);
    }

    #[test]
    fn not_mp4_is_rejected() {
        assert!(crate::compress(b"definitely not a video").is_none());
        assert!(crate::compress(&[0u8; 1000]).is_none());
    }

    /// Pins the exact output so that any platform or refactoring that changes the coded
    /// bytes fails loudly (decoders must reproduce the encoder's arithmetic exactly).
    #[test]
    fn output_is_pinned() {
        let file = synthetic_mp4(7, 8, 6, 4);
        let (packed, _) = crate::compress(&file).unwrap();
        assert_eq!((file.len(), crc32(&file)), PINNED_INPUT, "synthetic input changed");
        assert_eq!((packed.len(), crc32(&packed)), PINNED_OUTPUT, "compressed output changed");
    }

    /// The older `PPM3` layout (no decoded pictures) must not change either: files written
    /// by earlier releases are decoded with it.
    #[test]
    fn ppm3_output_is_pinned() {
        let file = synthetic_mp4(7, 8, 6, 4);
        let opts = crate::Options { no_pixels: true, ..Default::default() };
        let (packed, _) = crate::compress_with(&file, &opts, &mut |_, _| true).unwrap().unwrap();
        assert_eq!((packed.len(), crc32(&packed)), PINNED_OUTPUT_PPM3, "PPM3 output changed");
    }

    const PINNED_INPUT: (usize, u32) = (3342, 1160179443);
    const PINNED_OUTPUT: (usize, u32) = (1417, 1144486891);
    const PINNED_OUTPUT_PPM3: (usize, u32) = (1407, 1290394572);
}
