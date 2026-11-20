//! Synthetic HEVC test files.
//!
//! [`synthetic_mp4`] builds a small MP4 whose video track holds syntactically valid HEVC
//! slice segments, produced like the H.264 generator's: the slice parser is driven with
//! pseudo-random bins and the bins are CABAC-encoded. A [`Config`] switches on the coding
//! tools to exercise (wavefronts, tiles, dependent slice segments, AMP, transform skip,
//! lossless CUs, sign hiding, QP deltas, SAO, 10-bit, monochrome). The pictures are noise,
//! but the files exercise the recompressor end to end without shipping third-party video.

use super::ctu::{kind, Picture, SliceParser};
use super::{Geometry, ParamSets, B, I, P};
use crate::bits::{escape, BitWriter, R};
use crate::h264::cabac::Encoder;
use crate::h264::mb::{BinIo, Hint};

/// Coding tools and picture layout of a synthetic stream.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub width: u32,
    pub height: u32,
    pub log2_ctb: u32,
    pub log2_min_cb: u32,
    pub log2_min_tb: u32,
    pub log2_max_tb: u32,
    pub depth_inter: u32,
    pub depth_intra: u32,
    /// 0 = monochrome, 1 = 4:2:0.
    pub chroma: u32,
    pub bit_depth: u32,
    pub amp: bool,
    pub sao: bool,
    pub transform_skip: bool,
    pub transquant_bypass: bool,
    pub sign_hiding: bool,
    /// diff_cu_qp_delta_depth, or `None` for no CU QP deltas.
    pub cu_qp_delta: Option<u32>,
    pub cabac_init_present: bool,
    pub wpp: bool,
    /// Uniformly spaced tile columns and rows.
    pub tiles: Option<(u32, u32)>,
    pub dependent_slices: bool,
    pub max_merge_cand: u32,
    pub scaling_lists: bool,
    /// Parameter sets only in the `hvcC` box (`hvc1`) instead of also in band (`hev1`).
    pub out_of_band: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            width: 136,
            height: 72,
            log2_ctb: 5,
            log2_min_cb: 3,
            log2_min_tb: 2,
            log2_max_tb: 4,
            depth_inter: 2,
            depth_intra: 1,
            chroma: 1,
            bit_depth: 8,
            amp: true,
            sao: true,
            transform_skip: true,
            transquant_bypass: false,
            sign_hiding: true,
            cu_qp_delta: Some(1),
            cabac_init_present: true,
            wpp: false,
            tiles: None,
            dependent_slices: false,
            max_merge_cand: 5,
            scaling_lists: false,
            out_of_band: false,
        }
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n.max(1)
    }
}

fn ue(w: &mut BitWriter, v: u32) {
    let x = v as u64 + 1;
    let n = 64 - x.leading_zeros();
    w.put_bits(0, n - 1);
    for i in (0..n).rev() {
        w.put(((x >> i) & 1) as u32);
    }
}

fn se(w: &mut BitWriter, v: i32) {
    ue(w, if v > 0 { (2 * v - 1) as u32 } else { (-2 * v) as u32 });
}

fn trailing(mut w: BitWriter) -> Vec<u8> {
    w.put(1);
    w.finish()
}

fn nal(typ: u32, tid: u32, rbsp: &[u8]) -> Vec<u8> {
    let mut out = vec![(typ << 1) as u8, (tid + 1) as u8];
    escape(rbsp, &mut out);
    out
}

fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

fn profile_tier_level(w: &mut BitWriter, profile: u32) {
    w.put_bits(profile, 8); // general_profile_space 0, tier 0, profile_idc
    w.put_bits(1 << (31 - profile), 32);
    w.put_bits(0b1001 << 12, 16); // progressive, frame-only; no constraint flags
    w.put_bits(0, 32);
    w.put_bits(93, 8); // level 3.1
}

fn vps(c: &Config) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4);
    w.put(1);
    w.put(1);
    w.put_bits(0, 6);
    w.put_bits(0, 3);
    w.put(1);
    w.put_bits(0xFFFF, 16);
    profile_tier_level(&mut w, if c.bit_depth > 8 { 2 } else { 1 });
    w.put(1);
    ue(&mut w, 4);
    ue(&mut w, 2);
    ue(&mut w, 0);
    w.put_bits(0, 6);
    ue(&mut w, 0);
    w.put(0);
    w.put(0);
    nal(32, 0, &trailing(w))
}

fn sps(c: &Config) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_bits(0, 4); // vps id
    w.put_bits(0, 3); // max_sub_layers_minus1
    w.put(1);
    profile_tier_level(&mut w, if c.bit_depth > 8 { 2 } else { 1 });
    ue(&mut w, 0); // sps id
    ue(&mut w, c.chroma);
    ue(&mut w, c.width);
    ue(&mut w, c.height);
    w.put(0); // no conformance window
    ue(&mut w, c.bit_depth - 8);
    ue(&mut w, c.bit_depth - 8);
    ue(&mut w, 4); // log2_max_pic_order_cnt_lsb = 8
    w.put(1);
    ue(&mut w, 4);
    ue(&mut w, 2);
    ue(&mut w, 0);
    ue(&mut w, c.log2_min_cb - 3);
    ue(&mut w, c.log2_ctb - c.log2_min_cb);
    ue(&mut w, c.log2_min_tb - 2);
    ue(&mut w, c.log2_max_tb - c.log2_min_tb);
    ue(&mut w, c.depth_inter);
    ue(&mut w, c.depth_intra);
    w.put(c.scaling_lists as u32);
    if c.scaling_lists {
        w.put(1); // sps_scaling_list_data_present_flag
        for size_id in 0..4 {
            let step = if size_id == 3 { 3 } else { 1 };
            let mut m = 0;
            while m < 6 {
                if (size_id + m) % 2 == 0 {
                    w.put(0);
                    ue(&mut w, 0);
                } else {
                    w.put(1);
                    let n = (1usize << (4 + (size_id << 1))).min(64);
                    if size_id > 1 {
                        se(&mut w, 3);
                    }
                    for i in 0..n {
                        se(&mut w, (i % 3) as i32 - 1);
                    }
                }
                m += step;
            }
        }
    }
    w.put(c.amp as u32);
    w.put(c.sao as u32);
    w.put(0); // pcm
              // Short-term reference picture sets: {-1}, {-1, -2} (predicted from set 0), {-1, +1}.
    ue(&mut w, 3);
    ue(&mut w, 1);
    ue(&mut w, 0);
    ue(&mut w, 0);
    w.put(1);
    w.put(1); // set 1: inter_ref_pic_set_prediction_flag
    w.put(1); // delta_rps_sign
    ue(&mut w, 0); // abs_delta_rps_minus1 -> deltaRps = -1
    w.put(1); // used_by_curr_pic_flag[0]
    w.put(1); // used_by_curr_pic_flag[1]
    w.put(0); // set 2: explicit
    ue(&mut w, 1);
    ue(&mut w, 1);
    ue(&mut w, 0);
    w.put(1);
    ue(&mut w, 0);
    w.put(1);
    w.put(1); // long_term_ref_pics_present_flag
    ue(&mut w, 1); // num_long_term_ref_pics_sps
    w.put_bits(0, 8);
    w.put(0);
    w.put(1); // sps_temporal_mvp_enabled_flag
    w.put(1); // strong_intra_smoothing_enabled_flag
    w.put(1); // vui_parameters_present_flag
    w.put(0); // aspect ratio
    w.put(0); // overscan
    w.put(1); // video signal type
    w.put_bits(5, 3);
    w.put(0);
    w.put(1);
    w.put_bits(0x010101, 24);
    w.put(0); // chroma loc
    w.put_bits(0, 3);
    w.put(0); // default display window
    w.put(1); // timing
    w.put_bits(1001, 32);
    w.put_bits(30000, 32);
    w.put(0);
    w.put(1); // hrd parameters
    w.put(1); // nal_hrd_parameters_present_flag
    w.put(0);
    w.put(0); // sub_pic_hrd_params_present_flag
    w.put_bits(0, 4);
    w.put_bits(0, 4);
    w.put_bits(23, 5);
    w.put_bits(23, 5);
    w.put_bits(23, 5);
    w.put(0); // fixed_pic_rate_general_flag
    w.put(0); // fixed_pic_rate_within_cvs_flag
    w.put(0); // low_delay_hrd_flag
    ue(&mut w, 1); // cpb_cnt_minus1
    for _ in 0..2 {
        ue(&mut w, 1000);
        ue(&mut w, 2000);
        w.put(0);
    }
    w.put(1); // bitstream_restriction_flag
    w.put_bits(0, 3);
    for v in [0, 2, 1, 15, 15] {
        ue(&mut w, v);
    }
    w.put(0); // sps_extension_present_flag
    nal(33, 0, &trailing(w))
}

fn pps(c: &Config) -> Vec<u8> {
    let mut w = BitWriter::new();
    ue(&mut w, 0);
    ue(&mut w, 0);
    w.put(c.dependent_slices as u32);
    w.put(0); // output_flag_present_flag
    w.put_bits(1, 3); // num_extra_slice_header_bits
    w.put(c.sign_hiding as u32);
    w.put(c.cabac_init_present as u32);
    ue(&mut w, 1); // two L0 references by default
    ue(&mut w, 0);
    se(&mut w, 0);
    w.put(0);
    w.put(c.transform_skip as u32);
    w.put(c.cu_qp_delta.is_some() as u32);
    if let Some(d) = c.cu_qp_delta {
        ue(&mut w, d);
    }
    se(&mut w, 1);
    se(&mut w, -1);
    w.put(1); // pps_slice_chroma_qp_offsets_present_flag
    w.put(1); // weighted_pred_flag
    w.put(0);
    w.put(c.transquant_bypass as u32);
    w.put(c.tiles.is_some() as u32);
    w.put(c.wpp as u32);
    if let Some((cols, rows)) = c.tiles {
        ue(&mut w, cols - 1);
        ue(&mut w, rows - 1);
        w.put(1);
        w.put(1);
    }
    w.put(1); // pps_loop_filter_across_slices_enabled_flag
    w.put(1); // deblocking_filter_control_present_flag
    w.put(1);
    w.put(0);
    se(&mut w, 0);
    se(&mut w, 0);
    w.put(0); // pps_scaling_list_data_present_flag
    w.put(1); // lists_modification_present_flag
    ue(&mut w, 0);
    w.put(0);
    w.put(0);
    nal(34, 0, &trailing(w))
}

/// Draws bins, ends the segment after `ctus` coding tree units and records where each CABAC
/// substream ends.
///
/// Each bin's probability is a fixed pseudo-random function of its syntax element and CABAC
/// context only, so the generated streams do not change when the model's hints are tuned.
struct GenIo<'a> {
    rng: &'a mut Rng,
    enc: Encoder,
    ctus: u32,
    subs: Vec<usize>,
}

impl BinIo for GenIo<'_> {
    fn decision(&mut self, state: u8, h: &Hint) -> R<u32> {
        const LEVELS: [u32; 6] = [1500, 6000, 20000, 45000, 59000, 64000];
        let mut x = (h.kind as u64) << 56 ^ (h.ctx as u64) << 40;
        x = (x ^ (x >> 31)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let p1 = LEVELS[((x >> 40) % 6) as usize];
        let p1 = match h.kind {
            kind::DQP => p1 / 16,
            kind::REF_IDX | kind::LAST => p1 / 3,
            kind::SIG | kind::GT1 | kind::CSBF => p1 / 2,
            _ => p1,
        };
        let bin = ((self.rng.next() & 0xFFFF) < p1) as u32;
        self.enc.decision(state, bin);
        Ok(bin)
    }

    fn bypass(&mut self, h: &Hint) -> R<u32> {
        // Keep unary prefixes short, as in real streams.
        let p1 = match h.kind {
            kind::REM if h.a & 16 == 0 => 22000,
            kind::MVD_SUF if h.a < 32 => 22000,
            kind::DQP => 8000,
            _ => 32768,
        };
        let bin = ((self.rng.next() & 0xFFFF) < p1) as u32;
        self.enc.bypass(bin);
        Ok(bin)
    }

    fn terminate(&mut self, h: &Hint) -> R<u32> {
        let bin = if h.kind == kind::END {
            self.ctus -= 1;
            (self.ctus == 0) as u32
        } else {
            0 // never PCM
        };
        self.enc.terminate(bin);
        Ok(bin)
    }

    fn end_substream(&mut self) -> R<()> {
        self.enc.terminate(1);
        self.enc.restart();
        self.subs.push(self.enc.w.buf.len());
        Ok(())
    }
}

/// Fields of a slice segment header the generator varies.
struct Seg {
    nal_type: u32,
    first: bool,
    dependent: bool,
    address: u32,
    slice_type: u32,
    poc: u32,
    qp_delta: i32,
    cabac_init: bool,
    variant: u32,
}

fn slice_header(c: &Config, n_ctb: u32, s: &Seg, entry_points: &[u32]) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put(s.first as u32);
    if (16..=23).contains(&s.nal_type) {
        w.put(0);
    }
    ue(&mut w, 0);
    if !s.first {
        if c.dependent_slices {
            w.put(s.dependent as u32);
        }
        w.put_bits(s.address, ceil_log2(n_ctb));
    }
    if !s.dependent {
        w.put(s.variant & 1); // slice_reserved_flag
        ue(&mut w, s.slice_type);
        let idr = s.nal_type == 19 || s.nal_type == 20;
        // NumPicTotalCurr of the reference picture set chosen below.
        let total_curr = if idr {
            0
        } else if s.variant % 3 == 2 {
            3
        } else if s.slice_type == B {
            2
        } else {
            [1, 2][(s.variant % 2) as usize]
        };
        if !idr {
            w.put_bits(s.poc & 255, 8);
            if s.variant % 3 == 2 {
                // An explicit set, predicted from SPS set 1: {-1, -2} shifted by -1 -> {-2, -3}
                // plus -1.
                w.put(0);
                w.put(1);
                ue(&mut w, 1); // delta_idx_minus1 -> set 1
                w.put(1);
                ue(&mut w, 0);
                for _ in 0..3 {
                    w.put(1);
                }
            } else {
                w.put(1);
                w.put_bits(if s.slice_type == B { 2 } else { s.variant % 2 }, 2);
            }
            ue(&mut w, 0); // num_long_term_sps
            ue(&mut w, (s.variant % 5 == 1) as u32); // num_long_term_pics
            if s.variant % 5 == 1 {
                w.put_bits(7, 8);
                w.put(0);
                w.put(1);
                ue(&mut w, 1);
            }
            w.put(1); // slice_temporal_mvp_enabled_flag
        }
        if c.sao {
            w.put((s.variant % 4 != 3) as u32);
            if c.chroma != 0 {
                w.put((s.variant % 3 != 1) as u32);
            }
        }
        let refs = if s.slice_type == P { 1 + s.variant % 2 } else { 1 };
        if s.slice_type != I {
            w.put(1); // num_ref_idx_active_override_flag
            ue(&mut w, refs - 1);
            if s.slice_type == B {
                ue(&mut w, 0);
            }
            if total_curr > 1 {
                // ref_pic_lists_modification() (lists_modification_present_flag is set)
                let bits = ceil_log2(total_curr);
                w.put(1);
                for i in 0..refs {
                    w.put_bits(i % total_curr, bits);
                }
                if s.slice_type == B {
                    w.put(s.variant & 1);
                    if s.variant & 1 == 1 {
                        w.put_bits(1, bits);
                    }
                }
            }
            if s.slice_type == B {
                w.put(s.variant % 2); // mvd_l1_zero_flag
            }
            if c.cabac_init_present {
                w.put(s.cabac_init as u32);
            }
            if s.slice_type == B {
                w.put(1); // collocated_from_l0_flag
            }
            if refs > 1 {
                ue(&mut w, 0);
            }
            if s.slice_type == P {
                // pred_weight_table
                ue(&mut w, 6);
                if c.chroma != 0 {
                    se(&mut w, 0);
                }
                for i in 0..refs {
                    w.put(i & 1);
                }
                if c.chroma != 0 {
                    for _ in 0..refs {
                        w.put(1);
                    }
                }
                for i in 0..refs {
                    if i & 1 == 1 {
                        se(&mut w, 2);
                        se(&mut w, -3);
                    }
                    if c.chroma != 0 {
                        for _ in 0..4 {
                            se(&mut w, 1);
                        }
                    }
                }
            }
            ue(&mut w, 5 - c.max_merge_cand);
        }
        se(&mut w, s.qp_delta);
        se(&mut w, 1);
        se(&mut w, -2);
        w.put(s.variant % 2); // deblocking_filter_override_flag
        if s.variant % 2 == 1 {
            w.put(0);
            se(&mut w, 1);
            se(&mut w, -1);
        }
        w.put(1); // slice_loop_filter_across_slices_enabled_flag
    }
    if c.tiles.is_some() || c.wpp {
        ue(&mut w, entry_points.len() as u32);
        if !entry_points.is_empty() {
            let max = entry_points.iter().map(|&e| e - 1).max().unwrap_or(0);
            let len = (32 - max.leading_zeros()).max(1);
            ue(&mut w, len - 1);
            for &e in entry_points {
                w.put_bits(e - 1, len);
            }
        }
    }
    w.put(1);
    while w.bit_len() & 7 != 0 {
        w.put(0);
    }
    w.finish()
}

/// Splits a picture into slice segments: `(first tile-scan address, CTUs, dependent)`.
fn plan_picture(c: &Config, geo: &Geometry, rng: &mut Rng, f: usize) -> Vec<(u32, u32, bool)> {
    let n = (geo.w_ctb * geo.h_ctb) as u32;
    let w = geo.w_ctb as u32;
    // Allowed cut points (tile-scan addresses where a segment may start).
    let cuts: Vec<u32> = if let Some((cols, rows)) = c.tiles {
        // Whole tiles.
        let _ = (cols, rows);
        (1..n).filter(|&ts| geo.tile_ts(ts as usize) != geo.tile_ts(ts as usize - 1)).collect()
    } else if c.wpp {
        (1..n).filter(|&ts| ts % w == 0).collect()
    } else {
        (1..n).collect()
    };
    let mut starts = vec![0u32];
    if f % 2 == 1 {
        for &cut in &cuts {
            if rng.below(3) == 0 {
                starts.push(cut);
            }
        }
    }
    let mut out = Vec::new();
    for (i, &s) in starts.iter().enumerate() {
        let e = starts.get(i + 1).copied().unwrap_or(n);
        let dependent = c.dependent_slices && i > 0 && rng.below(2) == 0;
        out.push((s, e - s, dependent));
    }
    out
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

fn hvcc(c: &Config, params: &[Vec<u8>]) -> Vec<u8> {
    let profile = if c.bit_depth > 8 { 2u8 } else { 1 };
    let mut v = vec![1, profile];
    v.extend_from_slice(&(1u32 << (31 - profile)).to_be_bytes());
    v.extend_from_slice(&[0x90, 0, 0, 0, 0, 0, 93, 0xF0, 0x00, 0xFC | ((c.wpp as u8) * 3)]);
    v.push(0xFC | c.chroma as u8);
    v.push(0xF8 | (c.bit_depth - 8) as u8);
    v.push(0xF8 | (c.bit_depth - 8) as u8);
    v.extend_from_slice(&[0, 0, 0x0F]); // avgFrameRate, 1 temporal layer, nested, 4-byte lengths
    v.push(params.len() as u8);
    for p in params {
        v.push(0x80 | (p[0] >> 1)); // array_completeness, NAL unit type
        v.extend_from_slice(&1u16.to_be_bytes());
        v.extend_from_slice(&(p.len() as u16).to_be_bytes());
        v.extend_from_slice(p);
    }
    v
}

/// Parameter sets, samples (length-prefixed NAL units) and the generator state after them.
type Stream = ([Vec<u8>; 3], Vec<Vec<u8>>, Rng);

/// A synthetic HEVC MP4 with `frames` pictures.
pub fn synthetic_mp4(seed: u64, frames: usize, c: &Config) -> Vec<u8> {
    let (params, samples, rng) = stream(seed, frames, c);
    mp4(c, &params, &samples, rng)
}

/// The same stream in a Matroska file (`V_MPEGH/ISO/HEVC`).
pub fn synthetic_mkv(seed: u64, frames: usize, c: &Config) -> Vec<u8> {
    let (params, samples, _) = stream(seed, frames, c);
    crate::mkv::write_mkv_codec(b"V_MPEGH/ISO/HEVC", Some(&hvcc(c, &params)), &samples, c.width, c.height)
}

fn stream(seed: u64, frames: usize, c: &Config) -> Stream {
    let mut rng = Rng(seed ^ 0x2545_F491_4F6C_DD1D);
    let params = [vps(c), sps(c), pps(c)];
    let mut ps = ParamSets::default();
    for p in &params {
        ps.add_nal(p);
    }
    let sps_ref = ps.sps[0].as_ref().expect("synthetic SPS");
    let pps_ref = ps.pps[0].as_ref().expect("synthetic PPS");
    assert!(sps_ref.supported() && !pps_ref.unsupported);
    let geo = Geometry::new(sps_ref, pps_ref).expect("synthetic geometry");
    let n_ctb = (geo.w_ctb * geo.h_ctb) as u32;
    let mut pic = Picture::default();
    let mut samples: Vec<Vec<u8>> = Vec::new();
    for f in 0..frames {
        let mut sample = Vec::new();
        let push = |sample: &mut Vec<u8>, n: &[u8]| {
            sample.extend_from_slice(&(n.len() as u32).to_be_bytes());
            sample.extend_from_slice(n);
        };
        let idr = f == 0;
        if idr && !c.out_of_band {
            for p in &params {
                push(&mut sample, p);
            }
        }
        if f % 4 == 1 {
            // A prefix SEI that stays raw.
            let sei: Vec<u8> = (0..20).map(|_| rng.below(255) as u8 + 1).collect();
            push(&mut sample, &nal(39, 0, &sei));
        }
        let (slice_type, nal_type, tid) = if idr {
            (I, 19, 0)
        } else if f % 7 == 5 {
            (I, 21, 0) // CRA
        } else if f % 3 == 2 {
            (B, 0, 1)
        } else {
            (P, 1, 0)
        };
        let mut last_indep = None;
        for (start, count, dependent) in plan_picture(c, &geo, &mut rng, f) {
            let seg = Seg {
                nal_type,
                first: start == 0,
                dependent,
                address: geo.ts_to_rs[start as usize],
                slice_type,
                poc: 2 * f as u32,
                qp_delta: rng.below(9) as i32 - 4,
                cabac_init: f % 2 == 0,
                variant: rng.below(30),
            };
            // Parse the header once without entry points to drive the parser.
            let probe = slice_header(c, n_ctb, &seg, &[]);
            let sh = ps.slice_header(&probe, nal_type, tid, last_indep.as_ref()).expect("synthetic header");
            assert_eq!(sh.data_bit, probe.len() * 8);
            let mut io = GenIo { rng: &mut rng, enc: Encoder::new(), ctus: count, subs: Vec::new() };
            let mut p = SliceParser::new(sps_ref, pps_ref, &sh, &geo, &mut pic).expect("parser");
            p.run(&mut io).expect("synthetic slice");
            p.commit();
            let subs = io.subs;
            let data = io.enc.w.finish();
            // entry_point_offset_minus1 + 1 counts the bytes of each substream as it appears
            // in the NAL unit, emulation-prevention bytes included.
            let mut eps = Vec::new();
            let mut prev = 0;
            for &end in &subs {
                let mut esc = Vec::new();
                escape(&data[prev..end], &mut esc);
                eps.push(esc.len() as u32);
                prev = end;
            }
            let mut rbsp = slice_header(c, n_ctb, &seg, &eps);
            rbsp.extend_from_slice(&data);
            if f % 5 == 3 {
                rbsp.extend_from_slice(&[0, 0, 0, 0]); // cabac_zero_words
            }
            if !sh.dependent {
                last_indep = Some(sh);
            }
            push(&mut sample, &nal(nal_type, tid, &rbsp));
        }
        samples.push(sample);
    }
    (params, samples, rng)
}

fn mp4(c: &Config, params: &[Vec<u8>; 3], samples: &[Vec<u8>], mut rng: Rng) -> Vec<u8> {
    let ftyp = mp4_box(b"ftyp", b"isom\0\0\x02\0isomiso2");
    let free = mp4_box(b"free", &[0u8; 900]);
    let mut mdat_body = Vec::new();
    let mut rel_offsets = Vec::new();
    for (i, s) in samples.iter().enumerate() {
        rel_offsets.push(mdat_body.len());
        mdat_body.extend_from_slice(s);
        let gap: Vec<u8> = (0..(i * 5 % 17)).map(|_| rng.below(256) as u8).collect();
        mdat_body.extend_from_slice(&gap);
    }
    let mdat_start = ftyp.len() + free.len() + 8;
    let mut entry = vec![0u8; 78];
    entry[7] = 1; // data_reference_index
    entry[24..26].copy_from_slice(&(c.width as u16).to_be_bytes());
    entry[26..28].copy_from_slice(&(c.height as u16).to_be_bytes());
    entry.extend_from_slice(&mp4_box(b"hvcC", &hvcc(c, params)));
    let mut stsd_body = 1u32.to_be_bytes().to_vec();
    stsd_body.extend_from_slice(&mp4_box(if c.out_of_band { b"hvc1" } else { b"hev1" }, &entry));
    let stsd = full_box(b"stsd", &stsd_body);
    let mut stsz_body = 0u32.to_be_bytes().to_vec();
    stsz_body.extend_from_slice(&(samples.len() as u32).to_be_bytes());
    for s in samples {
        stsz_body.extend_from_slice(&(s.len() as u32).to_be_bytes());
    }
    let stsz = full_box(b"stsz", &stsz_body);
    let mut stsc_body = 1u32.to_be_bytes().to_vec();
    stsc_body.extend_from_slice(&[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
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

/// Named configurations the tests (and `ppmedia synth-hevc`) use.
pub fn config(name: &str) -> Option<Config> {
    let d = Config::default();
    Some(match name {
        "main" => d,
        "wpp" => Config { wpp: true, dependent_slices: true, width: 200, height: 104, ..d },
        "tiles" => Config { tiles: Some((3, 2)), width: 224, height: 128, log2_ctb: 5, ..d },
        "lossless" => Config { transquant_bypass: true, amp: false, max_merge_cand: 1, cabac_init_present: false, ..d },
        "ctb64" => Config {
            log2_ctb: 6,
            log2_max_tb: 5,
            depth_inter: 3,
            depth_intra: 3,
            width: 192,
            height: 136,
            scaling_lists: true,
            ..d
        },
        "ctb16" => Config {
            log2_ctb: 4,
            log2_max_tb: 4,
            width: 88,
            height: 40,
            cu_qp_delta: Some(0),
            sao: false,
            out_of_band: true,
            ..d
        },
        "mono10" => {
            Config { chroma: 0, bit_depth: 10, transform_skip: false, sign_hiding: false, cu_qp_delta: None, ..d }
        }
        "depth0" => Config { depth_inter: 0, depth_intra: 0, log2_max_tb: 3, amp: false, max_merge_cand: 2, ..d },
        _ => return None,
    })
}

/// All named configurations.
pub const CONFIGS: [&str; 8] = ["main", "wpp", "tiles", "lossless", "ctb64", "ctb16", "mono10", "depth0"];

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
    fn synthetic_roundtrip() {
        for (i, name) in CONFIGS.iter().enumerate() {
            let c = config(name).unwrap();
            let file = synthetic_mp4(100 + i as u64, 9, &c);
            assert!(crate::probe(&file), "{name}");
            let (packed, st) = crate::compress(&file).unwrap_or_else(|| panic!("{name}: not recompressible"));
            assert!(st.hevc, "{name}");
            assert_eq!(&packed[..4], crate::MAGIC_HEVC);
            assert_eq!(st.recoded, st.slices, "{name}: some slices were stored raw: {:?}", st.fails);
            assert!(st.slices >= 9, "{name}");
            assert_eq!(crate::decompress(&packed).unwrap(), file, "{name}");
        }
    }

    #[test]
    fn unsupported_streams_are_declined_quickly() {
        // Parameter sets only in the hvcC box, so patching one copy of the SPS changes the stream.
        let c = config("ctb16").unwrap();
        let clean = synthetic_mp4(3, 2, &c);
        let track = crate::mp4::find_hevc_track(&clean).unwrap();
        assert!(crate::hevc::recode::quick_check(&clean, &track));
        // Make the SPS 4:2:2 (chroma_format_idc ue(1) = 010 -> ue(2) = 011), which the parser
        // does not handle: compress() must decline at once instead of coding every byte raw.
        let sps = super::sps(&c);
        let mut rbsp = crate::bits::unescape(&sps[2..]);
        let bit = 4 + 3 + 1 + 96 + 1 + 2; // vps id, sub layers, nesting, profile_tier_level, sps id, "01"
        rbsp[bit / 8] |= 0x80 >> (bit % 8);
        let mut bad = sps[..2].to_vec();
        crate::bits::escape(&rbsp, &mut bad);
        assert_eq!(bad.len(), sps.len());
        assert_eq!(crate::hevc::parse_sps(&rbsp).unwrap().chroma_format_idc, 2);
        let at = clean.windows(sps.len()).position(|w| w == sps.as_slice()).unwrap();
        let mut file = clean.clone();
        file[at..at + sps.len()].copy_from_slice(&bad);
        let track = crate::mp4::find_hevc_track(&file).unwrap();
        assert!(!crate::hevc::recode::quick_check(&file, &track));
        assert!(crate::compress(&file).is_none());
    }

    #[test]
    fn matroska_roundtrip() {
        let c = config("tiles").unwrap();
        let file = synthetic_mkv(41, 6, &c);
        assert!(crate::mp4::find_hevc_track_iso(&file).is_none());
        let (packed, st) = crate::compress(&file).expect("recompressible Matroska");
        assert_eq!(&packed[..4], crate::MAGIC_HEVC);
        assert_eq!(st.recoded, st.slices, "{:?}", st.fails);
        assert_eq!(crate::decompress(&packed).unwrap(), file);
    }

    #[test]
    fn corrupt_streams_fail_cleanly() {
        let file = synthetic_mp4(5, 4, &config("wpp").unwrap());
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
    fn damaged_slices_still_roundtrip() {
        // Damage slice headers and payloads (and parameter sets): every slice must either be
        // recoded with whatever bins it now holds or be stored raw, and the parser must never
        // panic on what it reads.
        let mut rng = Rng(77);
        for (i, name) in CONFIGS.iter().enumerate() {
            let clean = synthetic_mp4(300 + i as u64, 4, &config(name).unwrap());
            // The video data sits between the ftyp/free boxes and the moov box.
            let (lo, hi) = (950, clean.len() * 3 / 4);
            for round in 0..12 {
                let mut file = clean.clone();
                for _ in 0..1 + round % 3 {
                    let at = lo + rng.below((hi - lo) as u32) as usize;
                    file[at] ^= 1 << rng.below(8);
                }
                if let Some((packed, _)) = crate::compress(&file) {
                    assert_eq!(crate::decompress(&packed).unwrap(), file, "{name} round {round}");
                }
            }
        }
    }

    /// Pins the exact output so that any platform or refactoring that changes the coded bytes
    /// fails loudly (decoders must reproduce the encoder's arithmetic exactly).
    #[test]
    fn output_is_pinned() {
        let file = synthetic_mp4(7, 8, &config("wpp").unwrap());
        let (packed, _) = crate::compress(&file).unwrap();
        assert_eq!((file.len(), crc32(&file)), PINNED_INPUT, "synthetic input changed");
        assert_eq!((packed.len(), crc32(&packed)), PINNED_OUTPUT, "compressed output changed");
    }

    const PINNED_INPUT: (usize, u32) = (5135, 2748056033);
    const PINNED_OUTPUT: (usize, u32) = (3493, 331999308);
}
