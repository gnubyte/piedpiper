//! HEVC (H.265): parameter sets, slice segment headers, picture geometry and the CABAC
//! coding-tree parser ([`ctu`]).
//!
//! Supported: everything the Main, Main 10 and Main Still Picture profiles allow, and any
//! other 4:2:0 or 4:0:0 stream (8 to 12 bits) that keeps clear of the range-extension tools
//! that change CABAC parsing and of the SCC and 3-D extensions. Tiles or wavefront parallel
//! processing (not both in one picture), dependent slice segments, AMP, transform skip,
//! lossless CUs, sign data hiding, scaling lists and long-term references are handled. Only base-layer (`nuh_layer_id` 0)
//! slices are recoded; a slice with a PCM coding unit is stored unchanged.

pub mod ctu;
pub mod recode;
pub mod tables;
#[doc(hidden)]
pub mod testgen;

use crate::bits::{Bad, BitReader, R};

/// slice_type values (7.4.7.1).
pub const B: u32 = 0;
pub const P: u32 = 1;
pub const I: u32 = 2;

pub const NAL_VPS: u32 = 32;
pub const NAL_SPS: u32 = 33;
pub const NAL_PPS: u32 = 34;
const IDR_W_RADL: u32 = 19;
const IDR_N_LP: u32 = 20;

/// nal_unit_type of a NAL unit (header included).
#[inline]
pub fn nal_type(nal: &[u8]) -> u32 {
    ((nal[0] >> 1) & 63) as u32
}

/// nuh_layer_id of a NAL unit with a complete two-byte header.
#[inline]
pub fn layer_id(nal: &[u8]) -> u32 {
    (((nal[0] & 1) as u32) << 5) | (nal.get(1).copied().unwrap_or(0) >> 3) as u32
}

/// True for the NAL unit types that carry coded slice segments.
pub fn is_slice(t: u32) -> bool {
    t <= 9 || (16..=21).contains(&t)
}

/// True for intra random access point pictures (BLA, IDR, CRA and the reserved IRAP types).
pub fn is_irap(t: u32) -> bool {
    (16..=23).contains(&t)
}

fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

/// A short-term reference picture set (7.3.7), as far as parsing needs it.
#[derive(Clone, Copy, Debug, Default)]
pub struct StRps {
    pub n_neg: usize,
    pub n_pos: usize,
    pub poc_s0: [i32; 16],
    pub used_s0: [bool; 16],
    pub poc_s1: [i32; 16],
    pub used_s1: [bool; 16],
}

impl StRps {
    fn num_delta(&self) -> usize {
        self.n_neg + self.n_pos
    }

    fn num_used(&self) -> u32 {
        (self.used_s0[..self.n_neg].iter().filter(|&&u| u).count()
            + self.used_s1[..self.n_pos].iter().filter(|&&u| u).count()) as u32
    }

    fn push(&mut self, s1: bool, poc: i32, used: bool) -> R<()> {
        let (n, pocs, useds) = if s1 {
            (&mut self.n_pos, &mut self.poc_s1, &mut self.used_s1)
        } else {
            (&mut self.n_neg, &mut self.poc_s0, &mut self.used_s0)
        };
        if *n >= 16 {
            return Err(Bad);
        }
        pocs[*n] = poc;
        useds[*n] = used;
        *n += 1;
        Ok(())
    }
}

fn st_ref_pic_set(r: &mut BitReader, idx: usize, num: usize, sets: &[StRps]) -> R<StRps> {
    let mut out = StRps::default();
    if idx != 0 && r.flag()? {
        // inter_ref_pic_set_prediction_flag (7.4.8)
        let delta_idx = if idx == num { r.ue()? as usize + 1 } else { 1 };
        if delta_idx > idx {
            return Err(Bad);
        }
        let rf = *sets.get(idx - delta_idx).ok_or(Bad)?;
        let sign = r.flag()?;
        let abs = r.ue()?;
        if abs >= 1 << 15 {
            return Err(Bad);
        }
        let delta_rps = if sign { -(abs as i32 + 1) } else { abs as i32 + 1 };
        let nd = rf.num_delta();
        let mut used = [false; 33];
        let mut use_delta = [true; 33];
        for j in 0..=nd {
            used[j] = r.flag()?;
            if !used[j] {
                use_delta[j] = r.flag()?;
            }
        }
        for j in (0..rf.n_pos).rev() {
            let d = rf.poc_s1[j] + delta_rps;
            if d < 0 && use_delta[rf.n_neg + j] {
                out.push(false, d, used[rf.n_neg + j])?;
            }
        }
        if delta_rps < 0 && use_delta[nd] {
            out.push(false, delta_rps, used[nd])?;
        }
        for j in 0..rf.n_neg {
            let d = rf.poc_s0[j] + delta_rps;
            if d < 0 && use_delta[j] {
                out.push(false, d, used[j])?;
            }
        }
        for j in (0..rf.n_neg).rev() {
            let d = rf.poc_s0[j] + delta_rps;
            if d > 0 && use_delta[j] {
                out.push(true, d, used[j])?;
            }
        }
        if delta_rps > 0 && use_delta[nd] {
            out.push(true, delta_rps, used[nd])?;
        }
        for j in 0..rf.n_pos {
            let d = rf.poc_s1[j] + delta_rps;
            if d > 0 && use_delta[rf.n_neg + j] {
                out.push(true, d, used[rf.n_neg + j])?;
            }
        }
    } else {
        let nn = r.ue()? as usize;
        let np = r.ue()? as usize;
        if nn > 16 || np > 16 {
            return Err(Bad);
        }
        let mut poc = 0i32;
        for _ in 0..nn {
            poc -= (r.ue()?.min(1 << 16)) as i32 + 1;
            let u = r.flag()?;
            out.push(false, poc, u)?;
        }
        poc = 0;
        for _ in 0..np {
            poc += (r.ue()?.min(1 << 16)) as i32 + 1;
            let u = r.flag()?;
            out.push(true, poc, u)?;
        }
    }
    Ok(out)
}

fn profile_tier_level(r: &mut BitReader, max_sub_layers_minus1: u32) -> R<u32> {
    let p = r.u(8)? & 31; // general_profile_space, general_tier_flag, general_profile_idc
    r.u(32)?; // general_profile_compatibility_flag[32]
    r.u(32)?; // progressive/interlaced/non-packed/frame-only flags and the constraint flags
    r.u(16)?;
    r.u(8)?; // general_level_idc
    let n = max_sub_layers_minus1 as usize;
    let mut prof = [false; 8];
    let mut lev = [false; 8];
    for i in 0..n {
        prof[i] = r.flag()?;
        lev[i] = r.flag()?;
    }
    if n > 0 {
        for _ in n..8 {
            r.u(2)?;
        }
    }
    for i in 0..n {
        if prof[i] {
            r.u(8)?;
            r.u(32)?;
            r.u(32)?;
            r.u(16)?;
        }
        if lev[i] {
            r.u(8)?;
        }
    }
    Ok(p)
}

fn scaling_list_data(r: &mut BitReader) -> R<()> {
    for size_id in 0..4usize {
        let mut m = 0;
        while m < 6 {
            if !r.flag()? {
                r.ue()?; // scaling_list_pred_matrix_id_delta
            } else {
                let n = (1usize << (4 + (size_id << 1))).min(64);
                if size_id > 1 {
                    r.se()?;
                }
                for _ in 0..n {
                    r.se()?;
                }
            }
            m += if size_id == 3 { 3 } else { 1 };
        }
    }
    Ok(())
}

fn sub_layer_hrd(r: &mut BitReader, cpb_cnt: u32, sub_pic: bool) -> R<()> {
    for _ in 0..cpb_cnt {
        r.ue()?;
        r.ue()?;
        if sub_pic {
            r.ue()?;
            r.ue()?;
        }
        r.flag()?;
    }
    Ok(())
}

fn hrd_parameters(r: &mut BitReader, common: bool, max_sub_layers_minus1: u32) -> R<()> {
    let (mut nal, mut vcl, mut sub_pic) = (false, false, false);
    if common {
        nal = r.flag()?;
        vcl = r.flag()?;
        if nal || vcl {
            sub_pic = r.flag()?;
            if sub_pic {
                r.u(8)?;
                r.u(5)?;
                r.flag()?;
                r.u(5)?;
            }
            r.u(4)?;
            r.u(4)?;
            if sub_pic {
                r.u(4)?;
            }
            r.u(5)?;
            r.u(5)?;
            r.u(5)?;
        }
    }
    for _ in 0..=max_sub_layers_minus1 {
        let fixed_general = r.flag()?;
        let fixed_cvs = if !fixed_general { r.flag()? } else { true };
        let mut low_delay = false;
        if fixed_cvs {
            r.ue()?;
        } else {
            low_delay = r.flag()?;
        }
        let mut cpb_cnt = 1;
        if !low_delay {
            cpb_cnt = r.ue()? + 1;
            if cpb_cnt > 32 {
                return Err(Bad);
            }
        }
        if nal {
            sub_layer_hrd(r, cpb_cnt, sub_pic)?;
        }
        if vcl {
            sub_layer_hrd(r, cpb_cnt, sub_pic)?;
        }
    }
    Ok(())
}

fn vui_parameters(r: &mut BitReader, max_sub_layers_minus1: u32) -> R<()> {
    if r.flag()? && r.u(8)? == 255 {
        r.u(16)?;
        r.u(16)?;
    }
    if r.flag()? {
        r.flag()?;
    }
    if r.flag()? {
        r.u(3)?;
        r.flag()?;
        if r.flag()? {
            r.u(24)?;
        }
    }
    if r.flag()? {
        r.ue()?;
        r.ue()?;
    }
    r.u(3)?; // neutral_chroma_indication, field_seq, frame_field_info_present
    if r.flag()? {
        for _ in 0..4 {
            r.ue()?;
        }
    }
    if r.flag()? {
        r.u(32)?;
        r.u(32)?;
        if r.flag()? {
            r.ue()?;
        }
        if r.flag()? {
            hrd_parameters(r, true, max_sub_layers_minus1)?;
        }
    }
    if r.flag()? {
        r.u(3)?;
        for _ in 0..5 {
            r.ue()?;
        }
    }
    Ok(())
}

/// The sequence parameter set fields that parsing slices needs.
#[derive(Clone, Debug, Default)]
pub struct Sps {
    pub id: u32,
    pub profile_idc: u32,
    pub chroma_format_idc: u32,
    pub separate_colour_plane: bool,
    pub width: u32,
    pub height: u32,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub log2_max_poc_lsb: u32,
    pub log2_min_cb: u32,
    pub log2_ctb: u32,
    pub log2_min_tb: u32,
    pub log2_max_tb: u32,
    pub max_th_depth_inter: u32,
    pub max_th_depth_intra: u32,
    pub amp: bool,
    pub sao: bool,
    pub pcm: bool,
    pub log2_min_pcm: u32,
    pub log2_max_pcm: u32,
    pub st_rps: Vec<StRps>,
    pub long_term_refs: bool,
    /// used_by_curr_pic_lt_sps_flag of each long-term candidate.
    pub lt_used: Vec<bool>,
    pub temporal_mvp: bool,
    /// A range, SCC or 3-D extension tool that changes CABAC parsing is enabled.
    pub unsupported: bool,
}

impl Sps {
    pub fn chroma_array_type(&self) -> u32 {
        if self.separate_colour_plane {
            0
        } else {
            self.chroma_format_idc
        }
    }

    pub fn w_ctb(&self) -> u32 {
        self.width.div_ceil(1 << self.log2_ctb)
    }

    pub fn h_ctb(&self) -> u32 {
        self.height.div_ceil(1 << self.log2_ctb)
    }

    /// Whether slices using this SPS can be recoded.
    pub fn supported(&self) -> bool {
        let min_cb = 1u32 << self.log2_min_cb;
        !self.unsupported
            && !self.separate_colour_plane
            && self.chroma_format_idc <= 1
            && (8..=12).contains(&self.bit_depth_luma)
            && (8..=12).contains(&self.bit_depth_chroma)
            && (4..=6).contains(&self.log2_ctb)
            && self.log2_min_cb >= 3
            && self.log2_min_cb <= self.log2_ctb
            && self.log2_min_tb >= 2
            && self.log2_min_tb < self.log2_min_cb
            && self.log2_max_tb <= 5
            && self.log2_max_tb <= self.log2_ctb
            && self.log2_max_tb >= self.log2_min_tb
            && self.width > 0
            && self.height > 0
            && self.width.is_multiple_of(min_cb)
            && self.height.is_multiple_of(min_cb)
            && (self.width as u64 * self.height as u64) <= 8192 * 4352
            && (!self.pcm || (self.log2_min_pcm >= 3 && self.log2_max_pcm <= 5.min(self.log2_ctb)))
    }
}

/// Parses a sequence parameter set RBSP (after the two-byte NAL header).
pub fn parse_sps(rbsp: &[u8]) -> R<Sps> {
    let mut r = BitReader::new(rbsp);
    r.u(4)?; // sps_video_parameter_set_id
    let max_sub_layers_minus1 = r.u(3)?;
    if max_sub_layers_minus1 > 6 {
        return Err(Bad);
    }
    r.flag()?;
    let mut s = Sps { profile_idc: profile_tier_level(&mut r, max_sub_layers_minus1)?, ..Default::default() };
    s.id = r.ue()?;
    s.chroma_format_idc = r.ue()?;
    if s.id > 15 || s.chroma_format_idc > 3 {
        return Err(Bad);
    }
    if s.chroma_format_idc == 3 {
        s.separate_colour_plane = r.flag()?;
    }
    s.width = r.ue()?;
    s.height = r.ue()?;
    if r.flag()? {
        for _ in 0..4 {
            r.ue()?;
        }
    }
    s.bit_depth_luma = r.ue()? + 8;
    s.bit_depth_chroma = r.ue()? + 8;
    s.log2_max_poc_lsb = r.ue()? + 4;
    if s.bit_depth_luma > 16 || s.bit_depth_chroma > 16 || s.log2_max_poc_lsb > 16 {
        return Err(Bad);
    }
    let ordering = r.flag()?;
    for _ in (if ordering { 0 } else { max_sub_layers_minus1 })..=max_sub_layers_minus1 {
        r.ue()?;
        r.ue()?;
        r.ue()?;
    }
    s.log2_min_cb = r.ue()? + 3;
    s.log2_ctb = s.log2_min_cb + r.ue()?;
    s.log2_min_tb = r.ue()? + 2;
    s.log2_max_tb = s.log2_min_tb + r.ue()?;
    s.max_th_depth_inter = r.ue()?;
    s.max_th_depth_intra = r.ue()?;
    if s.log2_ctb > 6 || s.log2_max_tb > 5 || s.max_th_depth_inter > 4 || s.max_th_depth_intra > 4 {
        return Err(Bad);
    }
    if r.flag()? && r.flag()? {
        scaling_list_data(&mut r)?;
    }
    s.amp = r.flag()?;
    s.sao = r.flag()?;
    s.pcm = r.flag()?;
    if s.pcm {
        r.u(8)?;
        s.log2_min_pcm = r.ue()? + 3;
        s.log2_max_pcm = s.log2_min_pcm + r.ue()?;
        r.flag()?;
    }
    let nsets = r.ue()? as usize;
    if nsets > 64 {
        return Err(Bad);
    }
    for i in 0..nsets {
        let set = st_ref_pic_set(&mut r, i, nsets, &s.st_rps)?;
        s.st_rps.push(set);
    }
    s.long_term_refs = r.flag()?;
    if s.long_term_refs {
        let n = r.ue()?;
        if n > 32 {
            return Err(Bad);
        }
        for _ in 0..n {
            r.u(s.log2_max_poc_lsb)?;
            s.lt_used.push(r.flag()?);
        }
    }
    s.temporal_mvp = r.flag()?;
    r.flag()?; // strong_intra_smoothing_enabled_flag
    if r.flag()? {
        vui_parameters(&mut r, max_sub_layers_minus1)?;
    }
    if r.flag()? {
        let range = r.flag()?;
        let multilayer = r.flag()?;
        let ext3d = r.flag()?;
        let scc = r.flag()?;
        let ext4 = r.u(4)?;
        if range {
            let mut f = [false; 9];
            for v in f.iter_mut() {
                *v = r.flag()?;
            }
            // transform_skip_context, implicit_rdpcm, explicit_rdpcm, extended_precision,
            // persistent_rice_adaptation and cabac_bypass_alignment change parsing; rotation,
            // intra smoothing and high-precision offsets do not.
            if f[1] || f[2] || f[3] || f[4] || f[7] || f[8] {
                s.unsupported = true;
            }
        }
        if multilayer {
            r.flag()?;
        }
        if ext3d || scc || ext4 != 0 {
            s.unsupported = true;
        }
    }
    Ok(s)
}

/// sps_seq_parameter_set_id of an SPS RBSP, without parsing the rest.
pub fn sps_id(rbsp: &[u8]) -> R<u32> {
    let mut r = BitReader::new(rbsp);
    r.u(4)?;
    let n = r.u(3)?;
    r.flag()?;
    profile_tier_level(&mut r, n)?;
    let id = r.ue()?;
    if id > 15 {
        return Err(Bad);
    }
    Ok(id)
}

/// The picture parameter set fields that parsing slices needs.
#[derive(Clone, Debug, Default)]
pub struct Pps {
    pub id: u32,
    pub sps_id: u32,
    pub dependent_slice_segments: bool,
    pub output_flag_present: bool,
    pub num_extra_slice_header_bits: u32,
    pub sign_data_hiding: bool,
    pub cabac_init_present: bool,
    pub num_ref_idx_l0_default: u32,
    pub num_ref_idx_l1_default: u32,
    pub init_qp: i32,
    pub transform_skip: bool,
    pub cu_qp_delta: bool,
    pub diff_cu_qp_delta_depth: u32,
    pub slice_chroma_qp_offsets_present: bool,
    pub weighted_pred: bool,
    pub weighted_bipred: bool,
    pub transquant_bypass: bool,
    pub tiles: bool,
    pub entropy_coding_sync: bool,
    pub tile_cols: u32,
    pub tile_rows: u32,
    pub uniform_spacing: bool,
    /// Explicit tile column widths / row heights in CTBs (all but the last), when not uniform.
    pub col_widths: Vec<u32>,
    pub row_heights: Vec<u32>,
    pub loop_filter_across_slices: bool,
    pub deblocking_override_enabled: bool,
    pub deblocking_disabled: bool,
    pub lists_modification_present: bool,
    pub slice_header_extension_present: bool,
    /// A range, SCC or 3-D extension tool that changes parsing is enabled.
    pub unsupported: bool,
}

/// Parses a picture parameter set RBSP (after the two-byte NAL header).
pub fn parse_pps(rbsp: &[u8]) -> R<Pps> {
    let mut r = BitReader::new(rbsp);
    let mut p = Pps { id: r.ue()?, sps_id: r.ue()?, ..Default::default() };
    if p.id > 63 || p.sps_id > 15 {
        return Err(Bad);
    }
    p.dependent_slice_segments = r.flag()?;
    p.output_flag_present = r.flag()?;
    p.num_extra_slice_header_bits = r.u(3)?;
    p.sign_data_hiding = r.flag()?;
    p.cabac_init_present = r.flag()?;
    p.num_ref_idx_l0_default = r.ue()? + 1;
    p.num_ref_idx_l1_default = r.ue()? + 1;
    let qp = r.se()?;
    if !(-62..=25).contains(&qp) || p.num_ref_idx_l0_default > 15 || p.num_ref_idx_l1_default > 15 {
        return Err(Bad);
    }
    p.init_qp = 26 + qp;
    r.flag()?; // constrained_intra_pred_flag
    p.transform_skip = r.flag()?;
    p.cu_qp_delta = r.flag()?;
    if p.cu_qp_delta {
        p.diff_cu_qp_delta_depth = r.ue()?;
        if p.diff_cu_qp_delta_depth > 3 {
            return Err(Bad);
        }
    }
    r.se()?; // pps_cb_qp_offset
    r.se()?; // pps_cr_qp_offset
    p.slice_chroma_qp_offsets_present = r.flag()?;
    p.weighted_pred = r.flag()?;
    p.weighted_bipred = r.flag()?;
    p.transquant_bypass = r.flag()?;
    p.tiles = r.flag()?;
    p.entropy_coding_sync = r.flag()?;
    p.tile_cols = 1;
    p.tile_rows = 1;
    p.uniform_spacing = true;
    if p.tiles {
        p.tile_cols = r.ue()? + 1;
        p.tile_rows = r.ue()? + 1;
        if p.tile_cols > 64 || p.tile_rows > 64 {
            return Err(Bad);
        }
        p.uniform_spacing = r.flag()?;
        if !p.uniform_spacing {
            for _ in 1..p.tile_cols {
                p.col_widths.push(r.ue()?.saturating_add(1));
            }
            for _ in 1..p.tile_rows {
                p.row_heights.push(r.ue()?.saturating_add(1));
            }
        }
        r.flag()?; // loop_filter_across_tiles_enabled_flag
        // With both tiles and wavefronts, substreams restart at every CTB row of every tile;
        // the parser handles each tool on its own, so such streams are stored unchanged.
        if p.entropy_coding_sync {
            p.unsupported = true;
        }
    }
    p.loop_filter_across_slices = r.flag()?;
    if r.flag()? {
        p.deblocking_override_enabled = r.flag()?;
        p.deblocking_disabled = r.flag()?;
        if !p.deblocking_disabled {
            r.se()?;
            r.se()?;
        }
    }
    if r.flag()? {
        scaling_list_data(&mut r)?;
    }
    p.lists_modification_present = r.flag()?;
    r.ue()?; // log2_parallel_merge_level_minus2
    p.slice_header_extension_present = r.flag()?;
    if r.flag()? {
        let range = r.flag()?;
        let _multilayer = r.flag()?; // only affects layers above the base layer
        let ext3d = r.flag()?;
        let scc = r.flag()?;
        let ext4 = r.u(4)?;
        if range {
            let mut log2_max_ts = 2;
            if p.transform_skip {
                log2_max_ts = r.ue()? + 2;
            }
            let ccp = r.flag()?;
            let chroma_qp_offset_list = r.flag()?;
            if chroma_qp_offset_list {
                r.ue()?;
                let n = r.ue()?;
                if n > 5 {
                    return Err(Bad);
                }
                for _ in 0..=n {
                    r.se()?;
                    r.se()?;
                }
            }
            r.ue()?;
            r.ue()?;
            if ccp || chroma_qp_offset_list || log2_max_ts != 2 {
                p.unsupported = true;
            }
        }
        if ext3d || scc || ext4 != 0 {
            p.unsupported = true;
        }
    }
    Ok(p)
}

/// One slice segment header (7.3.6.1), with the fields inherited by dependent slice
/// segments filled in.
#[derive(Clone, Debug, Default)]
pub struct SliceHeader {
    pub nal_type: u32,
    pub temporal_id: u32,
    pub first_in_pic: bool,
    pub pps_id: u32,
    pub dependent: bool,
    /// slice_segment_address (raster-scan CTB address of the segment's first CTB).
    pub segment_address: u32,
    /// SliceAddrRs: address of the first CTB of the slice (the independent segment).
    pub slice_addr: u32,
    pub slice_type: u32,
    pub sao_luma: bool,
    pub sao_chroma: bool,
    pub num_ref_l0: u32,
    pub num_ref_l1: u32,
    pub mvd_l1_zero: bool,
    pub cabac_init_flag: bool,
    pub max_num_merge_cand: u32,
    /// SliceQpY.
    pub slice_qp: i32,
    /// entry_point_offset_minus1[i] + 1, in bytes of the escaped slice data.
    pub entry_points: Vec<u32>,
    /// Bit offset of slice_segment_data() in the RBSP (byte-aligned).
    pub data_bit: usize,
}

fn pred_weight_table(r: &mut BitReader, chroma: bool, slice_type: u32, n0: u32, n1: u32) -> R<()> {
    r.ue()?;
    if chroma {
        r.se()?;
    }
    for list in 0..(if slice_type == B { 2 } else { 1 }) {
        let n = if list == 0 { n0 } else { n1 } as usize;
        let mut lw = [false; 16];
        let mut cw = [false; 16];
        for f in lw.iter_mut().take(n) {
            *f = r.flag()?;
        }
        if chroma {
            for f in cw.iter_mut().take(n) {
                *f = r.flag()?;
            }
        }
        for i in 0..n {
            if lw[i] {
                r.se()?;
                r.se()?;
            }
            if cw[i] {
                for _ in 0..4 {
                    r.se()?;
                }
            }
        }
    }
    Ok(())
}

/// CTB address tables of a picture (6.5.1), derived from an SPS and a PPS.
#[derive(Clone, Debug, Default)]
pub struct Geometry {
    pub w_ctb: usize,
    pub h_ctb: usize,
    pub rs_to_ts: Vec<u32>,
    pub ts_to_rs: Vec<u32>,
    /// Tile index of every CTB, in raster order.
    pub tile_rs: Vec<u16>,
}

impl Geometry {
    pub fn new(sps: &Sps, pps: &Pps) -> R<Geometry> {
        let (w, h) = (sps.w_ctb(), sps.h_ctb());
        let sizes = |n: u32, total: u32, explicit: &[u32]| -> R<Vec<u32>> {
            if n > total {
                return Err(Bad);
            }
            let v: Vec<u32> = if pps.uniform_spacing {
                (0..n).map(|i| ((i + 1) * total) / n - (i * total) / n).collect()
            } else {
                let sum: u64 = explicit.iter().map(|&x| x as u64).sum();
                if sum >= total as u64 {
                    return Err(Bad);
                }
                let mut v = explicit.to_vec();
                v.push(total - sum as u32);
                v
            };
            if v.contains(&0) {
                return Err(Bad);
            }
            Ok(v)
        };
        let cols = sizes(pps.tile_cols, w, &pps.col_widths)?;
        let rows = sizes(pps.tile_rows, h, &pps.row_heights)?;
        let (w, h) = (w as usize, h as usize);
        let mut col_bd = vec![0usize];
        for &c in &cols {
            col_bd.push(col_bd.last().unwrap() + c as usize);
        }
        let mut row_bd = vec![0usize];
        for &r in &rows {
            row_bd.push(row_bd.last().unwrap() + r as usize);
        }
        let mut rs_to_ts = vec![0u32; w * h];
        let mut tile_rs = vec![0u16; w * h];
        for rs in 0..w * h {
            let (tx, ty) = (rs % w, rs / w);
            let tile_x = (0..cols.len()).rev().find(|&i| tx >= col_bd[i]).unwrap_or(0);
            let tile_y = (0..rows.len()).rev().find(|&j| ty >= row_bd[j]).unwrap_or(0);
            let mut v = 0usize;
            for &c in &cols[..tile_x] {
                v += rows[tile_y] as usize * c as usize;
            }
            for &r in &rows[..tile_y] {
                v += w * r as usize;
            }
            v += (ty - row_bd[tile_y]) * cols[tile_x] as usize + tx - col_bd[tile_x];
            rs_to_ts[rs] = v as u32;
            tile_rs[rs] = (tile_y * cols.len() + tile_x) as u16;
        }
        let mut ts_to_rs = vec![0u32; w * h];
        for (rs, &ts) in rs_to_ts.iter().enumerate() {
            ts_to_rs[ts as usize] = rs as u32;
        }
        Ok(Geometry { w_ctb: w, h_ctb: h, rs_to_ts, ts_to_rs, tile_rs })
    }

    /// Tile of the CTB at tile-scan address `ts`.
    #[inline]
    pub fn tile_ts(&self, ts: usize) -> u16 {
        self.tile_rs[self.ts_to_rs[ts] as usize]
    }
}

/// Parameter sets seen so far, indexed by id.
pub struct ParamSets {
    pub sps: Vec<Option<Sps>>,
    pub pps: Vec<Option<Pps>>,
}

impl Default for ParamSets {
    fn default() -> Self {
        ParamSets { sps: vec![None; 16], pps: vec![None; 64] }
    }
}

impl ParamSets {
    /// Records an SPS or PPS NAL unit (header included). Other NAL units, unparseable
    /// parameter sets and those of layers above the base layer are ignored. Returns true if
    /// a stored parameter set changed.
    pub fn add_nal(&mut self, nal: &[u8]) -> bool {
        if nal.len() < 3 || layer_id(nal) != 0 {
            return false;
        }
        let rbsp = crate::bits::unescape(&nal[2..]);
        match nal_type(nal) {
            NAL_SPS => match parse_sps(&rbsp) {
                Ok(s) => {
                    let id = s.id as usize;
                    self.sps[id] = Some(s);
                    true
                }
                Err(_) => {
                    // An SPS we cannot read replaces whatever had its id.
                    if let Ok(id) = sps_id(&rbsp) {
                        self.sps[id as usize] = None;
                    }
                    true
                }
            },
            NAL_PPS => {
                match parse_pps(&rbsp) {
                    Ok(p) => {
                        let id = p.id as usize;
                        self.pps[id] = Some(p);
                    }
                    Err(_) => {
                        if let Ok(id) = BitReader::new(&rbsp).ue() {
                            if let Some(slot) = self.pps.get_mut(id as usize) {
                                *slot = None;
                            }
                        }
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// Parses a slice segment header from a slice RBSP (after the two-byte NAL header).
    /// `prev` must be the header of the last independent slice segment of the same picture;
    /// dependent slice segments inherit most fields from it.
    pub fn slice_header(
        &self,
        rbsp: &[u8],
        nal_type: u32,
        temporal_id: u32,
        prev: Option<&SliceHeader>,
    ) -> R<SliceHeader> {
        let mut r = BitReader::new(rbsp);
        let first = r.flag()?;
        if is_irap(nal_type) {
            r.flag()?; // no_output_of_prior_pics_flag
        }
        let pps_id = r.ue()?;
        let pps = self.pps.get(pps_id as usize).and_then(|p| p.as_ref()).ok_or(Bad)?;
        let sps = self.sps.get(pps.sps_id as usize).and_then(|s| s.as_ref()).ok_or(Bad)?;
        if !sps.supported() || pps.unsupported {
            return Err(Bad);
        }
        let ctbs = sps.w_ctb() * sps.h_ctb();
        let mut dependent = false;
        let mut address = 0;
        if !first {
            if pps.dependent_slice_segments {
                dependent = r.flag()?;
            }
            address = r.u(ceil_log2(ctbs))?;
            if address == 0 || address >= ctbs {
                return Err(Bad);
            }
        }
        let mut sh = if dependent {
            let p = prev.ok_or(Bad)?;
            if p.pps_id != pps_id {
                return Err(Bad);
            }
            SliceHeader {
                nal_type,
                temporal_id,
                first_in_pic: false,
                dependent: true,
                segment_address: address,
                ..p.clone()
            }
        } else {
            for _ in 0..pps.num_extra_slice_header_bits {
                r.flag()?;
            }
            let slice_type = r.ue()?;
            if slice_type > 2 {
                return Err(Bad);
            }
            if pps.output_flag_present {
                r.flag()?;
            }
            let mut total_curr = 0;
            let mut temporal_mvp = false;
            if nal_type != IDR_W_RADL && nal_type != IDR_N_LP {
                r.u(sps.log2_max_poc_lsb)?;
                let rps = if !r.flag()? {
                    st_ref_pic_set(&mut r, sps.st_rps.len(), sps.st_rps.len(), &sps.st_rps)?
                } else {
                    let n = sps.st_rps.len() as u32;
                    let idx = if n > 1 { r.u(ceil_log2(n))? } else { 0 };
                    *sps.st_rps.get(idx as usize).ok_or(Bad)?
                };
                total_curr = rps.num_used();
                if sps.long_term_refs {
                    let n_sps = if !sps.lt_used.is_empty() { r.ue()? as usize } else { 0 };
                    let n_pics = r.ue()? as usize;
                    if n_sps > sps.lt_used.len() || n_pics > 32 {
                        return Err(Bad);
                    }
                    for i in 0..n_sps + n_pics {
                        if i < n_sps {
                            let n = sps.lt_used.len() as u32;
                            let idx = if n > 1 { r.u(ceil_log2(n))? } else { 0 };
                            if *sps.lt_used.get(idx as usize).ok_or(Bad)? {
                                total_curr += 1;
                            }
                        } else {
                            r.u(sps.log2_max_poc_lsb)?;
                            if r.flag()? {
                                total_curr += 1;
                            }
                        }
                        if r.flag()? {
                            r.ue()?;
                        }
                    }
                }
                if sps.temporal_mvp {
                    temporal_mvp = r.flag()?;
                }
            }
            let (mut sao_luma, mut sao_chroma) = (false, false);
            if sps.sao {
                sao_luma = r.flag()?;
                if sps.chroma_array_type() != 0 {
                    sao_chroma = r.flag()?;
                }
            }
            let (mut n0, mut n1) = (0, 0);
            let (mut mvd_l1_zero, mut cabac_init_flag, mut max_merge) = (false, false, 5);
            if slice_type != I {
                n0 = pps.num_ref_idx_l0_default;
                n1 = if slice_type == B { pps.num_ref_idx_l1_default } else { 0 };
                if r.flag()? {
                    n0 = r.ue()? + 1;
                    if slice_type == B {
                        n1 = r.ue()? + 1;
                    }
                }
                if n0 > 15 || n1 > 15 {
                    return Err(Bad);
                }
                if pps.lists_modification_present && total_curr > 1 {
                    let bits = ceil_log2(total_curr);
                    if r.flag()? {
                        for _ in 0..n0 {
                            r.u(bits)?;
                        }
                    }
                    if slice_type == B && r.flag()? {
                        for _ in 0..n1 {
                            r.u(bits)?;
                        }
                    }
                }
                if slice_type == B {
                    mvd_l1_zero = r.flag()?;
                }
                if pps.cabac_init_present {
                    cabac_init_flag = r.flag()?;
                }
                if temporal_mvp {
                    let from_l0 = if slice_type == B { r.flag()? } else { true };
                    if (from_l0 && n0 > 1) || (!from_l0 && n1 > 1) {
                        r.ue()?;
                    }
                }
                if (pps.weighted_pred && slice_type == P) || (pps.weighted_bipred && slice_type == B) {
                    pred_weight_table(&mut r, sps.chroma_array_type() != 0, slice_type, n0, n1)?;
                }
                let d = r.ue()?;
                if d > 4 {
                    return Err(Bad);
                }
                max_merge = 5 - d;
            }
            let qp_delta = r.se()?;
            if pps.slice_chroma_qp_offsets_present {
                r.se()?;
                r.se()?;
            }
            let mut deblocking_disabled = pps.deblocking_disabled;
            if pps.deblocking_override_enabled && r.flag()? {
                deblocking_disabled = r.flag()?;
                if !deblocking_disabled {
                    r.se()?;
                    r.se()?;
                }
            }
            if pps.loop_filter_across_slices && (sao_luma || sao_chroma || !deblocking_disabled) {
                r.flag()?;
            }
            let slice_qp = pps.init_qp + qp_delta;
            if slice_qp < -6 * (sps.bit_depth_luma as i32 - 8) || slice_qp > 51 {
                return Err(Bad);
            }
            SliceHeader {
                nal_type,
                temporal_id,
                first_in_pic: first,
                pps_id,
                dependent: false,
                segment_address: address,
                slice_addr: address,
                slice_type,
                sao_luma,
                sao_chroma,
                num_ref_l0: n0,
                num_ref_l1: n1,
                mvd_l1_zero,
                cabac_init_flag,
                max_num_merge_cand: max_merge,
                slice_qp,
                entry_points: Vec::new(),
                data_bit: 0,
            }
        };
        sh.entry_points.clear();
        if pps.tiles || pps.entropy_coding_sync {
            let n = r.ue()?;
            if n >= ctbs {
                return Err(Bad);
            }
            if n > 0 {
                let len = r.ue()? + 1;
                if len > 32 {
                    return Err(Bad);
                }
                for _ in 0..n {
                    sh.entry_points.push(r.u(len)?.wrapping_add(1));
                }
            }
        }
        if pps.slice_header_extension_present {
            let n = r.ue()?;
            if n > 256 {
                return Err(Bad);
            }
            for _ in 0..n {
                r.u(8)?;
            }
        }
        // byte_alignment(): a one bit, then zero bits up to the byte boundary.
        if r.bit()? != 1 {
            return Err(Bad);
        }
        while !r.pos().is_multiple_of(8) {
            if r.bit()? != 0 {
                return Err(Bad);
            }
        }
        sh.data_bit = r.pos();
        Ok(sh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log2_ceil() {
        assert_eq!([1, 2, 3, 4, 5, 8, 9].map(ceil_log2), [0, 1, 2, 2, 3, 3, 4]);
    }

    #[test]
    fn tiles_geometry() {
        let sps = Sps { width: 64 * 5, height: 64 * 3, log2_ctb: 6, ..Default::default() };
        let pps = Pps { tiles: true, tile_cols: 2, tile_rows: 2, uniform_spacing: true, ..Default::default() };
        let g = Geometry::new(&sps, &pps).unwrap();
        // Columns 2 + 3 CTBs, rows 1 + 2 CTBs.
        assert_eq!(g.rs_to_ts[..5], [0, 1, 2, 3, 4]);
        assert_eq!(g.rs_to_ts[5..10], [5, 6, 9, 10, 11]);
        assert_eq!(g.tile_rs[..5], [0, 0, 1, 1, 1]);
        assert_eq!(g.tile_rs[5..10], [2, 2, 3, 3, 3]);
        for ts in 0..15 {
            assert_eq!(g.rs_to_ts[g.ts_to_rs[ts] as usize] as usize, ts);
        }
    }
}
