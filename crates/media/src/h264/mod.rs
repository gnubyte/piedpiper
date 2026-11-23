//! H.264 parameter sets, slice headers and the macroblock-layer parsers (CABAC and CAVLC).

pub mod cabac;
pub mod cavlc;
pub mod mb;
pub mod recon;
pub mod tables;

use crate::bits::{Bad, BitReader, R};

pub const P: u32 = 0;
pub const B: u32 = 1;
pub const I: u32 = 2;
pub const SP: u32 = 3;
pub const SI: u32 = 4;

#[derive(Clone, Debug, Default)]
pub struct Sps {
    pub id: u32,
    pub profile_idc: u32,
    pub chroma_format_idc: u32,
    pub separate_colour_plane: bool,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub log2_max_frame_num: u32,
    pub poc_type: u32,
    pub log2_max_poc_lsb: u32,
    pub delta_pic_order_always_zero: bool,
    pub frame_mbs_only: bool,
    pub mb_adaptive_frame_field: bool,
    pub direct_8x8_inference: bool,
    pub width_mbs: u32,
    pub height_map_units: u32,
    pub max_num_ref_frames: u32,
}

impl Sps {
    pub fn chroma_array_type(&self) -> u32 {
        if self.separate_colour_plane {
            0
        } else {
            self.chroma_format_idc
        }
    }

    pub fn height_mbs(&self) -> u32 {
        self.height_map_units * if self.frame_mbs_only { 1 } else { 2 }
    }

    /// Whether slices using this SPS can be recoded.
    pub fn supported(&self) -> bool {
        self.frame_mbs_only
            && self.chroma_array_type() == 1
            && self.bit_depth_luma == 8
            && self.bit_depth_chroma == 8
            && self.width_mbs > 0
            && self.height_map_units > 0
            && (self.width_mbs as u64 * self.height_mbs() as u64) <= 139_264
    }
}

#[derive(Clone, Debug, Default)]
pub struct Pps {
    pub id: u32,
    pub sps_id: u32,
    pub cabac: bool,
    pub bottom_field_pic_order_in_frame_present: bool,
    pub num_slice_groups: u32,
    pub num_ref_idx_l0_default: u32,
    pub num_ref_idx_l1_default: u32,
    pub weighted_pred: bool,
    pub weighted_bipred_idc: u32,
    pub pic_init_qp: i32,
    pub deblocking_filter_control_present: bool,
    pub constrained_intra_pred: bool,
    pub redundant_pic_cnt_present: bool,
    pub transform_8x8_mode: bool,
    pub chroma_qp_offset: i32,
    pub chroma_qp_offset2: i32,
}

fn skip_scaling_list(r: &mut BitReader, size: usize) -> R<()> {
    let (mut last, mut next) = (8i32, 8i32);
    for _ in 0..size {
        if next != 0 {
            let delta = r.se()?;
            next = (last + delta + 256).rem_euclid(256);
        }
        if next != 0 {
            last = next;
        }
    }
    Ok(())
}

/// Parses a sequence parameter set RBSP (without the NAL header byte).
pub fn parse_sps(rbsp: &[u8]) -> R<Sps> {
    let mut r = BitReader::new(rbsp);
    let mut s = Sps { profile_idc: r.u(8)?, chroma_format_idc: 1, bit_depth_luma: 8, bit_depth_chroma: 8, ..Default::default() };
    r.u(16)?; // constraint flags, level_idc
    s.id = r.ue()?;
    if s.id > 31 {
        return Err(Bad);
    }
    if matches!(s.profile_idc, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135) {
        s.chroma_format_idc = r.ue()?;
        if s.chroma_format_idc > 3 {
            return Err(Bad);
        }
        if s.chroma_format_idc == 3 {
            s.separate_colour_plane = r.flag()?;
        }
        s.bit_depth_luma = r.ue()? + 8;
        s.bit_depth_chroma = r.ue()? + 8;
        r.flag()?; // qpprime_y_zero_transform_bypass_flag
        if r.flag()? {
            let n = if s.chroma_format_idc != 3 { 8 } else { 12 };
            for i in 0..n {
                if r.flag()? {
                    skip_scaling_list(&mut r, if i < 6 { 16 } else { 64 })?;
                }
            }
        }
    }
    s.log2_max_frame_num = r.ue()? + 4;
    s.poc_type = r.ue()?;
    match s.poc_type {
        0 => s.log2_max_poc_lsb = r.ue()? + 4,
        1 => {
            s.delta_pic_order_always_zero = r.flag()?;
            r.se()?;
            r.se()?;
            let n = r.ue()?;
            if n > 255 {
                return Err(Bad);
            }
            for _ in 0..n {
                r.se()?;
            }
        }
        2 => {}
        _ => return Err(Bad),
    }
    if s.log2_max_frame_num > 16 || s.log2_max_poc_lsb > 16 {
        return Err(Bad);
    }
    s.max_num_ref_frames = r.ue()?;
    r.flag()?; // gaps_in_frame_num_value_allowed_flag
    s.width_mbs = r.ue()? + 1;
    s.height_map_units = r.ue()? + 1;
    s.frame_mbs_only = r.flag()?;
    if !s.frame_mbs_only {
        s.mb_adaptive_frame_field = r.flag()?;
    }
    s.direct_8x8_inference = r.flag()?;
    Ok(s)
}

/// Parses a picture parameter set RBSP (without the NAL header byte).
pub fn parse_pps(rbsp: &[u8]) -> R<Pps> {
    let mut r = BitReader::new(rbsp);
    let mut p = Pps { id: r.ue()?, sps_id: r.ue()?, ..Default::default() };
    if p.id > 255 || p.sps_id > 31 {
        return Err(Bad);
    }
    p.cabac = r.flag()?;
    p.bottom_field_pic_order_in_frame_present = r.flag()?;
    p.num_slice_groups = r.ue()? + 1;
    if p.num_slice_groups > 1 {
        // Slice groups (FMO) are a Baseline feature that never appears with CABAC in
        // practice; such streams are stored unchanged.
        return Ok(p);
    }
    p.num_ref_idx_l0_default = r.ue()? + 1;
    p.num_ref_idx_l1_default = r.ue()? + 1;
    p.weighted_pred = r.flag()?;
    p.weighted_bipred_idc = r.u(2)?;
    p.pic_init_qp = 26 + r.se()?;
    r.se()?; // pic_init_qs_minus26
    p.chroma_qp_offset = r.se()?;
    p.chroma_qp_offset2 = p.chroma_qp_offset;
    p.deblocking_filter_control_present = r.flag()?;
    p.constrained_intra_pred = r.flag()?;
    p.redundant_pic_cnt_present = r.flag()?;
    if r.more_rbsp_data() {
        p.transform_8x8_mode = r.flag()?;
        // Only reconstruction (model context) uses what follows, so errors are ignored.
        let t8 = p.transform_8x8_mode;
        let mut rest = || -> R<i32> {
            if r.flag()? {
                let n = 6 + if t8 { 2 } else { 0 };
                for i in 0..n {
                    if r.flag()? {
                        skip_scaling_list(&mut r, if i < 6 { 16 } else { 64 })?;
                    }
                }
            }
            r.se()
        };
        if let Ok(v) = rest() {
            p.chroma_qp_offset2 = v;
        }
    }
    Ok(p)
}

#[derive(Clone, Debug, Default)]
pub struct SliceHeader {
    pub first_mb: u32,
    pub slice_type: u32,
    pub pps_id: u32,
    pub frame_num: u32,
    pub idr: bool,
    pub nal_ref_idc: u32,
    pub num_ref_l0: u32,
    pub num_ref_l1: u32,
    pub cabac_init_idc: u32,
    pub slice_qp: i32,
    pub direct_spatial: bool,
    /// Bit offset of slice_data() in the RBSP (CABAC streams align it to a byte after
    /// this; CAVLC data starts right here).
    pub data_bit: usize,
    /// Fields only picture reconstruction uses.
    pub ri: recon::RefInfo,
}

fn ref_pic_list_modification(r: &mut BitReader, mods: &mut Vec<recon::ListMod>) -> R<()> {
    if r.flag()? {
        for _ in 0..100 {
            match r.ue()? {
                v @ 0..=2 => {
                    mods.push((v, r.ue()?));
                }
                3 => return Ok(()),
                _ => return Err(Bad),
            }
        }
        return Err(Bad);
    }
    Ok(())
}

fn pred_weight_table(r: &mut BitReader, sps: &Sps, slice_type: u32, n0: u32, n1: u32, ri: &mut recon::RefInfo) -> R<()> {
    let ld = r.ue()?;
    let chroma = sps.chroma_array_type() != 0;
    let cd = if chroma { r.ue()? } else { 0 };
    if ld > 7 || cd > 7 {
        return Err(Bad);
    }
    ri.wp_denom = [ld, cd];
    let lists = if slice_type == B { 2 } else { 1 };
    for list in 0..lists {
        let n = if list == 0 { n0 } else { n1 };
        for _ in 0..n {
            let mut w = [(1 << ld, 0), (1 << cd, 0), (1 << cd, 0)];
            // Weights and offsets are -128..=127 in conforming streams.
            let wo = |r: &mut BitReader| -> R<(i32, i32)> { Ok((r.se()?.clamp(-128, 127), r.se()?.clamp(-128, 127))) };
            if r.flag()? {
                w[0] = wo(r)?;
            }
            if chroma && r.flag()? {
                for c in 1..3 {
                    w[c] = wo(r)?;
                }
            }
            ri.wp[list].push(w);
        }
    }
    Ok(())
}

fn dec_ref_pic_marking(r: &mut BitReader, idr: bool, ri: &mut recon::RefInfo) -> R<()> {
    if idr {
        r.flag()?;
        ri.long_term_reference = r.flag()?;
    } else if r.flag()? {
        ri.adaptive = true;
        for _ in 0..100 {
            match r.ue()? {
                0 => return Ok(()),
                op @ (1 | 2 | 4 | 6) => {
                    let a = r.ue()?;
                    ri.mmco.push((op, a, 0));
                }
                3 => {
                    let a = r.ue()?;
                    let b = r.ue()?;
                    ri.mmco.push((3, a, b));
                }
                5 => ri.mmco.push((5, 0, 0)),
                _ => return Err(Bad),
            }
        }
        return Err(Bad);
    }
    Ok(())
}

/// Parameter sets seen so far, indexed by id.
pub struct ParamSets {
    pub sps: Vec<Option<Sps>>,
    pub pps: Vec<Option<Pps>>,
}

impl Default for ParamSets {
    fn default() -> Self {
        ParamSets { sps: vec![None; 32], pps: vec![None; 256] }
    }
}

impl ParamSets {
    /// Records a parameter-set NAL unit (header byte included). Unparseable ones are
    /// ignored; slices that need them then fail to parse and are stored unchanged.
    pub fn add_nal(&mut self, nal: &[u8]) {
        if nal.is_empty() {
            return;
        }
        let rbsp = crate::bits::unescape(&nal[1..]);
        match nal[0] & 31 {
            7 => {
                if let Ok(s) = parse_sps(&rbsp) {
                    let id = s.id as usize;
                    self.sps[id] = Some(s);
                }
            }
            8 => {
                if let Ok(p) = parse_pps(&rbsp) {
                    let id = p.id as usize;
                    self.pps[id] = Some(p);
                }
            }
            _ => {}
        }
    }

    /// Parses a slice header from a slice RBSP (without the NAL header byte) and returns it
    /// with the parameter sets it refers to, if the slice is one we can recode.
    pub fn slice_header(&self, rbsp: &[u8], nal_type: u32, nal_ref_idc: u32) -> R<(SliceHeader, &Sps, &Pps)> {
        let mut r = BitReader::new(rbsp);
        let first_mb = r.ue()?;
        let st = r.ue()?;
        if st > 9 {
            return Err(Bad);
        }
        let slice_type = st % 5;
        if slice_type == SP || slice_type == SI {
            return Err(Bad);
        }
        let pps_id = r.ue()?;
        let pps = self.pps.get(pps_id as usize).and_then(|p| p.as_ref()).ok_or(Bad)?;
        let sps = self.sps.get(pps.sps_id as usize).and_then(|s| s.as_ref()).ok_or(Bad)?;
        if !sps.supported() || pps.num_slice_groups != 1 || nal_type == 20 || nal_type == 21 {
            return Err(Bad);
        }
        let frame_num = r.u(sps.log2_max_frame_num)?;
        let idr = nal_type == 5;
        if idr {
            r.ue()?;
        }
        let mut ri = recon::RefInfo::default();
        if sps.poc_type == 0 {
            ri.poc_lsb = r.u(sps.log2_max_poc_lsb)?;
            if pps.bottom_field_pic_order_in_frame_present {
                ri.delta_poc_bottom = r.se()?;
            }
        }
        if sps.poc_type == 1 && !sps.delta_pic_order_always_zero {
            r.se()?;
            if pps.bottom_field_pic_order_in_frame_present {
                r.se()?;
            }
        }
        if pps.redundant_pic_cnt_present {
            r.ue()?;
        }
        let mut direct_spatial = false;
        if slice_type == B {
            direct_spatial = r.flag()?;
        }
        let (mut n0, mut n1) = (pps.num_ref_idx_l0_default, pps.num_ref_idx_l1_default);
        if (slice_type == P || slice_type == B) && r.flag()? {
            n0 = r.ue()? + 1;
            if slice_type == B {
                n1 = r.ue()? + 1;
            }
        }
        if n0 > 32 || n1 > 32 {
            return Err(Bad);
        }
        if slice_type != I {
            ref_pic_list_modification(&mut r, &mut ri.mods[0])?;
        }
        if slice_type == B {
            ref_pic_list_modification(&mut r, &mut ri.mods[1])?;
        }
        if (pps.weighted_pred && slice_type == P) || (pps.weighted_bipred_idc == 1 && slice_type == B) {
            pred_weight_table(&mut r, sps, slice_type, n0, n1, &mut ri)?;
            ri.explicit_wp = true;
        }
        if nal_ref_idc != 0 {
            dec_ref_pic_marking(&mut r, idr, &mut ri)?;
        }
        let mut cabac_init_idc = 0;
        if pps.cabac && slice_type != I {
            cabac_init_idc = r.ue()?;
            if cabac_init_idc > 2 {
                return Err(Bad);
            }
        }
        let qp_delta = r.se()?;
        if pps.deblocking_filter_control_present {
            ri.dbk_idc = r.ue()?;
            if ri.dbk_idc != 1 {
                ri.dbk_a = 2 * r.se()?;
                ri.dbk_b = 2 * r.se()?;
            }
        }
        let slice_qp = pps.pic_init_qp + qp_delta;
        if !(0..=51).contains(&slice_qp) {
            return Err(Bad);
        }
        let hdr = SliceHeader {
            first_mb,
            slice_type,
            pps_id,
            frame_num,
            idr,
            nal_ref_idc,
            num_ref_l0: if slice_type == I { 0 } else { n0 },
            num_ref_l1: if slice_type == B { n1 } else { 0 },
            cabac_init_idc,
            slice_qp,
            direct_spatial,
            data_bit: r.pos(),
            ri,
        };
        Ok((hdr, sps, pps))
    }
}
