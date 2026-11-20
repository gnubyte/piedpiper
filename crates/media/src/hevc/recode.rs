//! Recoding HEVC slice segments: the per-stream state the container keeps, verification
//! (parse the original payload, re-encode it, compare) and reconstruction from the model.

use super::ctu::{Picture, SliceParser};
use super::{is_slice, layer_id, nal_type, Geometry, ParamSets, SliceHeader};
use crate::bits::{escape, unescape, Bad, R};
use crate::container::{tail_fixup, ModelIo, Rec, Verified, VerifyIo};
use crate::model::{ArithDec, Model};

/// Parser state carried from NAL unit to NAL unit within a segment.
#[derive(Default)]
pub struct State {
    pub ps: ParamSets,
    pic: Picture,
    /// Header of the last independent slice segment; dependent segments inherit from it.
    last_indep: Option<SliceHeader>,
    /// Every segment of the current slice so far was recoded, so a following dependent
    /// segment may be (it resumes the contexts and neighbour data its predecessors left).
    slice_ok: bool,
    /// Geometry of the last (sps id, pps id) pair; dropped whenever a parameter set changes.
    geo: Option<(u32, u32, Geometry)>,
}

/// temporal id (nuh_temporal_id_plus1 - 1) of a NAL unit.
fn temporal_id(nal: &[u8]) -> u32 {
    ((nal[1] & 7) as u32).saturating_sub(1)
}

impl State {
    /// Records a parameter-set NAL unit (other NAL units are ignored).
    pub fn add_param_nal(&mut self, nal: &[u8]) {
        if self.ps.add_nal(nal) {
            self.geo = None;
        }
    }

    /// Updates the slice bookkeeping after a slice NAL unit has been coded, raw or recoded.
    /// Both sides call this with identical bytes, so they stay in step.
    pub fn after_slice(&mut self, nal: &[u8], recoded: bool) {
        if nal.len() < 3 || layer_id(nal) != 0 || !is_slice(nal_type(nal)) {
            return;
        }
        // A header is at most a few hundred bytes; parse only a bounded prefix.
        let rbsp = unescape(&nal[2..nal.len().min(4096)]);
        match self.ps.slice_header(&rbsp, nal_type(nal), temporal_id(nal), self.last_indep.as_ref()) {
            Ok(sh) if !sh.dependent => {
                self.slice_ok = recoded;
                self.last_indep = Some(sh);
            }
            Ok(_) => self.slice_ok &= recoded,
            Err(_) => {
                self.slice_ok = false;
                if rbsp.first().is_some_and(|&b| b & 0x80 != 0) {
                    // An unreadable first segment of a picture: nothing after it can inherit.
                    self.last_indep = None;
                }
            }
        }
    }

    fn header(&self, rbsp: &[u8], nal_hdr: &[u8]) -> R<SliceHeader> {
        let sh = self.ps.slice_header(rbsp, nal_type(nal_hdr), temporal_id(nal_hdr), self.last_indep.as_ref())?;
        if sh.dependent && !self.slice_ok {
            return Err(Bad);
        }
        Ok(sh)
    }

    /// Makes sure `self.geo` describes the slice's parameter sets.
    fn ensure_geometry(&mut self, sh: &SliceHeader) -> R<()> {
        let pps = self.ps.pps[sh.pps_id as usize].as_ref().ok_or(Bad)?;
        if !matches!(&self.geo, Some((s, p, _)) if *s == pps.sps_id && *p == pps.id) {
            let sps = self.ps.sps[pps.sps_id as usize].as_ref().ok_or(Bad)?;
            self.geo = Some((pps.sps_id, pps.id, Geometry::new(sps, pps)?));
        }
        Ok(())
    }
}

/// Checks that a slice NAL unit can be rebuilt exactly from its bins (the HEVC counterpart
/// of the H.264 `verify_slice`), leaving the bins in `recs`. On success the segment becomes
/// part of the model's history. Errors are the container's failure reasons.
pub(crate) fn verify_slice(nal: &[u8], st: &mut State, recs: &mut Vec<Rec>) -> Result<Verified, usize> {
    if nal.len() < 3 || layer_id(nal) != 0 {
        return Err(1);
    }
    let rbsp = unescape(&nal[2..]);
    let mut chk = Vec::with_capacity(nal.len());
    escape(&rbsp, &mut chk);
    if chk != nal[2..] {
        return Err(0);
    }
    let sh = st.header(&rbsp, nal).map_err(|_| 1usize)?;
    st.ensure_geometry(&sh).map_err(|_| 1usize)?;
    let start = sh.data_bit / 8;
    if start > rbsp.len() {
        return Err(2);
    }
    let State { ps, pic, geo, .. } = st;
    let pps = ps.pps[sh.pps_id as usize].as_ref().ok_or(1usize)?;
    let sps = ps.sps[pps.sps_id as usize].as_ref().ok_or(1usize)?;
    let geo = &geo.as_ref().ok_or(1usize)?.2;
    let mut io = VerifyIo::new(&rbsp, start, recs);
    let mut p = SliceParser::new(sps, pps, &sh, geo, pic).map_err(|_| 1usize)?;
    p.run(&mut io).map_err(|_| 3usize)?;
    let data = io.finish();
    let tail = tail_fixup(&rbsp[start..], &data)?;
    p.commit();
    Ok(Verified { hdr_bits: sh.data_bit, rbsp, ..tail })
}

/// Rebuilds the CABAC payload of a recoded slice segment from the model, given its NAL
/// header and the decoded slice-header bits (`hdr_rbsp`, `hbits` bits long).
pub(crate) fn rebuild_payload(
    st: &mut State,
    nal_hdr: &[u8],
    hdr_rbsp: &[u8],
    hbits: usize,
    m: &mut Model,
    c: &mut ArithDec,
) -> R<Vec<u8>> {
    let mut probe = hdr_rbsp.to_vec();
    probe.extend_from_slice(&[0xFF; 8]);
    let sh = st.header(&probe, nal_hdr)?;
    if sh.data_bit != hbits {
        return Err(Bad);
    }
    st.ensure_geometry(&sh)?;
    let State { ps, pic, geo, .. } = st;
    let pps = ps.pps[sh.pps_id as usize].as_ref().ok_or(Bad)?;
    let sps = ps.sps[pps.sps_id as usize].as_ref().ok_or(Bad)?;
    let geo = &geo.as_ref().ok_or(Bad)?.2;
    let mut io = ModelIo::new(m, c);
    let mut p = SliceParser::new(sps, pps, &sh, geo, pic)?;
    p.run(&mut io)?;
    p.commit();
    Ok(io.enc.w.finish())
}

/// Cheap pre-check: true if any of the first slice segments of an HEVC track has a header this
/// module can parse with supported parameter sets. Files that fail it (4:2:2, 4:4:4, range or
/// SCC extension tools, broken parameter sets) are not worth running through the recoder.
pub fn quick_check(input: &[u8], track: &crate::mp4::AvcTrack) -> bool {
    let mut st = State::default();
    for n in &track.param_nals {
        st.add_param_nal(n);
    }
    let nls = track.nal_len_size;
    let mut tried = 0;
    for &(o, s) in track.samples.iter().take(64) {
        let Some(smp) = input.get(o as usize..(o + s as u64) as usize) else { return false };
        let mut i = 0;
        while i + nls <= smp.len() {
            let len = smp[i..i + nls].iter().fold(0usize, |a, &b| a << 8 | b as usize);
            i += nls;
            if len < 3 || i + len > smp.len() {
                break;
            }
            let nal = &smp[i..i + len];
            i += len;
            let t = nal_type(nal);
            if !is_slice(t) {
                st.add_param_nal(nal);
                continue;
            }
            if layer_id(nal) != 0 {
                continue;
            }
            let rbsp = unescape(&nal[2..nal.len().min(4096)]);
            if st.ps.slice_header(&rbsp, t, temporal_id(nal), st.last_indep.as_ref()).is_ok() {
                return true;
            }
            st.after_slice(nal, false);
            tried += 1;
            if tried >= 16 {
                return false;
            }
        }
    }
    false
}

/// Development aid: describes the stream's parameter sets and every slice segment that fails
/// to recode.
pub fn diagnose(input: &[u8], limit: usize) -> Vec<String> {
    let Some(track) = crate::mp4::find_hevc_track(input) else { return vec!["no HEVC track".into()] };
    let mut out = Vec::new();
    let mut st = State::default();
    for n in &track.param_nals {
        st.add_param_nal(n);
    }
    let describe = |st: &State, out: &mut Vec<String>| {
        for s in st.ps.sps.iter().flatten() {
            out.push(format!(
                "SPS {}: profile {} {}x{} chroma {} bits {}/{} ctb {} cb {}..{} tb {}..{} depth inter {} intra {} amp {} sao {} pcm {} rps {} lt {} tmvp {} supported {}",
                s.id, s.profile_idc, s.width, s.height, s.chroma_format_idc, s.bit_depth_luma, s.bit_depth_chroma,
                1 << s.log2_ctb, 1 << s.log2_min_cb, 1 << s.log2_ctb, 1 << s.log2_min_tb, 1 << s.log2_max_tb,
                s.max_th_depth_inter, s.max_th_depth_intra, s.amp, s.sao, s.pcm, s.st_rps.len(), s.long_term_refs,
                s.temporal_mvp, s.supported()
            ));
        }
        for p in st.ps.pps.iter().flatten() {
            out.push(format!(
                "PPS {}: dependent {} sign_hiding {} cabac_init {} tskip {} cu_qp_delta {}/{} wp {}/{} tq_bypass {} tiles {} ({}x{}) wpp {} lists_mod {} unsupported {}",
                p.id, p.dependent_slice_segments, p.sign_data_hiding, p.cabac_init_present, p.transform_skip,
                p.cu_qp_delta, p.diff_cu_qp_delta_depth, p.weighted_pred, p.weighted_bipred, p.transquant_bypass,
                p.tiles, p.tile_cols, p.tile_rows, p.entropy_coding_sync, p.lists_modification_present, p.unsupported
            ));
        }
    };
    describe(&st, &mut out);
    let nls = track.nal_len_size;
    let (mut ok, mut failed, mut idx) = ([0usize; 3], [0usize; 3], 0usize);
    let mut recs = Vec::new();
    let mut types = std::collections::BTreeMap::new();
    for &(o, s) in &track.samples {
        let smp = &input[o as usize..(o + s as u64) as usize];
        let mut i = 0;
        while i + nls <= smp.len() {
            let len = smp[i..i + nls].iter().fold(0usize, |a, &b| a << 8 | b as usize);
            i += nls;
            if i + len > smp.len() || len < 2 {
                break;
            }
            let nal = &smp[i..i + len];
            i += len;
            let t = nal_type(nal);
            *types.entry(t).or_insert(0usize) += 1;
            if !is_slice(t) {
                let before = st.ps.sps.iter().flatten().count() + st.ps.pps.iter().flatten().count();
                st.add_param_nal(nal);
                if idx == 0 && st.ps.sps.iter().flatten().count() + st.ps.pps.iter().flatten().count() != before {
                    out.push("in-band parameter sets:".into());
                    describe(&st, &mut out);
                }
                continue;
            }
            let rbsp = unescape(&nal[2..nal.len().min(4096)]);
            let hdr = st.ps.slice_header(&rbsp, t, temporal_id(nal), st.last_indep.as_ref());
            let stype = hdr.as_ref().map_or(2, |h| h.slice_type as usize).min(2);
            match verify_slice(nal, &mut st, &mut recs) {
                Ok(_) => {
                    ok[stype] += 1;
                    st.after_slice(nal, true);
                }
                Err(reason) => {
                    failed[stype] += 1;
                    if out.len() < limit {
                        let h = hdr.map(|h| {
                            format!(
                                "addr {} dep {} qp {} entry points {}",
                                h.segment_address,
                                h.dependent,
                                h.slice_qp,
                                h.entry_points.len()
                            )
                        });
                        out.push(format!(
                            "slice {idx} nal type {t} len {len}: {} | {:?}",
                            crate::FAIL_NAMES[reason.min(6)],
                            h
                        ));
                    }
                    st.after_slice(nal, false);
                }
            }
            idx += 1;
        }
    }
    out.push(format!("NAL unit types: {types:?}"));
    out.push(format!(
        "ok/failed by slice type: B {:?} P {:?} I {:?}",
        [ok[0], failed[0]],
        [ok[1], failed[1]],
        [ok[2], failed[2]]
    ));
    out
}
