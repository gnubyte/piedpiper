//! VP9 frame headers, probability contexts and the state carried from frame to frame,
//! following libvpx (`vp9/decoder/vp9_decodeframe.c`, `vp9_decoder.c`).
//!
//! Only what parsing needs is modelled: probabilities, reference slot sizes, motion vectors
//! of the previous frame and the flags that decide when they are used. Streams that need
//! backward probability adaptation (`error_resilient_mode == 0 && frame_parallel_decoding_mode
//! == 0`), segmentation, intra-only frames and 4:4:4 / 4:2:2 streams (profiles 1 and 3)
//! are not recoded; profiles 0 and 2 (8-bit and 10/12-bit 4:2:0) are.

pub mod bool;
pub mod stream;
#[doc(hidden)]
pub mod synth;
pub mod tables;
pub mod tile;

use crate::bits::{Bad, BitReader, R};
use bool::BoolDecoder;
use tables::*;

pub const KEY_FRAME: u8 = 0;
pub const SWITCHABLE: u8 = 4;
pub const TX_MODE_SELECT: u8 = 4;
pub const SINGLE_REFERENCE: u8 = 0;
pub const COMPOUND_REFERENCE: u8 = 1;
pub const REFERENCE_MODE_SELECT: u8 = 2;
pub const LAST_FRAME: i8 = 1;
pub const GOLDEN_FRAME: i8 = 2;
pub const ALTREF_FRAME: i8 = 3;

#[derive(Clone, Copy)]
pub struct MvComp {
    pub sign: u8,
    pub classes: [u8; 10],
    pub class0: u8,
    pub bits: [u8; 10],
    pub class0_fp: [[u8; 3]; 2],
    pub fp: [u8; 3],
    pub class0_hp: u8,
    pub hp: u8,
}

/// Coefficient probabilities: [tx_size][plane_type][ref][band][ctx][node].
pub type CoefProbs = [[[[[[u8; 3]; 6]; 6]; 2]; 2]; 4];

/// All adaptive probabilities of a frame (libvpx `FRAME_CONTEXT`).
#[derive(Clone)]
pub struct FrameCtx {
    pub tx8: [[u8; 1]; 2],
    pub tx16: [[u8; 2]; 2],
    pub tx32: [[u8; 3]; 2],
    pub coef: CoefProbs,
    pub skip: [u8; 3],
    pub inter_mode: [[u8; 3]; 7],
    pub interp: [[u8; 2]; 4],
    pub intra_inter: [u8; 4],
    pub comp_inter: [u8; 5],
    pub single_ref: [[u8; 2]; 5],
    pub comp_ref: [u8; 5],
    pub y_mode: [[u8; 9]; 4],
    pub uv_mode: [[u8; 9]; 10],
    pub partition: [[u8; 3]; 16],
    pub mv_joints: [u8; 3],
    pub mv: [MvComp; 2],
}

fn rows<const N: usize>(src: &[u8]) -> Vec<[u8; N]> {
    src.chunks(N).map(|c| c.try_into().unwrap()).collect()
}

impl FrameCtx {
    pub fn defaults() -> Self {
        let mut coef: CoefProbs = [[[[[[0u8; 3]; 6]; 6]; 2]; 2]; 4];
        for (tx, table) in [&DEFAULT_COEF_PROBS_4X4, &DEFAULT_COEF_PROBS_8X8, &DEFAULT_COEF_PROBS_16X16, &DEFAULT_COEF_PROBS_32X32]
            .iter()
            .enumerate()
        {
            let mut k = 0;
            for plane in 0..2 {
                for r in 0..2 {
                    for band in 0..6 {
                        let ctxs = if band == 0 { 3 } else { 6 };
                        for ctx in 0..ctxs {
                            for node in 0..3 {
                                coef[tx][plane][r][band][ctx][node] = table[k];
                                k += 1;
                            }
                        }
                    }
                }
            }
        }
        let t = &DEFAULT_TX_PROBS;
        let mut mv = [MvComp { sign: 0, classes: [0; 10], class0: 0, bits: [0; 10], class0_fp: [[0; 3]; 2], fp: [0; 3], class0_hp: 0, hp: 0 }; 2];
        let n = &DEFAULT_NMV;
        for (i, c) in mv.iter_mut().enumerate() {
            let o = 3 + 33 * i;
            c.sign = n[o];
            c.classes.copy_from_slice(&n[o + 1..o + 11]);
            c.class0 = n[o + 11];
            c.bits.copy_from_slice(&n[o + 12..o + 22]);
            c.class0_fp[0].copy_from_slice(&n[o + 22..o + 25]);
            c.class0_fp[1].copy_from_slice(&n[o + 25..o + 28]);
            c.fp.copy_from_slice(&n[o + 28..o + 31]);
            c.class0_hp = n[o + 31];
            c.hp = n[o + 32];
        }
        FrameCtx {
            tx32: [[t[0], t[1], t[2]], [t[3], t[4], t[5]]],
            tx16: [[t[6], t[7]], [t[8], t[9]]],
            tx8: [[t[10]], [t[11]]],
            coef,
            skip: DEFAULT_SKIP_PROBS,
            inter_mode: rows::<3>(&DEFAULT_INTER_MODE_PROBS).try_into().unwrap(),
            interp: rows::<2>(&DEFAULT_SWITCHABLE_INTERP_PROB).try_into().unwrap(),
            intra_inter: DEFAULT_INTRA_INTER_P,
            comp_inter: DEFAULT_COMP_INTER_P,
            single_ref: rows::<2>(&DEFAULT_SINGLE_REF_P).try_into().unwrap(),
            comp_ref: DEFAULT_COMP_REF_P,
            y_mode: rows::<9>(&DEFAULT_IF_Y_PROBS).try_into().unwrap(),
            uv_mode: rows::<9>(&DEFAULT_IF_UV_PROBS).try_into().unwrap(),
            partition: rows::<3>(&DEFAULT_PARTITION_PROBS).try_into().unwrap(),
            mv_joints: [n[0], n[1], n[2]],
            mv,
        }
    }
}

/// Parsed frame header (uncompressed + compressed parts).
#[derive(Clone, Default)]
pub struct FrameHeader {
    pub show_existing: bool,
    pub frame_type: u8,
    pub show_frame: bool,
    pub error_res: bool,
    pub intra_only: bool,
    pub refresh_flags: u8,
    pub width: u32,
    pub height: u32,
    pub allow_hp: bool,
    pub interp_filter: u8,
    pub refresh_ctx: bool,
    pub parallel: bool,
    pub ctx_idx: usize,
    pub lossless: bool,
    pub log2_tile_cols: u32,
    pub log2_tile_rows: u32,
    /// Bytes of the uncompressed header.
    pub uncomp_bytes: usize,
    pub comp_bytes: usize,
    pub tx_mode: u8,
    pub reference_mode: u8,
    pub comp_fixed_ref: i8,
    pub comp_var_ref: [i8; 2],
    pub sign_bias: [u8; 4],
    pub use_prev_mvs: bool,
    pub base_q: u32,
    /// 8, 10 or 12 (set by key frames, inherited by inter frames).
    pub bit_depth: u8,
}

impl FrameHeader {
    pub fn intra(&self) -> bool {
        self.frame_type == KEY_FRAME || self.intra_only
    }
    pub fn mi_cols(&self) -> usize {
        (self.width as usize + 7) >> 3
    }
    pub fn mi_rows(&self) -> usize {
        (self.height as usize + 7) >> 3
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct MvRef {
    pub ref_frame: [i8; 2],
    pub mv: [[i16; 2]; 2],
}

/// Decoder state carried between frames.
pub struct State {
    pub ctxs: Vec<FrameCtx>,
    pub slots: [(u32, u32); 8],
    pub prev_mvs: Vec<MvRef>,
    pub last_width: u32,
    pub last_height: u32,
    pub last_show_frame: bool,
    pub last_intra_only: bool,
    pub last_frame_type: u8,
    pub intra_only: bool,
    pub frame_type: u8,
    pub sign_bias: [u8; 4],
    pub bit_depth: u8,
    started: bool,
    /// Mode info and per-4x4 ends of block of the last shown frame (model context only).
    pub prev_grid: Vec<tile::Mi>,
    pub prev_eob: [Vec<u8>; 3],
}

impl Default for State {
    fn default() -> Self {
        State {
            ctxs: vec![FrameCtx::defaults(); 4],
            slots: [(0, 0); 8],
            prev_mvs: Vec::new(),
            last_width: 0,
            last_height: 0,
            last_show_frame: false,
            last_intra_only: false,
            last_frame_type: KEY_FRAME,
            intra_only: false,
            frame_type: KEY_FRAME,
            sign_bias: [0; 4],
            bit_depth: 8,
            started: false,
            prev_grid: Vec::new(),
            prev_eob: Default::default(),
        }
    }
}

fn signed(r: &mut BitReader, bits: u32) -> R<i32> {
    let v = r.u(bits)? as i32;
    Ok(if r.bit()? == 1 { -v } else { v })
}

/// Parses the uncompressed header. Updates the per-frame "last_*" bookkeeping exactly where
/// libvpx does (at the start of every header, shown-existing frames included).
pub fn uncompressed_header(st: &mut State, data: &[u8]) -> R<FrameHeader> {
    st.last_frame_type = st.frame_type;
    st.last_intra_only = st.intra_only;
    let mut r = BitReader::new(data);
    let mut h = FrameHeader::default();
    if r.u(2)? != 2 {
        return Err(Bad);
    }
    let profile = r.bit()? | r.bit()? << 1;
    if profile & 1 != 0 {
        return Err(Bad); // 4:4:4 / 4:2:2 (profiles 1 and 3)
    }
    if r.flag()? {
        h.show_existing = true;
        r.u(3)?;
        h.uncomp_bytes = r.pos().div_ceil(8);
        return Ok(h);
    }
    h.frame_type = r.bit()? as u8;
    st.frame_type = h.frame_type;
    h.show_frame = r.flag()?;
    h.error_res = r.flag()?;
    let mut refs = [0usize; 3];
    if h.frame_type == KEY_FRAME {
        if r.u(24)? != 0x49_8342 {
            return Err(Bad);
        }
        st.bit_depth = if profile >= 2 { 10 + 2 * r.bit()? as u8 } else { 8 };
        let cs = r.u(3)?;
        if cs == 7 {
            return Err(Bad); // sRGB needs 4:4:4 (profiles 1 and 3)
        }
        r.bit()?; // color range
        h.width = r.u(16)? + 1;
        h.height = r.u(16)? + 1;
        if r.flag()? {
            r.u(32)?; // render size
        }
        h.refresh_flags = 0xFF;
    } else {
        h.intra_only = if h.show_frame { false } else { r.flag()? };
        st.intra_only = h.intra_only;
        if !h.error_res {
            r.u(2)?; // reset_frame_context
        }
        if h.intra_only {
            return Err(Bad);
        }
        h.refresh_flags = r.u(8)? as u8;
        for (i, slot) in refs.iter_mut().enumerate() {
            *slot = r.u(3)? as usize;
            st.sign_bias[1 + i] = r.bit()? as u8;
        }
        let mut found = false;
        for &slot in &refs {
            if r.flag()? {
                (h.width, h.height) = st.slots[slot];
                found = true;
                break;
            }
        }
        if !found {
            h.width = r.u(16)? + 1;
            h.height = r.u(16)? + 1;
        }
        if h.width == 0 || h.height == 0 {
            return Err(Bad);
        }
        if r.flag()? {
            r.u(32)?;
        }
        h.allow_hp = r.flag()?;
        h.interp_filter = if r.flag()? { SWITCHABLE } else { [1, 0, 2, 3][r.u(2)? as usize] };
    }
    if !h.error_res {
        h.refresh_ctx = r.flag()?;
        h.parallel = r.flag()?;
    } else {
        h.parallel = true;
    }
    if !h.error_res && !h.parallel {
        return Err(Bad); // needs backward adaptation
    }
    h.ctx_idx = r.u(2)? as usize;
    if h.intra() || h.error_res {
        // vp9_setup_past_independence
        let fc = FrameCtx::defaults();
        if h.frame_type == KEY_FRAME || h.error_res {
            st.ctxs = vec![fc; 4];
        }
        st.sign_bias = [0; 4];
        h.ctx_idx = 0;
    }
    // loop filter
    r.u(6)?;
    r.u(3)?;
    if r.flag()? && r.flag()? {
        for _ in 0..6 {
            if r.flag()? {
                signed(&mut r, 6)?;
            }
        }
    }
    // quantization
    h.base_q = r.u(8)?;
    let mut deltas_zero = true;
    for _ in 0..3 {
        if r.flag()? && signed(&mut r, 4)? != 0 {
            deltas_zero = false;
        }
    }
    h.lossless = h.base_q == 0 && deltas_zero;
    // segmentation
    if r.flag()? {
        return Err(Bad);
    }
    // tiles
    let sb_cols = (h.mi_cols() + 7) >> 3;
    let mut min_log2 = 0u32;
    while (64usize << min_log2) < sb_cols {
        min_log2 += 1;
    }
    let mut max_log2 = 1u32;
    while (sb_cols >> max_log2) >= 4 {
        max_log2 += 1;
    }
    max_log2 -= 1;
    h.log2_tile_cols = min_log2;
    let mut ones = max_log2.saturating_sub(min_log2);
    while ones > 0 && r.flag()? {
        h.log2_tile_cols += 1;
        ones -= 1;
    }
    if h.log2_tile_cols > 6 {
        return Err(Bad);
    }
    h.log2_tile_rows = r.bit()?;
    if h.log2_tile_rows == 1 {
        h.log2_tile_rows += r.bit()?;
    }
    h.comp_bytes = r.u(16)? as usize;
    if h.comp_bytes == 0 {
        return Err(Bad);
    }
    h.uncomp_bytes = r.pos().div_ceil(8);
    h.sign_bias = st.sign_bias;
    h.bit_depth = st.bit_depth;
    h.use_prev_mvs = !h.error_res
        && h.width == st.last_width
        && h.height == st.last_height
        && !st.last_intra_only
        && st.last_show_frame
        && st.last_frame_type != KEY_FRAME;
    Ok(h)
}

fn diff_update(r: &mut BoolDecoder, p: &mut u8) {
    if r.read(252) == 1 {
        let v = if r.read(128) == 0 {
            r.literal(4)
        } else if r.read(128) == 0 {
            r.literal(4) + 16
        } else if r.read(128) == 0 {
            r.literal(5) + 32
        } else {
            let v = r.literal(7);
            (if v < 65 { v } else { (v << 1) - 65 + r.read(128) }) + 64
        };
        *p = inv_remap_prob(v as usize, *p as i32) as u8;
    }
}

fn inv_recenter_nonneg(v: i32, m: i32) -> i32 {
    if v > 2 * m {
        v
    } else if v & 1 == 1 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

static INV_MAP: [u8; 254] = {
    // libvpx vp9_dsubexp.c inv_map_table: 20 coarse values, then the rest in order.
    let mut t = [0u8; 254];
    let mut i = 0;
    while i < 20 {
        t[i] = (7 + 13 * i) as u8;
        i += 1;
    }
    let mut v = 1u32;
    let mut k = 20;
    while k < 253 {
        if v % 13 != 7 {
            t[k] = v as u8;
            k += 1;
        }
        v += 1;
    }
    t[253] = 253;
    t
};

fn inv_remap_prob(v: usize, m: i32) -> i32 {
    let v = INV_MAP[v.min(253)] as i32;
    let m = m - 1;
    if (m << 1) <= 255 {
        1 + inv_recenter_nonneg(v, m)
    } else {
        255 - inv_recenter_nonneg(v, 255 - 1 - m)
    }
}

fn update_mv_prob(r: &mut BoolDecoder, p: &mut u8) {
    if r.read(252) == 1 {
        *p = ((r.literal(7) << 1) | 1) as u8;
    }
}

/// Parses the compressed header into `fc` (a copy of the frame's context) and the frame-level
/// coding modes in `h`.
pub fn compressed_header(h: &mut FrameHeader, fc: &mut FrameCtx, data: &[u8]) -> R<()> {
    let mut r = BoolDecoder::new(data);
    if r.read(128) != 0 {
        return Err(Bad);
    }
    h.tx_mode = if h.lossless {
        0
    } else {
        let mut m = r.literal(2) as u8;
        if m == 3 {
            m += r.read(128) as u8;
        }
        m
    };
    if h.tx_mode == TX_MODE_SELECT {
        for i in 0..2 {
            diff_update(&mut r, &mut fc.tx8[i][0]);
        }
        for i in 0..2 {
            for j in 0..2 {
                diff_update(&mut r, &mut fc.tx16[i][j]);
            }
        }
        for i in 0..2 {
            for j in 0..3 {
                diff_update(&mut r, &mut fc.tx32[i][j]);
            }
        }
    }
    let max_tx = TX_MODE_TO_BIGGEST_TX_SIZE[h.tx_mode as usize] as usize;
    for tx in 0..=max_tx {
        if r.read(128) == 1 {
            for plane in 0..2 {
                for rf in 0..2 {
                    for band in 0..6 {
                        for ctx in 0..if band == 0 { 3 } else { 6 } {
                            for node in 0..3 {
                                diff_update(&mut r, &mut fc.coef[tx][plane][rf][band][ctx][node]);
                            }
                        }
                    }
                }
            }
        }
    }
    for k in 0..3 {
        diff_update(&mut r, &mut fc.skip[k]);
    }
    if !h.intra() {
        for i in 0..7 {
            for j in 0..3 {
                diff_update(&mut r, &mut fc.inter_mode[i][j]);
            }
        }
        if h.interp_filter == SWITCHABLE {
            for j in 0..4 {
                for i in 0..2 {
                    diff_update(&mut r, &mut fc.interp[j][i]);
                }
            }
        }
        for i in 0..4 {
            diff_update(&mut r, &mut fc.intra_inter[i]);
        }
        let sb = h.sign_bias;
        let compound_allowed = sb[2] != sb[1] || sb[3] != sb[1];
        h.reference_mode = if compound_allowed {
            if r.read(128) == 1 {
                if r.read(128) == 1 {
                    REFERENCE_MODE_SELECT
                } else {
                    COMPOUND_REFERENCE
                }
            } else {
                SINGLE_REFERENCE
            }
        } else {
            SINGLE_REFERENCE
        };
        if h.reference_mode != SINGLE_REFERENCE {
            if sb[1] == sb[2] {
                h.comp_fixed_ref = ALTREF_FRAME;
                h.comp_var_ref = [LAST_FRAME, GOLDEN_FRAME];
            } else if sb[1] == sb[3] {
                h.comp_fixed_ref = GOLDEN_FRAME;
                h.comp_var_ref = [LAST_FRAME, ALTREF_FRAME];
            } else {
                h.comp_fixed_ref = LAST_FRAME;
                h.comp_var_ref = [GOLDEN_FRAME, ALTREF_FRAME];
            }
        }
        if h.reference_mode == REFERENCE_MODE_SELECT {
            for i in 0..5 {
                diff_update(&mut r, &mut fc.comp_inter[i]);
            }
        }
        if h.reference_mode != COMPOUND_REFERENCE {
            for i in 0..5 {
                diff_update(&mut r, &mut fc.single_ref[i][0]);
                diff_update(&mut r, &mut fc.single_ref[i][1]);
            }
        }
        if h.reference_mode != SINGLE_REFERENCE {
            for i in 0..5 {
                diff_update(&mut r, &mut fc.comp_ref[i]);
            }
        }
        for j in 0..4 {
            for i in 0..9 {
                diff_update(&mut r, &mut fc.y_mode[j][i]);
            }
        }
        for j in 0..16 {
            for i in 0..3 {
                diff_update(&mut r, &mut fc.partition[j][i]);
            }
        }
        for i in 0..3 {
            update_mv_prob(&mut r, &mut fc.mv_joints[i]);
        }
        for c in fc.mv.iter_mut() {
            update_mv_prob(&mut r, &mut c.sign);
            for p in c.classes.iter_mut() {
                update_mv_prob(&mut r, p);
            }
            update_mv_prob(&mut r, &mut c.class0);
            for p in c.bits.iter_mut() {
                update_mv_prob(&mut r, p);
            }
        }
        for c in fc.mv.iter_mut() {
            for j in 0..2 {
                for p in c.class0_fp[j].iter_mut() {
                    update_mv_prob(&mut r, p);
                }
            }
            for p in c.fp.iter_mut() {
                update_mv_prob(&mut r, p);
            }
        }
        if h.allow_hp {
            for c in fc.mv.iter_mut() {
                update_mv_prob(&mut r, &mut c.class0_hp);
                update_mv_prob(&mut r, &mut c.hp);
            }
        }
    }
    if r.overrun() {
        return Err(Bad);
    }
    Ok(())
}

/// Updates the state after a frame has been parsed.
pub fn end_frame(st: &mut State, h: &FrameHeader, fc: &FrameCtx, frame_mvs: Vec<MvRef>) {
    if !h.show_existing {
        if h.refresh_ctx {
            st.ctxs[h.ctx_idx] = fc.clone();
        }
        for i in 0..8 {
            if (h.refresh_flags >> i) & 1 == 1 {
                st.slots[i] = (h.width, h.height);
            }
        }
        st.last_show_frame = h.show_frame;
        st.prev_mvs = frame_mvs;
        st.last_width = h.width;
        st.last_height = h.height;
    }
    st.started = true;
}

/// Development aid: (key frame, error resilient, frame parallel) of a profile 0/2 frame,
/// read without any state.
#[doc(hidden)]
pub fn frame_mode_flags(data: &[u8]) -> Option<(bool, bool, bool)> {
    let mut r = BitReader::new(data);
    if r.u(2).ok()? != 2 {
        return None;
    }
    let profile = r.bit().ok()? | r.bit().ok()? << 1;
    if profile == 3 {
        r.bit().ok()?;
    }
    if r.flag().ok()? {
        return None; // show_existing_frame
    }
    let key = r.bit().ok()? == 0;
    let show = r.flag().ok()?;
    let error_res = r.flag().ok()?;
    if key {
        r.u(24).ok()?;
        if profile >= 2 {
            r.bit().ok()?;
        }
        if r.u(3).ok()? != 7 {
            r.bit().ok()?;
            if profile & 1 == 1 {
                r.u(3).ok()?;
            }
        } else if profile & 1 == 1 {
            r.bit().ok()?;
        }
        r.u(32).ok()?;
        if r.flag().ok()? {
            r.u(32).ok()?;
        }
    } else {
        let intra_only = if show { false } else { r.flag().ok()? };
        if !error_res {
            r.u(2).ok()?;
        }
        if intra_only {
            return None;
        }
        r.u(8).ok()?;
        r.u(12).ok()?;
        let mut found = false;
        for _ in 0..3 {
            if r.flag().ok()? {
                found = true;
                break;
            }
        }
        if !found {
            r.u(32).ok()?;
        }
        if r.flag().ok()? {
            r.u(32).ok()?;
        }
        r.bit().ok()?;
        if !r.flag().ok()? {
            r.u(2).ok()?;
        }
    }
    let parallel = if error_res {
        true
    } else {
        r.bit().ok()?;
        r.flag().ok()?
    };
    Some((key, error_res, parallel))
}

/// Splits a WebM block into frames (superframes carry an index at the end).
pub fn split_superframe(data: &[u8]) -> (Vec<(usize, usize)>, usize) {
    if let Some(&marker) = data.last() {
        if marker & 0xE0 == 0xC0 {
            let frames = (marker & 7) as usize + 1;
            let mag = ((marker >> 3) & 3) as usize + 1;
            let index = 2 + mag * frames;
            if data.len() >= index && data[data.len() - index] == marker {
                let mut out = Vec::new();
                let mut pos = 0;
                let mut p = data.len() - index + 1;
                for _ in 0..frames {
                    let mut size = 0usize;
                    for k in 0..mag {
                        size |= (data[p + k] as usize) << (8 * k);
                    }
                    p += mag;
                    if pos + size > data.len() - index {
                        return (vec![(0, data.len())], 0);
                    }
                    out.push((pos, size));
                    pos += size;
                }
                if pos == data.len() - index {
                    return (out, index);
                }
            }
        }
    }
    (vec![(0, data.len())], 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_map_matches_libvpx_prefix() {
        let expect: [u8; 30] = [7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163, 176, 189, 202, 215, 228, 241, 254, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11];
        assert_eq!(&INV_MAP[..30], &expect);
        assert_eq!(INV_MAP[252], 252);
        assert_eq!(INV_MAP[253], 253);
    }
}
