//! HEVC CABAC context tables and coefficient scan orders.
//!
//! The `initValue` numbers are those of ITU-T H.265 (Tables 9-5 to 9-37), checked value by
//! value against the HM reference software's `TLibCommon/ContextTables.h` (BSD-3-Clause,
//! Copyright (c) 2010-2015, ITU/ISO/IEC). They are listed here in the specification's order:
//! one row per `initType` (0 = I slices, 1 and 2 = P/B slices, see [`init_type`]).
//!
//! All contexts of a slice live in one state array; the constants below are each syntax
//! element's first index in it. States use the same `pStateIdx << 1 | valMps` encoding as
//! the H.264 engine, whose arithmetic coder HEVC shares unchanged.

/// Value the standard gives contexts that a slice type never uses.
const CNU: u8 = 154;

pub const SAO_MERGE: usize = 0;
pub const SAO_TYPE: usize = 1;
pub const SPLIT_CU: usize = 2;
pub const TQ_BYPASS: usize = 5;
pub const SKIP: usize = 6;
pub const PRED_MODE: usize = 9;
pub const PART_MODE: usize = 10;
pub const PREV_INTRA: usize = 14;
pub const CHROMA_PRED: usize = 15;
pub const RQT_ROOT_CBF: usize = 16;
pub const MERGE_FLAG: usize = 17;
pub const MERGE_IDX: usize = 18;
pub const INTER_DIR: usize = 19;
pub const REF_IDX: usize = 24;
pub const MVP_FLAG: usize = 26;
pub const SPLIT_TU: usize = 27;
pub const CBF_LUMA: usize = 30;
pub const CBF_CHROMA: usize = 32;
pub const MVD_GT0: usize = 37;
pub const MVD_GT1: usize = 38;
pub const DQP: usize = 39;
pub const TSKIP: usize = 41;
pub const LAST_X: usize = 43;
pub const LAST_Y: usize = 61;
pub const CSBF: usize = 79;
pub const SIG: usize = 83;
pub const GT1: usize = 125;
pub const GT2: usize = 149;
/// Number of context variables per slice.
pub const NUM_CTX: usize = 155;

/// `(first index, initValue rows for initType 0, 1, 2)` per syntax element.
const GROUPS: &[(usize, [&[u8]; 3])] = &[
    (SAO_MERGE, [&[153], &[153], &[153]]),
    (SAO_TYPE, [&[200], &[185], &[160]]),
    (SPLIT_CU, [&[139, 141, 157], &[107, 139, 126], &[107, 139, 126]]),
    (TQ_BYPASS, [&[154], &[154], &[154]]),
    (SKIP, [&[CNU, CNU, CNU], &[197, 185, 201], &[197, 185, 201]]),
    (PRED_MODE, [&[CNU], &[149], &[134]]),
    (PART_MODE, [&[184, CNU, CNU, CNU], &[154, 139, 154, 154], &[154, 139, 154, 154]]),
    (PREV_INTRA, [&[184], &[154], &[183]]),
    (CHROMA_PRED, [&[63], &[152], &[152]]),
    (RQT_ROOT_CBF, [&[CNU], &[79], &[79]]),
    (MERGE_FLAG, [&[CNU], &[110], &[154]]),
    (MERGE_IDX, [&[CNU], &[122], &[137]]),
    (INTER_DIR, [&[CNU, CNU, CNU, CNU, CNU], &[95, 79, 63, 31, 31], &[95, 79, 63, 31, 31]]),
    (REF_IDX, [&[CNU, CNU], &[153, 153], &[153, 153]]),
    (MVP_FLAG, [&[CNU], &[168], &[168]]),
    (SPLIT_TU, [&[153, 138, 138], &[124, 138, 94], &[224, 167, 122]]),
    (CBF_LUMA, [&[111, 141], &[153, 111], &[153, 111]]),
    (CBF_CHROMA, [&[94, 138, 182, 154, 154], &[149, 107, 167, 154, 154], &[149, 92, 167, 154, 154]]),
    (MVD_GT0, [&[CNU], &[140], &[169]]),
    (MVD_GT1, [&[CNU], &[198], &[198]]),
    (DQP, [&[154, 154], &[154, 154], &[154, 154]]),
    (TSKIP, [&[139, 139], &[139, 139], &[139, 139]]),
    (LAST_X, [LAST_I, LAST_P, LAST_B]),
    (LAST_Y, [LAST_I, LAST_P, LAST_B]),
    (CSBF, [&[91, 171, 134, 141], &[121, 140, 61, 154], &[121, 140, 61, 154]]),
    (SIG, [SIG_I, SIG_P, SIG_B]),
    (GT1, [GT1_I, GT1_P, GT1_B]),
    (GT2, [&[138, 153, 136, 167, 152, 152], &[107, 167, 91, 122, 107, 167], &[107, 167, 91, 107, 107, 167]]),
];

const LAST_I: &[u8] = &[110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63];
const LAST_P: &[u8] = &[125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123, 108];
const LAST_B: &[u8] = &[125, 110, 124, 110, 95, 94, 125, 111, 111, 79, 125, 126, 111, 111, 79, 108, 123, 93];

/// sig_coeff_flag: 27 luma contexts, then 15 chroma contexts.
const SIG_I: &[u8] = &[
    111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107, 125, 141, 179, 153, 125, 107, 125,
    141, 179, 153, 125, 140, 139, 182, 182, 152, 136, 152, 136, 153, 136, 139, 111, 136, 139, 111,
];
const SIG_P: &[u8] = &[
    155, 154, 139, 153, 139, 123, 123, 63, 153, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136, 153, 154, 166, 183,
    140, 136, 153, 154, 170, 153, 123, 123, 107, 121, 107, 121, 167, 151, 183, 140, 151, 183, 140,
];
const SIG_B: &[u8] = &[
    170, 154, 139, 153, 139, 123, 123, 63, 124, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136, 153, 154, 166, 183,
    140, 136, 153, 154, 170, 153, 138, 138, 122, 121, 122, 121, 167, 151, 183, 140, 151, 183, 140,
];

/// coeff_abs_level_greater1_flag: 16 luma contexts (4 sets of 4), then 8 chroma contexts.
const GT1_I: &[u8] = &[
    140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107, 122, 152, 140, 179, 166, 182, 140, 227, 122, 197,
];
const GT1_P: &[u8] = &[
    154, 196, 196, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 137, 169, 194, 166, 167, 154, 167, 137,
    182,
];
const GT1_B: &[u8] = &[
    154, 196, 167, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 122, 169, 208, 166, 167, 154, 152, 167,
    182,
];

const fn build_init() -> [[u8; NUM_CTX]; 3] {
    let mut t = [[0u8; NUM_CTX]; 3];
    let mut g = 0;
    while g < GROUPS.len() {
        let (first, rows) = GROUPS[g];
        let mut it = 0;
        while it < 3 {
            let row = rows[it];
            let mut i = 0;
            while i < row.len() {
                t[it][first + i] = row[i];
                i += 1;
            }
            it += 1;
        }
        g += 1;
    }
    t
}

/// `initValue` of every context, per `initType`.
pub static INIT: [[u8; NUM_CTX]; 3] = build_init();

/// initType (9.3.2.2) of a slice.
pub fn init_type(slice_type: u32, cabac_init_flag: bool) -> usize {
    match slice_type {
        super::I => 0,
        super::P => 1 + cabac_init_flag as usize,
        _ => 2 - cabac_init_flag as usize,
    }
}

/// Initialises the context variables of a slice (9.3.2.2).
pub fn init_states(st: &mut [u8; NUM_CTX], slice_qp: i32, init_type: usize) {
    let qp = slice_qp.clamp(0, 51);
    for (s, &v) in st.iter_mut().zip(INIT[init_type].iter()) {
        let v = v as i32;
        let m = (v >> 4) * 5 - 45;
        let n = ((v & 15) << 3) - 16;
        let pre = (((m * qp) >> 4) + n).clamp(1, 126);
        *s = if pre <= 63 { ((63 - pre) << 1) as u8 } else { (((pre - 64) << 1) | 1) as u8 };
    }
}

/// ctxIdxMap of 9.3.4.2.5 (sig_coeff_flag in 4x4 transform blocks), indexed by `(y << 2) + x`.
pub const CTX_IDX_MAP: [u8; 16] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8];

/// Scan orders (6.5.3 - 6.5.5): `SCAN[log2 size][scanIdx][i] = (x, y)` for block sizes
/// 1, 2, 4 and 8, scanIdx 0 = up-right diagonal, 1 = horizontal, 2 = vertical.
pub static SCAN: [[[(u8, u8); 64]; 3]; 4] = build_scans();
/// Inverse of [`SCAN`]: position in scan order of `(x, y)`, indexed by `y * size + x`.
pub static SCAN_POS: [[[u8; 64]; 3]; 4] = build_scan_pos();

const fn build_scans() -> [[[(u8, u8); 64]; 3]; 4] {
    let mut t = [[[(0u8, 0u8); 64]; 3]; 4];
    let mut l = 0;
    while l < 4 {
        let n = 1i32 << l;
        // up-right diagonal
        let (mut i, mut x, mut y) = (0usize, 0i32, 0i32);
        while i < (n * n) as usize {
            while y >= 0 {
                if x < n && y < n {
                    t[l][0][i] = (x as u8, y as u8);
                    i += 1;
                }
                y -= 1;
                x += 1;
            }
            y = x;
            x = 0;
        }
        // horizontal and vertical
        let mut k = 0;
        while k < n * n {
            t[l][1][k as usize] = ((k % n) as u8, (k / n) as u8);
            t[l][2][k as usize] = ((k / n) as u8, (k % n) as u8);
            k += 1;
        }
        l += 1;
    }
    t
}

const fn build_scan_pos() -> [[[u8; 64]; 3]; 4] {
    let scans = build_scans();
    let mut t = [[[0u8; 64]; 3]; 4];
    let mut l = 0;
    while l < 4 {
        let n = 1usize << l;
        let mut s = 0;
        while s < 3 {
            let mut i = 0;
            while i < n * n {
                let (x, y) = scans[l][s][i];
                t[l][s][y as usize * n + x as usize] = i as u8;
                i += 1;
            }
            s += 1;
        }
        l += 1;
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_context_initialised() {
        // CNU (154) is a legal value, so check that no slot kept the 0 the builder starts with.
        for (it, row) in INIT.iter().enumerate() {
            for (i, &v) in row.iter().enumerate() {
                assert_ne!(v, 0, "initType {it} context {i}");
            }
        }
        assert_eq!(GT2 + 6, NUM_CTX);
    }

    #[test]
    fn scans() {
        // 4x4 up-right diagonal: (0,0) (0,1) (1,0) (0,2) (1,1) (2,0) ...
        assert_eq!(&SCAN[2][0][..6], &[(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]);
        assert_eq!(SCAN[2][0][15], (3, 3));
        assert_eq!(SCAN[1][1][..4], [(0, 0), (1, 0), (0, 1), (1, 1)]);
        assert_eq!(SCAN[1][2][..4], [(0, 0), (0, 1), (1, 0), (1, 1)]);
        for l in 0..4 {
            let n = 1usize << l;
            for s in 0..3 {
                for i in 0..n * n {
                    let (x, y) = SCAN[l][s][i];
                    assert_eq!(SCAN_POS[l][s][y as usize * n + x as usize] as usize, i);
                }
            }
        }
    }

    #[test]
    fn init_matches_spec_example() {
        // initValue 154: m = 0, n = 64, so preCtxState = 64 at any QP (pStateIdx 0, valMps 1).
        let mut st = [0u8; NUM_CTX];
        init_states(&mut st, 30, 0);
        assert_eq!(st[TQ_BYPASS], 1);
        // initValue 139 (transform_skip_flag) at QP 26: m = -5, n = 72, pre = 64 - 9 + ... = 64.
        let pre = ((-5 * 26) >> 4) + 72;
        assert_eq!(st.len(), NUM_CTX);
        init_states(&mut st, 26, 0);
        let want = if pre <= 63 { ((63 - pre) << 1) as u8 } else { (((pre - 64) << 1) | 1) as u8 };
        assert_eq!(st[TSKIP], want);
    }
}
