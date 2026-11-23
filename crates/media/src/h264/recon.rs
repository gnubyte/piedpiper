//! Picture reconstruction for model context.
//!
//! The recoder does not need decoded pictures to rebuild a bitstream, but the decoded
//! prediction of a block says a lot about its residual: textured predictions leave larger
//! residuals at the frequencies where they are textured. This module rebuilds pictures from
//! the parsed syntax (intra prediction, motion compensation, inverse transforms) so the
//! model can use them. The compressor and decompressor run exactly this code on exactly the
//! same syntax, so it only has to be deterministic, not bit-exact to the H.264 decoding
//! process: a mistake here (or a skipped step such as the deblocking filter) costs
//! compression, never correctness.

/// One 8-bit sample plane.
#[derive(Clone, Default)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub d: Vec<u8>,
}

impl Plane {
    pub fn new(w: usize, h: usize) -> Self {
        Plane { w, h, d: vec![128; w * h] }
    }

    #[inline]
    pub fn at(&self, x: i32, y: i32) -> u8 {
        let x = x.clamp(0, self.w as i32 - 1) as usize;
        let y = y.clamp(0, self.h as i32 - 1) as usize;
        self.d[y * self.w + x]
    }
}

/// Motion of one 4x4 luma block (for direct prediction from a co-located picture).
#[derive(Clone, Copy, Default)]
pub struct Motion {
    pub mv: [[i16; 2]; 2],
    /// Reference index per list, -1 when the list is unused (both -1 for intra blocks).
    pub refi: [i8; 2],
    /// Picture order count of the reference picture per list (temporal direct).
    pub rpoc: [i32; 2],
}

/// A decoded (or partly decoded) picture.
#[derive(Clone, Default)]
pub struct Pic {
    pub y: Plane,
    pub u: Plane,
    pub v: Plane,
    pub poc: i32,
    pub frame_num: u32,
    pub frame_num_wrap: i32,
    pub long_term: bool,
    pub lt_idx: u32,
    /// Motion field, one entry per 4x4 luma block in raster order.
    pub mf: Vec<Motion>,
    pub w4: usize,
    /// Per-macroblock facts the deblocking filter needs.
    pub db: Vec<DbInfo>,
}

/// What the deblocking filter needs to know about a macroblock.
#[derive(Clone, Copy, Default)]
pub struct DbInfo {
    /// Set once the macroblock was reconstructed by a slice that is kept.
    pub done: bool,
    pub qp: u8,
    pub qpc: [u8; 2],
    pub intra: bool,
    pub t8: bool,
    /// Luma 4x4 blocks (raster) with non-zero coefficients.
    pub nz: u16,
    pub slice: u32,
    pub idc: u8,
    pub off_a: i8,
    pub off_b: i8,
    /// Inter macroblock without luma coefficients whose 16 blocks share one motion: its
    /// inner edges have boundary strength 0, and so does an edge to another such macroblock
    /// with the same motion.
    pub uni: bool,
}

impl Pic {
    pub fn new(wmb: usize, hmb: usize) -> Self {
        Pic {
            y: Plane::new(wmb * 16, hmb * 16),
            u: Plane::new(wmb * 8, hmb * 8),
            v: Plane::new(wmb * 8, hmb * 8),
            mf: vec![Motion { refi: [-1, -1], ..Default::default() }; wmb * hmb * 16],
            w4: wmb * 4,
            db: vec![DbInfo::default(); wmb * hmb],
            ..Default::default()
        }
    }
}

#[inline]
pub fn clip1(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

// --- dequantisation and inverse transforms ------------------------------------------------

const NORM4: [[i32; 3]; 6] = [[10, 16, 13], [11, 18, 14], [13, 20, 16], [14, 23, 18], [16, 25, 20], [18, 29, 23]];
const NORM8: [[i32; 6]; 6] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];

/// Raster position (y * 4 + x) of each 4x4 frame zig-zag scan position.
pub const ZZ4: [u8; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
/// Raster position (y * 8 + x) of each 8x8 frame zig-zag scan position.
pub const ZZ8: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35,
    42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// LevelScale4x4 (flat scaling matrix) at raster position `r`.
#[inline]
pub fn level_scale4(m: usize, r: usize) -> i32 {
    let (x, y) = (r & 3, r >> 2);
    let c = if x % 2 == 0 && y % 2 == 0 {
        0
    } else if x % 2 == 1 && y % 2 == 1 {
        1
    } else {
        2
    };
    16 * NORM4[m][c]
}

#[inline]
pub fn level_scale8(m: usize, r: usize) -> i32 {
    let (x, y) = (r & 7, r >> 3);
    let c = if x % 4 == 0 && y % 4 == 0 {
        0
    } else if x % 2 == 1 && y % 2 == 1 {
        1
    } else if x % 4 == 2 && y % 4 == 2 {
        2
    } else if (x % 4 == 0 && y % 2 == 1) || (x % 2 == 1 && y % 4 == 0) {
        3
    } else if (x % 4 == 0 && y % 4 == 2) || (x % 4 == 2 && y % 4 == 0) {
        4
    } else {
        5
    };
    16 * NORM8[m][c]
}

/// Dequantises a 4x4 block given in scan order (`dc_given`: position 0 already holds the
/// scaled DC, as for Intra16x16 and chroma blocks) into raster order.
pub fn dequant4(levels: &[i32; 16], qp: i32, dc_given: Option<i32>, out: &mut [i32; 16]) {
    let (m, e) = ((qp % 6) as usize, qp / 6);
    for i in 0..16 {
        let r = ZZ4[i] as usize;
        let c = levels[i];
        let v = c.wrapping_mul(level_scale4(m, r));
        out[r] = if c == 0 {
            0
        } else if e >= 4 {
            v.wrapping_shl((e - 4) as u32)
        } else {
            v.wrapping_add(1 << (3 - e)) >> (4 - e)
        };
    }
    if let Some(dc) = dc_given {
        out[0] = dc;
    }
}

pub fn dequant8(levels: &[i32; 64], qp: i32, out: &mut [i32; 64]) {
    let (m, e) = ((qp % 6) as usize, qp / 6);
    for i in 0..64 {
        let r = ZZ8[i] as usize;
        let c = levels[i];
        let v = c.wrapping_mul(level_scale8(m, r));
        out[r] = if c == 0 {
            0
        } else if e >= 6 {
            v.wrapping_shl((e - 6) as u32)
        } else {
            v.wrapping_add(1 << (5 - e)) >> (6 - e)
        };
    }
}

/// Largest dequantised coefficient magnitude the transforms accept (conforming streams use
/// 16-bit values; larger ones only come from corrupt input and are clamped, so the sums
/// below cannot overflow).
const DMAX: i32 = 1 << 20;

/// Inverse 4x4 transform of dequantised coefficients (raster order) into residuals.
pub fn idct4(d: &[i32; 16], out: &mut [i32; 16]) {
    let d = &d.map(|v| v.clamp(-DMAX, DMAX));
    let mut t = [0i32; 16];
    for y in 0..4 {
        let r = &d[y * 4..y * 4 + 4];
        let e0 = r[0] + r[2];
        let e1 = r[0] - r[2];
        let e2 = (r[1] >> 1) - r[3];
        let e3 = r[1] + (r[3] >> 1);
        t[y * 4] = e0 + e3;
        t[y * 4 + 1] = e1 + e2;
        t[y * 4 + 2] = e1 - e2;
        t[y * 4 + 3] = e0 - e3;
    }
    for x in 0..4 {
        let (a, b, c, dd) = (t[x], t[4 + x], t[8 + x], t[12 + x]);
        let e0 = a + c;
        let e1 = a - c;
        let e2 = (b >> 1) - dd;
        let e3 = b + (dd >> 1);
        out[x] = (e0 + e3 + 32) >> 6;
        out[4 + x] = (e1 + e2 + 32) >> 6;
        out[8 + x] = (e1 - e2 + 32) >> 6;
        out[12 + x] = (e0 - e3 + 32) >> 6;
    }
}

fn idct8_1d(d: [i32; 8]) -> [i32; 8] {
    let e0 = d[0] + d[4];
    let e1 = -d[3] + d[5] - d[7] - (d[7] >> 1);
    let e2 = d[0] - d[4];
    let e3 = d[1] + d[7] - d[3] - (d[3] >> 1);
    let e4 = (d[2] >> 1) - d[6];
    let e5 = -d[1] + d[7] + d[5] + (d[5] >> 1);
    let e6 = d[2] + (d[6] >> 1);
    let e7 = d[3] + d[5] + d[1] + (d[1] >> 1);
    let f0 = e0 + e6;
    let f1 = e1 + (e7 >> 2);
    let f2 = e2 + e4;
    let f3 = e3 + (e5 >> 2);
    let f4 = e2 - e4;
    let f5 = (e3 >> 2) - e5;
    let f6 = e0 - e6;
    let f7 = e7 - (e1 >> 2);
    [f0 + f7, f2 + f5, f4 + f3, f6 + f1, f6 - f1, f4 - f3, f2 - f5, f0 - f7]
}

pub fn idct8(d: &[i32; 64], out: &mut [i32; 64]) {
    let d = &d.map(|v| v.clamp(-DMAX, DMAX));
    let mut t = [0i32; 64];
    for y in 0..8 {
        let mut r = [0i32; 8];
        r.copy_from_slice(&d[y * 8..y * 8 + 8]);
        t[y * 8..y * 8 + 8].copy_from_slice(&idct8_1d(r));
    }
    for x in 0..8 {
        let mut c = [0i32; 8];
        for y in 0..8 {
            c[y] = t[y * 8 + x];
        }
        let g = idct8_1d(c);
        for y in 0..8 {
            out[y * 8 + x] = (g[y] + 32) >> 6;
        }
    }
}

/// Intra16x16 luma DC: inverse Hadamard and scaling of the 16 DC levels (scan order in,
/// raster order of 4x4 blocks out: index y * 4 + x).
pub fn luma_dc(levels: &[i32; 16], qp: i32, out: &mut [i32; 16]) {
    let mut c = [0i32; 16];
    for i in 0..16 {
        c[ZZ4[i] as usize] = levels[i];
    }
    let mut t = [0i32; 16];
    for y in 0..4 {
        let r = &c[y * 4..y * 4 + 4];
        t[y * 4] = r[0] + r[1] + r[2] + r[3];
        t[y * 4 + 1] = r[0] + r[1] - r[2] - r[3];
        t[y * 4 + 2] = r[0] - r[1] - r[2] + r[3];
        t[y * 4 + 3] = r[0] - r[1] + r[2] - r[3];
    }
    let mut f = [0i32; 16];
    for x in 0..4 {
        let (a, b, cc, d) = (t[x], t[4 + x], t[8 + x], t[12 + x]);
        f[x] = a + b + cc + d;
        f[4 + x] = a + b - cc - d;
        f[8 + x] = a - b - cc + d;
        f[12 + x] = a - b + cc - d;
    }
    let (m, e) = ((qp % 6) as usize, qp / 6);
    let ls = level_scale4(m, 0);
    for i in 0..16 {
        let v = f[i].wrapping_mul(ls);
        out[i] = if e >= 6 { v.wrapping_shl((e - 6) as u32) } else { v.wrapping_add(1 << (5 - e)) >> (6 - e) };
    }
}

/// Chroma DC (4:2:0): 2x2 inverse transform and scaling.
pub fn chroma_dc(levels: &[i32; 4], qpc: i32, out: &mut [i32; 4]) {
    let c = levels;
    let f = [c[0] + c[1] + c[2] + c[3], c[0] - c[1] + c[2] - c[3], c[0] + c[1] - c[2] - c[3], c[0] - c[1] - c[2] + c[3]];
    let ls = level_scale4((qpc % 6) as usize, 0);
    for i in 0..4 {
        out[i] = f[i].wrapping_mul(ls).wrapping_shl((qpc / 6) as u32) >> 5;
    }
}

const QPC: [i32; 22] = [29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39];

/// QP'c for 8-bit video.
pub fn chroma_qp(qpy: i32, offset: i32) -> i32 {
    let qpi = (qpy + offset).clamp(0, 51);
    if qpi < 30 {
        qpi
    } else {
        QPC[(qpi - 30) as usize]
    }
}

// --- forward transform (model features) --------------------------------------------------

/// Forward 4x4 core transform of a 4x4 sample block (raster in, raster out).
pub fn fdct4(s: &[i32; 16], out: &mut [i32; 16]) {
    let mut t = [0i32; 16];
    for y in 0..4 {
        let r = &s[y * 4..y * 4 + 4];
        let (s03, d03, s12, d12) = (r[0] + r[3], r[0] - r[3], r[1] + r[2], r[1] - r[2]);
        t[y * 4] = s03 + s12;
        t[y * 4 + 1] = 2 * d03 + d12;
        t[y * 4 + 2] = s03 - s12;
        t[y * 4 + 3] = d03 - 2 * d12;
    }
    for x in 0..4 {
        let (a, b, c, d) = (t[x], t[4 + x], t[8 + x], t[12 + x]);
        let (s03, d03, s12, d12) = (a + d, a - d, b + c, b - c);
        out[x] = s03 + s12;
        out[4 + x] = 2 * d03 + d12;
        out[8 + x] = s03 - s12;
        out[12 + x] = d03 - 2 * d12;
    }
}

/// 8 x the 8x8 inverse transform basis: C8[x][u] (sample x, frequency u). Its columns
/// are orthogonal with squared norms N8.
pub const C8: [[i32; 8]; 8] = [
    [8, 12, 8, 10, 8, 6, 4, 3],
    [8, 10, 4, -3, -8, -12, -8, -6],
    [8, 6, -4, -12, -8, 3, 8, 10],
    [8, 3, -8, -6, 8, 10, -4, -12],
    [8, -3, -8, 6, 8, -10, -4, 12],
    [8, -6, -4, 12, -8, -3, 8, -10],
    [8, -10, 4, 3, -8, 12, -8, 6],
    [8, -12, 8, -10, 8, -6, 4, -3],
];
pub const N8: [i64; 8] = [512, 578, 320, 578, 512, 578, 320, 578];

/// Dequantisation step of 8x8 position `r` (x64, so low QPs keep their precision).
#[inline]
pub fn step8_x64(qp: i32, r: usize) -> i64 {
    let (m, e) = ((qp % 6) as usize, qp / 6);
    (level_scale8(m, r) as i64) << e
}

/// Magnitude of an 8x8 sample block in quantiser steps, x16 (raster order).
pub fn qmag8(s: &[i32; 64], qp: i32, out: &mut [u32; 64]) {
    // Separable projection on the basis: rows first.
    let mut t = [0i64; 64];
    for y in 0..8 {
        for u in 0..8 {
            let mut a = 0i64;
            for x in 0..8 {
                a += s[y * 8 + x] as i64 * C8[x][u] as i64;
            }
            t[y * 8 + u] = a;
        }
    }
    for v in 0..8 {
        for u in 0..8 {
            let mut a = 0i64;
            for y in 0..8 {
                a += t[y * 8 + u] * C8[y][v] as i64;
            }
            // d = 4096 * proj / (N8[u] N8[v]); level = d / step = d * 64 / step_x64.
            let r = v * 8 + u;
            let num = a.unsigned_abs() as u128 * 4096 * 64 * 16;
            let den = (N8[u] * N8[v]) as u128 * step8_x64(qp, r) as u128;
            out[r] = (num / den.max(1)).min(u32::MAX as u128) as u32;
        }
    }
}

const MF4: [[i32; 3]; 6] =
    [[13107, 5243, 8066], [11916, 4660, 7490], [10082, 4194, 6554], [9362, 3647, 5825], [8192, 3355, 5243], [7282, 2893, 4559]];

/// Quantisation multiplier of the forward 4x4 transform at raster position `r`.
#[inline]
pub fn mf4(m: usize, r: usize) -> i32 {
    let (x, y) = (r & 3, r >> 2);
    let c = if x % 2 == 0 && y % 2 == 0 {
        0
    } else if x % 2 == 1 && y % 2 == 1 {
        1
    } else {
        2
    };
    MF4[m][c]
}

/// Magnitude of a transformed 4x4 block in quantiser steps, x16 (raster order), as the
/// encoder's quantiser would see it.
pub fn qmag4(s: &[i32; 16], qp: i32, out: &mut [u32; 16]) {
    let mut w = [0i32; 16];
    fdct4(s, &mut w);
    let (m, e) = ((qp % 6) as usize, qp / 6);
    for r in 0..16 {
        out[r] = ((w[r].unsigned_abs() as u64 * mf4(m, r) as u64) >> (11 + e)) as u32;
    }
}

// --- intra prediction ------------------------------------------------------------------

/// Neighbouring samples of an intra block: `top[0..n*2]` (with the top-right part already
/// substituted when unavailable), `left[0..n]`, the corner, and their availability.
pub struct Edge {
    pub top: [i32; 32],
    pub left: [i32; 16],
    pub corner: i32,
    pub has_top: bool,
    pub has_left: bool,
    pub has_corner: bool,
}

/// Intra 4x4 prediction (`mode` 0..8) into `out` (raster 4x4).
pub fn pred4x4(e: &Edge, mode: u8, out: &mut [i32; 16]) {
    let t = |i: i32| -> i32 {
        if i < 0 {
            e.corner
        } else {
            e.top[i as usize]
        }
    };
    let l = |i: i32| -> i32 {
        if i < 0 {
            e.corner
        } else {
            e.left[i as usize]
        }
    };
    for y in 0..4i32 {
        for x in 0..4i32 {
            let v = match mode {
                0 => t(x),
                1 => l(y),
                2 => {
                    let st: i32 = (0..4).map(|i| e.top[i]).sum();
                    let sl: i32 = (0..4).map(|i| e.left[i]).sum();
                    match (e.has_top, e.has_left) {
                        (true, true) => (st + sl + 4) >> 3,
                        (false, true) => (sl + 2) >> 2,
                        (true, false) => (st + 2) >> 2,
                        _ => 128,
                    }
                }
                3 => {
                    if x == 3 && y == 3 {
                        (t(6) + 3 * t(7) + 2) >> 2
                    } else {
                        (t(x + y) + 2 * t(x + y + 1) + t(x + y + 2) + 2) >> 2
                    }
                }
                4 => {
                    if x > y {
                        (t(x - y - 2) + 2 * t(x - y - 1) + t(x - y) + 2) >> 2
                    } else if x < y {
                        (l(y - x - 2) + 2 * l(y - x - 1) + l(y - x) + 2) >> 2
                    } else {
                        (t(0) + 2 * e.corner + l(0) + 2) >> 2
                    }
                }
                5 => {
                    let z = 2 * x - y;
                    if z >= 0 && z & 1 == 0 {
                        (t(x - (y >> 1) - 1) + t(x - (y >> 1)) + 1) >> 1
                    } else if z >= 0 {
                        (t(x - (y >> 1) - 2) + 2 * t(x - (y >> 1) - 1) + t(x - (y >> 1)) + 2) >> 2
                    } else if z == -1 {
                        (l(0) + 2 * e.corner + t(0) + 2) >> 2
                    } else {
                        (l(y - 1) + 2 * l(y - 2) + l(y - 3) + 2) >> 2
                    }
                }
                6 => {
                    let z = 2 * y - x;
                    if z >= 0 && z & 1 == 0 {
                        (l(y - (x >> 1) - 1) + l(y - (x >> 1)) + 1) >> 1
                    } else if z >= 0 {
                        (l(y - (x >> 1) - 2) + 2 * l(y - (x >> 1) - 1) + l(y - (x >> 1)) + 2) >> 2
                    } else if z == -1 {
                        (l(0) + 2 * e.corner + t(0) + 2) >> 2
                    } else {
                        (t(x - 1) + 2 * t(x - 2) + t(x - 3) + 2) >> 2
                    }
                }
                7 => {
                    if y & 1 == 0 {
                        (t(x + (y >> 1)) + t(x + (y >> 1) + 1) + 1) >> 1
                    } else {
                        (t(x + (y >> 1)) + 2 * t(x + (y >> 1) + 1) + t(x + (y >> 1) + 2) + 2) >> 2
                    }
                }
                _ => {
                    let z = x + 2 * y;
                    if z < 5 && z & 1 == 0 {
                        (l(y + (x >> 1)) + l(y + (x >> 1) + 1) + 1) >> 1
                    } else if z < 5 {
                        (l(y + (x >> 1)) + 2 * l(y + (x >> 1) + 1) + l(y + (x >> 1) + 2) + 2) >> 2
                    } else if z == 5 {
                        (l(2) + 3 * l(3) + 2) >> 2
                    } else {
                        l(3)
                    }
                }
            };
            out[(y * 4 + x) as usize] = v;
        }
    }
}

/// Intra 8x8 prediction with reference sample filtering (`top` holds 16 samples).
pub fn pred8x8(e: &Edge, mode: u8, out: &mut [i32; 64]) {
    // Filtered references (8.3.2.2.1).
    let mut tp = [0i32; 16];
    let mut lp = [0i32; 8];
    let mut cp = e.corner;
    if e.has_top {
        let t = &e.top;
        tp[0] = if e.has_corner { (e.corner + 2 * t[0] + t[1] + 2) >> 2 } else { (3 * t[0] + t[1] + 2) >> 2 };
        for x in 1..15 {
            tp[x] = (t[x - 1] + 2 * t[x] + t[x + 1] + 2) >> 2;
        }
        tp[15] = (t[14] + 3 * t[15] + 2) >> 2;
    }
    if e.has_corner {
        cp = match (e.has_top, e.has_left) {
            (true, true) => (e.top[0] + 2 * e.corner + e.left[0] + 2) >> 2,
            (true, false) => (3 * e.corner + e.top[0] + 2) >> 2,
            (false, true) => (3 * e.corner + e.left[0] + 2) >> 2,
            _ => e.corner,
        };
    }
    if e.has_left {
        let l = &e.left;
        lp[0] = if e.has_corner { (e.corner + 2 * l[0] + l[1] + 2) >> 2 } else { (3 * l[0] + l[1] + 2) >> 2 };
        for y in 1..7 {
            lp[y] = (l[y - 1] + 2 * l[y] + l[y + 1] + 2) >> 2;
        }
        lp[7] = (l[6] + 3 * l[7] + 2) >> 2;
    }
    let t = |i: i32| -> i32 {
        if i < 0 {
            cp
        } else {
            tp[i.min(15) as usize]
        }
    };
    let l = |i: i32| -> i32 {
        if i < 0 {
            cp
        } else {
            lp[i.min(7) as usize]
        }
    };
    let st: i32 = tp[..8].iter().sum();
    let sl: i32 = lp.iter().sum();
    for y in 0..8i32 {
        for x in 0..8i32 {
            let v = match mode {
                0 => t(x),
                1 => l(y),
                2 => match (e.has_top, e.has_left) {
                    (true, true) => (st + sl + 8) >> 4,
                    (false, true) => (sl + 4) >> 3,
                    (true, false) => (st + 4) >> 3,
                    _ => 128,
                },
                3 => {
                    if x == 7 && y == 7 {
                        (t(14) + 3 * t(15) + 2) >> 2
                    } else {
                        (t(x + y) + 2 * t(x + y + 1) + t(x + y + 2) + 2) >> 2
                    }
                }
                4 => {
                    if x > y {
                        (t(x - y - 2) + 2 * t(x - y - 1) + t(x - y) + 2) >> 2
                    } else if x < y {
                        (l(y - x - 2) + 2 * l(y - x - 1) + l(y - x) + 2) >> 2
                    } else {
                        (t(0) + 2 * cp + l(0) + 2) >> 2
                    }
                }
                5 => {
                    let z = 2 * x - y;
                    if z >= 0 && z & 1 == 0 {
                        (t(x - (y >> 1) - 1) + t(x - (y >> 1)) + 1) >> 1
                    } else if z >= 0 {
                        (t(x - (y >> 1) - 2) + 2 * t(x - (y >> 1) - 1) + t(x - (y >> 1)) + 2) >> 2
                    } else if z == -1 {
                        (l(0) + 2 * cp + t(0) + 2) >> 2
                    } else {
                        (l(y - 2 * x - 1) + 2 * l(y - 2 * x - 2) + l(y - 2 * x - 3) + 2) >> 2
                    }
                }
                6 => {
                    let z = 2 * y - x;
                    if z >= 0 && z & 1 == 0 {
                        (l(y - (x >> 1) - 1) + l(y - (x >> 1)) + 1) >> 1
                    } else if z >= 0 {
                        (l(y - (x >> 1) - 2) + 2 * l(y - (x >> 1) - 1) + l(y - (x >> 1)) + 2) >> 2
                    } else if z == -1 {
                        (l(0) + 2 * cp + t(0) + 2) >> 2
                    } else {
                        (t(x - 2 * y - 1) + 2 * t(x - 2 * y - 2) + t(x - 2 * y - 3) + 2) >> 2
                    }
                }
                7 => {
                    if y & 1 == 0 {
                        (t(x + (y >> 1)) + t(x + (y >> 1) + 1) + 1) >> 1
                    } else {
                        (t(x + (y >> 1)) + 2 * t(x + (y >> 1) + 1) + t(x + (y >> 1) + 2) + 2) >> 2
                    }
                }
                _ => {
                    let z = x + 2 * y;
                    if z < 13 && z & 1 == 0 {
                        (l(y + (x >> 1)) + l(y + (x >> 1) + 1) + 1) >> 1
                    } else if z < 13 {
                        (l(y + (x >> 1)) + 2 * l(y + (x >> 1) + 1) + l(y + (x >> 1) + 2) + 2) >> 2
                    } else if z == 13 {
                        (l(6) + 3 * l(7) + 2) >> 2
                    } else {
                        l(7)
                    }
                }
            };
            out[(y * 8 + x) as usize] = v;
        }
    }
}

/// Intra 16x16 prediction (`mode` 0..3) into `out` (raster 16x16).
pub fn pred16x16(e: &Edge, mode: u32, out: &mut [i32; 256]) {
    match mode {
        0 => {
            for y in 0..16 {
                for x in 0..16 {
                    out[y * 16 + x] = e.top[x];
                }
            }
        }
        1 => {
            for y in 0..16 {
                for x in 0..16 {
                    out[y * 16 + x] = e.left[y];
                }
            }
        }
        2 => {
            let st: i32 = e.top[..16].iter().sum();
            let sl: i32 = e.left.iter().sum();
            let v = match (e.has_top, e.has_left) {
                (true, true) => (st + sl + 16) >> 5,
                (false, true) => (sl + 8) >> 4,
                (true, false) => (st + 8) >> 4,
                _ => 128,
            };
            out.iter_mut().for_each(|o| *o = v);
        }
        _ => {
            let t = |i: i32| if i < 0 { e.corner } else { e.top[i as usize] };
            let l = |i: i32| if i < 0 { e.corner } else { e.left[i as usize] };
            let mut hh = 0;
            let mut vv = 0;
            for k in 0..8 {
                hh += (k + 1) * (t(8 + k) - t(6 - k));
                vv += (k + 1) * (l(8 + k) - l(6 - k));
            }
            let a = 16 * (l(15) + t(15));
            let b = (5 * hh + 32) >> 6;
            let c = (5 * vv + 32) >> 6;
            for y in 0..16i32 {
                for x in 0..16i32 {
                    out[(y * 16 + x) as usize] = clip1((a + b * (x - 7) + c * (y - 7) + 16) >> 5) as i32;
                }
            }
        }
    }
}

/// Intra chroma prediction of one 8x8 (4:2:0) component; `mode` 0 DC, 1 horizontal, 2
/// vertical, 3 plane.
pub fn pred_chroma(e: &Edge, mode: u32, out: &mut [i32; 64]) {
    match mode {
        0 => {
            for by in 0..2 {
                for bx in 0..2 {
                    let st: i32 = e.top[bx * 4..bx * 4 + 4].iter().sum();
                    let sl: i32 = e.left[by * 4..by * 4 + 4].iter().sum();
                    let (ht, hl) = (e.has_top, e.has_left);
                    let v = if (bx, by) == (0, 0) || (bx, by) == (1, 1) {
                        match (ht, hl) {
                            (true, true) => (st + sl + 4) >> 3,
                            (false, true) => (sl + 2) >> 2,
                            (true, false) => (st + 2) >> 2,
                            _ => 128,
                        }
                    } else if (bx, by) == (1, 0) {
                        if ht {
                            (st + 2) >> 2
                        } else if hl {
                            (sl + 2) >> 2
                        } else {
                            128
                        }
                    } else if hl {
                        (sl + 2) >> 2
                    } else if ht {
                        (st + 2) >> 2
                    } else {
                        128
                    };
                    for y in 0..4 {
                        for x in 0..4 {
                            out[(by * 4 + y) * 8 + bx * 4 + x] = v;
                        }
                    }
                }
            }
        }
        1 => {
            for y in 0..8 {
                for x in 0..8 {
                    out[y * 8 + x] = e.left[y];
                }
            }
        }
        2 => {
            for y in 0..8 {
                for x in 0..8 {
                    out[y * 8 + x] = e.top[x];
                }
            }
        }
        _ => {
            let t = |i: i32| if i < 0 { e.corner } else { e.top[i as usize] };
            let l = |i: i32| if i < 0 { e.corner } else { e.left[i as usize] };
            let mut hh = 0;
            let mut vv = 0;
            for k in 0..4 {
                hh += (k + 1) * (t(4 + k) - t(2 - k));
                vv += (k + 1) * (l(4 + k) - l(2 - k));
            }
            let a = 16 * (l(7) + t(7));
            let b = (34 * hh + 32) >> 6;
            let c = (34 * vv + 32) >> 6;
            for y in 0..8i32 {
                for x in 0..8i32 {
                    out[(y * 8 + x) as usize] = clip1((a + b * (x - 3) + c * (y - 3) + 16) >> 5) as i32;
                }
            }
        }
    }
}

// --- motion compensation -------------------------------------------------------------

#[inline]
fn tap6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    a - 5 * b + 20 * c + 20 * d - 5 * e + f
}

/// Scratch space of [`mc_luma`] (reused, so no call has to clear it).
pub struct McScratch {
    win: [i32; 21 * 21],
    b1: [i32; 21 * 16],
    bh: [i32; 17 * 16],
    hv: [i32; 16 * 17],
    jv: [i32; 16 * 16],
}

impl Default for McScratch {
    fn default() -> Self {
        McScratch { win: [0; 21 * 21], b1: [0; 21 * 16], bh: [0; 17 * 16], hv: [0; 16 * 17], jv: [0; 16 * 16] }
    }
}

/// Luma prediction of a `w`x`h` block (at most 16x16) at (`x`, `y`) with quarter-sample
/// motion (`mvx`, `mvy`), written to `out` with stride `ostride`.
#[allow(clippy::too_many_arguments)]
pub fn mc_luma(sc: &mut McScratch, p: &Plane, x: i32, y: i32, mvx: i32, mvy: i32, w: usize, h: usize, out: &mut [i32], ostride: usize) {
    const WS: usize = 21;
    let (fx, fy) = (mvx & 3, mvy & 3);
    let x0 = x + (mvx >> 2) - 2;
    let y0 = y + (mvy >> 2) - 2;
    let inside = x0 >= 0 && y0 >= 0 && x0 as usize + w + 5 <= p.w && y0 as usize + h + 5 <= p.h;
    if fx == 0 && fy == 0 {
        for j in 0..h {
            for i in 0..w {
                out[j * ostride + i] = if inside {
                    p.d[(y0 as usize + 2 + j) * p.w + x0 as usize + 2 + i] as i32
                } else {
                    p.at(x0 + 2 + i as i32, y0 + 2 + j as i32) as i32
                };
            }
        }
        return;
    }
    let McScratch { win, b1, bh, hv, jv } = sc;
    for j in 0..h + 5 {
        if inside {
            let row = &p.d[(y0 as usize + j) * p.w + x0 as usize..][..w + 5];
            for (i, &v) in row.iter().enumerate() {
                win[j * WS + i] = v as i32;
            }
        } else {
            for i in 0..w + 5 {
                win[j * WS + i] = p.at(x0 + i as i32, y0 + j as i32) as i32;
            }
        }
    }
    let g = |i: usize, j: usize| win[(j + 2) * WS + i + 2];
    let need_b = fx != 0 && fy != 2;
    let need_h = fy != 0 && fx != 2;
    let need_j = (fx == 2 && fy != 0) || (fy == 2 && fx != 0);
    // Unclipped horizontal half samples of every window row (columns 0..w), for j.
    if need_j || need_b {
        let rows = if need_j { 0..h + 5 } else { 2..h + 4 };
        for jj in rows {
            for i in 0..w {
                let r = &win[jj * WS + i..jj * WS + i + 6];
                b1[jj * 16 + i] = tap6(r[0], r[1], r[2], r[3], r[4], r[5]);
            }
        }
    }
    // b (rows 0..=h, for s) and h (columns 0..=w, for m), clipped.
    if need_b {
        for j in 0..=h.min(15) {
            if j + 2 >= h + 5 {
                break;
            }
            for i in 0..w {
                bh[j * 16 + i] = clip1((b1[(j + 2) * 16 + i] + 16) >> 5) as i32;
            }
        }
    }
    if need_h {
        for j in 0..h {
            for i in 0..=w {
                let c = i + 2;
                let v = tap6(win[j * WS + c], win[(j + 1) * WS + c], win[(j + 2) * WS + c], win[(j + 3) * WS + c], win[(j + 4) * WS + c], win[(j + 5) * WS + c]);
                hv[j * 17 + i] = clip1((v + 16) >> 5) as i32;
            }
        }
    }
    if need_j {
        for j in 0..h {
            for i in 0..w {
                let v = tap6(b1[j * 16 + i], b1[(j + 1) * 16 + i], b1[(j + 2) * 16 + i], b1[(j + 3) * 16 + i], b1[(j + 4) * 16 + i], b1[(j + 5) * 16 + i]);
                jv[j * 16 + i] = clip1((v + 512) >> 10) as i32;
            }
        }
    }
    let bget = |i: usize, j: usize| -> i32 {
        if j < 16 {
            bh[j * 16 + i]
        } else {
            // Row 16 (s below a 16-row block): computed on demand.
            let jj = j + 2;
            let r = &win[jj * WS + i..jj * WS + i + 6];
            clip1((tap6(r[0], r[1], r[2], r[3], r[4], r[5]) + 16) >> 5) as i32
        }
    };
    let avg = |a: i32, b: i32| (a + b + 1) >> 1;
    // One loop per sub-sample position, so no decision is made per sample.
    #[inline(always)]
    fn fill(out: &mut [i32], ostride: usize, w: usize, h: usize, f: impl Fn(usize, usize) -> i32) {
        for j in 0..h {
            let row = &mut out[j * ostride..j * ostride + w];
            for (i, o) in row.iter_mut().enumerate() {
                *o = f(i, j);
            }
        }
    }
    let (bh, hv, jv) = (&*bh, &*hv, &*jv);
    match (fx, fy) {
        (0, 1) => fill(out, ostride, w, h, |i, j| avg(g(i, j), hv[j * 17 + i])),
        (0, 2) => fill(out, ostride, w, h, |i, j| hv[j * 17 + i]),
        (0, 3) => fill(out, ostride, w, h, |i, j| avg(g(i, j + 1), hv[j * 17 + i])),
        (1, 0) => fill(out, ostride, w, h, |i, j| avg(g(i, j), bh[j * 16 + i])),
        (2, 0) => fill(out, ostride, w, h, |i, j| bh[j * 16 + i]),
        (3, 0) => fill(out, ostride, w, h, |i, j| avg(g(i + 1, j), bh[j * 16 + i])),
        (1, 1) => fill(out, ostride, w, h, |i, j| avg(bh[j * 16 + i], hv[j * 17 + i])),
        (3, 1) => fill(out, ostride, w, h, |i, j| avg(bh[j * 16 + i], hv[j * 17 + i + 1])),
        (1, 3) => fill(out, ostride, w, h, |i, j| avg(hv[j * 17 + i], bget(i, j + 1))),
        (3, 3) => fill(out, ostride, w, h, |i, j| avg(hv[j * 17 + i + 1], bget(i, j + 1))),
        (2, 2) => fill(out, ostride, w, h, |i, j| jv[j * 16 + i]),
        (2, 1) => fill(out, ostride, w, h, |i, j| avg(bh[j * 16 + i], jv[j * 16 + i])),
        (2, 3) => fill(out, ostride, w, h, |i, j| avg(jv[j * 16 + i], bget(i, j + 1))),
        (1, 2) => fill(out, ostride, w, h, |i, j| avg(hv[j * 17 + i], jv[j * 16 + i])),
        _ => fill(out, ostride, w, h, |i, j| avg(jv[j * 16 + i], hv[j * 17 + i + 1])), // (3, 2)
    }
}

/// The direct (slow) form of [`mc_luma`], kept as the test oracle.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn mc_luma_ref(p: &Plane, x: i32, y: i32, mvx: i32, mvy: i32, w: usize, h: usize, out: &mut [i32], ostride: usize) {
    let (fx, fy) = (mvx & 3, mvy & 3);
    let x0 = x + (mvx >> 2) - 2;
    let y0 = y + (mvy >> 2) - 2;
    // Window of (w + 5) x (h + 5) samples with edge clamping.
    const WS: usize = 21;
    let mut win = [0i32; WS * WS];
    let inside = x0 >= 0 && y0 >= 0 && x0 as usize + w + 5 <= p.w && y0 as usize + h + 5 <= p.h;
    for j in 0..h + 5 {
        for i in 0..w + 5 {
            win[j * WS + i] = if inside {
                p.d[(y0 as usize + j) * p.w + x0 as usize + i] as i32
            } else {
                p.at(x0 + i as i32, y0 + j as i32) as i32
            };
        }
    }
    let g = |i: usize, j: usize| win[(j + 2) * WS + i + 2];
    // Unclipped horizontal half sample between G(i, j) and G(i + 1, j) (j may be -2..h+2).
    let b1 = |i: usize, jj: usize| {
        let r = &win[jj * WS + i..jj * WS + i + 6];
        tap6(r[0], r[1], r[2], r[3], r[4], r[5])
    };
    let h1 = |ii: usize, j: usize| {
        tap6(win[j * WS + ii], win[(j + 1) * WS + ii], win[(j + 2) * WS + ii], win[(j + 3) * WS + ii], win[(j + 4) * WS + ii], win[(j + 5) * WS + ii])
    };
    let bh = |i: usize, j: usize| clip1((b1(i, j + 2) + 16) >> 5) as i32; // b at (i, j)
    let hv = |i: usize, j: usize| clip1((h1(i + 2, j) + 16) >> 5) as i32; // h at (i, j)
    let jc = |i: usize, j: usize| {
        let v = tap6(b1(i, j), b1(i, j + 1), b1(i, j + 2), b1(i, j + 3), b1(i, j + 4), b1(i, j + 5));
        clip1((v + 512) >> 10) as i32
    };
    let avg = |a: i32, b: i32| (a + b + 1) >> 1;
    for j in 0..h {
        for i in 0..w {
            let v = match (fx, fy) {
                (0, 0) => g(i, j),
                (0, 1) => avg(g(i, j), hv(i, j)),
                (0, 2) => hv(i, j),
                (0, 3) => avg(g(i, j + 1), hv(i, j)),
                (1, 0) => avg(g(i, j), bh(i, j)),
                (2, 0) => bh(i, j),
                (3, 0) => avg(g(i + 1, j), bh(i, j)),
                (1, 1) => avg(bh(i, j), hv(i, j)),
                (3, 1) => avg(bh(i, j), hv(i + 1, j)),
                (1, 3) => avg(hv(i, j), bh(i, j + 1)),
                (3, 3) => avg(hv(i + 1, j), bh(i, j + 1)),
                (2, 2) => jc(i, j),
                (2, 1) => avg(bh(i, j), jc(i, j)),
                (2, 3) => avg(jc(i, j), bh(i, j + 1)),
                (1, 2) => avg(hv(i, j), jc(i, j)),
                _ => avg(jc(i, j), hv(i + 1, j)), // (3, 2)
            };
            out[j * ostride + i] = v;
        }
    }
}

/// Chroma (4:2:0) prediction of a `w`x`h` block at chroma position (`x`, `y`) with the luma
/// motion vector (eighth-sample units for chroma).
#[allow(clippy::too_many_arguments)]
pub fn mc_chroma(p: &Plane, x: i32, y: i32, mvx: i32, mvy: i32, w: usize, h: usize, out: &mut [i32], ostride: usize) {
    let (fx, fy) = (mvx & 7, mvy & 7);
    let x0 = x + (mvx >> 3);
    let y0 = y + (mvy >> 3);
    let inside = x0 >= 0 && y0 >= 0 && x0 as usize + w < p.w && y0 as usize + h < p.h;
    if inside && fx == 0 && fy == 0 {
        for j in 0..h {
            let o = (y0 as usize + j) * p.w + x0 as usize;
            for (v, &s) in out[j * ostride..j * ostride + w].iter_mut().zip(&p.d[o..o + w]) {
                *v = s as i32;
            }
        }
        return;
    }
    if inside {
        let (wa, wb, wc, wd) = ((8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy);
        for j in 0..h {
            let o = (y0 as usize + j) * p.w + x0 as usize;
            let (r0, r1) = (&p.d[o..o + w + 1], &p.d[o + p.w..o + p.w + w + 1]);
            let row = &mut out[j * ostride..j * ostride + w];
            for (i, v) in row.iter_mut().enumerate() {
                *v = (wa * r0[i] as i32 + wb * r0[i + 1] as i32 + wc * r1[i] as i32 + wd * r1[i + 1] as i32 + 32) >> 6;
            }
        }
        return;
    }
    for j in 0..h {
        for i in 0..w {
            let (xx, yy) = (x0 + i as i32, y0 + j as i32);
            let (a, b, c, d) = if inside {
                let o = yy as usize * p.w + xx as usize;
                (p.d[o] as i32, p.d[o + 1] as i32, p.d[o + p.w] as i32, p.d[o + p.w + 1] as i32)
            } else {
                (p.at(xx, yy) as i32, p.at(xx + 1, yy) as i32, p.at(xx, yy + 1) as i32, p.at(xx + 1, yy + 1) as i32)
            };
            out[j * ostride + i] = ((8 - fx) * (8 - fy) * a + fx * (8 - fy) * b + (8 - fx) * fy * c + fx * fy * d + 32) >> 6;
        }
    }
}

// --- reference picture management ------------------------------------------------------

/// A reference list modification command (modification_of_pic_nums_idc, value).
pub type ListMod = (u32, u32);

/// A memory management control operation (op, value 1, value 2).
pub type Mmco = (u32, u32, u32);

/// Slice header fields reconstruction needs beyond what parsing needs.
#[derive(Clone, Debug, Default)]
pub struct RefInfo {
    /// Explicit weighted prediction: log2 denominators (luma, chroma) and per list and
    /// reference index the (weight, offset) of Y, Cb and Cr.
    pub explicit_wp: bool,
    pub wp_denom: [u32; 2],
    pub wp: [Vec<[(i32, i32); 3]>; 2],
    /// disable_deblocking_filter_idc and FilterOffsetA/B.
    pub dbk_idc: u32,
    pub dbk_a: i32,
    pub dbk_b: i32,
    pub poc_lsb: u32,
    pub delta_poc_bottom: i32,
    pub mods: [Vec<ListMod>; 2],
    pub long_term_reference: bool,
    pub adaptive: bool,
    pub mmco: Vec<Mmco>,
}

/// Decoded pictures of one segment.
#[derive(Default)]
pub struct Recon {
    /// The picture being decoded.
    pub cur: Pic,
    pub cur_ref: bool,
    pub cur_idr: bool,
    pub cur_mmco: Vec<Mmco>,
    pub cur_adaptive: bool,
    pub cur_lt: bool,
    pub have_cur: bool,
    /// Reference pictures.
    pub dpb: Vec<Pic>,
    /// Reference lists of the current slice (indices into `dpb`).
    pub list: [Vec<usize>; 2],
    prev_poc_msb: i32,
    prev_poc_lsb: i32,
    prev_frame_num: u32,
    prev_frame_num_offset: i32,
    max_frame_num: u32,
    max_refs: usize,
    spare: Vec<Pic>,
    /// Pictures finished so far (debug dumps).
    pub count: u32,
    /// Macroblocks of the current picture rebuilt by kept (recoded) slices.
    pub covered: Vec<bool>,
}

/// Debug aid (feature `recon-debug`): appends "seq poc hash" of every finished picture to the
/// file PPM_HASH names
/// (FNV-1a over the top PPM_HASH_H luma rows and half as many chroma rows).
#[cfg(feature = "recon-debug")]
fn hash_line(p: &Pic, n: u32) {
    let Ok(path) = std::env::var("PPM_HASH") else { return };
    let h = std::env::var("PPM_HASH_H").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(p.y.h).min(p.y.h);
    // Position-weighted sums (cheap to compute with numpy on the other side).
    let (mut s1, mut s2, mut i) = (0u64, 0u64, 0u64);
    let mut eat = |b: &[u8]| {
        for &v in b {
            i += 1;
            s1 = s1.wrapping_add(i.wrapping_mul(v as u64));
            s2 = s2.wrapping_add((v as u64 * v as u64).wrapping_mul(i % 65521 + 1));
        }
    };
    let w = std::env::var("PPM_HASH_W").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(p.y.w).min(p.y.w);
    for y in 0..h {
        eat(&p.y.d[y * p.y.w..y * p.y.w + w]);
    }
    for pl in [&p.u, &p.v] {
        for y in 0..h / 2 {
            eat(&pl.d[y * pl.w..y * pl.w + w / 2]);
        }
    }
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{n} {} {s1:016x}{s2:016x}", p.poc);
    }
}

/// Debug aid (feature `recon-debug`): writes picture `n` as a PPM (or raw YUV with
/// PPM_DUMP_YUV) when PPM_DUMP=n,m,... is set.
#[cfg(feature = "recon-debug")]
fn dump(p: &Pic, n: u32) {
    let Ok(list) = std::env::var("PPM_DUMP") else { return };
    if !list.split(',').any(|v| v.trim().parse() == Ok(n)) {
        return;
    }
    let (w, h) = (p.y.w, p.y.h);
    let dir = std::env::var("PPM_DUMP_DIR").unwrap_or_else(|_| ".".into());
    if std::env::var_os("PPM_DUMP_YUV").is_some() {
        let mut out = p.y.d.clone();
        out.extend_from_slice(&p.u.d);
        out.extend_from_slice(&p.v.d);
        let _ = std::fs::write(format!("{dir}/pic{n:05}_p{}.yuv", p.poc), out);
        return;
    }
    let mut out = format!("P6
{w} {h}
255
").into_bytes();
    for y in 0..h {
        for x in 0..w {
            let yy = p.y.d[y * w + x] as i32 - 16;
            let u = p.u.d[(y / 2) * p.u.w + x / 2] as i32 - 128;
            let v = p.v.d[(y / 2) * p.v.w + x / 2] as i32 - 128;
            let r = (298 * yy + 459 * v + 128) >> 8;
            let g = (298 * yy - 55 * u - 136 * v + 128) >> 8;
            let b = (298 * yy + 541 * u + 128) >> 8;
            out.extend_from_slice(&[clip1(r), clip1(g), clip1(b)]);
        }
    }
    let _ = std::fs::write(format!("{dir}/pic{n:05}.ppm"), out);
}

pub struct SliceInfo<'a> {
    pub first_mb: u32,
    pub slice_type: u32,
    pub frame_num: u32,
    pub idr: bool,
    pub nal_ref_idc: u32,
    pub num_ref: [u32; 2],
    pub log2_max_frame_num: u32,
    pub poc_type: u32,
    pub log2_max_poc_lsb: u32,
    pub max_num_ref_frames: u32,
    pub wmb: usize,
    pub hmb: usize,
    pub ri: &'a RefInfo,
}

impl Recon {
    /// Prepares the picture and reference lists for a slice. Call for every slice (also
    /// slices that are not recoded), in stream order, on both sides.
    pub fn begin_slice(&mut self, s: &SliceInfo) {
        let new_pic = s.first_mb == 0 || !self.have_cur;
        if new_pic {
            self.finish_picture();
            self.max_frame_num = 1 << s.log2_max_frame_num;
            self.max_refs = (s.max_num_ref_frames as usize).clamp(1, 16);
            if s.idr {
                self.dpb.clear();
                self.prev_poc_msb = 0;
                self.prev_poc_lsb = 0;
                self.prev_frame_num_offset = 0;
            }
            let poc = match s.poc_type {
                0 => {
                    let max_lsb = 1i32 << s.log2_max_poc_lsb;
                    let lsb = s.ri.poc_lsb as i32;
                    let msb = if lsb < self.prev_poc_lsb && self.prev_poc_lsb - lsb >= max_lsb / 2 {
                        self.prev_poc_msb.wrapping_add(max_lsb)
                    } else if lsb > self.prev_poc_lsb && lsb - self.prev_poc_lsb > max_lsb / 2 {
                        self.prev_poc_msb.wrapping_sub(max_lsb)
                    } else {
                        self.prev_poc_msb
                    };
                    if s.nal_ref_idc != 0 {
                        self.prev_poc_msb = msb;
                        self.prev_poc_lsb = lsb;
                    }
                    let top = msb.wrapping_add(lsb);
                    top.min(top.wrapping_add(s.ri.delta_poc_bottom))
                }
                _ => {
                    let off = if s.idr {
                        0
                    } else if self.prev_frame_num > s.frame_num {
                        self.prev_frame_num_offset.wrapping_add(self.max_frame_num as i32)
                    } else {
                        self.prev_frame_num_offset
                    };
                    self.prev_frame_num_offset = off;
                    if s.idr {
                        0
                    } else if s.nal_ref_idc == 0 {
                        off.wrapping_add(s.frame_num as i32).wrapping_mul(2).wrapping_sub(1)
                    } else {
                        off.wrapping_add(s.frame_num as i32).wrapping_mul(2)
                    }
                }
            };
            self.prev_frame_num = s.frame_num;
            let mut p = self.spare.pop().unwrap_or_default();
            if p.y.w != s.wmb * 16 || p.y.h != s.hmb * 16 {
                p = Pic::new(s.wmb, s.hmb);
            }
            p.poc = poc;
            p.frame_num = s.frame_num;
            p.long_term = false;
            p.db.iter_mut().for_each(|d| d.done = false);
            self.covered.clear();
            self.covered.resize(s.wmb * s.hmb, false);
            self.cur = p;
            self.have_cur = true;
            self.cur_ref = s.nal_ref_idc != 0;
            self.cur_idr = s.idr;
            self.cur_mmco = s.ri.mmco.clone();
            self.cur_adaptive = s.ri.adaptive;
            self.cur_lt = s.ri.long_term_reference;
        }
        // Reference lists.
        let fnum = s.frame_num as i32;
        let maxf = self.max_frame_num as i32;
        for p in self.dpb.iter_mut() {
            p.frame_num_wrap = if p.frame_num as i32 > fnum { p.frame_num as i32 - maxf } else { p.frame_num as i32 };
        }
        self.list = [Vec::new(), Vec::new()];
        if s.slice_type == super::I {
            return;
        }
        let mut st: Vec<usize> = (0..self.dpb.len()).filter(|&i| !self.dpb[i].long_term).collect();
        let mut lt: Vec<usize> = (0..self.dpb.len()).filter(|&i| self.dpb[i].long_term).collect();
        lt.sort_by_key(|&i| self.dpb[i].lt_idx);
        let poc = self.cur.poc;
        if s.slice_type == super::P {
            st.sort_by_key(|&i| -self.dpb[i].frame_num_wrap);
            let mut l = st;
            l.extend_from_slice(&lt);
            self.list[0] = l;
        } else {
            let mut before: Vec<usize> = st.iter().copied().filter(|&i| self.dpb[i].poc < poc).collect();
            let mut after: Vec<usize> = st.iter().copied().filter(|&i| self.dpb[i].poc > poc).collect();
            before.sort_by_key(|&i| -self.dpb[i].poc);
            after.sort_by_key(|&i| self.dpb[i].poc);
            let mut l0 = before.clone();
            l0.extend_from_slice(&after);
            l0.extend_from_slice(&lt);
            let mut l1 = after;
            l1.extend_from_slice(&before);
            l1.extend_from_slice(&lt);
            if l1.len() > 1 && l1 == l0 {
                l1.swap(0, 1);
            }
            self.list = [l0, l1];
        }
        let lists = if s.slice_type == super::B { 2 } else { 1 };
        for li in 0..lists {
            let n = s.num_ref[li] as usize;
            let mut l = std::mem::take(&mut self.list[li]);
            // Entries beyond the initial list are "no reference picture" (usize::MAX).
            l.resize(n, usize::MAX);
            // Modifications (8.2.4.3).
            let mut pred = fnum;
            for (k, &(idc, v)) in s.ri.mods[li].iter().enumerate() {
                if k >= n {
                    break;
                }
                let target = match idc {
                    0 | 1 => {
                        let d = v as i32 + 1;
                        let mut nw = if idc == 0 { pred - d } else { pred + d };
                        if nw < 0 {
                            nw += maxf;
                        } else if nw >= maxf {
                            nw -= maxf;
                        }
                        pred = nw;
                        let num = if nw > fnum { nw - maxf } else { nw };
                        self.dpb.iter().position(|p| !p.long_term && p.frame_num_wrap == num)
                    }
                    2 => self.dpb.iter().position(|p| p.long_term && p.lt_idx == v),
                    _ => None,
                };
                if let Some(t) = target {
                    // Insert at k, then drop later copies of the same picture.
                    l.insert(k, t);
                    let mut out: Vec<usize> = l[..=k].to_vec();
                    out.extend(l[k + 1..].iter().copied().filter(|&i| i != t));
                    out.resize(n, usize::MAX);
                    l = out;
                }
            }
            self.list[li] = l;
        }
    }

    /// Marks the finished picture and keeps it if it is a reference.
    pub fn finish_picture(&mut self) {
        if !self.have_cur {
            return;
        }
        self.have_cur = false;
        let mut p = std::mem::take(&mut self.cur);
        self.fill_uncovered(&mut p);
        if self.cur_ref || cfg!(feature = "recon-debug") {
            deblock(&mut p);
        }
        #[cfg(feature = "recon-debug")]
        {
            static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            dump(&p, seq);
            hash_line(&p, seq);
        }
        self.count += 1;
        if !self.cur_ref {
            self.spare.push(p);
            return;
        }
        let maxf = self.max_frame_num as i32;
        let fnum = p.frame_num as i32;
        if self.cur_idr {
            self.dpb.clear();
            if self.cur_lt {
                p.long_term = true;
                p.lt_idx = 0;
            }
        } else if self.cur_adaptive {
            for &(op, a, b) in &self.cur_mmco.clone() {
                match op {
                    1 => {
                        let num = fnum - (a as i32 + 1);
                        self.drop_where(|q| !q.long_term && q.frame_num_wrap == num);
                    }
                    2 => self.drop_where(|q| q.long_term && q.lt_idx == a),
                    3 => {
                        let num = fnum - (a as i32 + 1);
                        self.drop_where(|q| q.long_term && q.lt_idx == b);
                        if let Some(q) = self.dpb.iter_mut().find(|q| !q.long_term && q.frame_num_wrap == num) {
                            q.long_term = true;
                            q.lt_idx = b;
                        }
                    }
                    4 => self.drop_where(|q| q.long_term && q.lt_idx + 1 > a),
                    5 => {
                        self.dpb.clear();
                        self.prev_poc_msb = 0;
                        self.prev_poc_lsb = 0;
                        self.prev_frame_num_offset = 0;
                        self.prev_frame_num = 0;
                    }
                    6 => {
                        self.drop_where(|q| q.long_term && q.lt_idx == a);
                        p.long_term = true;
                        p.lt_idx = a;
                    }
                    _ => {}
                }
            }
        }
        if !p.long_term && !self.cur_adaptive && !self.cur_idr {
            // Sliding window.
            while self.dpb.len() >= self.max_refs {
                let Some(i) = (0..self.dpb.len()).filter(|&i| !self.dpb[i].long_term).min_by_key(|&i| {
                    let q = &self.dpb[i];
                    if q.frame_num as i32 > fnum {
                        q.frame_num as i32 - maxf
                    } else {
                        q.frame_num as i32
                    }
                }) else {
                    break;
                };
                let q = self.dpb.remove(i);
                self.spare.push(q);
            }
        }
        while self.dpb.len() >= 16 {
            let q = self.dpb.remove(0);
            self.spare.push(q);
        }
        self.dpb.push(p);
    }

    /// Gives the macroblocks no kept slice rebuilt (slices stored raw, which the
    /// decompressor never parses) the same content on both sides: the co-located samples of
    /// the latest reference picture, or grey.
    fn fill_uncovered(&self, p: &mut Pic) {
        let wmb = p.y.w / 16;
        let n = wmb * (p.y.h / 16);
        if self.covered.len() != n {
            return;
        }
        let src = self.dpb.last().filter(|q| q.y.w == p.y.w && q.y.h == p.y.h);
        for m in 0..n {
            if self.covered[m] {
                continue;
            }
            let (mx, my) = (m % wmb, m / wmb);
            for (dst, sp, sz) in [(&mut p.y, src.map(|q| &q.y), 16usize), (&mut p.u, src.map(|q| &q.u), 8), (&mut p.v, src.map(|q| &q.v), 8)] {
                let w = dst.w;
                for j in 0..sz {
                    let o = (my * sz + j) * w + mx * sz;
                    match sp {
                        Some(s) => dst.d[o..o + sz].copy_from_slice(&s.d[o..o + sz]),
                        None => dst.d[o..o + sz].fill(128),
                    }
                }
            }
            p.db[m].done = false;
            for k in 0..16 {
                p.mf[(my * 4 + k / 4) * p.w4 + mx * 4 + k % 4] = Motion { refi: [-1, -1], ..Default::default() };
            }
        }
    }

    fn drop_where(&mut self, f: impl Fn(&Pic) -> bool) {
        let mut i = 0;
        while i < self.dpb.len() {
            if f(&self.dpb[i]) {
                let q = self.dpb.remove(i);
                self.spare.push(q);
            } else {
                i += 1;
            }
        }
    }

    /// The reference picture `idx` of list `l` for the current slice.
    #[inline]
    pub fn refpic(&self, l: usize, idx: i32) -> Option<&Pic> {
        if idx < 0 {
            return None;
        }
        self.list[l].get(idx as usize).filter(|&&i| i != usize::MAX).map(|&i| &self.dpb[i])
    }
}

// --- deblocking filter (H.264 clause 8.7, frames) ---------------------------------------

/// Table 8-16 of Rec. ITU-T H.264: alpha' and beta' by indexA / indexB.
const ALPHA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20, 22, 25, 28, 32, 36, 40, 45, 50, 56,
    63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226, 255, 255,
];
const BETA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];
/// Table 8-17: tC0 by indexA for bS = 1, 2, 3.
const TC0: [[u8; 3]; 52] = [
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1],
    [0, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 2], [1, 1, 2], [1, 1, 2], [1, 1, 2], [1, 2, 3], [1, 2, 3],
    [2, 2, 3], [2, 2, 4], [2, 3, 4], [2, 3, 4], [3, 3, 5], [3, 4, 6], [3, 4, 6], [4, 5, 7], [4, 5, 8], [4, 6, 9], [5, 7, 10],
    [6, 8, 11], [6, 8, 13], [7, 10, 14], [8, 11, 16], [9, 12, 18], [10, 13, 20], [11, 15, 23], [13, 17, 25],
];

/// Boundary strength between 4x4 blocks `p` and `q` (picture 4x4 positions), not intra.
fn bs_motion(pic: &Pic, pm: usize, qm: usize, pb: usize, qb: usize) -> u8 {
    let (dp, dq) = (&pic.db[pm], &pic.db[qm]);
    let bit = |d: &DbInfo, b: usize| -> bool {
        let (x, y) = (b % pic.w4 % 4, (b / pic.w4) % 4);
        (d.nz >> (y * 4 + x)) & 1 == 1
    };
    if bit(dp, pb) || bit(dq, qb) {
        return 2;
    }
    let (a, b) = (&pic.mf[pb], &pic.mf[qb]);
    let refs = |m: &Motion| -> [i32; 2] { [if m.refi[0] >= 0 { m.rpoc[0] } else { i32::MIN }, if m.refi[1] >= 0 { m.rpoc[1] } else { i32::MIN }] };
    let (ra, rb) = (refs(a), refs(b));
    let na = ra.iter().filter(|&&r| r != i32::MIN).count();
    let nb = rb.iter().filter(|&&r| r != i32::MIN).count();
    if na != nb {
        return 1;
    }
    let far = |u: [i16; 2], v: [i16; 2]| (u[0] as i32 - v[0] as i32).abs() >= 4 || (u[1] as i32 - v[1] as i32).abs() >= 4;
    if na == 1 {
        let (la, lb) = (if ra[0] != i32::MIN { 0 } else { 1 }, if rb[0] != i32::MIN { 0 } else { 1 });
        return (ra[la] != rb[lb] || far(a.mv[la], b.mv[lb])) as u8;
    }
    let mut sa = ra;
    let mut sb = rb;
    sa.sort_unstable();
    sb.sort_unstable();
    if sa != sb {
        return 1;
    }
    if ra[0] != ra[1] {
        // Pair the vectors by reference picture.
        let (m0, m1) = if ra[0] == rb[0] { (b.mv[0], b.mv[1]) } else { (b.mv[1], b.mv[0]) };
        return (far(a.mv[0], m0) || far(a.mv[1], m1)) as u8;
    }
    ((far(a.mv[0], b.mv[0]) || far(a.mv[1], b.mv[1])) && (far(a.mv[0], b.mv[1]) || far(a.mv[1], b.mv[0]))) as u8
}

/// Filters one line of samples across an edge: `s[0..4]` = p0..p3 (outwards), `s[4..8]` =
/// q0..q3, `luma` selects the luma filters.
#[inline]
fn filter_line(s: &mut [i32; 8], bs: u8, alpha: i32, beta: i32, tc0: i32, luma: bool) {
    let (p0, p1, p2, p3) = (s[0], s[1], s[2], s[3]);
    let (q0, q1, q2, q3) = (s[4], s[5], s[6], s[7]);
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return;
    }
    let ap = (p2 - p0).abs();
    let aq = (q2 - q0).abs();
    if bs < 4 {
        let tc = if luma { tc0 + (ap < beta) as i32 + (aq < beta) as i32 } else { tc0 + 1 };
        let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
        s[0] = clip1(p0 + delta) as i32;
        s[4] = clip1(q0 - delta) as i32;
        if luma {
            if ap < beta {
                s[1] = p1 + ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0);
            }
            if aq < beta {
                s[5] = q1 + ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0);
            }
        }
    } else if luma {
        let strong = (p0 - q0).abs() < ((alpha >> 2) + 2);
        if ap < beta && strong {
            s[0] = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3;
            s[1] = (p2 + p1 + p0 + q0 + 2) >> 2;
            s[2] = (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3;
        } else {
            s[0] = (2 * p1 + p0 + q1 + 2) >> 2;
        }
        if aq < beta && strong {
            s[4] = (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3;
            s[5] = (p0 + q0 + q1 + q2 + 2) >> 2;
            s[6] = (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3;
        } else {
            s[4] = (2 * q1 + q0 + p1 + 2) >> 2;
        }
    } else {
        s[0] = (2 * p1 + p0 + q1 + 2) >> 2;
        s[4] = (2 * q1 + q0 + p1 + 2) >> 2;
    }
}

/// Whether two `uni` macroblocks have the same motion (same reference pictures and vectors).
fn same_motion(p: &Pic, a: usize, b: usize) -> bool {
    let wmb = p.w4 / 4;
    let first = |m: usize| p.mf[(m / wmb) * 4 * p.w4 + (m % wmb) * 4];
    let (x, y) = (first(a), first(b));
    let refs = |m: &Motion| [if m.refi[0] >= 0 { m.rpoc[0] } else { i32::MIN }, if m.refi[1] >= 0 { m.rpoc[1] } else { i32::MIN }];
    let mvs = |m: &Motion| [if m.refi[0] >= 0 { m.mv[0] } else { [0, 0] }, if m.refi[1] >= 0 { m.mv[1] } else { [0, 0] }];
    refs(&x) == refs(&y) && mvs(&x) == mvs(&y)
}

/// Deblocks a finished picture in place.
pub fn deblock(p: &mut Pic) {
    let wmb = p.y.w / 16;
    let hmb = p.y.h / 16;
    for mby in 0..hmb {
        for mbx in 0..wmb {
            let m = mby * wmb + mbx;
            let cur = p.db[m];
            if !cur.done || cur.idc == 1 {
                continue;
            }
            for dir in 0..2 {
                // dir 0: vertical edges (filter horizontally), 1: horizontal edges.
                let nb = if dir == 0 { (mbx > 0).then(|| m - 1) } else { (mby > 0).then(|| m - wmb) };
                let nb_ok = nb.is_some_and(|n| p.db[n].done && (cur.idc != 2 || p.db[n].slice == cur.slice));
                for e in 0..4 {
                    if e == 0 && !nb_ok {
                        continue;
                    }
                    if e > 0 && cur.t8 && e != 2 {
                        continue;
                    }
                    let qn = if e == 0 { nb.expect("checked") } else { m };
                    let pdb = p.db[qn];
                    if cur.uni && (e > 0 || (pdb.uni && same_motion(p, qn, m))) {
                        continue;
                    }
                    // Boundary strength per 4-sample segment.
                    let mut bs = [0u8; 4];
                    for (k, b) in bs.iter_mut().enumerate() {
                        let (qx, qy) = if dir == 0 { (mbx * 4 + e, mby * 4 + k) } else { (mbx * 4 + k, mby * 4 + e) };
                        let (px, py) = if dir == 0 { (qx - 1, qy) } else { (qx, qy - 1) };
                        let qb = qy * p.w4 + qx;
                        let pb = py * p.w4 + px;
                        *b = if cur.intra || pdb.intra {
                            if e == 0 {
                                4
                            } else {
                                3
                            }
                        } else {
                            bs_motion(p, qn, m, pb, qb)
                        };
                    }
                    if bs == [0; 4] {
                        continue;
                    }
                    // Luma.
                    let qpav = (pdb.qp as i32 + cur.qp as i32 + 1) >> 1;
                    let ia = (qpav + cur.off_a as i32).clamp(0, 51) as usize;
                    let ib = (qpav + cur.off_b as i32).clamp(0, 51) as usize;
                    let (alpha, beta) = (ALPHA[ia] as i32, BETA[ib] as i32);
                    if alpha > 0 && beta > 0 {
                        let w = p.y.w;
                        for i in 0..16 {
                            let b = bs[i / 4];
                            if b == 0 {
                                continue;
                            }
                            let tc0 = if b < 4 { TC0[ia][b as usize - 1] as i32 } else { 0 };
                            let (x0, y0) = if dir == 0 { (mbx * 16 + e * 4, mby * 16 + i) } else { (mbx * 16 + i, mby * 16 + e * 4) };
                            let idx = |k: i32| -> usize {
                                if dir == 0 {
                                    y0 * w + (x0 as i32 + k) as usize
                                } else {
                                    (y0 as i32 + k) as usize * w + x0
                                }
                            };
                            let mut s = [0i32; 8];
                            for k in 0..4 {
                                s[k] = p.y.d[idx(-1 - k as i32)] as i32;
                                s[4 + k] = p.y.d[idx(k as i32)] as i32;
                            }
                            let before = s;
                            filter_line(&mut s, b, alpha, beta, tc0, true);
                            for k in 0..3 {
                                if s[k] != before[k] {
                                    p.y.d[idx(-1 - k as i32)] = s[k] as u8;
                                }
                                if s[4 + k] != before[4 + k] {
                                    p.y.d[idx(k as i32)] = s[4 + k] as u8;
                                }
                            }
                        }
                    }
                    // Chroma: edges 0 and 2 (luma) are chroma edges 0 and 4.
                    if e % 2 == 1 {
                        continue;
                    }
                    for c in 0..2 {
                        let qpav = (pdb.qpc[c] as i32 + cur.qpc[c] as i32 + 1) >> 1;
                        let ia = (qpav + cur.off_a as i32).clamp(0, 51) as usize;
                        let ib = (qpav + cur.off_b as i32).clamp(0, 51) as usize;
                        let (alpha, beta) = (ALPHA[ia] as i32, BETA[ib] as i32);
                        if alpha == 0 || beta == 0 {
                            continue;
                        }
                        let pl = if c == 0 { &mut p.u } else { &mut p.v };
                        let w = pl.w;
                        for i in 0..8 {
                            let b = bs[i / 2];
                            if b == 0 {
                                continue;
                            }
                            let tc0 = if b < 4 { TC0[ia][b as usize - 1] as i32 } else { 0 };
                            let (x0, y0) = if dir == 0 { (mbx * 8 + e * 2, mby * 8 + i) } else { (mbx * 8 + i, mby * 8 + e * 2) };
                            let idx = |k: i32| -> usize {
                                if dir == 0 {
                                    y0 * w + (x0 as i32 + k) as usize
                                } else {
                                    (y0 as i32 + k) as usize * w + x0
                                }
                            };
                            let mut s = [0i32; 8];
                            for k in 0..2 {
                                s[k] = pl.d[idx(-1 - k as i32)] as i32;
                                s[4 + k] = pl.d[idx(k as i32)] as i32;
                            }
                            s[2] = s[1];
                            s[3] = s[1];
                            s[6] = s[5];
                            s[7] = s[5];
                            filter_line(&mut s, b, alpha, beta, tc0, false);
                            pl.d[idx(-1)] = s[0] as u8;
                            pl.d[idx(0)] = s[4] as u8;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mc_luma_matches_direct_form() {
        let mut p = Plane::new(48, 40);
        let mut x = 12345u32;
        for v in p.d.iter_mut() {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            *v = (x >> 16) as u8;
        }
        for (w, h) in [(4, 4), (8, 8), (16, 16), (16, 8), (8, 16), (4, 8), (8, 4)] {
            for mvy in -12..12 {
                for mvx in -12..12 {
                    for (bx, by) in [(0, 0), (16, 8), (32, 24)] {
                        let mut a = [0i32; 256];
                        let mut b = [0i32; 256];
                        let mut sc = McScratch::default();
                        mc_luma(&mut sc, &p, bx, by, mvx * 7 + 3, mvy * 5 - 1, w, h, &mut a, 16);
                        mc_luma_ref(&p, bx, by, mvx * 7 + 3, mvy * 5 - 1, w, h, &mut b, 16);
                        assert_eq!(a, b, "{w}x{h} at ({bx},{by}) mv ({}, {})", mvx * 7 + 3, mvy * 5 - 1);
                    }
                }
            }
        }
    }
}
