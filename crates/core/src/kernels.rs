//! Dense kernels with a fixed, platform-independent order of floating-point operations.
//!
//! Every kernel defines the arithmetic of each output element exactly:
//!
//! * [`matmul`] / [`matmul_tn`]: `y = Σ_i x_i·w_i`, accumulated in f32 in ascending inner
//!   index starting from `0.0`;
//! * [`matmul_nt`] / [`dot`]: eight lane accumulators over consecutive 8-element chunks,
//!   reduced as `((l0+l4)+(l1+l5))+((l2+l6)+(l3+l7))`, then the tail added in order.
//!
//! The kernels are written once against the 8-lane [`V8`] abstraction (explicit SIMD per
//! instruction set, lane-wise IEEE ops, never FMA) and instantiated per backend. Tiling,
//! vector width, threading and batch shape only change *which* elements are computed
//! together, never the arithmetic of one element — so results are bit-identical across
//! CPUs, operating systems, thread counts and languages.

use crate::par::{self, SyncPtr};
use crate::simd::{Base, V8};

// ---------------------------------------------------------------------------------------
// generic building blocks (inline(always): they inherit the caller's target features)
// ---------------------------------------------------------------------------------------

/// `Σ a[i]·b[i]` in the fixed 8-lane order.
#[inline(always)]
pub(crate) fn dot_v<V: V8>(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let mut acc = V::splat(0.0);
    // SAFETY: c*8+8 <= n for every chunk.
    unsafe {
        for c in 0..chunks {
            acc = acc.add(V::load(a.as_ptr().add(c * 8)).mul(V::load(b.as_ptr().add(c * 8))));
        }
    }
    let mut s = acc.reduce();
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
}

/// `Σ a[i]` in the fixed 8-lane order.
#[inline(always)]
pub(crate) fn sum_v<V: V8>(a: &[f32]) -> f32 {
    let chunks = a.len() / 8;
    let mut acc = V::splat(0.0);
    unsafe {
        for c in 0..chunks {
            acc = acc.add(V::load(a.as_ptr().add(c * 8)));
        }
    }
    let mut s = acc.reduce();
    for &v in &a[chunks * 8..] {
        s += v;
    }
    s
}

/// `v[i] = exp(max(v[i] - m, floor))` (NaN kept until `exp` maps it), returning the
/// sequential sum `((e0 + e1) + e2) + …` — the scalar softmax recipe, vectorised.
#[inline(always)]
pub(crate) fn exp_shift_v<V: V8>(v: &mut [f32], m: f32, floor: f32, sequential_sum: bool) -> f32 {
    let n = v.len();
    let chunks = n / 8;
    let (vm, vf) = (V::splat(m), V::splat(floor));
    let mut s = 0f32;
    // SAFETY: c*8+8 <= n.
    unsafe {
        for c in 0..chunks {
            let p = v.as_mut_ptr().add(c * 8);
            let e = V::load(p).sub(vm).clamp_below(vf).exp();
            e.store(p);
            if sequential_sum {
                for x in e.lanes() {
                    s += x;
                }
            }
        }
    }
    for x in &mut v[chunks * 8..] {
        let mut z = *x - m;
        if z < floor {
            z = floor;
        }
        let e = crate::fpmath::exp(z);
        *x = e;
        if sequential_sum {
            s += e;
        }
    }
    s
}

/// `sg[i] = (g·σ(g))·u` over `gu = [g | u]` (SwiGLU forward).
#[inline(always)]
pub(crate) fn swiglu_v<V: V8>(gu: &[f32], sg: &mut [f32]) {
    let dff = sg.len();
    let chunks = dff / 8;
    let one = V::splat(1.0);
    unsafe {
        for c in 0..chunks {
            let o = c * 8;
            let g = V::load(gu.as_ptr().add(o));
            let u = V::load(gu.as_ptr().add(dff + o));
            let s = one.div(one.add(g.neg().exp()));
            g.mul(s).mul(u).store(sg.as_mut_ptr().add(o));
        }
    }
    for i in chunks * 8..dff {
        let g = gu[i];
        let s = crate::fpmath::sigmoid(g);
        sg[i] = (g * s) * gu[dff + i];
    }
}

/// SwiGLU backward: `out = [ds·u·(σ·(1 + g·(1−σ))) | ds·(g·σ)]`.
#[inline(always)]
pub(crate) fn swiglu_bwd_v<V: V8>(gu: &[f32], ds: &[f32], out: &mut [f32]) {
    let dff = ds.len();
    let chunks = dff / 8;
    let one = V::splat(1.0);
    unsafe {
        for c in 0..chunks {
            let o = c * 8;
            let g = V::load(gu.as_ptr().add(o));
            let u = V::load(gu.as_ptr().add(dff + o));
            let d = V::load(ds.as_ptr().add(o));
            let s = one.div(one.add(g.neg().exp()));
            let silu = g.mul(s);
            d.mul(u).mul(s.mul(one.add(g.mul(one.sub(s))))).store(out.as_mut_ptr().add(o));
            d.mul(silu).store(out.as_mut_ptr().add(dff + o));
        }
    }
    for i in chunks * 8..dff {
        let g = gu[i];
        let u = gu[dff + i];
        let s = crate::fpmath::sigmoid(g);
        let silu = g * s;
        out[i] = ds[i] * u * (s * (1.0 + g * (1.0 - s)));
        out[dff + i] = ds[i] * silu;
    }
}

/// `y[i] += a · x[i]` (per element: `y + (a·x)`).
#[inline(always)]
pub(crate) fn axpy_v<V: V8>(y: &mut [f32], a: f32, x: &[f32]) {
    let n = y.len().min(x.len());
    let chunks = n / 8;
    let va = V::splat(a);
    unsafe {
        for c in 0..chunks {
            let p = y.as_mut_ptr().add(c * 8);
            V::load(p).add(va.mul(V::load(x.as_ptr().add(c * 8)))).store(p);
        }
    }
    for i in chunks * 8..n {
        y[i] += a * x[i];
    }
}

/// Dot product with the fixed 8-lane pattern (baseline backend; same bits as any other).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    dot_v::<Base>(a, b)
}

/// Sum with the fixed 8-lane pattern.
#[inline]
pub fn sum(a: &[f32]) -> f32 {
    sum_v::<Base>(a)
}

// ---------------------------------------------------------------------------------------
// y = x · w   (axpy order)
// ---------------------------------------------------------------------------------------

/// `y[rows x n] = x[rows x k] · w[k x n]` (all row-major, `y` overwritten).
pub fn matmul(y: &mut [f32], x: &[f32], w: &[f32], rows: usize, k: usize, n: usize) {
    assert!(y.len() >= rows * n && x.len() >= rows * k && w.len() >= k * n);
    if rows == 0 || n == 0 {
        return;
    }
    if k == 0 {
        y[..rows * n].fill(0.0);
        return;
    }
    let yp = SyncPtr(y.as_mut_ptr());
    let (xp, wp) = (x.as_ptr() as usize, w.as_ptr() as usize);
    let run = move |r0: usize, r1: usize, c0: usize, c1: usize| mm_range(yp.get() as usize, xp, wp, k, 1, k, n, r0, r1, c0, c1);
    if rows >= 4 * par::threads() || n < 64 {
        par::for_ranges(rows, k * n, 4, move |r0, r1| run(r0, r1, 0, n));
    } else {
        par::for_ranges(n, k * rows, 16, move |c0, c1| run(0, rows, c0, c1));
    }
}

/// `y[kd x n] = a[rows x kd]ᵀ · b[rows x n]` without materialising the transpose;
/// element `(k, j) = Σ_{r ascending} a[r][k]·b[r][j]` (same bits as `matmul(aᵀ, b)`).
pub fn matmul_tn(y: &mut [f32], a: &[f32], b: &[f32], rows: usize, kd: usize, n: usize) {
    assert!(y.len() >= kd * n && a.len() >= rows * kd && b.len() >= rows * n);
    if kd == 0 || n == 0 {
        return;
    }
    if rows == 0 {
        y[..kd * n].fill(0.0);
        return;
    }
    let yp = SyncPtr(y.as_mut_ptr());
    let (ap, bp) = (a.as_ptr() as usize, b.as_ptr() as usize);
    // x(k, r) = a[r*kd + k]: row stride 1, inner stride kd.
    let run = move |k0: usize, k1: usize, c0: usize, c1: usize| mm_range(yp.get() as usize, ap, bp, 1, kd, rows, n, k0, k1, c0, c1);
    if kd >= 4 * par::threads() || n < 64 {
        par::for_ranges(kd, rows * n, 4, move |k0, k1| run(k0, k1, 0, n));
    } else {
        par::for_ranges(n, rows * kd, 16, move |c0, c1| run(0, kd, c0, c1));
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn mm_range(y: usize, x: usize, w: usize, xrs: usize, xis: usize, k: usize, n: usize, r0: usize, r1: usize, c0: usize, c1: usize) {
    let (y, x, w) = (y as *mut f32, x as *const f32, w as *const f32);
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn avx2(y: *mut f32, x: *const f32, w: *const f32, xrs: usize, xis: usize, k: usize, n: usize, r0: usize, r1: usize, c0: usize, c1: usize) {
            mm_block::<crate::simd::Avx, 2>(y, x, w, xrs, xis, k, n, r0, r1, c0, c1)
        }
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 present; callers pass in-bounds, non-overlapping regions.
            return unsafe { avx2(y, x, w, xrs, xis, k, n, r0, r1, c0, c1) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    const NV: usize = 2;
    #[cfg(not(target_arch = "aarch64"))]
    const NV: usize = 1;
    // SAFETY: callers pass in-bounds, non-overlapping regions.
    unsafe { mm_block::<Base, NV>(y, x, w, xrs, xis, k, n, r0, r1, c0, c1) }
}

/// `y[r][c] = Σ_kk x(r, kk)·w[kk][c]` for `r in r0..r1`, `c in c0..c1`, where
/// `x(r, kk) = x[r*xrs + kk*xis]`. Tiles of up to 4 rows × `8·NV` columns.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn mm_block<V: V8, const NV: usize>(
    y: *mut f32,
    x: *const f32,
    w: *const f32,
    xrs: usize,
    xis: usize,
    k: usize,
    n: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) {
    let mut r = r0;
    while r + 4 <= r1 {
        mm_rows::<V, 4, NV>(y, x, w, xrs, xis, k, n, r, c0, c1);
        r += 4;
    }
    if r + 2 <= r1 {
        mm_rows::<V, 2, NV>(y, x, w, xrs, xis, k, n, r, c0, c1);
        r += 2;
    }
    if r < r1 {
        mm_rows::<V, 1, NV>(y, x, w, xrs, xis, k, n, r, c0, c1);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn mm_rows<V: V8, const R: usize, const NV: usize>(
    y: *mut f32,
    x: *const f32,
    w: *const f32,
    xrs: usize,
    xis: usize,
    k: usize,
    n: usize,
    r: usize,
    c0: usize,
    c1: usize,
) {
    let t = 8 * NV;
    let full_end = c0 + (c1 - c0) / t * t;
    let mut xr = [x; R];
    for (i, p) in xr.iter_mut().enumerate() {
        *p = x.add((r + i) * xrs);
    }
    let mut c = c0;
    while c < full_end {
        let mut acc = [[V::splat(0.0); NV]; R];
        let mut wp = w.add(c);
        let mut xo = 0usize;
        for _ in 0..k {
            let mut wv = [V::splat(0.0); NV];
            for (j, v) in wv.iter_mut().enumerate() {
                *v = V::load(wp.add(8 * j));
            }
            for i in 0..R {
                let xv = V::splat(*xr[i].add(xo));
                for j in 0..NV {
                    acc[i][j] = acc[i][j].add(xv.mul(wv[j]));
                }
            }
            wp = wp.add(n);
            xo += xis;
        }
        for i in 0..R {
            for j in 0..NV {
                acc[i][j].store(y.add((r + i) * n + c + 8 * j));
            }
        }
        c += t;
    }
    // Remaining columns: full 8-wide vectors, then scalars (same per-element order).
    while c + 8 <= c1 {
        let mut acc = [V::splat(0.0); R];
        let mut wp = w.add(c);
        let mut xo = 0usize;
        for _ in 0..k {
            let wv = V::load(wp);
            for i in 0..R {
                acc[i] = acc[i].add(V::splat(*xr[i].add(xo)).mul(wv));
            }
            wp = wp.add(n);
            xo += xis;
        }
        for (i, a) in acc.iter().enumerate() {
            a.store(y.add((r + i) * n + c));
        }
        c += 8;
    }
    for cc in c..c1 {
        for (i, &xp) in xr.iter().enumerate() {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += *xp.add(kk * xis) * *w.add(kk * n + cc);
            }
            *y.add((r + i) * n + cc) = acc;
        }
    }
}

// ---------------------------------------------------------------------------------------
// y = a · bᵀ   (dot order)
// ---------------------------------------------------------------------------------------

/// `y[rows x kd] = a[rows x n] · b[kd x n]ᵀ`; every element is exactly
/// `dot(a_row, b_row)`. Used for input gradients (`dX = dY · Wᵀ`) without transposed
/// weight copies.
pub fn matmul_nt(y: &mut [f32], a: &[f32], b: &[f32], rows: usize, n: usize, kd: usize) {
    assert!(y.len() >= rows * kd && a.len() >= rows * n && b.len() >= kd * n);
    if rows == 0 || kd == 0 {
        return;
    }
    let yp = SyncPtr(y.as_mut_ptr());
    let (ap, bp) = (a.as_ptr() as usize, b.as_ptr() as usize);
    let run = move |r0: usize, r1: usize, k0: usize, k1: usize| nt_range(yp.get() as usize, ap, bp, n, kd, r0, r1, k0, k1);
    if rows >= 4 * par::threads() || kd < 64 {
        par::for_ranges(rows, n * kd, 4, move |r0, r1| run(r0, r1, 0, kd));
    } else {
        par::for_ranges(kd, n * rows, 8, move |k0, k1| run(0, rows, k0, k1));
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn nt_range(y: usize, a: usize, b: usize, n: usize, kd: usize, r0: usize, r1: usize, k0: usize, k1: usize) {
    let (y, a, b) = (y as *mut f32, a as *const f32, b as *const f32);
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn avx2(y: *mut f32, a: *const f32, b: *const f32, n: usize, kd: usize, r0: usize, r1: usize, k0: usize, k1: usize) {
            nt_block::<crate::simd::Avx, 4>(y, a, b, n, kd, r0, r1, k0, k1)
        }
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 present; callers pass in-bounds regions.
            return unsafe { avx2(y, a, b, n, kd, r0, r1, k0, k1) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    const R: usize = 4;
    #[cfg(not(target_arch = "aarch64"))]
    const R: usize = 2;
    // SAFETY: callers pass in-bounds regions.
    unsafe { nt_block::<Base, R>(y, a, b, n, kd, r0, r1, k0, k1) }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn nt_block<V: V8, const R: usize>(y: *mut f32, a: *const f32, b: *const f32, n: usize, kd: usize, r0: usize, r1: usize, k0: usize, k1: usize) {
    let mut r = r0;
    while r + R <= r1 {
        nt_rows::<V, R>(y, a, b, n, kd, r, k0, k1);
        r += R;
    }
    while r < r1 {
        nt_rows::<V, 1>(y, a, b, n, kd, r, k0, k1);
        r += 1;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn nt_rows<V: V8, const R: usize>(y: *mut f32, a: *const f32, b: *const f32, n: usize, kd: usize, r: usize, k0: usize, k1: usize) {
    let chunks = n / 8;
    let mut k = k0;
    while k + 2 <= k1 {
        let mut acc = [[V::splat(0.0); 2]; R];
        let b0 = b.add(k * n);
        let b1 = b.add((k + 1) * n);
        for c in 0..chunks {
            let w0 = V::load(b0.add(c * 8));
            let w1 = V::load(b1.add(c * 8));
            for (i, accr) in acc.iter_mut().enumerate() {
                let av = V::load(a.add((r + i) * n + c * 8));
                accr[0] = accr[0].add(av.mul(w0));
                accr[1] = accr[1].add(av.mul(w1));
            }
        }
        for (i, accr) in acc.iter().enumerate() {
            let ar = a.add((r + i) * n);
            for (jj, bj) in [b0, b1].into_iter().enumerate() {
                let mut s = accr[jj].reduce();
                for t in chunks * 8..n {
                    s += *ar.add(t) * *bj.add(t);
                }
                *y.add((r + i) * kd + k + jj) = s;
            }
        }
        k += 2;
    }
    if k < k1 {
        let bk = std::slice::from_raw_parts(b.add(k * n), n);
        for i in 0..R {
            let ar = std::slice::from_raw_parts(a.add((r + i) * n), n);
            *y.add((r + i) * kd + k) = dot_v::<V>(ar, bk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn naive(x: &[f32], w: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
        let mut y = vec![0f32; rows * n];
        for r in 0..rows {
            for c in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += x[r * k + kk] * w[kk * n + c];
                }
                y[r * n + c] = acc;
            }
        }
        y
    }

    /// Reference for the 8-lane dot order, written with plain scalars.
    fn dot_ref(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let mut l = [0f32; 8];
        for c in 0..n / 8 {
            for j in 0..8 {
                l[j] += a[c * 8 + j] * b[c * 8 + j];
            }
        }
        let mut s = ((l[0] + l[4]) + (l[1] + l[5])) + ((l[2] + l[6]) + (l[3] + l[7]));
        for i in n / 8 * 8..n {
            s += a[i] * b[i];
        }
        s
    }

    fn rand(rng: &mut Rng, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.uniform()).collect()
    }

    const SHAPES: &[(usize, usize, usize)] =
        &[(1, 1, 1), (2, 3, 5), (3, 5, 7), (4, 16, 16), (6, 9, 70), (9, 33, 70), (7, 17, 23), (64, 128, 384), (5, 300, 257)];

    #[test]
    fn matmul_is_bit_exact_with_naive_order() {
        let mut rng = Rng::new(7);
        for &(rows, k, n) in SHAPES {
            let x = rand(&mut rng, rows * k);
            let w = rand(&mut rng, k * n);
            let mut y = vec![0f32; rows * n];
            matmul(&mut y, &x, &w, rows, k, n);
            let want = naive(&x, &w, rows, k, n);
            for i in 0..rows * n {
                assert_eq!(y[i].to_bits(), want[i].to_bits(), "({rows},{k},{n}) at {i}");
            }
            for r in 0..rows {
                let mut y1 = vec![0f32; n];
                matmul(&mut y1, &x[r * k..(r + 1) * k], &w, 1, k, n);
                assert_eq!(&y1[..], &y[r * n..(r + 1) * n]);
            }
        }
    }

    #[test]
    fn matmul_tn_equals_matmul_of_transpose() {
        let mut rng = Rng::new(8);
        for &(rows, kd, n) in SHAPES {
            let a = rand(&mut rng, rows * kd);
            let b = rand(&mut rng, rows * n);
            let mut at = vec![0f32; kd * rows];
            for r in 0..rows {
                for k in 0..kd {
                    at[k * rows + r] = a[r * kd + k];
                }
            }
            let want = naive(&at, &b, kd, rows, n);
            let mut y = vec![0f32; kd * n];
            matmul_tn(&mut y, &a, &b, rows, kd, n);
            for i in 0..kd * n {
                assert_eq!(y[i].to_bits(), want[i].to_bits(), "({rows},{kd},{n}) at {i}");
            }
        }
    }

    #[test]
    fn matmul_nt_and_dot_match_reference() {
        let mut rng = Rng::new(9);
        for &(rows, n, kd) in SHAPES {
            let a = rand(&mut rng, rows * n);
            let b = rand(&mut rng, kd * n);
            let mut y = vec![0f32; rows * kd];
            matmul_nt(&mut y, &a, &b, rows, n, kd);
            for r in 0..rows {
                for k in 0..kd {
                    let (ar, bk) = (&a[r * n..(r + 1) * n], &b[k * n..(k + 1) * n]);
                    let want = dot_ref(ar, bk);
                    assert_eq!(y[r * kd + k].to_bits(), want.to_bits(), "nt ({rows},{n},{kd}) at ({r},{k})");
                    assert_eq!(dot(ar, bk).to_bits(), want.to_bits());
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_kernels() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut rng = Rng::new(1);
        for &(rows, n, kd) in &[(2, 96, 288), (8, 96, 288), (8, 288, 96), (8, 96, 256), (64, 288, 96), (256, 128, 384)] {
            let a = rand(&mut rng, rows * n);
            let w = rand(&mut rng, kd * n);
            let wf = rand(&mut rng, n * kd);
            let mut y = vec![0f32; rows * kd];
            let iters = (400_000_000 / (rows * n * kd)).max(5);
            let gf = |t: f64| 2.0 * (rows * n * kd) as f64 * iters as f64 / t / 1e9;
            let t0 = Instant::now();
            for _ in 0..iters {
                matmul(black_box(&mut y), black_box(&a), black_box(&wf), rows, n, kd);
            }
            let mm = gf(t0.elapsed().as_secs_f64());
            let t1 = Instant::now();
            for _ in 0..iters {
                matmul_nt(black_box(&mut y), black_box(&a), black_box(&w), rows, n, kd);
            }
            let nt = gf(t1.elapsed().as_secs_f64());
            eprintln!("rows {rows:4} n {n:4} kd {kd:4}: matmul {mm:6.1} GF/s  matmul_nt {nt:6.1} GF/s");
        }
    }
}
