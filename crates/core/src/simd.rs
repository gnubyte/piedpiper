//! An 8-lane f32 vector with one implementation per instruction set. Every operation is a
//! lane-wise IEEE-754 single-precision `+ − × ÷ √` (never FMA), so all backends — AVX2,
//! SSE2, AArch64 NEON, WebAssembly SIMD128 and the scalar fallback — produce identical
//! bits. Kernels are written once, generic over [`V8`], and instantiated per backend.
//!
//! 32-bit ARM NEON is *not* used: it flushes subnormals and lacks IEEE division/sqrt.

// The exp constants must stay bit-identical to `fpmath::exp`, not follow std::consts.
#![allow(clippy::missing_safety_doc, clippy::approx_constant)]

pub(crate) trait V8: Copy {
    unsafe fn load(p: *const f32) -> Self;
    unsafe fn store(self, p: *mut f32);
    fn splat(x: f32) -> Self;
    fn add(self, b: Self) -> Self;
    fn sub(self, b: Self) -> Self;
    fn mul(self, b: Self) -> Self;
    fn div(self, b: Self) -> Self;
    fn sqrt(self) -> Self;
    /// Lanes strictly below `thr` become 0 (NaN lanes are kept, like the scalar code).
    fn zero_below(self, thr: Self) -> Self;
    /// Lanes strictly below `lo` become `lo` (NaN lanes are kept).
    fn clamp_below(self, lo: Self) -> Self;
    /// Sign flip (exact, like unary minus).
    fn neg(self) -> Self;
    fn lanes(self) -> [f32; 8];
    /// `((l0+l4)+(l1+l5))+((l2+l6)+(l3+l7))` — the shared reduction tree.
    #[inline(always)]
    fn reduce(self) -> f32 {
        reduce8(self.lanes())
    }
    /// Lane-wise [`crate::fpmath::exp`], bit-identical to the scalar function.
    #[inline(always)]
    fn exp(self) -> Self {
        let mut a = self.lanes();
        for x in a.iter_mut() {
            *x = crate::fpmath::exp(*x);
        }
        // SAFETY: `a` holds 8 floats.
        unsafe { Self::load(a.as_ptr()) }
    }
}

// Constants of `fpmath::exp`, shared by the vector versions.
const EXP_LOG2E: f32 = 1.442_695;
const EXP_LN2_HI: f32 = 0.693_145_75;
const EXP_LN2_LO: f32 = 1.428_606_8e-6;
const EXP_C: [f32; 4] = [0.166_666_67, 0.041_666_668, 0.008_333_334, 0.001_388_888_9];

/// The fixed reduction tree shared by every dot product / sum.
#[inline(always)]
pub(crate) fn reduce8(a: [f32; 8]) -> f32 {
    ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))
}

// ----------------------------------------------------------------------------------------
// Scalar (reference semantics; used on targets without an IEEE-exact SIMD unit)
// ----------------------------------------------------------------------------------------
#[derive(Clone, Copy)]
#[allow(dead_code)] // the backend on targets without IEEE-exact SIMD; also the test reference
pub(crate) struct Scalar([f32; 8]);

impl V8 for Scalar {
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        Scalar(*(p as *const [f32; 8]))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        *(p as *mut [f32; 8]) = self.0;
    }
    #[inline(always)]
    fn splat(x: f32) -> Self {
        Scalar([x; 8])
    }
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        let mut r = self.0;
        for i in 0..8 {
            r[i] += b.0[i];
        }
        Scalar(r)
    }
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        let mut r = self.0;
        for i in 0..8 {
            r[i] -= b.0[i];
        }
        Scalar(r)
    }
    #[inline(always)]
    fn mul(self, b: Self) -> Self {
        let mut r = self.0;
        for i in 0..8 {
            r[i] *= b.0[i];
        }
        Scalar(r)
    }
    #[inline(always)]
    fn div(self, b: Self) -> Self {
        let mut r = self.0;
        for i in 0..8 {
            r[i] /= b.0[i];
        }
        Scalar(r)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        let mut r = self.0;
        for x in r.iter_mut() {
            *x = x.sqrt();
        }
        Scalar(r)
    }
    #[inline(always)]
    fn zero_below(self, thr: Self) -> Self {
        let mut r = self.0;
        for i in 0..8 {
            if r[i] < thr.0[i] {
                r[i] = 0.0;
            }
        }
        Scalar(r)
    }
    #[inline(always)]
    fn clamp_below(self, lo: Self) -> Self {
        let mut r = self.0;
        for i in 0..8 {
            if r[i] < lo.0[i] {
                r[i] = lo.0[i];
            }
        }
        Scalar(r)
    }
    #[inline(always)]
    fn neg(self) -> Self {
        let mut r = self.0;
        for x in r.iter_mut() {
            *x = -*x;
        }
        Scalar(r)
    }
    #[inline(always)]
    fn lanes(self) -> [f32; 8] {
        self.0
    }
}

// ----------------------------------------------------------------------------------------
// x86-64: AVX2 (runtime-detected) and SSE2 (baseline)
// ----------------------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::V8;
    use core::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub(crate) struct Avx(__m256);

    // SAFETY (whole impl): only instantiated inside `#[target_feature(enable = "avx2")]`
    // functions that are entered after runtime detection.
    impl V8 for Avx {
        #[inline(always)]
        unsafe fn load(p: *const f32) -> Self {
            Avx(_mm256_loadu_ps(p))
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut f32) {
            _mm256_storeu_ps(p, self.0)
        }
        #[inline(always)]
        fn splat(x: f32) -> Self {
            unsafe { Avx(_mm256_set1_ps(x)) }
        }
        #[inline(always)]
        fn add(self, b: Self) -> Self {
            unsafe { Avx(_mm256_add_ps(self.0, b.0)) }
        }
        #[inline(always)]
        fn sub(self, b: Self) -> Self {
            unsafe { Avx(_mm256_sub_ps(self.0, b.0)) }
        }
        #[inline(always)]
        fn mul(self, b: Self) -> Self {
            unsafe { Avx(_mm256_mul_ps(self.0, b.0)) }
        }
        #[inline(always)]
        fn div(self, b: Self) -> Self {
            unsafe { Avx(_mm256_div_ps(self.0, b.0)) }
        }
        #[inline(always)]
        fn sqrt(self) -> Self {
            unsafe { Avx(_mm256_sqrt_ps(self.0)) }
        }
        #[inline(always)]
        fn zero_below(self, thr: Self) -> Self {
            unsafe {
                let lt = _mm256_cmp_ps::<_CMP_LT_OQ>(self.0, thr.0);
                Avx(_mm256_andnot_ps(lt, self.0))
            }
        }
        #[inline(always)]
        fn clamp_below(self, lo: Self) -> Self {
            unsafe {
                let lt = _mm256_cmp_ps::<_CMP_LT_OQ>(self.0, lo.0);
                Avx(_mm256_blendv_ps(self.0, lo.0, lt))
            }
        }
        #[inline(always)]
        fn neg(self) -> Self {
            unsafe { Avx(_mm256_xor_ps(self.0, _mm256_set1_ps(-0.0))) }
        }
        #[inline(always)]
        fn lanes(self) -> [f32; 8] {
            let mut a = [0f32; 8];
            unsafe { _mm256_storeu_ps(a.as_mut_ptr(), self.0) };
            a
        }
        #[inline(always)]
        fn reduce(self) -> f32 {
            unsafe {
                let s = _mm_add_ps(_mm256_castps256_ps128(self.0), _mm256_extractf128_ps::<1>(self.0));
                super::reduce4_sse(s)
            }
        }
        #[inline(always)]
        fn exp(self) -> Self {
            unsafe {
                let s = |v: f32| _mm256_set1_ps(v);
                let x = self.0;
                let x = _mm256_blendv_ps(s(-80.0), x, _mm256_cmp_ps::<_CMP_GE_OQ>(x, s(-80.0)));
                let x = _mm256_blendv_ps(s(80.0), x, _mm256_cmp_ps::<_CMP_LE_OQ>(x, s(80.0)));
                let t = _mm256_add_ps(_mm256_mul_ps(x, s(super::EXP_LOG2E)), s(0.5));
                let n0 = _mm256_cvttps_epi32(t);
                let gt = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_cvtepi32_ps(n0), t);
                let n = _mm256_add_epi32(n0, _mm256_castps_si256(gt));
                let nf = _mm256_cvtepi32_ps(n);
                let r = _mm256_sub_ps(
                    _mm256_sub_ps(x, _mm256_mul_ps(nf, s(super::EXP_LN2_HI))),
                    _mm256_mul_ps(nf, s(super::EXP_LN2_LO)),
                );
                let c = super::EXP_C;
                let mut p = _mm256_add_ps(s(c[2]), _mm256_mul_ps(r, s(c[3])));
                p = _mm256_add_ps(s(c[1]), _mm256_mul_ps(r, p));
                p = _mm256_add_ps(s(c[0]), _mm256_mul_ps(r, p));
                p = _mm256_add_ps(s(0.5), _mm256_mul_ps(r, p));
                p = _mm256_add_ps(s(1.0), _mm256_mul_ps(r, p));
                p = _mm256_add_ps(s(1.0), _mm256_mul_ps(r, p));
                let scale = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n, _mm256_set1_epi32(127))));
                Avx(_mm256_mul_ps(p, scale))
            }
        }
    }

    #[derive(Clone, Copy)]
    pub(crate) struct Sse(__m128, __m128);

    /// `mask ? a : b` with SSE2 bit operations.
    #[inline(always)]
    unsafe fn sel(mask: __m128, a: __m128, b: __m128) -> __m128 {
        _mm_or_ps(_mm_and_ps(mask, a), _mm_andnot_ps(mask, b))
    }

    /// Four lanes of `fpmath::exp` (SSE2).
    #[inline(always)]
    unsafe fn exp4(x: __m128) -> __m128 {
        let s = |v: f32| _mm_set1_ps(v);
        let x = sel(_mm_cmpge_ps(x, s(-80.0)), x, s(-80.0));
        let x = sel(_mm_cmple_ps(x, s(80.0)), x, s(80.0));
        let t = _mm_add_ps(_mm_mul_ps(x, s(super::EXP_LOG2E)), s(0.5));
        let n0 = _mm_cvttps_epi32(t);
        let gt = _mm_cmpgt_ps(_mm_cvtepi32_ps(n0), t);
        let n = _mm_add_epi32(n0, _mm_castps_si128(gt));
        let nf = _mm_cvtepi32_ps(n);
        let r = _mm_sub_ps(_mm_sub_ps(x, _mm_mul_ps(nf, s(super::EXP_LN2_HI))), _mm_mul_ps(nf, s(super::EXP_LN2_LO)));
        let c = super::EXP_C;
        let mut p = _mm_add_ps(s(c[2]), _mm_mul_ps(r, s(c[3])));
        p = _mm_add_ps(s(c[1]), _mm_mul_ps(r, p));
        p = _mm_add_ps(s(c[0]), _mm_mul_ps(r, p));
        p = _mm_add_ps(s(0.5), _mm_mul_ps(r, p));
        p = _mm_add_ps(s(1.0), _mm_mul_ps(r, p));
        p = _mm_add_ps(s(1.0), _mm_mul_ps(r, p));
        let scale = _mm_castsi128_ps(_mm_slli_epi32::<23>(_mm_add_epi32(n, _mm_set1_epi32(127))));
        _mm_mul_ps(p, scale)
    }

    // SSE2 is part of the x86-64 baseline.
    impl V8 for Sse {
        #[inline(always)]
        unsafe fn load(p: *const f32) -> Self {
            Sse(_mm_loadu_ps(p), _mm_loadu_ps(p.add(4)))
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut f32) {
            _mm_storeu_ps(p, self.0);
            _mm_storeu_ps(p.add(4), self.1);
        }
        #[inline(always)]
        fn splat(x: f32) -> Self {
            unsafe { Sse(_mm_set1_ps(x), _mm_set1_ps(x)) }
        }
        #[inline(always)]
        fn add(self, b: Self) -> Self {
            unsafe { Sse(_mm_add_ps(self.0, b.0), _mm_add_ps(self.1, b.1)) }
        }
        #[inline(always)]
        fn sub(self, b: Self) -> Self {
            unsafe { Sse(_mm_sub_ps(self.0, b.0), _mm_sub_ps(self.1, b.1)) }
        }
        #[inline(always)]
        fn mul(self, b: Self) -> Self {
            unsafe { Sse(_mm_mul_ps(self.0, b.0), _mm_mul_ps(self.1, b.1)) }
        }
        #[inline(always)]
        fn div(self, b: Self) -> Self {
            unsafe { Sse(_mm_div_ps(self.0, b.0), _mm_div_ps(self.1, b.1)) }
        }
        #[inline(always)]
        fn sqrt(self) -> Self {
            unsafe { Sse(_mm_sqrt_ps(self.0), _mm_sqrt_ps(self.1)) }
        }
        #[inline(always)]
        fn zero_below(self, thr: Self) -> Self {
            unsafe {
                Sse(
                    _mm_andnot_ps(_mm_cmplt_ps(self.0, thr.0), self.0),
                    _mm_andnot_ps(_mm_cmplt_ps(self.1, thr.1), self.1),
                )
            }
        }
        #[inline(always)]
        fn clamp_below(self, lo: Self) -> Self {
            unsafe { Sse(sel(_mm_cmplt_ps(self.0, lo.0), lo.0, self.0), sel(_mm_cmplt_ps(self.1, lo.1), lo.1, self.1)) }
        }
        #[inline(always)]
        fn neg(self) -> Self {
            unsafe {
                let m = _mm_set1_ps(-0.0);
                Sse(_mm_xor_ps(self.0, m), _mm_xor_ps(self.1, m))
            }
        }
        #[inline(always)]
        fn reduce(self) -> f32 {
            unsafe { super::reduce4_sse(_mm_add_ps(self.0, self.1)) }
        }
        #[inline(always)]
        fn exp(self) -> Self {
            unsafe { Sse(exp4(self.0), exp4(self.1)) }
        }
        #[inline(always)]
        fn lanes(self) -> [f32; 8] {
            let mut a = [0f32; 8];
            unsafe {
                _mm_storeu_ps(a.as_mut_ptr(), self.0);
                _mm_storeu_ps(a.as_mut_ptr().add(4), self.1);
            }
            a
        }
    }
}
#[cfg(target_arch = "x86_64")]
pub(crate) use x86::{Avx, Sse};

/// `(s0+s1)+(s2+s3)` for `s = (l0+l4, l1+l5, l2+l6, l3+l7)`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn reduce4_sse(s: core::arch::x86_64::__m128) -> f32 {
    use core::arch::x86_64::*;
    let t = _mm_add_ps(s, _mm_shuffle_ps::<0b10_11_00_01>(s, s));
    _mm_cvtss_f32(_mm_add_ss(t, _mm_movehl_ps(t, t)))
}

// ----------------------------------------------------------------------------------------
// AArch64 NEON (IEEE-compliant, part of the baseline)
// ----------------------------------------------------------------------------------------
#[cfg(target_arch = "aarch64")]
mod arm64 {
    use super::V8;
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub(crate) struct Neon(float32x4_t, float32x4_t);

    /// Four lanes of `fpmath::exp` (NEON).
    #[inline(always)]
    unsafe fn exp4(x: float32x4_t) -> float32x4_t {
        let s = |v: f32| vdupq_n_f32(v);
        let x = vbslq_f32(vcgeq_f32(x, s(-80.0)), x, s(-80.0));
        let x = vbslq_f32(vcleq_f32(x, s(80.0)), x, s(80.0));
        let t = vaddq_f32(vmulq_f32(x, s(super::EXP_LOG2E)), s(0.5));
        let n0 = vcvtq_s32_f32(t);
        let gt = vcgtq_f32(vcvtq_f32_s32(n0), t);
        let n = vaddq_s32(n0, vreinterpretq_s32_u32(gt));
        let nf = vcvtq_f32_s32(n);
        let r = vsubq_f32(vsubq_f32(x, vmulq_f32(nf, s(super::EXP_LN2_HI))), vmulq_f32(nf, s(super::EXP_LN2_LO)));
        let c = super::EXP_C;
        let mut p = vaddq_f32(s(c[2]), vmulq_f32(r, s(c[3])));
        p = vaddq_f32(s(c[1]), vmulq_f32(r, p));
        p = vaddq_f32(s(c[0]), vmulq_f32(r, p));
        p = vaddq_f32(s(0.5), vmulq_f32(r, p));
        p = vaddq_f32(s(1.0), vmulq_f32(r, p));
        p = vaddq_f32(s(1.0), vmulq_f32(r, p));
        let scale = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n, vdupq_n_s32(127))));
        vmulq_f32(p, scale)
    }

    impl V8 for Neon {
        #[inline(always)]
        unsafe fn load(p: *const f32) -> Self {
            Neon(vld1q_f32(p), vld1q_f32(p.add(4)))
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut f32) {
            vst1q_f32(p, self.0);
            vst1q_f32(p.add(4), self.1);
        }
        #[inline(always)]
        fn splat(x: f32) -> Self {
            unsafe { Neon(vdupq_n_f32(x), vdupq_n_f32(x)) }
        }
        #[inline(always)]
        fn add(self, b: Self) -> Self {
            unsafe { Neon(vaddq_f32(self.0, b.0), vaddq_f32(self.1, b.1)) }
        }
        #[inline(always)]
        fn sub(self, b: Self) -> Self {
            unsafe { Neon(vsubq_f32(self.0, b.0), vsubq_f32(self.1, b.1)) }
        }
        #[inline(always)]
        fn mul(self, b: Self) -> Self {
            unsafe { Neon(vmulq_f32(self.0, b.0), vmulq_f32(self.1, b.1)) }
        }
        #[inline(always)]
        fn div(self, b: Self) -> Self {
            unsafe { Neon(vdivq_f32(self.0, b.0), vdivq_f32(self.1, b.1)) }
        }
        #[inline(always)]
        fn sqrt(self) -> Self {
            unsafe { Neon(vsqrtq_f32(self.0), vsqrtq_f32(self.1)) }
        }
        #[inline(always)]
        fn zero_below(self, thr: Self) -> Self {
            unsafe {
                let z = vdupq_n_f32(0.0);
                Neon(
                    vbslq_f32(vcltq_f32(self.0, thr.0), z, self.0),
                    vbslq_f32(vcltq_f32(self.1, thr.1), z, self.1),
                )
            }
        }
        #[inline(always)]
        fn clamp_below(self, lo: Self) -> Self {
            unsafe {
                Neon(vbslq_f32(vcltq_f32(self.0, lo.0), lo.0, self.0), vbslq_f32(vcltq_f32(self.1, lo.1), lo.1, self.1))
            }
        }
        #[inline(always)]
        fn neg(self) -> Self {
            unsafe { Neon(vnegq_f32(self.0), vnegq_f32(self.1)) }
        }
        #[inline(always)]
        fn reduce(self) -> f32 {
            unsafe {
                let s = vaddq_f32(self.0, self.1);
                (vgetq_lane_f32::<0>(s) + vgetq_lane_f32::<1>(s)) + (vgetq_lane_f32::<2>(s) + vgetq_lane_f32::<3>(s))
            }
        }
        #[inline(always)]
        fn exp(self) -> Self {
            unsafe { Neon(exp4(self.0), exp4(self.1)) }
        }
        #[inline(always)]
        fn lanes(self) -> [f32; 8] {
            let mut a = [0f32; 8];
            unsafe {
                vst1q_f32(a.as_mut_ptr(), self.0);
                vst1q_f32(a.as_mut_ptr().add(4), self.1);
            }
            a
        }
    }
}
#[cfg(target_arch = "aarch64")]
pub(crate) use arm64::Neon;

// ----------------------------------------------------------------------------------------
// WebAssembly SIMD128 (deterministic subset; relaxed-simd is never used)
// ----------------------------------------------------------------------------------------
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use super::V8;
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    pub(crate) struct Wasm(v128, v128);

    /// Four lanes of `fpmath::exp` (SIMD128).
    #[inline(always)]
    fn exp4(x: v128) -> v128 {
        let s = f32x4_splat;
        let x = v128_bitselect(x, s(-80.0), f32x4_ge(x, s(-80.0)));
        let x = v128_bitselect(x, s(80.0), f32x4_le(x, s(80.0)));
        let t = f32x4_add(f32x4_mul(x, s(super::EXP_LOG2E)), s(0.5));
        let n0 = i32x4_trunc_sat_f32x4(t);
        let gt = f32x4_gt(f32x4_convert_i32x4(n0), t);
        let n = i32x4_add(n0, gt);
        let nf = f32x4_convert_i32x4(n);
        let r = f32x4_sub(f32x4_sub(x, f32x4_mul(nf, s(super::EXP_LN2_HI))), f32x4_mul(nf, s(super::EXP_LN2_LO)));
        let c = super::EXP_C;
        let mut p = f32x4_add(s(c[2]), f32x4_mul(r, s(c[3])));
        p = f32x4_add(s(c[1]), f32x4_mul(r, p));
        p = f32x4_add(s(c[0]), f32x4_mul(r, p));
        p = f32x4_add(s(0.5), f32x4_mul(r, p));
        p = f32x4_add(s(1.0), f32x4_mul(r, p));
        p = f32x4_add(s(1.0), f32x4_mul(r, p));
        let scale = i32x4_shl(i32x4_add(n, i32x4_splat(127)), 23);
        f32x4_mul(p, scale)
    }

    impl V8 for Wasm {
        #[inline(always)]
        unsafe fn load(p: *const f32) -> Self {
            Wasm(v128_load(p as *const v128), v128_load(p.add(4) as *const v128))
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut f32) {
            v128_store(p as *mut v128, self.0);
            v128_store(p.add(4) as *mut v128, self.1);
        }
        #[inline(always)]
        fn splat(x: f32) -> Self {
            Wasm(f32x4_splat(x), f32x4_splat(x))
        }
        #[inline(always)]
        fn add(self, b: Self) -> Self {
            Wasm(f32x4_add(self.0, b.0), f32x4_add(self.1, b.1))
        }
        #[inline(always)]
        fn sub(self, b: Self) -> Self {
            Wasm(f32x4_sub(self.0, b.0), f32x4_sub(self.1, b.1))
        }
        #[inline(always)]
        fn mul(self, b: Self) -> Self {
            Wasm(f32x4_mul(self.0, b.0), f32x4_mul(self.1, b.1))
        }
        #[inline(always)]
        fn div(self, b: Self) -> Self {
            Wasm(f32x4_div(self.0, b.0), f32x4_div(self.1, b.1))
        }
        #[inline(always)]
        fn sqrt(self) -> Self {
            Wasm(f32x4_sqrt(self.0), f32x4_sqrt(self.1))
        }
        #[inline(always)]
        fn zero_below(self, thr: Self) -> Self {
            Wasm(
                v128_andnot(self.0, f32x4_lt(self.0, thr.0)),
                v128_andnot(self.1, f32x4_lt(self.1, thr.1)),
            )
        }
        #[inline(always)]
        fn clamp_below(self, lo: Self) -> Self {
            Wasm(
                v128_bitselect(lo.0, self.0, f32x4_lt(self.0, lo.0)),
                v128_bitselect(lo.1, self.1, f32x4_lt(self.1, lo.1)),
            )
        }
        #[inline(always)]
        fn neg(self) -> Self {
            Wasm(f32x4_neg(self.0), f32x4_neg(self.1))
        }
        #[inline(always)]
        fn reduce(self) -> f32 {
            let s = f32x4_add(self.0, self.1);
            (f32x4_extract_lane::<0>(s) + f32x4_extract_lane::<1>(s)) + (f32x4_extract_lane::<2>(s) + f32x4_extract_lane::<3>(s))
        }
        #[inline(always)]
        fn exp(self) -> Self {
            Wasm(exp4(self.0), exp4(self.1))
        }
        #[inline(always)]
        fn lanes(self) -> [f32; 8] {
            let mut a = [0f32; 8];
            unsafe {
                v128_store(a.as_mut_ptr() as *mut v128, self.0);
                v128_store(a.as_mut_ptr().add(4) as *mut v128, self.1);
            }
            a
        }
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) use wasm::Wasm;

/// The best backend usable without runtime detection (used outside dispatched kernels).
#[cfg(target_arch = "x86_64")]
pub(crate) type Base = Sse;
#[cfg(target_arch = "aarch64")]
pub(crate) type Base = Neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) type Base = Wasm;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
pub(crate) type Base = Scalar;

/// Instantiates `$body` (which refers to the backend type as `$V`) for AVX2 when the CPU
/// supports it, else for the baseline backend.
macro_rules! dispatch {
    ($(#[$meta:meta])* $vis:vis fn $name:ident<$V:ident>($($arg:ident : $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block) => {
        $(#[$meta])*
        #[inline]
        #[allow(clippy::too_many_arguments)]
        $vis fn $name($($arg: $ty),*) $(-> $ret)? {
            #[inline(always)]
            #[allow(clippy::too_many_arguments, non_snake_case)]
            fn imp<$V: $crate::simd::V8>($($arg: $ty),*) $(-> $ret)? $body
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2")]
                #[allow(clippy::too_many_arguments)]
                unsafe fn avx2($($arg: $ty),*) $(-> $ret)? { imp::<$crate::simd::Avx>($($arg),*) }
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: the CPU supports AVX2.
                    return unsafe { avx2($($arg),*) };
                }
            }
            imp::<$crate::simd::Base>($($arg),*)
        }
    };
}
pub(crate) use dispatch;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn check<V: V8>() {
        let mut rng = Rng::new(3);
        for _ in 0..1000 {
            let a: [f32; 8] = std::array::from_fn(|_| rng.uniform() * 4.0);
            let b: [f32; 8] = std::array::from_fn(|_| rng.uniform() * 4.0 + 4.5);
            let (va, vb) = unsafe { (V::load(a.as_ptr()), V::load(b.as_ptr())) };
            let (sa, sb) = (Scalar(a), Scalar(b));
            assert_eq!(va.add(vb).lanes(), sa.add(sb).lanes());
            assert_eq!(va.sub(vb).lanes(), sa.sub(sb).lanes());
            assert_eq!(va.mul(vb).lanes(), sa.mul(sb).lanes());
            assert_eq!(va.div(vb).lanes(), sa.div(sb).lanes());
            assert_eq!(vb.sqrt().lanes(), sb.sqrt().lanes());
            let t = V::splat(0.5);
            assert_eq!(va.zero_below(t).lanes(), sa.zero_below(Scalar::splat(0.5)).lanes());
        }
        // exp / reduce / clamp / neg against the scalar definitions, incl. edge cases.
        let mut xs: Vec<f32> = vec![
            f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -80.0, 80.0, -80.000_01, 80.000_01, 0.0, -0.0,
            -70.0, 1e-30, -1e-30, 88.7, -103.0, 1e9, -1e9, 0.346_573_6, -0.346_573_6, 0.693_147_2,
        ];
        for _ in 0..20000 {
            xs.push(rng.uniform() * 100.0);
        }
        for ch in xs.chunks(8).filter(|c| c.len() == 8) {
            let v = unsafe { V::load(ch.as_ptr()) };
            let want: Vec<u32> = ch.iter().map(|&x| crate::fpmath::exp(x).to_bits()).collect();
            let got: Vec<u32> = v.exp().lanes().iter().map(|x| x.to_bits()).collect();
            assert_eq!(got, want, "exp {ch:?}");
            let arr: [f32; 8] = ch.try_into().unwrap();
            assert_eq!(v.reduce().to_bits(), reduce8(arr).to_bits());
            assert_eq!(
                v.clamp_below(V::splat(-70.0)).lanes().map(f32::to_bits),
                Scalar(arr).clamp_below(Scalar::splat(-70.0)).lanes().map(f32::to_bits)
            );
            assert_eq!(v.neg().lanes().map(f32::to_bits), arr.map(|x| (-x).to_bits()));
        }
        let n = unsafe { V::load([f32::NAN, 0.0, 1.0, -1.0, 1e-31, 2e-30, 0.5, f32::INFINITY].as_ptr()) };
        let z = n.zero_below(V::splat(1e-30)).lanes();
        assert!(z[0].is_nan());
        assert_eq!(&z[1..], &[0.0, 1.0, 0.0, 0.0, 2e-30, 0.5, f32::INFINITY]);
    }

    #[test]
    fn backends_match_scalar() {
        check::<Base>();
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                #[target_feature(enable = "avx2")]
                unsafe fn run() {
                    check::<Avx>()
                }
                unsafe { run() }
            }
        }
    }
}
