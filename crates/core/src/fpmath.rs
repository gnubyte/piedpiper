//! Deterministic transcendental functions.
//!
//! The platform `libm` (`f32::exp`, `f64::sin`, …) differs between operating systems,
//! C libraries and JavaScript engines, so the codec never calls it. Everything here is
//! built from IEEE-754 `+ - * /`, `sqrt`, integer conversions and bit casts, which Rust
//! (no fast-math, no FMA contraction) guarantees to evaluate identically everywhere.
//!
//! Accuracy is ~1 ulp-ish — plenty for a neural network; what matters is that the
//! encoder and decoder get the *same* answer.
//!
//! The constants are spelled out on purpose: they are part of the format (changing a
//! single bit of one changes every compressed stream), so they must never silently
//! follow `std::f32::consts`.
#![allow(clippy::approx_constant, clippy::excessive_precision)]

/// Lower clamp applied to softmax exponents. `exp(-70) ≈ 4e-31`, which keeps every
/// probability (even divided by a 65 536-symbol sum) a normal float — no subnormals.
pub const EXP_FLOOR: f32 = -70.0;

/// `e^x` for f32. Inputs are clamped to `[-80, 80]` (NaN maps to -80), so the result is
/// always a finite, normal number.
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    let x = if x >= -80.0 {
        if x <= 80.0 {
            x
        } else {
            80.0
        }
    } else {
        -80.0
    };
    // n = round(x / ln 2)
    let t = x * 1.442_695_f32 + 0.5;
    let mut n = t as i32;
    if (n as f32) > t {
        n -= 1;
    }
    let nf = n as f32;
    // Cody–Waite reduction: r = x - n ln2, |r| <= ln2/2.
    let r = (x - nf * 0.693_145_75_f32) - nf * 1.428_606_8e-6_f32;
    // Taylor series to r^6: relative error < 2e-7 on |r| <= 0.347.
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (0.166_666_67
                    + r * (0.041_666_668 + r * (0.008_333_334 + r * 0.001_388_888_9)))));
    p * f32::from_bits(((n + 127) as u32) << 23)
}

/// Logistic sigmoid; never returns a subnormal.
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}

#[inline(always)]
fn floor_f64(x: f64) -> f64 {
    let n = x as i64;
    let nf = n as f64;
    if nf > x {
        nf - 1.0
    } else {
        nf
    }
}

/// `(sin x, cos x)` for `0 <= x < 2^50`, deterministic on every platform.
pub fn sincos(x: f64) -> (f64, f64) {
    const FRAC_2_PI: f64 = 0.636_619_772_367_581_4;
    // pi/2 split into three 33-bit pieces (fdlibm pio2_1 / pio2_2 / pio2_3).
    const P1: f64 = 1.570_796_326_734_125_6;
    const P2: f64 = 6.077_100_506_303_966e-11;
    const P3: f64 = 2.022_266_248_711_166_5e-21;
    let k = floor_f64(x * FRAC_2_PI + 0.5);
    let r = ((x - k * P1) - k * P2) - k * P3;
    let r2 = r * r;
    let s = r
        + r * r2
            * (-0.166_666_666_666_666_66
                + r2 * (0.008_333_333_333_333_333
                    + r2 * (-0.000_198_412_698_412_698_4
                        + r2 * (2.755_731_922_398_589e-6
                            + r2 * (-2.505_210_838_544_172e-8 + r2 * 1.605_904_383_682_161_3e-10)))));
    let c = 1.0
        + r2 * (-0.5
            + r2 * (0.041_666_666_666_666_664
                + r2 * (-0.001_388_888_888_888_889
                    + r2 * (2.480_158_730_158_730_2e-5
                        + r2 * (-2.755_731_922_398_589e-7 + r2 * 2.087_675_698_786_809_7e-9)))));
    match (k as i64) & 3 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// RoPE inverse frequencies `base^(-2i/hd)` for `i in 0..hd/2`, computed with
/// multiplications only (bisection for the ratio), hence deterministic.
pub fn rope_theta(base: f64, hd: usize) -> Vec<f64> {
    let half = hd / 2;
    let target = 1.0 / base;
    // ratio^half == 1/base
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let mut p = 1.0f64;
        for _ in 0..half {
            p *= mid;
        }
        if p > target {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let ratio = 0.5 * (lo + hi);
    let mut out = Vec::with_capacity(half);
    let mut th = 1.0f64;
    for _ in 0..half {
        out.push(th);
        th *= ratio;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_is_accurate() {
        let mut x = -80.0f32;
        while x <= 80.0 {
            let a = exp(x) as f64;
            let b = (x as f64).exp();
            assert!(((a - b) / b).abs() < 1e-6, "x={x} got {a} want {b}");
            x += 0.0137;
        }
        assert_eq!(exp(-1000.0), exp(-80.0));
        assert_eq!(exp(f32::NAN), exp(-80.0));
        assert!(exp(EXP_FLOOR).is_normal());
    }

    #[test]
    fn sincos_is_accurate() {
        for i in 0..200_000u64 {
            let x = i as f64 * 0.731 + (i * i) as f64 * 1e-3;
            let (s, c) = sincos(x);
            // Reduction error grows with |x| (a few ulps of x); irrelevant for RoPE.
            let tol = 1e-10 + 1e-15 * x;
            assert!((s - x.sin()).abs() < tol, "sin {x}");
            assert!((c - x.cos()).abs() < tol, "cos {x}");
        }
    }

    #[test]
    fn theta_matches_powf() {
        let th = rope_theta(10000.0, 32);
        for (i, t) in th.iter().enumerate() {
            let want = 10000f64.powf(-2.0 * i as f64 / 32.0);
            assert!((t - want).abs() / want < 1e-12);
        }
    }
}
