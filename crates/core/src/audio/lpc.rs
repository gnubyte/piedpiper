//! Block predictors: forward-adaptive linear prediction, fitted by the encoder to one block
//! of one channel and sent in the stream.
//!
//! The adaptive cascade has to converge, and its constant adaptation adds noise. On nearly
//! noise-free synthetic sound (tones, sweeps) and on clips too short to learn from, a
//! predictor fitted to the block itself does far better, as in FLAC. The encoder fits one per
//! block and channel and uses it only where its residuals are cheaper than the cascade's.
//!
//! The analysis is deterministic like everything else (the golden vectors require identical
//! output from every encoder): integer windowed autocorrelation, binary64 Levinson-Durbin
//! with `+ - * /` only, and quantisation by exact operations.

/// Frames per block.
pub(super) const BLOCK: usize = 4096;
/// Largest order, coefficient width (bits, two's complement) and shift.
pub(super) const MAX_ORDER: usize = 32;
pub(super) const COEF_BITS: u32 = 15;
pub(super) const MAX_SHIFT: u32 = 15;
/// Predictions are clamped like the cascade's, so residuals stay below 2^21.
const LIMIT: i64 = 1 << 20;
/// Orders the encoder tries, besides the fixed predictors (the format allows up to
/// `MAX_ORDER`; higher orders gained nothing measurable).
const ORDERS: [usize; 3] = [4, 8, 16];
const SEARCH_ORDER: usize = 16;
/// Header bits of a block: the flag, then (for a block predictor) order, shift, coefficients.
const ORDER_BITS: u32 = 6;
const SHIFT_BITS: u32 = 4;

/// `x[n] ≈ clamp((Σ q[j]·x[n−1−j]) >> shift)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Lpc {
    pub q: Vec<i64>,
    pub shift: u32,
}

impl Lpc {
    /// Prediction from `hist`, the channel's earlier values, most recent first.
    #[inline]
    pub(super) fn predict(&self, hist: &[i64]) -> i64 {
        let mut s = 0i64;
        for (q, x) in self.q.iter().zip(hist) {
            s += q * x;
        }
        (s >> self.shift).clamp(-LIMIT, LIMIT)
    }

    /// Header bits after the flag.
    pub(super) fn bits(&self) -> u64 {
        (ORDER_BITS + SHIFT_BITS) as u64 + self.q.len() as u64 * COEF_BITS as u64
    }
}

/// Recent values of one channel, most recent first (zeros before the first frame).
#[derive(Clone)]
pub(super) struct History {
    buf: Vec<i64>,
    pos: usize,
}

impl History {
    pub(super) fn new() -> Self {
        History { buf: vec![0; 2 * MAX_ORDER], pos: MAX_ORDER }
    }

    #[inline]
    pub(super) fn get(&self) -> &[i64] {
        &self.buf[self.pos..self.pos + MAX_ORDER]
    }

    #[inline]
    pub(super) fn push(&mut self, v: i64) {
        if self.pos == 0 {
            self.buf.copy_within(0..MAX_ORDER - 1, MAX_ORDER + 1);
            self.pos = MAX_ORDER + 1;
        }
        self.pos -= 1;
        self.buf[self.pos] = v;
    }
}

fn zigzag(r: i64) -> u64 {
    ((r << 1) ^ (r >> 63)) as u64
}

/// Estimated bits to Rice-code `res` (FLAC's partitioned scheme, with escapes): the yardstick
/// for choosing between predictors.
fn rice_bits(res: &[i64]) -> u64 {
    let u: Vec<u64> = res.iter().map(|&r| zigzag(r)).collect();
    let n = u.len();
    let mut best = u64::MAX;
    for po in 0..=6u32 {
        let parts = 1usize << po;
        if n == 0 || (po > 0 && !n.is_multiple_of(parts)) {
            break;
        }
        let mut total = 6u64;
        for seg in u.chunks(n / parts) {
            let mean = seg.iter().sum::<u64>() / seg.len() as u64;
            let guess = if mean == 0 { 0 } else { 63 - mean.leading_zeros() as i64 };
            let mut seg_best = u64::MAX;
            for k in (guess - 1).clamp(0, 30)..=(guess + 1).clamp(0, 30) {
                let c = seg.len() as u64 * (k as u64 + 1) + seg.iter().map(|&v| v >> k).sum::<u64>();
                seg_best = seg_best.min(c + if k > 14 { 5 } else { 4 });
            }
            let width = 64 - seg.iter().max().copied().unwrap_or(0).leading_zeros() as u64;
            total += seg_best.min(9 + seg.len() as u64 * width);
        }
        best = best.min(total);
    }
    if n == 0 {
        0
    } else {
        best
    }
}

/// Autocorrelation of `x` under a Welch window, in integers (window in Q15).
fn autocorrelation(x: &[i64], max_lag: usize) -> Vec<i64> {
    let n = x.len() as i64;
    let nn = n * n;
    let xw: Vec<i64> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let d = 2 * i as i64 + 1 - n;
            (v * (((nn - d * d) << 15) / nn)) >> 15
        })
        .collect();
    (0..=max_lag).map(|k| (k..xw.len()).map(|i| xw[i] * xw[i - k]).sum()).collect()
}

/// Levinson-Durbin: the coefficients of every order up to `p`, stopping early if the
/// prediction error vanishes.
fn levinson(r: &[i64], p: usize) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    if r[0] <= 0 {
        return out;
    }
    let rf: Vec<f64> = r.iter().map(|&v| v as f64).collect();
    let mut a = vec![0.0f64; p + 1];
    let mut err = rf[0];
    for i in 1..=p {
        let mut acc = rf[i];
        for j in 1..i {
            acc -= a[j] * rf[i - j];
        }
        let k = acc / err;
        let prev = a.clone();
        a[i] = k;
        for j in 1..i {
            a[j] = prev[j] - k * prev[i - j];
        }
        err *= 1.0 - k * k;
        out.push(a[1..=i].to_vec());
        if err.is_nan() || err <= 0.0 {
            break;
        }
    }
    out
}

/// Rounds half away from zero with truncation and subtraction only (both exact).
fn round_half_away(x: f64) -> i64 {
    let t = x as i64;
    let f = x - t as f64;
    if f >= 0.5 {
        t + 1
    } else if f <= -0.5 {
        t - 1
    } else {
        t
    }
}

/// Quantises coefficients to `COEF_BITS`, carrying each rounding error into the next.
fn quantize(a: &[f64]) -> Option<Lpc> {
    let cmax = a.iter().fold(0.0f64, |m, &c| if c.abs() > m { c.abs() } else { m });
    if !cmax.is_finite() || cmax <= 0.0 {
        return None;
    }
    // floor(log2(cmax)) from the exponent bits (cmax is normal here).
    let log2 = ((cmax.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let shift = (COEF_BITS as i32 - 2 - log2).clamp(0, MAX_SHIFT as i32) as u32;
    let qmax = (1i64 << (COEF_BITS - 1)) - 1;
    let scale = (1u64 << shift) as f64;
    let mut q = Vec::with_capacity(a.len());
    let mut carry = 0.0f64;
    for &c in a {
        carry += c * scale;
        let v = round_half_away(carry).clamp(-qmax - 1, qmax);
        q.push(v);
        carry -= v as f64;
    }
    Some(Lpc { q, shift })
}

/// Residuals of `x[start..end]` under `lpc` (earlier values of `x` as history; zeros before
/// the first frame).
pub(super) fn block_residuals(lpc: &Lpc, x: &[i64], start: usize, end: usize) -> Vec<i64> {
    let mut hist = History::new();
    for &v in &x[start.saturating_sub(MAX_ORDER)..start] {
        hist.push(v);
    }
    (start..end)
        .map(|i| {
            let r = x[i] - lpc.predict(hist.get());
            hist.push(x[i]);
            r
        })
        .collect()
}

/// The block predictor to use for `x[start..end]` (one channel's values), or `None` if the
/// cascade's residuals there (`cascade`) are cheaper, by the Rice estimate plus header bits.
pub(super) fn choose(x: &[i64], start: usize, end: usize, cascade: &[i64]) -> Option<Lpc> {
    let n = end - start;
    if n < 2 {
        return None;
    }
    let mut cands = vec![Lpc { q: vec![], shift: 0 }, Lpc { q: vec![1], shift: 0 }, Lpc { q: vec![2, -1], shift: 0 }];
    let r = autocorrelation(&x[start..end], SEARCH_ORDER.min(n - 1));
    let sets = levinson(&r, SEARCH_ORDER.min(n - 1));
    for &order in &ORDERS {
        if let Some(lpc) = sets.get(order - 1).and_then(|a| quantize(a)) {
            cands.push(lpc);
        }
    }
    let mut best: Option<(Lpc, u64)> = None;
    for lpc in cands {
        let cost = rice_bits(&block_residuals(&lpc, x, start, end)) + lpc.bits();
        if best.as_ref().is_none_or(|(_, c)| cost < *c) {
            best = Some((lpc, cost));
        }
    }
    best.filter(|(_, cost)| *cost < rice_bits(cascade)).map(|(lpc, _)| lpc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_keeps_the_most_recent_values() {
        let mut h = History::new();
        for v in 1..=200 {
            h.push(v);
            let g = h.get();
            assert_eq!(g[0], v);
            assert_eq!(g[MAX_ORDER - 1], (v - MAX_ORDER as i64 + 1).max(0));
        }
    }

    #[test]
    fn a_pure_tone_takes_a_block_predictor() {
        // A tone from an integer oscillator: a 2-tap predictor is nearly exact, while the
        // cascade's residuals (stand-in: the first difference) are large.
        let (mut a, mut b) = (20000i64, 0i64);
        let x: Vec<i64> = (0..BLOCK)
            .map(|_| {
                a -= (b * 301) >> 12;
                b += (a * 301) >> 12;
                a
            })
            .collect();
        let diff: Vec<i64> = (0..BLOCK).map(|i| x[i] - if i > 0 { x[i - 1] } else { 0 }).collect();
        let lpc = choose(&x, 0, BLOCK, &diff).expect("block predictor chosen");
        let res = block_residuals(&lpc, &x, 0, BLOCK);
        assert!(res[64..].iter().all(|r| r.abs() < 64), "{:?}", &res[64..80]);
        // Silence never needs one.
        assert_eq!(choose(&[0; 500], 0, 500, &[0; 500]), None);
    }

    #[test]
    fn rice_bits_handle_extremes() {
        assert_eq!(rice_bits(&[]), 0);
        let loud: Vec<i64> = (0..4096).map(|i| if i % 2 == 0 { 1 << 20 } else { -(1 << 20) }).collect();
        let bits = rice_bits(&loud);
        assert!((4096 * 21..4096 * 23).contains(&bits), "{bits}");
    }
}
