//! Sample predictors of the audio mode.
//!
//! Both predictors work per channel on first differences `e0 = x - x_prev` and predict `e0`
//! with a cascade of normalised-LMS filters: the first stage predicts `e0` from its own past,
//! every later stage predicts the previous stage's error from that error's past, and the
//! prediction is the sum of the stage outputs. Encoder and decoder call `predict(c)` and then
//! `update(c, x)` for every sample, frame by frame and channel 0 first, so both see the same
//! state.
//!
//! Only IEEE-754 double-precision `+ - * /` in a fixed order are used, plus integers and a
//! rounding step built from truncation, so every platform computes the same bits.

/// Predictions are clamped to ±2^20, so every residual stays below 2^21 in magnitude.
const PRED_LIMIT: f64 = 1_048_576.0;

/// A per-stream sample predictor.
pub(super) trait Predict {
    /// Integer prediction of channel `c`'s next first difference.
    fn predict(&mut self, c: usize) -> i64;
    /// Feeds channel `c`'s actual sample (after `predict(c)`).
    fn update(&mut self, c: usize, x: i64);
    /// Channel `c`'s previous sample.
    fn prev(&self, c: usize) -> i64;
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

// ---------------------------------------------------------------------------- version 1

/// Taps of the version-1 cascade: stage 1 sees `K1` own and `KX` cross-channel values.
const K1: usize = 32;
const KX: usize = 16;
const N1: usize = K1 + KX;
const K2: usize = 16;
const K3: usize = 4;
const MU1: f64 = 0.003;
const MU2: f64 = 0.006;
const MU3: f64 = 0.02;

fn dot_seq(w: &[f64], h: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..w.len() {
        acc += w[i] * h[i];
    }
    acc
}

/// Normalised LMS step: `w += mu * err * h / (1 + |h|²)`.
fn nlms(w: &mut [f64], h: &[f64], err: f64, mu: f64) {
    let mut pow = 1.0;
    for &x in h {
        pow += x * x;
    }
    let g = mu * err / pow;
    for i in 0..w.len() {
        w[i] += g * h[i];
    }
}

fn shift_in(h: &mut [f64], v: f64) {
    h.copy_within(0..h.len() - 1, 1);
    h[0] = v;
}

#[derive(Clone)]
struct V1Channel {
    prev: i64,
    /// Stage-1 input: own first differences, then the other channel's (most recent first).
    h1: [f64; N1],
    w1: [f64; N1],
    h2: [f64; K2],
    w2: [f64; K2],
    h3: [f64; K3],
    w3: [f64; K3],
    /// Recent first differences, most recent first (read by the other channel).
    e0: [f64; KX],
    p1: f64,
    p2: f64,
    p3: f64,
}

/// The version-1 cascade (32 own + 16 cross-channel taps, then 16, then 4), kept to
/// decode version-1 streams.
pub(super) struct V1 {
    ch: Vec<V1Channel>,
}

impl V1 {
    pub(super) fn new(channels: usize) -> Self {
        let c = V1Channel {
            prev: 0,
            h1: [0.0; N1],
            w1: [0.0; N1],
            h2: [0.0; K2],
            w2: [0.0; K2],
            h3: [0.0; K3],
            w3: [0.0; K3],
            e0: [0.0; KX],
            p1: 0.0,
            p2: 0.0,
            p3: 0.0,
        };
        V1 { ch: vec![c; channels] }
    }
}

impl Predict for V1 {
    /// Channel 1 sees channel 0's difference of the current frame; channel 0 sees channel
    /// 1's of earlier frames.
    fn predict(&mut self, c: usize) -> i64 {
        if self.ch.len() == 2 {
            let other = self.ch[1 - c].e0;
            self.ch[c].h1[K1..].copy_from_slice(&other);
        }
        let s = &mut self.ch[c];
        s.p1 = dot_seq(&s.w1, &s.h1);
        s.p2 = dot_seq(&s.w2, &s.h2);
        s.p3 = dot_seq(&s.w3, &s.h3);
        round_half_away((s.p1 + s.p2 + s.p3).clamp(-PRED_LIMIT, PRED_LIMIT))
    }

    fn update(&mut self, c: usize, x: i64) {
        let s = &mut self.ch[c];
        let e0 = (x - s.prev) as f64;
        let e1 = e0 - s.p1;
        let e2 = e1 - s.p2;
        let e3 = e2 - s.p3;
        nlms(&mut s.w1, &s.h1, e1, MU1);
        nlms(&mut s.w2, &s.h2, e2, MU2);
        nlms(&mut s.w3, &s.h3, e3, MU3);
        shift_in(&mut s.h1[..K1], e0);
        shift_in(&mut s.h2, e1);
        shift_in(&mut s.h3, e2);
        shift_in(&mut s.e0, e0);
        s.prev = x;
    }

    fn prev(&self, c: usize) -> i64 {
        self.ch[c].prev
    }
}

// ---------------------------------------------------------------------------- version 2

/// Stage layouts (taps, step size) of the version-2 cascade, by preset: level 1, levels
/// 2–3, levels 4–9. A slow long first stage tracks the spectrum; the faster stages after it
/// follow changes and mop up what the earlier stages miss.
pub(super) const PRESETS: [&[(usize, f64)]; 3] = [
    &[(256, 0.032), (32, 0.02), (16, 0.025), (8, 0.03)],
    &[(512, 0.008), (256, 0.032), (32, 0.02), (16, 0.025), (8, 0.03)],
    &[(1024, 0.008), (256, 0.032), (32, 0.02), (16, 0.025), (8, 0.03)],
];

/// Step sizes start `1 + BOOST` times larger and decay towards their base values over the
/// first few multiples of `BOOST_FRAMES`, so short files converge quickly.
const BOOST: f64 = 4.0;
const BOOST_FRAMES: f64 = 2048.0;

/// Dot products accumulate element `i` into partial sum `i % LANES`, add the partial sums
/// pairwise (a balanced tree), then add any leftover elements in order. The fixed order makes
/// the result identical everywhere; the independent sums keep the adds pipelined, and map
/// onto SIMD lanes when the compiler vectorises the loop (which never changes the result).
const LANES: usize = 8;

#[inline]
fn tree(a: &[f64; LANES]) -> f64 {
    let mut v = *a;
    let mut n = LANES;
    while n > 1 {
        n /= 2;
        for i in 0..n {
            v[i] += v[i + n];
        }
    }
    v[0]
}

/// `w += g * old`, then returns `w · new` (summed as described at [`LANES`]).
#[inline]
fn step_dot(w: &mut [f64], g: f64, old: &[f64], new: &[f64]) -> f64 {
    let (wc, wr) = w.as_chunks_mut::<LANES>();
    let (oc, or) = old.as_chunks::<LANES>();
    let (nc, nr) = new.as_chunks::<LANES>();
    let mut d = [0.0f64; LANES];
    for ((a, o), n) in wc.iter_mut().zip(oc).zip(nc) {
        for k in 0..LANES {
            a[k] += g * o[k];
            d[k] += a[k] * n[k];
        }
    }
    let mut s = tree(&d);
    for ((a, o), n) in wr.iter_mut().zip(or).zip(nr) {
        *a += g * o;
        s += *a * n;
    }
    s
}

/// [`step_dot`] that also returns `new · new`, summed the same way.
#[inline]
fn step_dot_pow(w: &mut [f64], g: f64, old: &[f64], new: &[f64]) -> (f64, f64) {
    let (wc, wr) = w.as_chunks_mut::<LANES>();
    let (oc, or) = old.as_chunks::<LANES>();
    let (nc, nr) = new.as_chunks::<LANES>();
    let (mut d, mut p) = ([0.0f64; LANES], [0.0f64; LANES]);
    for ((a, o), n) in wc.iter_mut().zip(oc).zip(nc) {
        for k in 0..LANES {
            a[k] += g * o[k];
            d[k] += a[k] * n[k];
            p[k] += n[k] * n[k];
        }
    }
    let (mut ds, mut ps) = (tree(&d), tree(&p));
    for ((a, o), n) in wr.iter_mut().zip(or).zip(nr) {
        *a += g * o;
        ds += *a * n;
        ps += n * n;
    }
    (ds, ps)
}

/// The last `n` values pushed, most recent first, as one contiguous slice. Values are written
/// downwards through spare room; when it runs out the window moves back to the top (one copy
/// per `n` pushes instead of a shift per push). Right after a push, the window before it is
/// still readable one slot further up.
struct History {
    buf: Vec<f64>,
    pos: usize,
    n: usize,
}

impl History {
    fn new(n: usize) -> Self {
        let room = n.max(64);
        History { buf: vec![0.0; n + room], pos: room, n }
    }

    #[inline]
    fn get(&self) -> &[f64] {
        &self.buf[self.pos..self.pos + self.n]
    }

    /// The window before the last push.
    #[inline]
    fn before(&self) -> &[f64] {
        &self.buf[self.pos + 1..self.pos + 1 + self.n]
    }

    #[inline]
    fn oldest(&self) -> f64 {
        self.buf[self.pos + self.n - 1]
    }

    #[inline]
    fn push(&mut self, v: f64) {
        if self.pos == 0 {
            let len = self.buf.len();
            self.buf.copy_within(0..self.n, len - self.n);
            self.pos = len - self.n;
        }
        self.pos -= 1;
        self.buf[self.pos] = v;
    }
}

struct Stage {
    w: Vec<f64>,
    /// Past targets of this stage: the previous stage's errors.
    h: History,
    mu: f64,
    /// This stage's prediction of its next target, and `1 + |h|²` for its next step.
    p: f64,
    pow: f64,
}

struct Channel {
    prev: i64,
    /// Past first differences (the first stage's input).
    e0: History,
    /// Exact sum of squares of `e0` (integers), for the first stage's normalisation.
    sq: i64,
    w0: Vec<f64>,
    mu0: f64,
    /// The first stage's prediction of the next first difference.
    p0: f64,
    rest: Vec<Stage>,
}

/// The version-2 cascade. Each update also computes the stage outputs for the channel's next
/// sample (one pass over the weights instead of two), so `predict` only adds them up.
pub(super) struct Cascade {
    ch: Vec<Channel>,
    frame: u64,
    scale: f64,
}

impl Cascade {
    pub(super) fn new(preset: usize, channels: usize) -> Self {
        let stages = PRESETS[preset];
        let (n0, mu0) = stages[0];
        let channel = || Channel {
            prev: 0,
            e0: History::new(n0),
            sq: 0,
            w0: vec![0.0; n0],
            mu0,
            p0: 0.0,
            rest: stages[1..]
                .iter()
                .map(|&(n, mu)| Stage { w: vec![0.0; n], h: History::new(n), mu, p: 0.0, pow: 1.0 })
                .collect(),
        };
        Cascade { ch: (0..channels).map(|_| channel()).collect(), frame: 0, scale: 1.0 + BOOST }
    }
}

impl Predict for Cascade {
    #[inline]
    fn predict(&mut self, c: usize) -> i64 {
        let s = &self.ch[c];
        let mut total = s.p0;
        for st in &s.rest {
            total += st.p;
        }
        round_half_away(total.clamp(-PRED_LIMIT, PRED_LIMIT))
    }

    #[inline]
    fn update(&mut self, c: usize, x: i64) {
        let scale = self.scale;
        let s = &mut self.ch[c];
        let e0 = x - s.prev;
        let err = e0 as f64 - s.p0;
        let g = s.mu0 * scale * err / (1.0 + s.sq as f64);
        let old = s.e0.oldest() as i64;
        s.sq += e0 * e0 - old * old;
        s.e0.push(e0 as f64);
        s.p0 = step_dot(&mut s.w0, g, s.e0.before(), s.e0.get());
        let mut target = err;
        for st in &mut s.rest {
            let err = target - st.p;
            let g = st.mu * scale * err / st.pow;
            st.h.push(target);
            let (d, p) = step_dot_pow(&mut st.w, g, st.h.before(), st.h.get());
            st.p = d;
            st.pow = 1.0 + p;
            target = err;
        }
        s.prev = x;
        if c + 1 == self.ch.len() {
            self.frame += 1;
            self.scale = 1.0 + BOOST * BOOST_FRAMES / (self.frame as f64 + BOOST_FRAMES);
        }
    }

    fn prev(&self, c: usize) -> i64 {
        self.ch[c].prev
    }
}
