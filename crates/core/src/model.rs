//! The predictor: a decoder-only Transformer in the style of TufaZip v2 — Peri-LN
//! (RMSNorm before *and* after every sub-layer), per-head QK-norm, RoPE, SwiGLU, untied
//! embeddings, no biases — trained online with Adam while it codes.
//!
//! Encoder and decoder drive the model through exactly the same call sequence:
//! [`Model::begin_segment`], then [`Model::forward_step`] once per position (all streams in
//! lockstep), [`Model::set_target`] for every coded symbol, [`Model::train_step`] after the
//! segment, and [`Model::end_segment`] to roll the attention memory forward.
//!
//! Keys/values of earlier segments are kept (computed with the weights of their time) as
//! Transformer-XL style memory; gradients flow through the current segment only.

use crate::config::{ModelConfig, TrainConfig};
use crate::fpmath::{rope_theta, sincos, EXP_FLOOR};
use crate::kernels::{axpy_v, dot, dot_v, exp_shift_v, matmul, matmul_nt, matmul_tn, sum, swiglu_bwd_v, swiglu_v};
use crate::simd::Base;
use crate::simd::dispatch;
use crate::par::{self, SyncPtr};
use crate::rng::Rng;

const NORM_EPS: f32 = 1e-6;
const ROPE_BASE: f64 = 10_000.0;
const EMB_STD: f32 = 1.0;

/// Section timers (`--features profile`); compiled out otherwise.
#[cfg(feature = "profile")]
pub mod profiler {
    use std::cell::RefCell;
    thread_local! { static T: RefCell<[f64; 12]> = const { RefCell::new([0.0; 12]) }; }
    const NAMES: [&str; 12] =
        ["forward", "fwd_matmul", "fwd_attn", "fwd_head", "code", "grads", "bwd_matmul", "bwd_attn", "adam", "-", "end_seg", "-"];
    pub fn add(i: usize, t: std::time::Instant) {
        T.with(|v| v.borrow_mut()[i] += t.elapsed().as_secs_f64());
    }
    pub fn report() {
        T.with(|v| {
            let v = v.borrow();
            let s: Vec<String> = NAMES.iter().zip(v.iter()).filter(|(n, _)| **n != "-").map(|(n, t)| format!("{n}={t:.3}")).collect();
            eprintln!("PROFILE {}", s.join(" "));
        });
    }
}

macro_rules! prof {
    ($i:expr, $e:expr) => {{
        #[cfg(feature = "profile")]
        let __t = std::time::Instant::now();
        let __r = $e;
        #[cfg(feature = "profile")]
        $crate::model::profiler::add($i, __t);
        __r
    }};
}
pub(crate) use prof;

#[derive(Clone, Copy)]
struct LayerOff {
    g1: usize,
    wqkv: usize,
    gq: usize,
    gk: usize,
    wo: usize,
    g2: usize,
    g3: usize,
    wgu: usize,
    wd: usize,
    g4: usize,
}

/// Per-layer activations of the current segment (rows are `t * n_streams + b`).
struct Acts {
    xh1: Vec<f32>,
    r1: Vec<f32>,
    a1: Vec<f32>,
    qkv: Vec<f32>,
    qh: Vec<f32>,
    qri: Vec<f32>,
    kh: Vec<f32>,
    kri: Vec<f32>,
    qr: Vec<f32>,
    kr: Vec<f32>,
    att: Vec<f32>,
    o: Vec<f32>,
    xh2: Vec<f32>,
    r2: Vec<f32>,
    xh3: Vec<f32>,
    r3: Vec<f32>,
    fin: Vec<f32>,
    gu: Vec<f32>,
    sg: Vec<f32>,
    xh4: Vec<f32>,
    r4: Vec<f32>,
}

impl Acts {
    fn new(rows: usize, d: usize, nh: usize, dff: usize, win: usize) -> Self {
        let z = |n: usize| vec![0f32; n];
        Acts {
            xh1: z(rows * d),
            r1: z(rows),
            a1: z(rows * d),
            qkv: z(rows * 3 * d),
            qh: z(rows * d),
            qri: z(rows * nh),
            kh: z(rows * d),
            kri: z(rows * nh),
            qr: z(rows * d),
            kr: z(rows * d),
            att: z(rows * nh * win),
            o: z(rows * d),
            xh2: z(rows * d),
            r2: z(rows),
            xh3: z(rows * d),
            r3: z(rows),
            fin: z(rows * d),
            gu: z(rows * 2 * dff),
            sg: z(rows * dff),
            xh4: z(rows * d),
            r4: z(rows),
        }
    }
}

pub struct Model {
    pub vocab: usize,
    d: usize,
    nh: usize,
    hd: usize,
    dff: usize,
    win: usize,
    seg: usize,
    ns: usize,
    nl: usize,
    emb: usize,
    lay: Vec<LayerOff>,
    gf: usize,
    wout: usize,
    params: Vec<f32>,
    grads: Vec<f32>,
    adam_m: Vec<f32>,
    adam_v: Vec<f32>,
    tc: TrainConfig,
    updates: u64,
    b1pow: f32,
    b2pow: f32,
    theta: Vec<f64>,
    cs: Vec<f32>,
    att_scale: f32,
    acts: Vec<Acts>,
    xhf: Vec<f32>,
    rf: Vec<f32>,
    z: Vec<f32>,
    probs: Vec<f32>,
    tok_in: Vec<u32>,
    target: Vec<u32>,
    active: Vec<bool>,
    seg_start: u64,
    hist_k: Vec<f32>,
    hist_v: Vec<f32>,
    h: Vec<f32>,
    sd: Vec<f32>,
    bh: Vec<f32>,
    bd1: Vec<f32>,
    bd2: Vec<f32>,
    bff: Vec<f32>,
    bgu: Vec<f32>,
    bqkv: Vec<f32>,
    bk: Vec<f32>,
    bv: Vec<f32>,
    batt: Vec<f32>,
}

#[inline(always)]
fn rms_inv(x: &[f32]) -> f32 {
    let ms = dot(x, x) / (x.len() as f32);
    1.0 / (ms + NORM_EPS).sqrt()
}

/// `xh = x * rinv`, `y = g * xh`; returns `rinv`.
#[inline(always)]
fn rmsnorm_fwd(x: &[f32], g: &[f32], xh: &mut [f32], y: &mut [f32]) -> f32 {
    let r = rms_inv(x);
    let n = x.len();
    let (g, xh, y) = (&g[..n], &mut xh[..n], &mut y[..n]);
    for i in 0..n {
        let v = x[i] * r;
        xh[i] = v;
        y[i] = g[i] * v;
    }
    r
}

/// Backward of `y = g * x * rinv(x)`: `dg += dy*xh`; `dx = rinv*(g*dy - xh*mean(g*dy*xh))`
/// (written, or added when `add`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn rmsnorm_bwd(
    dy: &[f32],
    xh: &[f32],
    rinv: f32,
    g: &[f32],
    dg: &mut [f32],
    dx: &mut [f32],
    add: bool,
    gdy: &mut [f32],
) {
    let n = dy.len();
    let (xh, g, dg, dx, gdy) = (&xh[..n], &g[..n], &mut dg[..n], &mut dx[..n], &mut gdy[..n]);
    for i in 0..n {
        dg[i] += dy[i] * xh[i];
        gdy[i] = g[i] * dy[i];
    }
    let m = dot(gdy, xh) / (n as f32);
    if add {
        for i in 0..n {
            dx[i] += rinv * (gdy[i] - xh[i] * m);
        }
    } else {
        for i in 0..n {
            dx[i] = rinv * (gdy[i] - xh[i] * m);
        }
    }
}

#[inline(always)]
fn rope(x: &[f32], cs: &[f32], y: &mut [f32]) {
    let half = x.len() / 2;
    let (c, s) = cs.split_at(half);
    for i in 0..half {
        let a = x[i];
        let b = x[i + half];
        y[i] = a * c[i] - b * s[i];
        y[i + half] = a * s[i] + b * c[i];
    }
}

#[inline(always)]
fn rope_inv(dy: &[f32], cs: &[f32], dx: &mut [f32]) {
    let half = dy.len() / 2;
    let (c, s) = cs.split_at(half);
    for i in 0..half {
        let a = dy[i];
        let b = dy[i + half];
        dx[i] = a * c[i] + b * s[i];
        dx[i + half] = b * c[i] - a * s[i];
    }
}

/// In-place softmax with deterministic exp and a fixed summation order.
#[inline]
pub fn softmax(v: &mut [f32]) {
    let mut m = v[0];
    for &x in v.iter() {
        if x > m {
            m = x;
        }
    }
    exp_shift_v::<Base>(v, m, EXP_FLOOR, false);
    let inv = 1.0 / sum(v);
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Everything the attention kernels need, as raw pointers so the work can be split over
/// threads by stream (each stream only touches its own rows).
#[derive(Clone, Copy)]
struct AttnCtx {
    qr: *const f32,
    kr: *const f32,
    qkv: *const f32,
    hist_k: *const f32,
    hist_v: *const f32,
    att: *mut f32,
    out: *mut f32,
    dout: *const f32,
    dk: *mut f32,
    dv: *mut f32,
    datt: *mut f32,
    d: usize,
    nh: usize,
    hd: usize,
    win: usize,
    ns: usize,
    layer: usize,
    seg_start: u64,
    scale: f32,
}
// SAFETY: tasks write disjoint rows (one stream each); reads are of finished data.
unsafe impl Send for AttnCtx {}
unsafe impl Sync for AttnCtx {}

impl AttnCtx {
    /// Key/value window of stream `b` at local position `t`: `(n_cur, n_hist, first_slot)`.
    /// Keys are visited newest first: `n_cur` from the current segment (local positions
    /// `t, t-1, …`), then `n_hist` from memory (ring slots `first_slot, first_slot-1, …`).
    #[inline(always)]
    fn window(&self, t: usize) -> (usize, usize, usize) {
        let p = self.seg_start + t as u64;
        let nk = (p + 1).min(self.win as u64) as usize;
        let n_cur = nk.min(t + 1);
        let n_hist = nk - n_cur;
        let first_slot = if n_hist > 0 { ((self.seg_start - 1) % self.win as u64) as usize } else { 0 };
        (n_cur, n_hist, first_slot)
    }

    #[inline(always)]
    unsafe fn hist_base(&self, b: usize) -> usize {
        (self.layer * self.ns + b) * self.win * self.d
    }
}

dispatch! {
    /// Forward attention for streams `b0..b1` at local position `t`.
    fn attn_forward<V>(c: AttnCtx, t: usize, b0: usize, b1: usize) {
        let (d, hd, nh, ns) = (c.d, c.hd, c.nh, c.ns);
        let (n_cur, n_hist, first_slot) = c.window(t);
        let nk = n_cur + n_hist;
        let mut acc = [0f32; 1024];
        for b in b0..b1 {
            let r = t * ns + b;
            // SAFETY: indices are within the activation / memory buffers sized at construction.
            unsafe {
                let hb = c.hist_base(b);
                for hh in 0..nh {
                    let off = hh * hd;
                    let q = std::slice::from_raw_parts(c.qr.add(r * d + off), hd);
                    let att = std::slice::from_raw_parts_mut(c.att.add((r * nh + hh) * c.win), nk);
                    let mut m = f32::NEG_INFINITY;
                    for w in 0..n_cur {
                        let rr = (t - w) * ns + b;
                        let k = std::slice::from_raw_parts(c.kr.add(rr * d + off), hd);
                        let s = dot_v::<V>(q, k) * c.scale;
                        att[w] = s;
                        if s > m {
                            m = s;
                        }
                    }
                    let mut slot = first_slot;
                    for w in n_cur..nk {
                        let k = std::slice::from_raw_parts(c.hist_k.add(hb + slot * d + off), hd);
                        let s = dot_v::<V>(q, k) * c.scale;
                        att[w] = s;
                        if s > m {
                            m = s;
                        }
                        slot = if slot == 0 { c.win - 1 } else { slot - 1 };
                    }
                    let sum_e = exp_shift_v::<V>(att, m, EXP_FLOOR, true);
                    let inv = 1.0 / sum_e;
                    let acc = &mut acc[..hd];
                    acc.fill(0.0);
                    for w in 0..n_cur {
                        att[w] *= inv;
                        let aw = att[w];
                        let rr = (t - w) * ns + b;
                        let v = std::slice::from_raw_parts(c.qkv.add(rr * 3 * d + 2 * d + off), hd);
                        axpy_v::<V>(acc, aw, v);
                    }
                    let mut slot = first_slot;
                    for w in n_cur..nk {
                        att[w] *= inv;
                        let aw = att[w];
                        let v = std::slice::from_raw_parts(c.hist_v.add(hb + slot * d + off), hd);
                        axpy_v::<V>(acc, aw, v);
                        slot = if slot == 0 { c.win - 1 } else { slot - 1 };
                    }
                    std::slice::from_raw_parts_mut(c.out.add(r * d + off), hd).copy_from_slice(acc);
                }
            }
        }
    }
}

dispatch! {
    /// Backward attention for streams `b0..b1` over local positions `0..lseg`. Writes
    /// `dq` (into `out`) and accumulates `dk`/`dv` for current-segment keys of the same
    /// stream. `active` holds one flag per row.
    fn attn_backward<V>(c: AttnCtx, lseg: usize, active: &[bool], b0: usize, b1: usize) {
        let (d, hd, nh, ns) = (c.d, c.hd, c.nh, c.ns);
        for b in b0..b1 {
            // SAFETY: stream b owns datt row b and rows t*ns+b of dq/dk/dv.
            unsafe {
                let hb = c.hist_base(b);
                let datt = std::slice::from_raw_parts_mut(c.datt.add(b * c.win), c.win);
                for t in 0..lseg {
                    let r = t * ns + b;
                    let dq_row = std::slice::from_raw_parts_mut(c.out.add(r * d), d);
                    if !active[r] {
                        dq_row.fill(0.0);
                        continue;
                    }
                    let (n_cur, n_hist, first_slot) = c.window(t);
                    let nk = n_cur + n_hist;
                    for hh in 0..nh {
                        let off = hh * hd;
                        let doh = std::slice::from_raw_parts(c.dout.add(r * d + off), hd);
                        let att = std::slice::from_raw_parts(c.att.add((r * nh + hh) * c.win), nk);
                        for w in 0..n_cur {
                            let rr = (t - w) * ns + b;
                            datt[w] = dot_v::<V>(doh, std::slice::from_raw_parts(c.qkv.add(rr * 3 * d + 2 * d + off), hd));
                        }
                        let mut slot = first_slot;
                        for w in n_cur..nk {
                            datt[w] = dot_v::<V>(doh, std::slice::from_raw_parts(c.hist_v.add(hb + slot * d + off), hd));
                            slot = if slot == 0 { c.win - 1 } else { slot - 1 };
                        }
                        let mut sad = 0f32;
                        for w in 0..nk {
                            sad += att[w] * datt[w];
                        }
                        let dq = &mut dq_row[off..off + hd];
                        dq.fill(0.0);
                        let q = std::slice::from_raw_parts(c.qr.add(r * d + off), hd);
                        for w in 0..n_cur {
                            let aw = att[w];
                            let ds = aw * (datt[w] - sad) * c.scale;
                            let rr = (t - w) * ns + b;
                            let k = std::slice::from_raw_parts(c.kr.add(rr * d + off), hd);
                            axpy_v::<V>(dq, ds, k);
                            let dk = std::slice::from_raw_parts_mut(c.dk.add(rr * d + off), hd);
                            let dv = std::slice::from_raw_parts_mut(c.dv.add(rr * d + off), hd);
                            axpy_v::<V>(dk, ds, q);
                            axpy_v::<V>(dv, aw, doh);
                        }
                        let mut slot = first_slot;
                        for w in n_cur..nk {
                            let ds = att[w] * (datt[w] - sad) * c.scale;
                            let k = std::slice::from_raw_parts(c.hist_k.add(hb + slot * d + off), hd);
                            axpy_v::<V>(dq, ds, k);
                            slot = if slot == 0 { c.win - 1 } else { slot - 1 };
                        }
                    }
                }
            }
        }
    }
}

dispatch! {
    /// Adam with β1 = 0 over one chunk (per element exactly the scalar recipe of the tail).
    fn adam_rms<V>(p: &mut [f32], v: &mut [f32], g: &[f32], gscale: f32, b2: f32, omb2: f32, bc2: f32, lr: f32, eps: f32) {
        let n = p.len().min(v.len()).min(g.len());
        let chunks = n / 8;
        let (vgs, vb2, vomb2, vbc2) = (V::splat(gscale), V::splat(b2), V::splat(omb2), V::splat(bc2));
        let (vlr, veps, thr) = (V::splat(lr), V::splat(eps), V::splat(1e-30));
        // SAFETY: chunk ranges are in bounds of all slices.
        unsafe {
            for c in 0..chunks {
                let o = c * 8;
                let gi = V::load(g.as_ptr().add(o)).mul(vgs);
                let vi = vb2.mul(V::load(v.as_ptr().add(o))).add(vomb2.mul(gi.mul(gi))).zero_below(thr);
                vi.store(v.as_mut_ptr().add(o));
                let pi = V::load(p.as_ptr().add(o));
                pi.sub(vlr.mul(gi).div(vi.div(vbc2).sqrt().add(veps))).store(p.as_mut_ptr().add(o));
            }
        }
        for i in chunks * 8..n {
            let gi = g[i] * gscale;
            let mut vi = b2 * v[i] + omb2 * (gi * gi);
            if vi < 1e-30 {
                vi = 0.0;
            }
            v[i] = vi;
            p[i] -= (lr * gi) / ((vi / bc2).sqrt() + eps);
        }
    }
}

dispatch! {
    /// Adam with momentum over one chunk.
    fn adam_full<V>(p: &mut [f32], v: &mut [f32], m: &mut [f32], g: &[f32], gscale: f32, b1: f32, omb1: f32, bc1: f32, b2: f32, omb2: f32, bc2: f32, lr: f32, eps: f32) {
        let n = p.len().min(v.len()).min(m.len()).min(g.len());
        let chunks = n / 8;
        let (vgs, vb1, vomb1, vbc1) = (V::splat(gscale), V::splat(b1), V::splat(omb1), V::splat(bc1));
        let (vb2, vomb2, vbc2) = (V::splat(b2), V::splat(omb2), V::splat(bc2));
        let (vlr, veps, thr) = (V::splat(lr), V::splat(eps), V::splat(1e-30));
        // SAFETY: chunk ranges are in bounds of all slices.
        unsafe {
            for c in 0..chunks {
                let o = c * 8;
                let gi = V::load(g.as_ptr().add(o)).mul(vgs);
                let mi = vb1.mul(V::load(m.as_ptr().add(o))).add(vomb1.mul(gi));
                mi.store(m.as_mut_ptr().add(o));
                let vi = vb2.mul(V::load(v.as_ptr().add(o))).add(vomb2.mul(gi.mul(gi))).zero_below(thr);
                vi.store(v.as_mut_ptr().add(o));
                let pi = V::load(p.as_ptr().add(o));
                pi.sub(vlr.mul(mi.div(vbc1)).div(vi.div(vbc2).sqrt().add(veps))).store(p.as_mut_ptr().add(o));
            }
        }
        for i in chunks * 8..n {
            let gi = g[i] * gscale;
            let mi = b1 * m[i] + omb1 * gi;
            m[i] = mi;
            let mut vi = b2 * v[i] + omb2 * (gi * gi);
            if vi < 1e-30 {
                vi = 0.0;
            }
            v[i] = vi;
            p[i] -= (lr * (mi / bc1)) / ((vi / bc2).sqrt() + eps);
        }
    }
}

/// Raw pointers to one layer's activation buffers (rows are written by exactly one task).
struct LayerPtrs {
    xh1: *mut f32,
    r1: *mut f32,
    a1: *mut f32,
    qkv: *mut f32,
    qh: *mut f32,
    qri: *mut f32,
    kh: *mut f32,
    kri: *mut f32,
    qr: *mut f32,
    kr: *mut f32,
    o: *mut f32,
    xh2: *mut f32,
    r2: *mut f32,
    xh3: *mut f32,
    r3: *mut f32,
    fin: *mut f32,
    gu: *mut f32,
    sg: *mut f32,
    xh4: *mut f32,
    r4: *mut f32,
}

impl LayerPtrs {
    fn new(a: &mut Acts) -> Self {
        LayerPtrs {
            xh1: a.xh1.as_mut_ptr(),
            r1: a.r1.as_mut_ptr(),
            a1: a.a1.as_mut_ptr(),
            qkv: a.qkv.as_mut_ptr(),
            qh: a.qh.as_mut_ptr(),
            qri: a.qri.as_mut_ptr(),
            kh: a.kh.as_mut_ptr(),
            kri: a.kri.as_mut_ptr(),
            qr: a.qr.as_mut_ptr(),
            kr: a.kr.as_mut_ptr(),
            o: a.o.as_mut_ptr(),
            xh2: a.xh2.as_mut_ptr(),
            r2: a.r2.as_mut_ptr(),
            xh3: a.xh3.as_mut_ptr(),
            r3: a.r3.as_mut_ptr(),
            fin: a.fin.as_mut_ptr(),
            gu: a.gu.as_mut_ptr(),
            sg: a.sg.as_mut_ptr(),
            xh4: a.xh4.as_mut_ptr(),
            r4: a.r4.as_mut_ptr(),
        }
    }
}

/// Everything one forward position needs; `rows` runs the whole network for a group of
/// streams.
struct FwdCtx<'a> {
    d: usize,
    nh: usize,
    hd: usize,
    dff: usize,
    ns: usize,
    v: usize,
    gf: usize,
    wout: usize,
    params: &'a [f32],
    lay: &'a [LayerOff],
    cs: &'a [f32],
    layers: Vec<LayerPtrs>,
    attn: Vec<AttnCtx>,
    h: *mut f32,
    sd: *mut f32,
    xhf: *mut f32,
    rf: *mut f32,
    z: *mut f32,
    probs: *mut f32,
}
// SAFETY: tasks write disjoint stream rows (see `forward_step`).
unsafe impl Send for FwdCtx<'_> {}
unsafe impl Sync for FwdCtx<'_> {}

#[inline(always)]
unsafe fn sl<'a>(p: *mut f32, off: usize, len: usize) -> &'a mut [f32] {
    std::slice::from_raw_parts_mut(p.add(off), len)
}

impl FwdCtx<'_> {
    /// Forward pass of streams `b0..b1` at local position `t`.
    unsafe fn rows(&self, t: usize, b0: usize, b1: usize) {
        let (d, nh, hd, dff, ns, v) = (self.d, self.nh, self.hd, self.dff, self.ns, self.v);
        let nb = b1 - b0;
        let ra = t * ns + b0;
        let pr = self.params;
        let h = sl(self.h, b0 * d, nb * d);
        let sd = sl(self.sd, b0 * d, nb * d);
        let mut tmp = [0f32; 1024];
        for (l, a) in self.layers.iter().enumerate() {
            let lo = self.lay[l];
            // --- attention sub-layer: h += N2(Wo · Attn(N1(h))) ---
            for i in 0..nb {
                let r = ra + i;
                *a.r1.add(r) = rmsnorm_fwd(&h[i * d..(i + 1) * d], &pr[lo.g1..lo.g1 + d], sl(a.xh1, r * d, d), sl(a.a1, r * d, d));
            }
            prof!(1, matmul(sl(a.qkv, ra * 3 * d, nb * 3 * d), sl(a.a1, ra * d, nb * d), &pr[lo.wqkv..lo.wqkv + 3 * d * d], nb, d, 3 * d));
            for i in 0..nb {
                let r = ra + i;
                let qkv = sl(a.qkv, r * 3 * d, 3 * d);
                for hh in 0..nh {
                    let o = r * d + hh * hd;
                    let q = &qkv[hh * hd..(hh + 1) * hd];
                    let ri = rms_inv(q);
                    *a.qri.add(r * nh + hh) = ri;
                    let qh = sl(a.qh, o, hd);
                    for j in 0..hd {
                        let x = q[j] * ri;
                        qh[j] = x;
                        tmp[j] = pr[lo.gq + j] * x;
                    }
                    rope(&tmp[..hd], self.cs, sl(a.qr, o, hd));
                    let k = &qkv[d + hh * hd..d + (hh + 1) * hd];
                    let ri = rms_inv(k);
                    *a.kri.add(r * nh + hh) = ri;
                    let kh = sl(a.kh, o, hd);
                    for j in 0..hd {
                        let x = k[j] * ri;
                        kh[j] = x;
                        tmp[j] = pr[lo.gk + j] * x;
                    }
                    rope(&tmp[..hd], self.cs, sl(a.kr, o, hd));
                }
            }
            prof!(2, attn_forward(self.attn[l], t, b0, b1));
            prof!(1, matmul(sd, sl(a.o, ra * d, nb * d), &pr[lo.wo..lo.wo + d * d], nb, d, d));
            let g2 = &pr[lo.g2..lo.g2 + d];
            for i in 0..nb {
                let r = ra + i;
                let x = &sd[i * d..(i + 1) * d];
                let ri = rms_inv(x);
                *a.r2.add(r) = ri;
                let xh = sl(a.xh2, r * d, d);
                let hr = &mut h[i * d..(i + 1) * d];
                for j in 0..d {
                    let val = x[j] * ri;
                    xh[j] = val;
                    hr[j] += g2[j] * val;
                }
            }
            // --- feed-forward sub-layer: h += N4(Wd · (silu(Wg x) * Wu x)), x = N3(h) ---
            for i in 0..nb {
                let r = ra + i;
                *a.r3.add(r) = rmsnorm_fwd(&h[i * d..(i + 1) * d], &pr[lo.g3..lo.g3 + d], sl(a.xh3, r * d, d), sl(a.fin, r * d, d));
            }
            prof!(1, matmul(sl(a.gu, ra * 2 * dff, nb * 2 * dff), sl(a.fin, ra * d, nb * d), &pr[lo.wgu..lo.wgu + d * 2 * dff], nb, d, 2 * dff));
            for i in 0..nb {
                let r = ra + i;
                swiglu_v::<Base>(sl(a.gu, r * 2 * dff, 2 * dff), sl(a.sg, r * dff, dff));
            }
            prof!(1, matmul(sd, sl(a.sg, ra * dff, nb * dff), &pr[lo.wd..lo.wd + dff * d], nb, dff, d));
            let g4 = &pr[lo.g4..lo.g4 + d];
            for i in 0..nb {
                let r = ra + i;
                let x = &sd[i * d..(i + 1) * d];
                let ri = rms_inv(x);
                *a.r4.add(r) = ri;
                let xh = sl(a.xh4, r * d, d);
                let hr = &mut h[i * d..(i + 1) * d];
                for j in 0..d {
                    let val = x[j] * ri;
                    xh[j] = val;
                    hr[j] += g4[j] * val;
                }
            }
        }
        for i in 0..nb {
            let r = ra + i;
            *self.rf.add(r) = rmsnorm_fwd(&h[i * d..(i + 1) * d], &pr[self.gf..self.gf + d], sl(self.xhf, r * d, d), sl(self.z, r * d, d));
        }
        prof!(3, {
            matmul(sl(self.probs, ra * v, nb * v), sl(self.z, ra * d, nb * d), &pr[self.wout..self.wout + d * v], nb, d, v);
            for i in 0..nb {
                softmax(sl(self.probs, (ra + i) * v, v));
            }
        });
    }
}

impl Model {
    pub fn new(vocab: usize, mc: &ModelConfig, tc: &TrainConfig) -> Model {
        let d = mc.d_model as usize;
        let nh = mc.n_head as usize;
        let hd = d / nh;
        let dff = mc.d_ff as usize;
        let win = mc.window as usize;
        let seg = mc.seg_len as usize;
        let ns = mc.n_streams as usize;
        let nl = mc.n_layer as usize;

        let mut off = 0usize;
        let mut take = |n: usize| {
            let o = off;
            off += n;
            o
        };
        let emb = take((vocab + 1) * d);
        let mut lay = Vec::with_capacity(nl);
        for _ in 0..nl {
            let g1 = take(d);
            let wqkv = take(d * 3 * d);
            let gq = take(hd);
            let gk = take(hd);
            let wo = take(d * d);
            let g2 = take(d);
            let g3 = take(d);
            let wgu = take(d * 2 * dff);
            let wd = take(dff * d);
            let g4 = take(d);
            lay.push(LayerOff { g1, wqkv, gq, gk, wo, g2, g3, wgu, wd, g4 });
        }
        let gf = take(d);
        let wout = take(d * vocab);
        let np = off;

        // Deterministic initialisation (fixed RNG draw order).
        let mut params = vec![0f32; np];
        let mut rng = Rng::new(tc.seed);
        let sq3 = 1.732_050_8f32;
        let mut fill = |p: &mut [f32], std: f32| {
            let a = std * sq3;
            for x in p.iter_mut() {
                *x = rng.uniform() * a;
            }
        };
        fill(&mut params[emb..emb + (vocab + 1) * d], EMB_STD);
        let s_d = tc.init_scale / (d as f32).sqrt();
        let s_ff = tc.init_scale / (dff as f32).sqrt();
        for lo in &lay {
            fill(&mut params[lo.wqkv..lo.wqkv + 3 * d * d], s_d);
            fill(&mut params[lo.wo..lo.wo + d * d], s_d);
            fill(&mut params[lo.wgu..lo.wgu + 2 * d * dff], s_d);
            fill(&mut params[lo.wd..lo.wd + dff * d], s_ff);
            for g in [lo.g1, lo.g2, lo.g3, lo.g4] {
                params[g..g + d].fill(1.0);
            }
            params[lo.gq..lo.gq + hd].fill(1.0);
            params[lo.gk..lo.gk + hd].fill(1.0);
        }
        params[gf..gf + d].fill(1.0);
        // w_out starts at zero: the very first prediction is uniform.

        let rows = seg * ns;
        Model {
            vocab,
            d,
            nh,
            hd,
            dff,
            win,
            seg,
            ns,
            nl,
            emb,
            lay,
            gf,
            wout,
            grads: vec![0f32; np],
            adam_m: if tc.beta1 > 0.0 { vec![0f32; np] } else { Vec::new() },
            adam_v: vec![0f32; np],
            params,
            tc: *tc,
            updates: 0,
            b1pow: 1.0,
            b2pow: 1.0,
            theta: rope_theta(ROPE_BASE, hd),
            cs: vec![0f32; seg * hd],
            att_scale: 1.0 / (hd as f32).sqrt(),
            acts: (0..nl).map(|_| Acts::new(rows, d, nh, dff, win)).collect(),
            xhf: vec![0f32; rows * d],
            rf: vec![0f32; rows],
            z: vec![0f32; rows * d],
            probs: vec![0f32; rows * vocab],
            tok_in: vec![0u32; rows],
            target: vec![0u32; rows],
            active: vec![false; rows],
            seg_start: 0,
            hist_k: vec![0f32; nl * ns * win * d],
            hist_v: vec![0f32; nl * ns * win * d],
            h: vec![0f32; ns * d],
            sd: vec![0f32; ns * d.max(dff)],
            bh: vec![0f32; rows * d],
            bd1: vec![0f32; rows * d],
            bd2: vec![0f32; rows * d],
            bff: vec![0f32; rows * dff],
            bgu: vec![0f32; rows * 2 * dff],
            bqkv: vec![0f32; rows * 3 * d],
            bk: vec![0f32; rows * d],
            bv: vec![0f32; rows * d],
            batt: vec![0f32; ns * win],
        }
    }

    /// Token id fed at the start of every stream.
    pub fn bos(&self) -> u32 {
        self.vocab as u32
    }

    pub fn params(&self) -> &[f32] {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut [f32] {
        &mut self.params
    }

    /// Must be called after editing parameters directly (kept for API stability; the
    /// model holds no derived copies of its weights any more).
    pub fn params_changed(&mut self) {}

    pub fn begin_segment(&mut self, seg_start: u64) {
        self.seg_start = seg_start;
    }

    /// Probability row of stream `b` at local position `t` (valid after `forward_step(t)`).
    #[inline]
    pub fn probs_row(&self, t: usize, b: usize) -> &[f32] {
        let r = t * self.ns + b;
        &self.probs[r * self.vocab..(r + 1) * self.vocab]
    }

    #[inline]
    pub fn set_target(&mut self, t: usize, b: usize, sym: u32) {
        self.target[t * self.ns + b] = sym;
    }

    fn attn_ctx(&self, l: usize) -> AttnCtx {
        let a = &self.acts[l];
        AttnCtx {
            qr: a.qr.as_ptr(),
            kr: a.kr.as_ptr(),
            qkv: a.qkv.as_ptr(),
            hist_k: self.hist_k.as_ptr(),
            hist_v: self.hist_v.as_ptr(),
            att: a.att.as_ptr() as *mut f32,
            out: std::ptr::null_mut(),
            dout: std::ptr::null(),
            dk: std::ptr::null_mut(),
            dv: std::ptr::null_mut(),
            datt: std::ptr::null_mut(),
            d: self.d,
            nh: self.nh,
            hd: self.hd,
            win: self.win,
            ns: self.ns,
            layer: l,
            seg_start: self.seg_start,
            scale: self.att_scale,
        }
    }

    /// Forward pass for local position `t` of the current segment, all streams at once.
    /// `inputs[b]` is the previous token of stream `b` (or [`Model::bos`]); inactive
    /// streams are computed but never coded or trained on. Large stream counts are split
    /// into groups that run on different threads; every row's arithmetic is unchanged.
    pub fn forward_step(&mut self, t: usize, inputs: &[u32], active: &[bool]) {
        let (d, nh, hd, ns, v) = (self.d, self.nh, self.hd, self.ns, self.vocab);
        let r0 = t * ns;
        let p = self.seg_start + t as u64;
        {
            let half = hd / 2;
            let cs = &mut self.cs[t * hd..(t + 1) * hd];
            for i in 0..half {
                let (s, c) = sincos(p as f64 * self.theta[i]);
                cs[i] = c as f32;
                cs[half + i] = s as f32;
            }
        }
        for b in 0..ns {
            let tok = inputs[b] as usize;
            self.h[b * d..(b + 1) * d].copy_from_slice(&self.params[self.emb + tok * d..self.emb + (tok + 1) * d]);
            self.tok_in[r0 + b] = inputs[b];
            self.active[r0 + b] = active[b];
        }
        let attn: Vec<AttnCtx> = (0..self.nl)
            .map(|l| {
                let mut c = self.attn_ctx(l);
                c.out = self.acts[l].o.as_ptr() as *mut f32;
                c
            })
            .collect();
        let layers: Vec<LayerPtrs> = self.acts.iter_mut().map(LayerPtrs::new).collect();
        let ctx = FwdCtx {
            d,
            nh,
            hd,
            dff: self.dff,
            ns,
            v,
            gf: self.gf,
            wout: self.wout,
            params: &self.params,
            lay: &self.lay,
            cs: &self.cs[t * hd..(t + 1) * hd],
            layers,
            attn,
            h: self.h.as_mut_ptr(),
            sd: self.sd.as_mut_ptr(),
            xhf: self.xhf.as_mut_ptr(),
            rf: self.rf.as_mut_ptr(),
            z: self.z.as_mut_ptr(),
            probs: self.probs.as_mut_ptr(),
        };
        let nk = (p + 1).min(self.win as u64) as usize;
        let per_stream = self.params.len() + self.nl * nh * nk * hd * 2;
        let ctx = &ctx;
        par::for_ranges(ns, per_stream, 4, move |b0, b1| {
            // SAFETY: each task owns streams b0..b1 — their residual rows, scratch rows and
            // activation rows — and only reads shared weights and finished keys/values.
            unsafe { ctx.rows(t, b0, b1) }
        });
    }

    /// Back-propagates the online loss of the first `lseg` positions and takes one Adam
    /// step. Returns `false` (and changes nothing) if no position was active.
    pub fn train_step(&mut self, lseg: usize) -> bool {
        if !prof!(5, self.compute_grads(lseg)) {
            return false;
        }
        prof!(8, self.adam_step());
        true
    }

    /// Mean cross-entropy gradient of the coded targets, written to `self.grads`.
    pub fn compute_grads(&mut self, lseg: usize) -> bool {
        let (d, v, nl) = (self.d, self.vocab, self.nl);
        let rows = lseg * self.ns;
        let n_act = self.active[..rows].iter().filter(|&&a| a).count();
        if n_act == 0 {
            return false;
        }
        let inv_n = 1.0 / n_act as f32;
        // Matrix gradients are overwritten by their kernels; gains and embeddings
        // accumulate, so clear everything once.
        self.grads.fill(0.0);

        // dlogits = (p - onehot) / n, in place over the probabilities.
        for r in 0..rows {
            let row = &mut self.probs[r * v..(r + 1) * v];
            if self.active[r] {
                row[self.target[r] as usize] -= 1.0;
                for x in row.iter_mut() {
                    *x *= inv_n;
                }
            } else {
                row.fill(0.0);
            }
        }
        prof!(6, matmul_tn(&mut self.grads[self.wout..self.wout + d * v], &self.z[..rows * d], &self.probs[..rows * v], rows, d, v));
        prof!(6, matmul_nt(&mut self.bd1[..rows * d], &self.probs[..rows * v], &self.params[self.wout..self.wout + d * v], rows, v, d));
        let mut gdy = vec![0f32; d.max(self.hd)];
        for r in 0..rows {
            rmsnorm_bwd(
                &self.bd1[r * d..(r + 1) * d],
                &self.xhf[r * d..(r + 1) * d],
                self.rf[r],
                &self.params[self.gf..self.gf + d],
                &mut self.grads[self.gf..self.gf + d],
                &mut self.bh[r * d..(r + 1) * d],
                false,
                &mut gdy,
            );
        }
        for l in (0..nl).rev() {
            self.layer_backward(l, lseg, &mut gdy);
        }
        for r in 0..rows {
            if !self.active[r] {
                continue;
            }
            let tok = self.tok_in[r] as usize;
            let g = &mut self.grads[self.emb + tok * d..self.emb + (tok + 1) * d];
            let dh = &self.bh[r * d..(r + 1) * d];
            for i in 0..d {
                g[i] += dh[i];
            }
        }
        true
    }

    fn layer_backward(&mut self, l: usize, lseg: usize, gdy: &mut [f32]) {
        let (d, nh, hd, dff, ns) = (self.d, self.nh, self.hd, self.dff, self.ns);
        let rows = lseg * ns;
        let lo = self.lay[l];

        // Core attention backward first needs d(out) — computed below — so build the
        // context up front (pointers into buffers that are not reallocated).
        let mut ctx = self.attn_ctx(l);

        let a = &self.acts[l];
        let pr = &self.params;
        let gr = &mut self.grads;

        // ----- feed-forward sub-layer -----
        for r in 0..rows {
            rmsnorm_bwd(
                &self.bh[r * d..(r + 1) * d],
                &a.xh4[r * d..(r + 1) * d],
                a.r4[r],
                &pr[lo.g4..lo.g4 + d],
                &mut gr[lo.g4..lo.g4 + d],
                &mut self.bd1[r * d..(r + 1) * d],
                false,
                gdy,
            );
        }
        prof!(6, matmul_tn(&mut gr[lo.wd..lo.wd + dff * d], &a.sg[..rows * dff], &self.bd1[..rows * d], rows, dff, d));
        prof!(6, matmul_nt(&mut self.bff[..rows * dff], &self.bd1[..rows * d], &pr[lo.wd..lo.wd + dff * d], rows, d, dff));
        for r in 0..rows {
            swiglu_bwd_v::<Base>(
                &a.gu[r * 2 * dff..(r + 1) * 2 * dff],
                &self.bff[r * dff..(r + 1) * dff],
                &mut self.bgu[r * 2 * dff..(r + 1) * 2 * dff],
            );
        }
        prof!(6, matmul_tn(&mut gr[lo.wgu..lo.wgu + d * 2 * dff], &a.fin[..rows * d], &self.bgu[..rows * 2 * dff], rows, d, 2 * dff));
        prof!(6, matmul_nt(&mut self.bd1[..rows * d], &self.bgu[..rows * 2 * dff], &pr[lo.wgu..lo.wgu + d * 2 * dff], rows, 2 * dff, d));
        for r in 0..rows {
            rmsnorm_bwd(
                &self.bd1[r * d..(r + 1) * d],
                &a.xh3[r * d..(r + 1) * d],
                a.r3[r],
                &pr[lo.g3..lo.g3 + d],
                &mut gr[lo.g3..lo.g3 + d],
                &mut self.bh[r * d..(r + 1) * d],
                true,
                gdy,
            );
        }

        // ----- attention sub-layer -----
        for r in 0..rows {
            rmsnorm_bwd(
                &self.bh[r * d..(r + 1) * d],
                &a.xh2[r * d..(r + 1) * d],
                a.r2[r],
                &pr[lo.g2..lo.g2 + d],
                &mut gr[lo.g2..lo.g2 + d],
                &mut self.bd1[r * d..(r + 1) * d],
                false,
                gdy,
            );
        }
        prof!(6, matmul_tn(&mut gr[lo.wo..lo.wo + d * d], &a.o[..rows * d], &self.bd1[..rows * d], rows, d, d));
        prof!(6, matmul_nt(&mut self.bd2[..rows * d], &self.bd1[..rows * d], &pr[lo.wo..lo.wo + d * d], rows, d, d));

        self.bk[..rows * d].fill(0.0);
        self.bv[..rows * d].fill(0.0);
        ctx.out = self.bd1.as_mut_ptr();
        ctx.dout = self.bd2.as_ptr();
        ctx.dk = self.bk.as_mut_ptr();
        ctx.dv = self.bv.as_mut_ptr();
        ctx.datt = self.batt.as_mut_ptr();
        {
            let active = &self.active;
            let cost = lseg * nh * self.win.min(lseg + self.win) * hd * 4;
            prof!(7, par::for_ranges(ns, cost, 1, move |b0, b1| attn_backward(ctx, lseg, active, b0, b1)));
        }

        // QK-norm + RoPE backward, assemble d(qkv).
        let mut t1 = [0f32; 1024];
        let mut t2 = [0f32; 1024];
        for t in 0..lseg {
            let cs = &self.cs[t * hd..(t + 1) * hd];
            for b in 0..ns {
                let r = t * ns + b;
                let row = &mut self.bqkv[r * 3 * d..(r + 1) * 3 * d];
                if !self.active[r] {
                    row.fill(0.0);
                    continue;
                }
                for hh in 0..nh {
                    let o = r * d + hh * hd;
                    // query
                    rope_inv(&self.bd1[o..o + hd], cs, &mut t1[..hd]);
                    for i in 0..hd {
                        gr[lo.gq + i] += t1[i] * a.qh[o + i];
                        t2[i] = pr[lo.gq + i] * t1[i];
                    }
                    let m = dot(&t2[..hd], &a.qh[o..o + hd]) / hd as f32;
                    let ri = a.qri[r * nh + hh];
                    for i in 0..hd {
                        row[hh * hd + i] = ri * (t2[i] - a.qh[o + i] * m);
                    }
                    // key
                    rope_inv(&self.bk[o..o + hd], cs, &mut t1[..hd]);
                    for i in 0..hd {
                        gr[lo.gk + i] += t1[i] * a.kh[o + i];
                        t2[i] = pr[lo.gk + i] * t1[i];
                    }
                    let m = dot(&t2[..hd], &a.kh[o..o + hd]) / hd as f32;
                    let ri = a.kri[r * nh + hh];
                    for i in 0..hd {
                        row[d + hh * hd + i] = ri * (t2[i] - a.kh[o + i] * m);
                    }
                    // value
                    row[2 * d + hh * hd..2 * d + (hh + 1) * hd].copy_from_slice(&self.bv[o..o + hd]);
                }
            }
        }
        prof!(6, matmul_tn(&mut gr[lo.wqkv..lo.wqkv + d * 3 * d], &a.a1[..rows * d], &self.bqkv[..rows * 3 * d], rows, d, 3 * d));
        prof!(6, matmul_nt(&mut self.bd1[..rows * d], &self.bqkv[..rows * 3 * d], &pr[lo.wqkv..lo.wqkv + d * 3 * d], rows, 3 * d, d));
        for r in 0..rows {
            rmsnorm_bwd(
                &self.bd1[r * d..(r + 1) * d],
                &a.xh1[r * d..(r + 1) * d],
                a.r1[r],
                &pr[lo.g1..lo.g1 + d],
                &mut gr[lo.g1..lo.g1 + d],
                &mut self.bh[r * d..(r + 1) * d],
                true,
                gdy,
            );
        }
    }

    fn lr_at(&self, step: u64) -> f32 {
        let tc = &self.tc;
        let s = step as f64;
        let warm = tc.warmup as f64;
        let lr = if s < warm {
            tc.lr as f64 * (s + 1.0) / warm
        } else {
            let x = s - warm;
            let dec = tc.decay as f64;
            let frac = if dec > 0.0 { dec / (dec + x) } else { 1.0 };
            tc.lr_min as f64 + (tc.lr as f64 - tc.lr_min as f64) * frac
        };
        lr as f32
    }

    fn adam_step(&mut self) {
        let lr = self.lr_at(self.updates);
        self.updates += 1;
        let tc = self.tc;
        let mut gscale = 1.0f32;
        if tc.clip > 0.0 {
            let n2 = dot(&self.grads, &self.grads);
            let norm = n2.sqrt();
            if norm > tc.clip {
                gscale = tc.clip / norm;
            }
        }
        let b2 = tc.beta2;
        self.b2pow *= b2;
        let bc2 = 1.0 - self.b2pow;
        let omb2 = 1.0 - b2;
        let eps = tc.eps;
        let n = self.params.len();
        let pp = SyncPtr(self.params.as_mut_ptr());
        let vp = SyncPtr(self.adam_v.as_mut_ptr());
        let grads = &self.grads;
        if tc.beta1 == 0.0 {
            par::for_ranges(n, 4, 1024, |s, e| {
                let (p, v) = unsafe { (pp.slice(s, e - s), vp.slice(s, e - s)) };
                adam_rms(p, v, &grads[s..e], gscale, b2, omb2, bc2, lr, eps);
            });
        } else {
            let b1 = tc.beta1;
            self.b1pow *= b1;
            let bc1 = 1.0 - self.b1pow;
            let omb1 = 1.0 - b1;
            let mp = SyncPtr(self.adam_m.as_mut_ptr());
            par::for_ranges(n, 6, 1024, |s, e| {
                let (p, v, m) = unsafe { (pp.slice(s, e - s), vp.slice(s, e - s), mp.slice(s, e - s)) };
                adam_full(p, v, m, &grads[s..e], gscale, b1, omb1, bc1, b2, omb2, bc2, lr, eps);
            });
        }
    }

    /// Moves the keys/values of the finished segment into the attention memory.
    pub fn end_segment(&mut self, lseg: usize) {
        let (d, ns, win) = (self.d, self.ns, self.win);
        let t0 = lseg.saturating_sub(win);
        for l in 0..self.nl {
            let a = &self.acts[l];
            for t in t0..lseg {
                let slot = ((self.seg_start + t as u64) % win as u64) as usize;
                for b in 0..ns {
                    let r = t * ns + b;
                    let hi = ((l * ns + b) * win + slot) * d;
                    self.hist_k[hi..hi + d].copy_from_slice(&a.kr[r * d..(r + 1) * d]);
                    self.hist_v[hi..hi + d].copy_from_slice(&a.qkv[r * 3 * d + 2 * d..(r + 1) * 3 * d]);
                }
            }
        }
    }

    pub fn segment_len(&self) -> usize {
        self.seg
    }

    pub fn n_streams(&self) -> usize {
        self.ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::preset;

    fn tiny() -> (ModelConfig, TrainConfig) {
        let p = preset(1);
        let mc = ModelConfig { d_model: 16, n_layer: 2, n_head: 2, d_ff: 24, window: 5, seg_len: 4, n_streams: 2 };
        let mut tc = p.train;
        tc.init_scale = 1.0;
        (mc, tc)
    }

    /// Mean cross-entropy over the last segment. Earlier segments run with the base
    /// parameters (their keys/values become stop-gradient attention memory); `params`, if
    /// given, replaces the weights only for the last segment — exactly the semantics the
    /// analytic gradient implements.
    fn loss_and_model(params: Option<&[f32]>, data: &[[u32; 2]], vocab: usize) -> (f64, Model) {
        let (mc, tc) = tiny();
        let mut m = Model::new(vocab, &mc, &tc);
        // Make the head non-zero so gradients reach every layer.
        let mut rng = Rng::new(99);
        for x in &mut m.params[m.wout..m.wout + mc.d_model as usize * vocab] {
            *x = rng.uniform() * 0.5;
        }
        let seg = mc.seg_len as usize;
        let ns = mc.n_streams as usize;
        let mut loss = 0f64;
        let mut n = 0;
        let nseg = data.len() / seg;
        for s in 0..nseg {
            if s == nseg - 1 {
                if let Some(p) = params {
                    m.params.copy_from_slice(p);
                }
            }
            m.begin_segment((s * seg) as u64);
            for t in 0..seg {
                let pos = s * seg + t;
                let inputs: Vec<u32> = (0..ns).map(|b| if pos == 0 { m.bos() } else { data[pos - 1][b] }).collect();
                let active: Vec<bool> = (0..ns).map(|b| !(b == 1 && pos == data.len() - 1)).collect();
                m.forward_step(t, &inputs, &active);
                for b in 0..ns {
                    if active[b] {
                        let tgt = data[pos][b];
                        m.set_target(t, b, tgt);
                        if s == nseg - 1 {
                            loss -= (m.probs_row(t, b)[tgt as usize] as f64).ln();
                            n += 1;
                        }
                    }
                }
            }
            if s + 1 < nseg {
                m.end_segment(seg);
            }
        }
        (loss / n as f64, m)
    }

    #[test]
    fn gradients_match_finite_differences() {
        let vocab = 7;
        let mut rng = Rng::new(5);
        let data: Vec<[u32; 2]> = (0..12)
            .map(|_| [(rng.next_u64() % vocab as u64) as u32, (rng.next_u64() % vocab as u64) as u32])
            .collect();
        let (_, mut m) = loss_and_model(None, &data, vocab);
        let seg = 4;
        assert!(m.compute_grads(seg));
        let analytic = m.grads.clone();
        let base = m.params.clone();
        let mut worst = 0f64;
        let mut pick = Rng::new(11);
        for _ in 0..400 {
            let i = (pick.next_u64() % base.len() as u64) as usize;
            let g = analytic[i] as f64;
            let h = 2e-3f32;
            let mut p = base.clone();
            p[i] = base[i] + h;
            let (lp, _) = loss_and_model(Some(&p), &data, vocab);
            p[i] = base[i] - h;
            let (lm, _) = loss_and_model(Some(&p), &data, vocab);
            let num = (lp - lm) / (2.0 * h as f64);
            let err = (num - g).abs() / (num.abs() + g.abs()).max(1e-2);
            if err > worst {
                worst = err;
            }
            assert!(err < 0.08, "param {i}: analytic {g} numeric {num}");
        }
        eprintln!("worst relative gradient error {worst:.4}");
    }
}
