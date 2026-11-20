//! PiedPiper's Transformer on a Vulkan GPU, driven like crates/core/src/model.rs: begin a
//! segment, forward every position, set the coded targets, one Adam step, roll the attention
//! memory forward. Every kernel reproduces the CPU's per-element arithmetic, so probabilities
//! and weights stay bit-identical to the CPU model.
//!
//! Two drivers:
//! * `run_segment` (encoder): all inputs of a segment are known in advance, so the forward
//!   pass covers every position at once (causal attention makes that equal to position by
//!   position) and forward + backward + Adam + memory update go out in one submission.
//! * `forward_position` + `finish_segment` (decoder): the next input is only known once the
//!   previous symbol has been decoded, so each position is its own submission and readback.

use crate::config::{ModelConfig, TrainConfig};
use crate::fpmath::{rope_theta, sincos};
use crate::shaders;
use crate::vkrt::{Buffer, Gpu, Kernel, Res};

const ROPE_BASE: f64 = 10_000.0;
/// Segments one submission may carry (encoder batching).
pub const KMAX: usize = 8;

/// One encoder segment: rows t*ns+b for t < lseg.
pub struct SegIo<'a> {
    pub pos: u64,
    pub lseg: usize,
    pub inputs: &'a [u32],
    pub active: &'a [bool],
    pub targets: &'a [u32],
    pub train: bool,
}

struct Kernels {
    matmul: Kernel,
    matmul_nt: Kernel,
    matmul16: Kernel,
    matmul_nt16: Kernel,
    embed: Kernel,
    rmsnorm: Kernel,
    postnorm: Kernel,
    qknorm_rope: Kernel,
    attn_fwd: Kernel,
    swiglu: Kernel,
    softmax: Kernel,
    dlogits: Kernel,
    rmsnorm_bwd: Kernel,
    gainsum: Kernel,
    swiglu_bwd: Kernel,
    attn_bwd_q: Kernel,
    attn_bwd_kv: Kernel,
    qknorm_bwd: Kernel,
    embed_grad: Kernel,
    adam: Kernel,
    end_segment: Kernel,
    fill: Kernel,
}

impl Kernels {
    fn new(g: &mut Gpu) -> Res<Self> {
        Ok(Kernels {
            matmul: g.kernel("matmul", shaders::MATMUL)?,
            matmul_nt: g.kernel("matmul_nt", shaders::MATMUL_NT)?,
            matmul16: g.kernel("matmul16", shaders::MATMUL16)?,
            matmul_nt16: g.kernel("matmul_nt16", shaders::MATMUL_NT16)?,
            embed: g.kernel("embed", shaders::EMBED)?,
            rmsnorm: g.kernel("rmsnorm", shaders::RMSNORM)?,
            postnorm: g.kernel("postnorm", shaders::POSTNORM)?,
            qknorm_rope: g.kernel("qknorm_rope", shaders::QKNORM_ROPE)?,
            attn_fwd: g.kernel("attn_fwd", shaders::ATTN_FWD)?,
            swiglu: g.kernel("swiglu", shaders::SWIGLU)?,
            softmax: g.kernel("softmax", shaders::SOFTMAX)?,
            dlogits: g.kernel("dlogits", shaders::DLOGITS)?,
            rmsnorm_bwd: g.kernel("rmsnorm_bwd", shaders::RMSNORM_BWD)?,
            gainsum: g.kernel("gainsum", shaders::GAINSUM)?,
            swiglu_bwd: g.kernel("swiglu_bwd", shaders::SWIGLU_BWD)?,
            attn_bwd_q: g.kernel("attn_bwd_q", shaders::ATTN_BWD_Q)?,
            attn_bwd_kv: g.kernel("attn_bwd_kv", shaders::ATTN_BWD_KV)?,
            qknorm_bwd: g.kernel("qknorm_bwd", shaders::QKNORM_BWD)?,
            embed_grad: g.kernel("embed_grad", shaders::EMBED_GRAD)?,
            adam: g.kernel("adam", shaders::ADAM)?,
            end_segment: g.kernel("end_segment", shaders::END_SEGMENT)?,
            fill: g.kernel("fill", shaders::FILL)?,
        })
    }
}

/// Parameter offsets relative to the start of the parameter vector (same layout as Model::new).
#[derive(Clone, Copy, Default)]
struct Lay {
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

/// Absolute pool offsets of one layer's activations.
#[derive(Clone, Copy, Default)]
struct Act {
    xh1: usize,
    r1: usize,
    a1: usize,
    qkv: usize,
    qh: usize,
    qri: usize,
    kh: usize,
    kri: usize,
    qr: usize,
    kr: usize,
    att: usize,
    o: usize,
    xh2: usize,
    r2: usize,
    xh3: usize,
    r3: usize,
    fin: usize,
    gu: usize,
    sg: usize,
    xh4: usize,
    r4: usize,
}

fn u(x: usize) -> u32 {
    u32::try_from(x).expect("offset exceeds 32 bits")
}

/// Grids of fewer 32x32 tiles than this use the 16x16 kernels (about two workgroups per
/// compute unit on a 72-CU GPU).
const SMALL_GRID: usize = 144;

#[allow(clippy::too_many_arguments)]
fn mm(g: &mut Gpu, k: &Kernels, m: usize, n: usize, kk: usize, a: (usize, usize, usize), b: (usize, usize, usize), y: (usize, usize)) {
    let push = [u(m), u(n), u(kk), u(a.0), u(a.1), u(a.2), u(b.0), u(b.1), u(b.2), u(y.0), u(y.1)];
    if n.div_ceil(32) * m.div_ceil(32) < SMALL_GRID {
        g.dispatch(&k.matmul16, [u(n.div_ceil(16)), u(m.div_ceil(16)), 1], &push);
    } else {
        g.dispatch(&k.matmul, [u(n.div_ceil(32)), u(m.div_ceil(32)), 1], &push);
    }
}

#[allow(clippy::too_many_arguments)]
fn mm_nt(g: &mut Gpu, k: &Kernels, m: usize, n: usize, kk: usize, a: usize, b: usize, y: usize) {
    let push = [u(m), u(n), u(kk), u(a), u(b), u(y)];
    if n.div_ceil(32) * m.div_ceil(32) < SMALL_GRID {
        g.dispatch(&k.matmul_nt16, [u(n.div_ceil(16)), u(m.div_ceil(16)), 1], &push);
    } else {
        g.dispatch(&k.matmul_nt, [u(n.div_ceil(32)), u(m.div_ceil(32)), 1], &push);
    }
}

pub struct GpuModel {
    pub gpu: Gpu,
    k: Kernels,
    pool: Buffer,
    up: Buffer,
    down: Buffer,
    pub vocab: usize,
    d: usize,
    nh: usize,
    hd: usize,
    dff: usize,
    win: usize,
    ns: usize,
    nl: usize,
    rows: usize,
    pub np: usize,
    p0: usize,
    g0: usize,
    v0: usize,
    emb: usize,
    lay: Vec<Lay>,
    gf: usize,
    wout: usize,
    acts: Vec<Act>,
    xhf: usize,
    rf: usize,
    z: usize,
    probs: usize,
    h: usize,
    sd: usize,
    bh: usize,
    bd1: usize,
    bd2: usize,
    bff: usize,
    bgu: usize,
    bqkv: usize,
    bk: usize,
    bv: usize,
    ds: usize,
    /// Per-gain product buffers (rows x d): gf, then per layer g4, g3, g2, g1, gq, gk.
    prod_gf: usize,
    prod: Vec<[usize; 6]>,
    gtab: usize,
    ngains: usize,
    down_params: usize,
    hist_k: usize,
    hist_v: usize,
    /// Upload region: [tok rows | tgt rows | act rows | cs seg*hd | groups 3*rows | grows rows].
    io: usize,
    io_words: usize,
    /// Which of the KMAX upload / readback regions the current segment uses.
    cur: usize,
    tc: TrainConfig,
    updates: u64,
    b2pow: f32,
    theta: Vec<f64>,
    att_scale: f32,
    seg_start: u64,
    active: Vec<bool>,
    tok_in: Vec<u32>,
    pub pool_bytes: u64,
}

impl GpuModel {
    /// `init` is the CPU model's initial parameter vector (Model::params()).
    pub fn new(mut gpu: Gpu, vocab: usize, mc: &ModelConfig, tc: &TrainConfig, init: &[f32]) -> Res<Self> {
        if tc.beta1 != 0.0 || tc.clip != 0.0 {
            return Err("the experiment covers the default optimiser (beta1 = 0, no clipping)".into());
        }
        let d = mc.d_model as usize;
        let nh = mc.n_head as usize;
        let hd = d / nh;
        let dff = mc.d_ff as usize;
        let win = mc.window as usize;
        let seg = mc.seg_len as usize;
        let ns = mc.n_streams as usize;
        let nl = mc.n_layer as usize;
        // Every preset has hd = 32, window <= 320 and vocab <= 4096.
        if d > 2048 || hd > 64 || win > 512 || vocab > 4096 {
            return Err("shapes beyond the experiment's shared-memory limits (d <= 2048, head dim <= 64, window <= 512, vocab <= 4096)".into());
        }
        let rows = seg * ns;

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
            lay.push(Lay { g1, wqkv, gq, gk, wo, g2, g3, wgu, wd, g4 });
        }
        let gf = take(d);
        let wout = take(d * vocab);
        let np = off;
        if np != init.len() {
            return Err(format!("parameter count {np} != CPU model's {}", init.len()));
        }

        let mut top = 0usize;
        let mut alloc = |n: usize| {
            let o = top;
            top += n.div_ceil(64) * 64;
            o
        };
        let p0 = alloc(np);
        let g0 = alloc(np);
        let v0 = alloc(np);
        let mut acts = Vec::with_capacity(nl);
        for _ in 0..nl {
            acts.push(Act {
                xh1: alloc(rows * d),
                r1: alloc(rows),
                a1: alloc(rows * d),
                qkv: alloc(rows * 3 * d),
                qh: alloc(rows * d),
                qri: alloc(rows * nh),
                kh: alloc(rows * d),
                kri: alloc(rows * nh),
                qr: alloc(rows * d),
                kr: alloc(rows * d),
                att: alloc(rows * nh * win),
                o: alloc(rows * d),
                xh2: alloc(rows * d),
                r2: alloc(rows),
                xh3: alloc(rows * d),
                r3: alloc(rows),
                fin: alloc(rows * d),
                gu: alloc(rows * 2 * dff),
                sg: alloc(rows * dff),
                xh4: alloc(rows * d),
                r4: alloc(rows),
            });
        }
        let xhf = alloc(rows * d);
        let rf = alloc(rows);
        let z = alloc(rows * d);
        let probs = alloc(rows * vocab);
        let h = alloc(rows * d);
        let sd = alloc(rows * d);
        let bh = alloc(rows * d);
        let bd1 = alloc(rows * d);
        let bd2 = alloc(rows * d);
        let bff = alloc(rows * dff);
        let bgu = alloc(rows * 2 * dff);
        let bqkv = alloc(rows * 3 * d);
        let bk = alloc(rows * d);
        let bv = alloc(rows * d);
        let ds = alloc(rows * nh * win);
        let prod_gf = alloc(rows * d);
        let prod: Vec<[usize; 6]> = (0..nl).map(|_| [0; 6].map(|_| alloc(rows * d))).collect();
        let ngains = 1 + 6 * nl;
        let gtab = alloc(8 * ngains);
        let hist_k = alloc(nl * ns * win * d);
        let hist_v = alloc(nl * ns * win * d);
        let io_words = 3 * rows + seg * hd + 4 * rows;
        let io = alloc(io_words * KMAX);
        if top >= 1 << 30 {
            return Err("model too large for 32-bit pool offsets".into());
        }
        let k = Kernels::new(&mut gpu)?;
        let pool = gpu.buffer((top * 4) as u64, false)?;
        gpu.bind_pool(&pool);
        let up = gpu.buffer((io_words * KMAX * 4) as u64, true)?;
        // Readback buffer: [probabilities | parameters]. At start-up the first region also
        // stages the gain table and the second one the initial weights.
        let down_params = (KMAX * rows * vocab).max(8 * ngains).max(1);
        let down = gpu.buffer((down_params + np) as u64 * 4, true)?;
        // Gain table for gainsum.comp: prod, ncols, row stride, out, rows per flag, use flags,
        // rows per segment row, unused. Offsets are static, so it is written once.
        let mut tab: Vec<u32> = vec![u(prod_gf), u(d), u(d), u(g0 + gf), 1, 0, 1, 0];
        for (l, lo) in lay.iter().enumerate() {
            let pr = prod[l];
            for (i, gain) in [lo.g4, lo.g3, lo.g2, lo.g1].into_iter().enumerate() {
                tab.extend_from_slice(&[u(pr[i]), u(d), u(d), u(g0 + gain), 1, 0, 1, 0]);
            }
            tab.extend_from_slice(&[u(pr[4]), u(hd), u(hd), u(g0 + lo.gq), u(nh), 1, u(nh), 0]);
            tab.extend_from_slice(&[u(pr[5]), u(hd), u(hd), u(g0 + lo.gk), u(nh), 1, u(nh), 0]);
        }
        down.write_u32(0, &tab);
        down.write_f32(down_params, init);
        gpu.begin();
        gpu.dispatch_1d(&k.fill, top, &[0, u(top)]);
        gpu.copy(&down, down_params, &pool, p0, np);
        gpu.copy(&down, 0, &pool, gtab, tab.len());
        gpu.submit();
        Ok(GpuModel {
            gpu,
            k,
            pool,
            up,
            down,
            vocab,
            d,
            nh,
            hd,
            dff,
            win,
            ns,
            nl,
            rows,
            np,
            p0,
            g0,
            v0,
            emb,
            lay,
            gf,
            wout,
            acts,
            xhf,
            rf,
            z,
            probs,
            h,
            sd,
            bh,
            bd1,
            bd2,
            bff,
            bgu,
            bqkv,
            bk,
            bv,
            ds,
            prod_gf,
            prod,
            gtab,
            ngains,
            down_params,
            hist_k,
            hist_v,
            io,
            io_words,
            cur: 0,
            tc: *tc,
            updates: 0,
            b2pow: 1.0,
            theta: rope_theta(ROPE_BASE, hd),
            att_scale: 1.0 / (hd as f32).sqrt(),
            seg_start: 0,
            active: vec![false; rows],
            tok_in: vec![0; rows],
            pool_bytes: (top * 4) as u64,
        })
    }

    pub fn begin_segment(&mut self, seg_start: u64) {
        assert!(seg_start + (self.rows as u64) < u32::MAX as u64);
        self.seg_start = seg_start;
    }

    /// Pool, upload-buffer and readback offsets of the current segment's regions.
    fn iob(&self) -> usize {
        self.io + self.cur * self.io_words
    }
    fn upb(&self) -> usize {
        self.cur * self.io_words
    }
    fn downb(&self) -> usize {
        self.cur * self.rows * self.vocab
    }

    // io sub-regions (offsets within one upload region)
    fn io_tok(&self) -> usize {
        0
    }
    fn io_tgt(&self) -> usize {
        self.rows
    }
    fn io_act(&self) -> usize {
        2 * self.rows
    }
    fn io_cs(&self) -> usize {
        3 * self.rows
    }
    fn io_groups(&self) -> usize {
        3 * self.rows + self.io_words - 7 * self.rows
    }
    fn io_grows(&self) -> usize {
        self.io_groups() + 3 * self.rows
    }

    /// cos / sin row of local position t, exactly as Model::forward_step computes it.
    fn cs_row(&self, t: usize) -> Vec<f32> {
        let half = self.hd / 2;
        let p = self.seg_start + t as u64;
        let mut cs = vec![0f32; self.hd];
        for i in 0..half {
            let (s, c) = sincos(p as f64 * self.theta[i]);
            cs[i] = c as f32;
            cs[half + i] = s as f32;
        }
        cs
    }

    /// Model::lr_at, verbatim.
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

    /// Active rows grouped by input token, ascending within each group, for embed_grad.
    /// Writes the groups into the upload buffer and returns the number of tokens.
    fn stage_groups(&mut self, n: usize) -> usize {
        let mut pairs: Vec<(u32, u32)> = (0..n).filter(|&r| self.active[r]).map(|r| (self.tok_in[r], r as u32)).collect();
        pairs.sort_by_key(|p| p.0); // stable: rows stay ascending within a token
        let mut groups: Vec<u32> = Vec::new();
        let mut rows: Vec<u32> = Vec::with_capacity(pairs.len());
        for (i, &(tok, r)) in pairs.iter().enumerate() {
            if i == 0 || pairs[i - 1].0 != tok {
                groups.extend_from_slice(&[tok, rows.len() as u32, 0]);
            }
            let last = groups.len() - 1;
            groups[last] += 1;
            rows.push(r);
        }
        self.up.write_u32(self.upb() + self.io_groups(), &groups);
        self.up.write_u32(self.upb() + self.io_grows(), &rows);
        groups.len() / 3
    }

    fn rec_forward(&mut self, t0: usize, t1: usize) {
        let (d, nh, hd, dff, ns, v, win) = (self.d, self.nh, self.hd, self.dff, self.ns, self.vocab, self.win);
        let (io_tok, io_cs) = (self.iob() + self.io_tok(), self.iob() + self.io_cs());
        let (g, k) = (&mut self.gpu, &self.k);
        let r0 = t0 * ns;
        let n = (t1 - t0) * ns;
        let (p0, h, sd) = (self.p0, self.h, self.sd);
        g.dispatch_1d(&k.embed, n * d, &[u(r0), u(n), u(d), u(p0 + self.emb), u(h), u(io_tok)]);
        for l in 0..self.nl {
            let (lo, a) = (self.lay[l], self.acts[l]);
            let (hk, hv) = (self.hist_k + l * ns * win * d, self.hist_v + l * ns * win * d);
            g.dispatch(&k.rmsnorm, [u(n), 1, 1], &[u(r0), u(d), u(h), u(p0 + lo.g1), u(a.xh1), u(a.a1), u(a.r1)]);
            mm(g, k, n, 3 * d, d, (a.a1 + r0 * d, d, 1), (p0 + lo.wqkv, 3 * d, 1), (a.qkv + r0 * 3 * d, 3 * d));
            g.dispatch(
                &k.qknorm_rope,
                [u(n * nh), 1, 1],
                &[
                    u(r0),
                    u(ns),
                    u(d),
                    u(nh),
                    u(hd),
                    u(a.qkv),
                    u(p0 + lo.gq),
                    u(p0 + lo.gk),
                    u(a.qh),
                    u(a.qri),
                    u(a.kh),
                    u(a.kri),
                    u(a.qr),
                    u(a.kr),
                    u(io_cs),
                ],
            );
            g.dispatch(
                &k.attn_fwd,
                [u(n * nh), 1, 1],
                &[
                    u(r0),
                    u(ns),
                    u(d),
                    u(nh),
                    u(hd),
                    u(win),
                    self.seg_start as u32,
                    self.att_scale.to_bits(),
                    u(a.qr),
                    u(a.kr),
                    u(a.qkv),
                    u(hk),
                    u(hv),
                    u(a.att),
                    u(a.o),
                ],
            );
            mm(g, k, n, d, d, (a.o + r0 * d, d, 1), (p0 + lo.wo, d, 1), (sd + r0 * d, d));
            g.dispatch(&k.postnorm, [u(n), 1, 1], &[u(r0), u(d), u(sd), u(p0 + lo.g2), u(a.xh2), u(h), u(a.r2)]);
            g.dispatch(&k.rmsnorm, [u(n), 1, 1], &[u(r0), u(d), u(h), u(p0 + lo.g3), u(a.xh3), u(a.fin), u(a.r3)]);
            mm(g, k, n, 2 * dff, d, (a.fin + r0 * d, d, 1), (p0 + lo.wgu, 2 * dff, 1), (a.gu + r0 * 2 * dff, 2 * dff));
            g.dispatch_1d(&k.swiglu, n * dff, &[u(r0), u(n), u(dff), u(a.gu), u(a.sg)]);
            mm(g, k, n, d, dff, (a.sg + r0 * dff, dff, 1), (p0 + lo.wd, d, 1), (sd + r0 * d, d));
            g.dispatch(&k.postnorm, [u(n), 1, 1], &[u(r0), u(d), u(sd), u(p0 + lo.g4), u(a.xh4), u(h), u(a.r4)]);
        }
        g.dispatch(&k.rmsnorm, [u(n), 1, 1], &[u(r0), u(d), u(h), u(p0 + self.gf), u(self.xhf), u(self.z), u(self.rf)]);
        mm(g, k, n, v, d, (self.z + r0 * d, d, 1), (p0 + self.wout, v, 1), (self.probs + r0 * v, v));
        g.dispatch(&k.softmax, [u(n), 1, 1], &[u(r0), u(v), u(self.probs)]);
    }

    fn rec_backward(&mut self, lseg: usize, inv_n: f32, ntok: usize) {
        let (d, nh, hd, dff, ns, v, win) = (self.d, self.nh, self.hd, self.dff, self.ns, self.vocab, self.win);
        let io = self.iob();
        let (tgt, act, cs) = (io + self.io_tgt(), io + self.io_act(), io + self.io_cs());
        let (groups, grows) = (io + self.io_groups(), io + self.io_grows());
        let (g, k) = (&mut self.gpu, &self.k);
        let rows = lseg * ns;
        let (p0, g0) = (self.p0, self.g0);
        let (bh, bd1, bd2, bff, bgu, bqkv, bk, bv) = (self.bh, self.bd1, self.bd2, self.bff, self.bgu, self.bqkv, self.bk, self.bv);
        g.dispatch_1d(&k.fill, self.np, &[u(g0), u(self.np)]);
        g.dispatch(&k.dlogits, [u(rows), 1, 1], &[u(v), u(self.probs), u(tgt), u(act), inv_n.to_bits()]);
        mm(g, k, d, v, rows, (self.z, 1, d), (self.probs, v, 1), (g0 + self.wout, v));
        mm_nt(g, k, rows, d, v, self.probs, p0 + self.wout, bd1);
        g.dispatch(&k.rmsnorm_bwd, [u(rows), 1, 1], &[u(d), u(bd1), u(self.xhf), u(self.rf), u(p0 + self.gf), u(bh), 0, u(self.prod_gf)]);
        for l in (0..self.nl).rev() {
            let (lo, a) = (self.lay[l], self.acts[l]);
            let (hk, hv) = (self.hist_k + l * ns * win * d, self.hist_v + l * ns * win * d);
            let pr = self.prod[l];
            // feed-forward sub-layer
            g.dispatch(&k.rmsnorm_bwd, [u(rows), 1, 1], &[u(d), u(bh), u(a.xh4), u(a.r4), u(p0 + lo.g4), u(bd1), 0, u(pr[0])]);
            mm(g, k, dff, d, rows, (a.sg, 1, dff), (bd1, d, 1), (g0 + lo.wd, d));
            mm_nt(g, k, rows, dff, d, bd1, p0 + lo.wd, bff);
            g.dispatch_1d(&k.swiglu_bwd, rows * dff, &[u(rows), u(dff), u(a.gu), u(bff), u(bgu)]);
            mm(g, k, d, 2 * dff, rows, (a.fin, 1, d), (bgu, 2 * dff, 1), (g0 + lo.wgu, 2 * dff));
            mm_nt(g, k, rows, d, 2 * dff, bgu, p0 + lo.wgu, bd1);
            g.dispatch(&k.rmsnorm_bwd, [u(rows), 1, 1], &[u(d), u(bd1), u(a.xh3), u(a.r3), u(p0 + lo.g3), u(bh), 1, u(pr[1])]);
            // attention sub-layer
            g.dispatch(&k.rmsnorm_bwd, [u(rows), 1, 1], &[u(d), u(bh), u(a.xh2), u(a.r2), u(p0 + lo.g2), u(bd1), 0, u(pr[2])]);
            mm(g, k, d, d, rows, (a.o, 1, d), (bd1, d, 1), (g0 + lo.wo, d));
            mm_nt(g, k, rows, d, d, bd1, p0 + lo.wo, bd2);
            g.dispatch(
                &k.attn_bwd_q,
                [u(rows * nh), 1, 1],
                &[
                    u(ns),
                    u(d),
                    u(nh),
                    u(hd),
                    u(win),
                    self.seg_start as u32,
                    self.att_scale.to_bits(),
                    u(a.kr),
                    u(a.qkv),
                    u(hk),
                    u(hv),
                    u(a.att),
                    u(bd2),
                    u(bd1),
                    u(self.ds),
                    u(act),
                ],
            );
            g.dispatch_1d(
                &k.attn_bwd_kv,
                rows * d,
                &[u(lseg), u(ns), u(d), u(nh), u(hd), u(win), self.seg_start as u32, u(a.qr), u(a.att), u(self.ds), u(bd2), u(bk), u(bv), u(act)],
            );
            g.dispatch(
                &k.qknorm_bwd,
                [u(rows * nh), 1, 1],
                &[
                    u(ns),
                    u(d),
                    u(nh),
                    u(hd),
                    u(bd1),
                    u(bk),
                    u(bv),
                    u(a.qh),
                    u(a.kh),
                    u(a.qri),
                    u(a.kri),
                    u(p0 + lo.gq),
                    u(p0 + lo.gk),
                    u(bqkv),
                    u(pr[4]),
                    u(pr[5]),
                    u(cs),
                    u(act),
                ],
            );
            mm(g, k, d, 3 * d, rows, (a.a1, 1, d), (bqkv, 3 * d, 1), (g0 + lo.wqkv, 3 * d));
            mm_nt(g, k, rows, d, 3 * d, bqkv, p0 + lo.wqkv, bd1);
            g.dispatch(&k.rmsnorm_bwd, [u(rows), 1, 1], &[u(d), u(bd1), u(a.xh1), u(a.r1), u(p0 + lo.g1), u(bh), 1, u(pr[3])]);
        }
        // Every gain gradient at once (they feed nothing but Adam).
        g.dispatch(&k.gainsum, [u(d.max(hd).div_ceil(64)), u(self.ngains), 1], &[u(rows), u(self.gtab), u(act)]);
        g.dispatch_1d(&k.embed_grad, ntok * d, &[u(ntok), u(d), u(bh), u(g0 + self.emb), u(groups), u(grows)]);
    }

    /// Model::adam_step for beta1 = 0 and no clipping: host-side schedule, GPU update.
    fn rec_adam(&mut self) {
        let lr = self.lr_at(self.updates);
        self.updates += 1;
        let b2 = self.tc.beta2;
        self.b2pow *= b2;
        let bc2 = 1.0 - self.b2pow;
        let omb2 = 1.0 - b2;
        let eps = self.tc.eps;
        let (g, k) = (&mut self.gpu, &self.k);
        g.dispatch_1d(
            &k.adam,
            self.np,
            &[u(self.np), u(self.p0), u(self.v0), u(self.g0), 1.0f32.to_bits(), b2.to_bits(), omb2.to_bits(), bc2.to_bits(), lr.to_bits(), eps.to_bits()],
        );
    }

    fn rec_end_segment(&mut self, lseg: usize) {
        let (d, ns, win) = (self.d, self.ns, self.win);
        let t0 = lseg.saturating_sub(win);
        let (g, k) = (&mut self.gpu, &self.k);
        for l in 0..self.nl {
            let a = self.acts[l];
            let (hk, hv) = (self.hist_k + l * ns * win * d, self.hist_v + l * ns * win * d);
            g.dispatch_1d(
                &k.end_segment,
                (lseg - t0) * ns * d,
                &[u(t0), u(lseg), u(ns), u(d), u(win), self.seg_start as u32, u(a.kr), u(a.qkv), u(hk), u(hv)],
            );
        }
    }

    /// Trains on the segment if any row was active (Model::train_step semantics). The token
    /// groups must already be in the upload region (stage_groups) and copied to the pool.
    fn rec_train(&mut self, lseg: usize, ntok: usize) {
        let n_act = self.active[..lseg * self.ns].iter().filter(|&&a| a).count();
        if n_act == 0 {
            return;
        }
        let inv_n = 1.0 / n_act as f32;
        self.rec_backward(lseg, inv_n, ntok);
        self.rec_adam();
    }

    /// Stages and records one encoder segment into the open command buffer (region `cur`).
    fn rec_segment(&mut self, sg: &SegIo) {
        self.begin_segment(sg.pos);
        let (ns, hd, lseg) = (self.ns, self.hd, sg.lseg);
        let n = lseg * ns;
        let act: Vec<u32> = sg.active[..n].iter().map(|&a| a as u32).collect();
        self.active[..n].copy_from_slice(&sg.active[..n]);
        self.tok_in[..n].copy_from_slice(&sg.inputs[..n]);
        let ub = self.upb();
        self.up.write_u32(ub + self.io_tok(), &sg.inputs[..n]);
        self.up.write_u32(ub + self.io_tgt(), &sg.targets[..n]);
        self.up.write_u32(ub + self.io_act(), &act);
        for t in 0..lseg {
            let cs = self.cs_row(t);
            self.up.write_f32(ub + self.io_cs() + t * hd, &cs);
        }
        let ntok = self.stage_groups(n);
        let (iob, downb) = (self.iob(), self.downb());
        self.gpu.copy(&self.up, ub, &self.pool, iob, self.io_words);
        self.rec_forward(0, lseg);
        self.gpu.copy(&self.pool, self.probs, &self.down, downb, n * self.vocab);
        if sg.train {
            self.rec_train(lseg, ntok);
        }
        self.rec_end_segment(lseg);
    }

    /// Encoder path: inputs, active flags and targets for rows t*ns+b, t < lseg.
    pub fn run_segment(&mut self, lseg: usize, inputs: &[u32], active: &[bool], targets: &[u32], train: bool) {
        let pos = self.seg_start;
        self.run_segments(&[SegIo { pos, lseg, inputs, active, targets, train }]);
    }

    /// Encoder path, several consecutive segments in one submission (their inputs are all
    /// known in advance). Probabilities of segment k: `probs_row_k(k, t, b)`.
    pub fn run_segments(&mut self, segs: &[SegIo]) {
        assert!(!segs.is_empty() && segs.len() <= KMAX);
        self.gpu.begin();
        for (k, sg) in segs.iter().enumerate() {
            self.cur = k;
            self.rec_segment(sg);
        }
        self.gpu.submit();
        self.cur = 0;
    }

    /// Decoder path: forward of local position t for all streams; its probabilities are
    /// readable afterwards with `probs_row(t, b)`.
    pub fn forward_position(&mut self, t: usize, inputs: &[u32], active: &[bool]) {
        let (ns, hd) = (self.ns, self.hd);
        let act: Vec<u32> = active[..ns].iter().map(|&a| a as u32).collect();
        self.active[t * ns..(t + 1) * ns].copy_from_slice(&active[..ns]);
        self.tok_in[t * ns..(t + 1) * ns].copy_from_slice(&inputs[..ns]);
        self.up.write_u32(self.io_tok() + t * ns, &inputs[..ns]);
        self.up.write_u32(self.io_act() + t * ns, &act);
        let cs = self.cs_row(t);
        self.up.write_f32(self.io_cs() + t * hd, &cs);
        let (tok, actp, csp) = (self.io_tok() + t * ns, self.io_act() + t * ns, self.io_cs() + t * hd);
        self.gpu.begin();
        self.gpu.copy(&self.up, tok, &self.pool, self.io + tok, ns);
        self.gpu.copy(&self.up, actp, &self.pool, self.io + actp, ns);
        self.gpu.copy(&self.up, csp, &self.pool, self.io + csp, hd);
        self.rec_forward(t, t + 1);
        self.gpu.copy(&self.pool, self.probs + t * ns * self.vocab, &self.down, t * ns * self.vocab, ns * self.vocab);
        self.gpu.submit();
    }

    /// Decoder path: once the segment's symbols are known, train and roll the memory forward.
    pub fn finish_segment(&mut self, lseg: usize, targets: &[u32], train: bool) {
        let n = lseg * self.ns;
        self.up.write_u32(self.io_tgt(), &targets[..n]);
        let ntok = self.stage_groups(n);
        let (tgt, groups) = (self.io_tgt(), self.io_groups());
        self.gpu.begin();
        self.gpu.copy(&self.up, tgt, &self.pool, self.io + tgt, n);
        self.gpu.copy(&self.up, groups, &self.pool, self.io + groups, 4 * self.rows);
        if train {
            self.rec_train(lseg, ntok);
        }
        self.rec_end_segment(lseg);
        self.gpu.submit();
    }

    pub fn probs_row(&self, t: usize, b: usize) -> &[f32] {
        self.probs_row_k(0, t, b)
    }

    pub fn probs_row_k(&self, k: usize, t: usize, b: usize) -> &[f32] {
        self.down.f32s(k * self.rows * self.vocab + (t * self.ns + b) * self.vocab, self.vocab)
    }

    pub fn read_params(&mut self) -> Vec<f32> {
        let off = self.down_params;
        self.gpu.begin();
        self.gpu.copy(&self.pool, self.p0, &self.down, off, self.np);
        self.gpu.submit();
        self.down.f32s(off, self.np).to_vec()
    }
}
