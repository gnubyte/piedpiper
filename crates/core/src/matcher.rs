//! Match model: long-range copying for the Transformer.
//!
//! The network only attends to the last `window` tokens. Text such as logs, CSV, JSON and
//! source code repeats itself far beyond that. Like the match models of PAQ/cmix, this
//! finds the most recent earlier occurrence of the current context (hash of the last
//! `min_len` tokens, verified) and predicts the token that followed it. Its reliability
//! is learned from integer hit/miss counts per match-length bucket, and the predicted
//! token's probability is raised to that estimate when it beats the network's own:
//!
//! ```text
//! p_hit = (hits[b][c] + 0.5) / (tries[b][c] + 1)    b: match-length bucket,
//!                                                   c: bin of the network's p_nn[t]
//! if p_hit > p_nn[t]:  λ = (p_hit − p_nn[t]) / (1 − p_nn[t]);  p = (1 − λ)·p_nn + λ·onehot(t)
//! ```
//!
//! Conditioning on the network's own confidence matters: the network already predicts the
//! structural parts of logs/JSON almost perfectly, so the match model is mostly consulted
//! where the network is unsure (random ids, numbers) — exactly where an old occurrence is
//! least reliable. Per-bin counts learn that instead of over-trusting the match.
//!
//! All state is integer and the mixing is a fixed sequence of f32 operations, so encoder
//! and decoder agree bit-for-bit on every platform. One model per stream.

const BUCKETS: usize = 24;
const PBINS: usize = 10;

/// Bin of the network's probability for the predicted token (fixed f32 comparisons).
#[inline]
fn pbin(p: f32) -> usize {
    const EDGES: [f32; PBINS - 1] = [0.003_906_25, 0.015_625, 0.0625, 0.125, 0.25, 0.5, 0.75, 0.9, 0.97];
    let mut b = 0;
    while b < EDGES.len() && p >= EDGES[b] {
        b += 1;
    }
    b
}

#[inline]
fn bucket(len: u32) -> usize {
    // 0..16 exact, then log-spaced up to 23.
    if len < 16 {
        len as usize
    } else {
        (16 + (31 - len.leading_zeros()) as usize - 4).min(BUCKETS - 1)
    }
}

pub struct MatchModel {
    min_len: usize,
    mask: usize,
    table: Vec<u32>,
    /// Stream index of the token the current match predicts (valid while `len > 0`).
    ptr: usize,
    len: u32,
    hits: [[u32; PBINS]; BUCKETS],
    tries: [[u32; PBINS]; BUCKETS],
    /// Context of the last prediction offered to `mix` (consumed by `update`).
    pending: Option<(usize, usize)>,
}

impl MatchModel {
    /// `stream_len` sizes the hash table (at most `1 << max_bits` entries).
    pub fn new(min_len: usize, stream_len: usize, max_bits: u32) -> Self {
        let bits = (usize::BITS - stream_len.max(1).leading_zeros()).clamp(10, max_bits.max(10));
        MatchModel {
            min_len: min_len.max(1),
            mask: (1usize << bits) - 1,
            table: vec![0u32; 1usize << bits],
            ptr: 0,
            len: 0,
            hits: [[0; PBINS]; BUCKETS],
            tries: [[0; PBINS]; BUCKETS],
            pending: None,
        }
    }

    #[inline]
    fn hash(&self, ctx: &[u16]) -> usize {
        let mut h: u32 = 0x9E37_79B1;
        for &t in ctx {
            h = (h ^ t as u32).wrapping_mul(0x0100_0193).rotate_left(5);
        }
        (h.wrapping_mul(0x9E37_79B1) >> 7) as usize & self.mask
    }

    /// The token predicted for stream position `pos` (all of `hist[..pos]` is known).
    #[inline]
    pub fn predict(&self, hist: &[u16], pos: usize) -> Option<(u16, usize)> {
        if self.len > 0 && self.ptr < pos {
            Some((hist[self.ptr], bucket(self.len)))
        } else {
            None
        }
    }

    /// Mixes the prediction into `p` (a probability row) and returns whether it did.
    #[inline]
    pub fn mix(&mut self, p: &mut [f32], tok: u16, b: usize) -> bool {
        let t = tok as usize;
        if t >= p.len() {
            return false;
        }
        let pn = p[t];
        let c = pbin(pn);
        self.pending = Some((b, c));
        let ph = (self.hits[b][c] as f32 + 0.5) / (self.tries[b][c] as f32 + 1.0);
        // Only boost when the estimate beats the network (false for NaN, too).
        if ph.partial_cmp(&pn) != Some(std::cmp::Ordering::Greater) {
            return false;
        }
        let lam = (ph - pn) / (1.0 - pn);
        let keep = 1.0 - lam;
        for x in p.iter_mut() {
            *x *= keep;
        }
        p[t] += lam;
        true
    }

    /// Records that `hist[pos]` was coded (call after coding position `pos`).
    #[inline]
    pub fn update(&mut self, hist: &[u16], pos: usize) {
        let actual = hist[pos];
        if self.len > 0 && self.ptr < pos {
            let hit = hist[self.ptr] == actual;
            if let Some((b, c)) = self.pending.take() {
                self.tries[b][c] += 1;
                if hit {
                    self.hits[b][c] += 1;
                }
                if self.tries[b][c] >= 1024 {
                    self.tries[b][c] >>= 1;
                    self.hits[b][c] >>= 1;
                }
            }
            if hit {
                self.len = (self.len + 1).min(65535);
                self.ptr += 1;
            } else {
                self.len = 0;
            }
        }
        self.pending = None;
        let n = pos + 1; // tokens known now
        if n >= self.min_len {
            let ctx = &hist[n - self.min_len..n];
            let h = self.hash(ctx);
            if self.len == 0 {
                let cand = self.table[h] as usize;
                if cand >= self.min_len && cand < n && hist[cand - self.min_len..cand] == *ctx {
                    // Extend backwards to measure the real match length (bounded).
                    let mut l = self.min_len;
                    while l < 32 && cand > l && hist[cand - l - 1] == hist[n - l - 1] {
                        l += 1;
                    }
                    self.ptr = cand;
                    self.len = l as u32;
                }
            }
            self.table[h] = n as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_repeats_and_learns_confidence() {
        let line: Vec<u16> = b"2026-11-20 00:00:01 INFO request served in 12 ms\n".iter().map(|&b| b as u16).collect();
        let mut hist = Vec::new();
        for _ in 0..40 {
            hist.extend_from_slice(&line);
        }
        let mut m = MatchModel::new(6, hist.len(), 16);
        let mut correct = 0;
        let mut predicted = 0;
        for pos in 0..hist.len() {
            if let Some((t, b)) = m.predict(&hist, pos) {
                predicted += 1;
                if t == hist[pos] {
                    correct += 1;
                }
                let mut p = vec![1.0 / 256.0; 256];
                m.mix(&mut p, t, b);
            }
            m.update(&hist, pos);
        }
        assert!(predicted > hist.len() * 9 / 10, "{predicted}");
        assert!(correct * 100 > predicted * 98);
        // Mixing raises the predicted token's probability towards the learned hit rate.
        let mut p = vec![1.0 / 256.0; 256];
        let (t, b) = m.predict(&hist, hist.len()).unwrap();
        assert!(m.mix(&mut p, t, b));
        assert!(p[t as usize] > 0.9);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-4);
    }
}
