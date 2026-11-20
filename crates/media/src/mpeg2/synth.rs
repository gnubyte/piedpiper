//! Synthetic MPEG-2 streams for tests: random but valid syntax (I, P and B pictures, frame
//! and field motion, quantiser changes, coded block patterns, intra DC, run/level codes from
//! both coefficient tables, escapes) written with the codes of annex B.

use super::parse::{BACK, FOR, INTRA, PAT, QUANT};
use crate::bits::BitWriter;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

// the code tables are private to parse.rs; the generator keeps the few rows it needs
const MBT_I: [(u32, u32, u8); 2] = [(1, 1, INTRA), (1, 2, QUANT | INTRA)];
const MBT_P: [(u32, u32, u8); 7] =
    [(3, 5, INTRA), (1, 2, PAT), (1, 3, FOR), (1, 1, FOR | PAT), (1, 6, QUANT | INTRA), (1, 5, QUANT | PAT), (2, 5, QUANT | FOR | PAT)];
const MBT_B: [(u32, u32, u8); 11] = [
    (3, 5, INTRA),
    (2, 3, BACK),
    (3, 3, BACK | PAT),
    (2, 4, FOR),
    (3, 4, FOR | PAT),
    (2, 2, FOR | BACK),
    (3, 2, FOR | BACK | PAT),
    (1, 6, QUANT | INTRA),
    (2, 6, QUANT | BACK | PAT),
    (3, 6, QUANT | FOR | PAT),
    (2, 5, QUANT | FOR | BACK | PAT),
];
/// coded_block_pattern codes for cbp 1..=8 and 60..=63 (enough variety)
const CBP: [(u32, u32, u32); 8] = [(1, 0xb, 5), (2, 0x9, 5), (4, 0xd, 4), (8, 0xc, 4), (16, 0xb, 4), (32, 0xa, 4), (60, 0x7, 3), (63, 0xc, 6)];
/// motion_code magnitudes 0..=3
const MV: [(u32, u32); 4] = [(0x1, 1), (0x1, 2), (0x1, 3), (0x1, 4)];
const DC_LUM: [(u32, u32); 4] = [(0x4, 3), (0x0, 2), (0x1, 2), (0x5, 3)];
const DC_CHROMA: [(u32, u32); 4] = [(0x0, 2), (0x1, 2), (0x2, 2), (0x6, 3)];
/// (run, level, B.14 code, length, B.15 code, length)
const COEF: [(u32, u32, u32, u32, u32, u32); 6] =
    [(0, 1, 0x3, 2, 0x2, 2), (0, 2, 0x4, 4, 0x6, 3), (1, 1, 0x3, 3, 0x2, 3), (0, 3, 0x5, 5, 0x7, 4), (2, 1, 0x5, 4, 0x5, 5), (1, 2, 0x6, 6, 0x6, 5)];

fn align(w: &mut BitWriter) {
    while !w.bit_len().is_multiple_of(8) {
        w.put(0);
    }
}

fn start(out: &mut Vec<u8>, code: u8, w: BitWriter) {
    out.extend_from_slice(&[0, 0, 1, code]);
    out.extend_from_slice(&w.finish());
}

/// `n` frames of `mbw` x `mbh` macroblocks (GOP I B P B ..., a sequence header first).
pub fn frames(seed: u64, mbw: u32, mbh: u32, n: usize) -> Vec<Vec<u8>> {
    let mut r = Rng(seed ^ 0x2545_F491_4F6C_DD1D);
    let mut out = Vec::new();
    for i in 0..n {
        let mut f = Vec::new();
        let ptype = match i % 4 {
            0 => 1,
            2 => 2,
            _ => 3,
        };
        if i == 0 {
            let mut w = BitWriter::new();
            w.put_bits(mbw * 16, 12);
            w.put_bits(mbh * 16, 12);
            w.put_bits(2, 4); // aspect
            w.put_bits(4, 4); // 29.97
            w.put_bits(20000, 18);
            w.put(1);
            w.put_bits(112, 10);
            w.put(0);
            w.put(0);
            w.put(0);
            start(&mut f, 0xB3, w);
            let mut w = BitWriter::new();
            w.put_bits(1, 4);
            w.put_bits(0x48, 8);
            w.put(0); // interlaced sequence
            w.put_bits(1, 2); // 4:2:0
            w.put_bits(0, 4);
            w.put_bits(0, 12);
            w.put(1);
            w.put_bits(0, 8);
            w.put(0);
            w.put_bits(0, 7);
            align(&mut w);
            start(&mut f, 0xB5, w);
        }
        let mut w = BitWriter::new();
        w.put_bits(i as u32 & 1023, 10);
        w.put_bits(ptype, 3);
        w.put_bits(0xFFFF, 16);
        if ptype >= 2 {
            w.put(0);
            w.put_bits(7, 3);
        }
        if ptype == 3 {
            w.put(0);
            w.put_bits(7, 3);
        }
        w.put(0);
        align(&mut w);
        start(&mut f, 0x00, w);
        let fpfd = r.below(2) == 1;
        let intra_vlc = r.below(2) == 1;
        let f_code = |s: u32| if ptype == 1 || (ptype == 2 && s == 1) { 15 } else { 2 };
        let mut w = BitWriter::new();
        w.put_bits(8, 4);
        for s in 0..2 {
            w.put_bits(f_code(s), 4);
            w.put_bits(f_code(s), 4);
        }
        w.put_bits(0, 2); // dc precision
        w.put_bits(3, 2); // frame picture
        w.put(1);
        w.put(fpfd as u32);
        w.put(0); // no concealment vectors
        w.put(0);
        w.put(intra_vlc as u32);
        w.put(0);
        w.put(0);
        w.put(0);
        w.put(fpfd as u32);
        w.put(0);
        align(&mut w);
        start(&mut f, 0xB5, w);
        for row in 0..mbh {
            let mut w = BitWriter::new();
            w.put_bits(1 + r.below(31), 5);
            w.put(0);
            for _ in 0..mbw {
                w.put(1); // address increment 1
                let (code, len, mbt) = match ptype {
                    1 => MBT_I[r.below(2) as usize],
                    2 => MBT_P[r.below(7) as usize],
                    _ => MBT_B[r.below(11) as usize],
                };
                w.put_bits(code, len);
                let intra = mbt & INTRA != 0;
                let mut field = false;
                if mbt & (FOR | BACK) != 0 && !fpfd {
                    field = r.below(2) == 1;
                    w.put_bits(if field { 1 } else { 2 }, 2);
                }
                if !fpfd && (intra || mbt & PAT != 0) {
                    w.put(r.below(2));
                }
                if mbt & QUANT != 0 {
                    w.put_bits(1 + r.below(31), 5);
                }
                for s in 0..2 {
                    if mbt & [FOR, BACK][s] == 0 {
                        continue;
                    }
                    for _ in 0..if field { 2 } else { 1 } {
                        if field {
                            w.put(r.below(2));
                        }
                        for _ in 0..2 {
                            let m = r.below(4);
                            w.put_bits(MV[m as usize].0, MV[m as usize].1);
                            if m != 0 {
                                w.put(r.below(2)); // sign
                                w.put(r.below(2)); // residual (f_code 2)
                            }
                        }
                    }
                }
                let cbp = if mbt & PAT != 0 {
                    let (v, code, len) = CBP[r.below(8) as usize];
                    w.put_bits(code, len);
                    v
                } else if intra {
                    63
                } else {
                    0
                };
                for b in 0..6 {
                    if cbp & (1 << (5 - b)) == 0 {
                        continue;
                    }
                    if intra {
                        let size = r.below(4);
                        let (code, len) = if b < 4 { DC_LUM[size as usize] } else { DC_CHROMA[size as usize] };
                        w.put_bits(code, len);
                        if size > 0 {
                            w.put_bits(r.below(1 << size), size);
                        }
                    }
                    let b15 = intra && intra_vlc;
                    let mut pos = if intra { 1 } else { 0 };
                    let count = if intra { r.below(4) } else { 1 + r.below(4) };
                    for k in 0..count {
                        if r.below(10) == 0 {
                            // escape: run 0..3, level 1..300
                            let run = r.below(4);
                            if pos + run > 63 {
                                break;
                            }
                            w.put_bits(1, 6);
                            w.put_bits(run, 6);
                            let lv = 1 + r.below(300);
                            w.put_bits(if r.below(2) == 1 { 4096 - lv } else { lv }, 12);
                            pos += run + 1;
                            continue;
                        }
                        let (run, _, c14, l14, c15, l15) = COEF[r.below(6) as usize];
                        if pos + run > 63 {
                            break;
                        }
                        if k == 0 && !intra && run == 0 && c14 == 0x3 {
                            w.put(1); // first-coefficient code for run 0, level 1
                        } else if b15 {
                            w.put_bits(c15, l15);
                        } else {
                            w.put_bits(c14, l14);
                        }
                        w.put(r.below(2)); // sign
                        pos += run + 1;
                    }
                    if b15 {
                        w.put_bits(0x6, 4);
                    } else {
                        w.put_bits(0x2, 2);
                    }
                }
            }
            align(&mut w);
            start(&mut f, (row + 1) as u8, w);
        }
        out.push(f);
    }
    out
}

/// The frames in a Matroska file with codec ID `V_MPEG2`.
pub fn synthetic_mkv(seed: u64, mbw: u32, mbh: u32, n: usize) -> Vec<u8> {
    let f = frames(seed, mbw, mbh, n);
    crate::mkv::write_mkv_codec(b"V_MPEG2", None, &f, mbw * 16, mbh * 16)
}
