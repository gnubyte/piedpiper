//! Reversible tokenizers.
//!
//! * [`Tokenizer::Bytes`] — 256 symbols, works for any data.
//! * [`Tokenizer::Bpe`] — word-aware byte-pair encoding learned from the input itself
//!   (TufaZip uses a 4K-symbol word vocabulary for enwik9). Pieces are split at word
//!   boundaries first (letters / digits / punctuation, with a leading space attached), then
//!   merged. The merge list is stored in the file; the decoder only needs `token → bytes`.
//!
//! Only the encoder ever *builds* a vocabulary, so this code need not be bit-exact across
//! platforms — but it is deterministic anyway (no hash-order dependence), which keeps
//! compressed output reproducible.

use crate::error::{Error, Result};
use crate::format::{get_varint, put_varint};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

#[derive(Clone)]
pub enum Tokenizer {
    Bytes,
    Bpe(Bpe),
}

#[derive(Clone)]
pub struct Bpe {
    merges: Vec<(u16, u16)>,
    tokens: Vec<Vec<u8>>,
}

impl Tokenizer {
    pub fn vocab_size(&self) -> usize {
        match self {
            Tokenizer::Bytes => 256,
            Tokenizer::Bpe(b) => 256 + b.merges.len(),
        }
    }

    pub fn kind(&self) -> u8 {
        match self {
            Tokenizer::Bytes => 0,
            Tokenizer::Bpe(_) => 1,
        }
    }

    pub fn name(&self) -> String {
        match self {
            Tokenizer::Bytes => "bytes".into(),
            Tokenizer::Bpe(b) => format!("bpe-{}", 256 + b.merges.len()),
        }
    }

    pub fn encode(&self, data: &[u8]) -> Vec<u16> {
        match self {
            Tokenizer::Bytes => data.iter().map(|&b| b as u16).collect(),
            Tokenizer::Bpe(b) => b.encode(data),
        }
    }

    pub fn decode_into(&self, tokens: &[u16], out: &mut Vec<u8>) {
        match self {
            Tokenizer::Bytes => out.extend(tokens.iter().map(|&t| t as u8)),
            Tokenizer::Bpe(b) => {
                for &t in tokens {
                    out.extend_from_slice(&b.tokens[t as usize]);
                }
            }
        }
    }

    pub fn write(&self, out: &mut Vec<u8>) {
        out.push(self.kind());
        if let Tokenizer::Bpe(b) = self {
            put_varint(out, b.merges.len() as u64);
            for &(x, y) in &b.merges {
                put_varint(out, x as u64);
                put_varint(out, y as u64);
            }
        }
    }

    pub fn read(buf: &[u8], pos: &mut usize) -> Result<Tokenizer> {
        let kind = *buf.get(*pos).ok_or_else(|| Error::Format("truncated header".into()))?;
        *pos += 1;
        match kind {
            0 => Ok(Tokenizer::Bytes),
            1 => {
                let n = get_varint(buf, pos)? as usize;
                if n > 65535 - 256 {
                    return Err(Error::Format("BPE vocabulary too large".into()));
                }
                let mut merges = Vec::with_capacity(n);
                for i in 0..n {
                    let a = get_varint(buf, pos)?;
                    let b = get_varint(buf, pos)?;
                    let lim = 256 + i as u64;
                    if a >= lim || b >= lim {
                        return Err(Error::Format("invalid BPE merge".into()));
                    }
                    merges.push((a as u16, b as u16));
                }
                Ok(Tokenizer::Bpe(Bpe::from_merges(merges)))
            }
            k => Err(Error::Format(format!("unknown tokenizer kind {k}"))),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Letter,
    Digit,
    Space,
    Newline,
    Other,
}

#[inline]
fn class(b: u8) -> Class {
    match b {
        b'a'..=b'z' | b'A'..=b'Z' | 0x80..=0xFF => Class::Letter,
        b'0'..=b'9' => Class::Digit,
        b' ' => Class::Space,
        b'\n' | b'\r' => Class::Newline,
        _ => Class::Other,
    }
}

const MAX_RUN: usize = 24;

#[inline]
fn run_end(data: &[u8], i: usize, c: Class) -> usize {
    let lim = match c {
        Class::Digit => 3,
        Class::Other => 4,
        _ => MAX_RUN,
    };
    let mut j = i;
    while j < data.len() && class(data[j]) == c && j - i < lim {
        j += 1;
    }
    j
}

/// Splits `data` into word-like pieces: `" word"`, `"word"`, digit groups of up to 3,
/// short punctuation runs (optionally with a leading space), newline runs, space runs.
pub fn for_each_piece<'a>(data: &'a [u8], mut f: impl FnMut(&'a [u8])) {
    let n = data.len();
    let mut i = 0;
    while i < n {
        let start = i;
        match class(data[i]) {
            Class::Space => {
                if i + 1 < n && !matches!(class(data[i + 1]), Class::Space | Class::Newline) {
                    i = run_end(data, i + 1, class(data[i + 1]));
                } else {
                    let mut j = i;
                    while j < n && data[j] == b' ' && j - i < MAX_RUN {
                        j += 1;
                    }
                    if j < n && j - i > 1 && !matches!(class(data[j]), Class::Space | Class::Newline) {
                        j -= 1; // leave one space for the following word
                    }
                    i = j;
                }
            }
            Class::Newline => {
                while i < n && class(data[i]) == Class::Newline && i - start < MAX_RUN {
                    i += 1;
                }
            }
            c => i = run_end(data, i, c),
        }
        f(&data[start..i]);
    }
}

impl Bpe {
    fn from_merges(merges: Vec<(u16, u16)>) -> Bpe {
        let mut tokens: Vec<Vec<u8>> = (0..256u16).map(|b| vec![b as u8]).collect();
        for &(a, b) in &merges {
            let mut t = tokens[a as usize].clone();
            t.extend_from_slice(&tokens[b as usize]);
            tokens.push(t);
        }
        Bpe { merges, tokens }
    }

    /// Learns up to `vocab - 256` merges from `data`, stopping early once the best pair
    /// occurs fewer than `min_count` times.
    pub fn train(data: &[u8], vocab: usize, min_count: u64) -> Bpe {
        let target = vocab.clamp(256, 65535) - 256;
        let mut counts: HashMap<&[u8], u64> = HashMap::new();
        for_each_piece(data, |p| {
            if p.len() >= 2 {
                *counts.entry(p).or_insert(0) += 1;
            }
        });
        let mut words: Vec<(&[u8], u64)> = counts.into_iter().collect();
        words.sort_unstable();
        let freqs: Vec<i64> = words.iter().map(|w| w.1 as i64).collect();
        let mut seqs: Vec<Vec<u16>> = words.iter().map(|w| w.0.iter().map(|&b| b as u16).collect()).collect();
        drop(words);

        let mut pair_count: HashMap<(u16, u16), i64> = HashMap::new();
        let mut pair_words: HashMap<(u16, u16), Vec<u32>> = HashMap::new();
        for (wi, s) in seqs.iter().enumerate() {
            for k in 0..s.len() - 1 {
                let pr = (s[k], s[k + 1]);
                *pair_count.entry(pr).or_insert(0) += freqs[wi];
                pair_words.entry(pr).or_default().push(wi as u32);
            }
        }
        let mut heap: BinaryHeap<(i64, Reverse<(u16, u16)>)> =
            pair_count.iter().map(|(&pr, &c)| (c, Reverse(pr))).collect();

        let mut merges: Vec<(u16, u16)> = Vec::with_capacity(target);
        while merges.len() < target {
            let Some((c, Reverse(pr))) = heap.pop() else { break };
            let cur = pair_count.get(&pr).copied().unwrap_or(0);
            if cur != c {
                if cur > 0 {
                    heap.push((cur, Reverse(pr)));
                }
                continue;
            }
            if (c as u64) < min_count.max(2) {
                break;
            }
            let new_id = (256 + merges.len()) as u16;
            merges.push(pr);
            pair_count.remove(&pr);
            let mut wl = pair_words.remove(&pr).unwrap_or_default();
            wl.sort_unstable();
            wl.dedup();
            for wi in wl {
                let wi_us = wi as usize;
                let s = &seqs[wi_us];
                if !s.windows(2).any(|w| w[0] == pr.0 && w[1] == pr.1) {
                    continue;
                }
                let f = freqs[wi_us];
                for k in 0..s.len() - 1 {
                    let p2 = (s[k], s[k + 1]);
                    if p2 != pr {
                        if let Some(e) = pair_count.get_mut(&p2) {
                            *e -= f;
                        }
                    }
                }
                let mut merged = Vec::with_capacity(s.len());
                let mut i = 0;
                while i < s.len() {
                    if i + 1 < s.len() && s[i] == pr.0 && s[i + 1] == pr.1 {
                        merged.push(new_id);
                        i += 2;
                    } else {
                        merged.push(s[i]);
                        i += 1;
                    }
                }
                for k in 0..merged.len().saturating_sub(1) {
                    let p2 = (merged[k], merged[k + 1]);
                    let e = pair_count.entry(p2).or_insert(0);
                    *e += f;
                    if p2.0 == new_id || p2.1 == new_id {
                        heap.push((*e, Reverse(p2)));
                        pair_words.entry(p2).or_default().push(wi);
                    }
                }
                seqs[wi_us] = merged;
            }
        }
        Bpe::from_merges(merges)
    }

    fn encode(&self, data: &[u8]) -> Vec<u16> {
        let ranks: HashMap<(u16, u16), u16> =
            self.merges.iter().enumerate().map(|(i, &p)| (p, i as u16)).collect();
        let mut cache: HashMap<&[u8], Vec<u16>> = HashMap::new();
        let mut out = Vec::with_capacity(data.len() / 3 + 16);
        for_each_piece(data, |p| {
            if p.len() == 1 {
                out.push(p[0] as u16);
            } else if let Some(t) = cache.get(p) {
                out.extend_from_slice(t);
            } else {
                let t = self.encode_piece(p, &ranks);
                out.extend_from_slice(&t);
                cache.insert(p, t);
            }
        });
        out
    }

    fn encode_piece(&self, p: &[u8], ranks: &HashMap<(u16, u16), u16>) -> Vec<u16> {
        let mut s: Vec<u16> = p.iter().map(|&b| b as u16).collect();
        while s.len() > 1 {
            let mut best: Option<u16> = None;
            for k in 0..s.len() - 1 {
                if let Some(&r) = ranks.get(&(s[k], s[k + 1])) {
                    if best.is_none_or(|b| r < b) {
                        best = Some(r);
                    }
                }
            }
            let Some(r) = best else { break };
            let pr = self.merges[r as usize];
            let id = 256 + r;
            let mut merged = Vec::with_capacity(s.len());
            let mut i = 0;
            while i < s.len() {
                if i + 1 < s.len() && s[i] == pr.0 && s[i + 1] == pr.1 {
                    merged.push(id);
                    i += 2;
                } else {
                    merged.push(s[i]);
                    i += 1;
                }
            }
            s = merged;
        }
        s
    }
}

/// Length of the valid UTF-8 sequence starting at `s[0]` (a byte >= 0x80), or 0.
fn utf8_len(s: &[u8]) -> usize {
    let need = match s[0] {
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => return 0,
    };
    if s.len() < need || !s[1..need].iter().all(|&c| (0x80..=0xBF).contains(&c)) {
        return 0;
    }
    need
}

/// Heuristic: is this data text (prose, code, markup) rather than binary?
///
/// Text has almost no control bytes, some whitespace, plenty of letters, and its bytes
/// at or above 0x80 mostly form valid UTF-8 (a few stray Latin-1 bytes are tolerated). Raw
/// images with light backgrounds (long runs of 0xF0-0xFF) or mid-gray tones
/// (printable-looking bytes but no whitespace) are rejected.
pub fn looks_like_text(data: &[u8]) -> bool {
    let s = &data[..data.len().min(1 << 22)];
    let n = s.len();
    if n == 0 {
        return false;
    }
    let (mut ctrl, mut ws, mut letters, mut invalid) = (0usize, 0usize, 0usize, 0usize);
    let mut digits = 0usize;
    let mut i = 0;
    while i < n {
        match s[i] {
            b'\t' | b'\n' | b'\r' | b' ' => ws += 1,
            0x00..=0x1F | 0x7F => ctrl += 1,
            b'a'..=b'z' | b'A'..=b'Z' => letters += 1,
            b'0'..=b'9' => digits += 1,
            0x21..=0x7E => {}
            _ => {
                let len = utf8_len(&s[i..]);
                if len >= 2 {
                    letters += len;
                    i += len;
                    continue;
                }
                invalid += 1;
            }
        }
        i += 1;
    }
    // Letters and digits together: CSV and JSON records are mostly numbers (19-27% letters)
    // yet gain 5-10% from the text priors; binary data fails the control-byte and UTF-8 tests.
    ctrl * 100 <= n && invalid * 10 <= n && ws * 50 >= n && (letters + digits) * 4 >= n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_detection() {
        assert!(looks_like_text(b"The quick brown fox jumps over the lazy dog.\n".repeat(50).as_slice()));
        assert!(looks_like_text("Ein sch\u{f6}ner Tag, tr\u{e8}s caf\u{e9}, \u{65e5}\u{672c}\u{8a9e} OK. ".repeat(50).as_bytes()));
        // Latin-1 (not UTF-8) accents are tolerated.
        let latin1: Vec<u8> = b"Un caf\xe9 tr\xe8s chaud, s'il vous pla\xeet. ".repeat(50);
        assert!(looks_like_text(&latin1));
        assert!(looks_like_text(b"fn main() {\n    println!(\"hi\");\n}\n".repeat(50).as_slice()));
        // Number-heavy records are text too.
        assert!(looks_like_text(b"2026-11-20,10432,17.95,3,shipped\n4411,2026-11-21,8.50,12,open\n".repeat(50).as_slice()));
        assert!(looks_like_text(b"{\"id\": 10432, \"qty\": 3, \"price\": 17.95}\n".repeat(50).as_slice()));
        // Raw light-background RGB (0xF0-0xFF), mid-gray pixels, and random bytes are binary.
        let light: Vec<u8> = (0..30000u32).map(|i| 0xF0 + (i * 7 % 16) as u8).collect();
        assert!(!looks_like_text(&light));
        let gray: Vec<u8> = (0..30000u32).map(|i| 0x40 + (i * 13 % 60) as u8).collect();
        assert!(!looks_like_text(&gray));
        let mut x = 1u32;
        let noise: Vec<u8> = (0..30000).map(|_| { x = x.wrapping_mul(1664525).wrapping_add(1013904223); (x >> 24) as u8 }).collect();
        assert!(!looks_like_text(&noise));
        assert!(!looks_like_text(b""));
    }

    #[test]
    fn pieces_cover_input() {
        let text = b"Hello,  world!! 12345 foo\n\n  bar\tbaz caf\xc3\xa9   x";
        let mut joined = Vec::new();
        let mut ps = Vec::new();
        for_each_piece(text, |p| {
            joined.extend_from_slice(p);
            ps.push(String::from_utf8_lossy(p).into_owned());
        });
        assert_eq!(&joined[..], &text[..]);
        assert!(ps.contains(&" world".to_string()));
        assert!(ps.contains(&" bar".to_string()));
    }

    #[test]
    fn bpe_roundtrip() {
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(200)
            + "Then the other fox, the brown one, jumped over the quick dog 1234567.\n";
        let bpe = Bpe::train(text.as_bytes(), 400, 2);
        assert!(bpe.merges.len() > 20);
        let tok = Tokenizer::Bpe(bpe);
        let ids = tok.encode(text.as_bytes());
        assert!(ids.len() < text.len() / 3);
        let mut out = Vec::new();
        tok.decode_into(&ids, &mut out);
        assert_eq!(out, text.as_bytes());
        let mut hdr = Vec::new();
        tok.write(&mut hdr);
        let mut pos = 0;
        let back = Tokenizer::read(&hdr, &mut pos).unwrap();
        assert_eq!(pos, hdr.len());
        assert_eq!(back.encode(text.as_bytes()), ids);
    }
}
