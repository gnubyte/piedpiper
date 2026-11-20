//! Synthetic VP9 streams for tests. Headers are written field by field; tile data comes from
//! pushing pseudo-random decisions through the tile parser itself and coding them with
//! libvpx's boolean coder, so every syntax path the parser knows gets exercised without
//! third-party clips. Decisions lean towards the more likely branch more often than the
//! coded probabilities say, which leaves room for the model to compress them.

use super::bool::BoolEncoder;
use super::stream::{frame_headers, parse_tiles};
use super::tables::TX_MODE_TO_BIGGEST_TX_SIZE;
use super::tile::{kind, BoolIo, TileIo};
use super::State;
use crate::bits::{BitWriter, R};
use crate::h264::mb::Hint;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

struct GenIo {
    rng: Rng,
    enc: BoolEncoder,
    first: bool,
    tiles: Vec<Vec<u8>>,
}

impl BoolIo for GenIo {
    fn read(&mut self, prob: u8, h: &Hint) -> R<u32> {
        let r = self.rng.next();
        // The tile marker must be 0; class and integer bits of motion vectors stay 0 to keep
        // vectors small.
        let bit = if self.first || h.kind == kind::MV_CLASS || h.kind == kind::MV_BITS {
            0
        } else if r & 3 != 0 {
            (prob < 128) as u32
        } else {
            ((r >> 8) & 255 >= prob as u32) as u32
        };
        self.first = false;
        self.enc.write(bit, prob);
        Ok(bit)
    }
}

impl TileIo for GenIo {
    fn begin(&mut self, _tile: usize) -> R<()> {
        self.enc = BoolEncoder::new_raw();
        self.first = true;
        Ok(())
    }

    fn end(&mut self, _tile: usize) -> R<()> {
        let t = std::mem::replace(&mut self.enc, BoolEncoder::new_raw()).finish();
        self.tiles.push(t);
        Ok(())
    }
}

/// What one synthetic frame looks like.
#[derive(Clone, Copy)]
pub struct FrameSpec {
    pub key: bool,
    pub show: bool,
    /// Reference slots refreshed (inter frames).
    pub refresh: u8,
    pub q: u32,
    /// 0..=4 (4 = per-block selection); ignored when `q == 0` (lossless).
    pub tx_mode: u8,
    pub switchable: bool,
    pub allow_hp: bool,
    /// ALTREF gets the opposite sign bias, enabling per-block compound prediction.
    pub compound: bool,
}

impl FrameSpec {
    pub fn key(q: u32) -> Self {
        FrameSpec { key: true, show: true, refresh: 0xFF, q, tx_mode: 4, switchable: false, allow_hp: false, compound: false }
    }
    pub fn inter(q: u32) -> Self {
        FrameSpec { key: false, show: true, refresh: 1, q, tx_mode: 4, switchable: true, allow_hp: true, compound: false }
    }
}

fn tile_bits(width: u32) -> (u32, u32) {
    let sb_cols = (width as usize).div_ceil(8).div_ceil(8);
    let mut min_log2 = 0u32;
    while (64usize << min_log2) < sb_cols {
        min_log2 += 1;
    }
    let mut max_log2 = 1u32;
    while (sb_cols >> max_log2) >= 4 {
        max_log2 += 1;
    }
    (min_log2, max_log2 - 1)
}

fn compressed_header(s: &FrameSpec) -> Vec<u8> {
    let mut e = BoolEncoder::new();
    let tx_mode = if s.q == 0 { 0 } else { s.tx_mode.min(4) };
    if s.q != 0 {
        let lit = tx_mode.min(3) as u32;
        e.write(lit >> 1, 128);
        e.write(lit & 1, 128);
        if lit == 3 {
            e.write((tx_mode == 4) as u32, 128);
        }
    }
    let no_update = |e: &mut BoolEncoder, n: usize| (0..n).for_each(|_| e.write(0, 252));
    if tx_mode == 4 {
        no_update(&mut e, 12);
    }
    for _ in 0..=TX_MODE_TO_BIGGEST_TX_SIZE[tx_mode as usize] {
        e.write(0, 128); // no coefficient probability updates
    }
    no_update(&mut e, 3);
    if !s.key {
        no_update(&mut e, 21);
        if s.switchable {
            no_update(&mut e, 8);
        }
        no_update(&mut e, 4);
        if s.compound {
            e.write(1, 128);
            e.write(1, 128); // REFERENCE_MODE_SELECT
            no_update(&mut e, 5 + 10 + 5);
        } else {
            no_update(&mut e, 10);
        }
        no_update(&mut e, 36 + 48);
        no_update(&mut e, 3 + 2 * 22 + 2 * 9 + if s.allow_hp { 4 } else { 0 });
    }
    e.finish()
}

#[allow(clippy::too_many_arguments)]
fn uncompressed_header(s: &FrameSpec, w: u32, h: u32, log2_cols: u32, log2_rows: u32, comp_len: usize) -> Vec<u8> {
    let mut b = BitWriter::new();
    b.put_bits(2, 2);
    b.put_bits(0, 2); // profile 0
    b.put(0); // show_existing_frame
    b.put(!s.key as u32);
    b.put(s.show as u32);
    b.put(0); // error_resilient_mode
    if s.key {
        b.put_bits(0x49_8342, 24);
        b.put_bits(2, 3); // BT.709
        b.put(0);
        b.put_bits(w - 1, 16);
        b.put_bits(h - 1, 16);
        b.put(0);
    } else {
        if !s.show {
            b.put(0); // intra_only
        }
        b.put_bits(0, 2); // reset_frame_context
        b.put_bits(s.refresh as u32, 8);
        for i in 0..3 {
            b.put_bits(i, 3);
            b.put((s.compound && i == 2) as u32);
        }
        b.put(1); // size from LAST
        b.put(0);
        b.put(s.allow_hp as u32);
        if s.switchable {
            b.put(1);
        } else {
            b.put(0);
            b.put_bits(2, 2); // EIGHTTAP_SHARP
        }
    }
    b.put(1); // refresh_frame_context
    b.put(1); // frame_parallel_decoding_mode
    b.put_bits(0, 2);
    b.put_bits(10, 6); // loop filter level
    b.put_bits(0, 3);
    b.put(1); // mode_ref_delta_enabled
    b.put(1); // update
    for k in 0..6 {
        b.put((k == 1) as u32);
        if k == 1 {
            b.put_bits(3, 6);
            b.put(1);
        }
    }
    b.put_bits(s.q, 8);
    b.put(0);
    b.put(0);
    b.put(0);
    b.put(0); // segmentation
    let (min_log2, max_log2) = tile_bits(w);
    let want = log2_cols.clamp(min_log2, max_log2);
    for k in min_log2..max_log2 {
        b.put((k < want) as u32);
        if k >= want {
            break;
        }
    }
    b.put((log2_rows > 0) as u32);
    if log2_rows > 0 {
        b.put((log2_rows > 1) as u32);
    }
    b.put_bits(comp_len as u32, 16);
    b.finish()
}

/// Generates the frames of a synthetic VP9 stream. A `None` entry is a frame that shows
/// reference slot 2 again (a one-byte show-existing frame). Frames listed in `superframes`
/// (pairs of consecutive indices) are packed into one block with a superframe index.
pub fn frames(seed: u64, w: u32, h: u32, log2_cols: u32, log2_rows: u32, specs: &[Option<FrameSpec>]) -> Vec<Vec<u8>> {
    let mut st = State::default();
    let mut io = GenIo { rng: Rng(seed ^ 0x9E37_79B9_7F4A_7C15), enc: BoolEncoder::new_raw(), first: true, tiles: Vec::new() };
    let mut out = Vec::new();
    for spec in specs {
        let Some(s) = spec else {
            out.push(vec![0x88 | 2]);
            let _ = frame_headers(&mut st, &[0x88 | 2]);
            continue;
        };
        let comp = compressed_header(s);
        let mut frame = uncompressed_header(s, w, h, log2_cols, log2_rows, comp.len());
        frame.extend_from_slice(&comp);
        let (hdr, fc) = frame_headers(&mut st, &frame).expect("synthetic header parses").expect("not show-existing");
        io.tiles.clear();
        parse_tiles(&mut st, &hdr, &fc, &mut io).expect("random decisions always parse");
        let n = io.tiles.len();
        for (i, t) in io.tiles.iter().enumerate() {
            if i + 1 < n {
                frame.extend_from_slice(&(t.len() as u32).to_be_bytes());
            }
            frame.extend_from_slice(t);
        }
        out.push(frame);
    }
    out
}

/// Packs frames into one block with a superframe index (4-byte sizes).
pub fn superframe(frames: &[Vec<u8>]) -> Vec<u8> {
    let marker = 0xC0 | (3 << 3) | (frames.len() as u8 - 1);
    let mut out: Vec<u8> = frames.concat();
    out.push(marker);
    for f in frames {
        out.extend_from_slice(&(f.len() as u32).to_le_bytes());
    }
    out.push(marker);
    out
}

/// A small WebM file covering key, inter, hidden, compound, show-existing, superframe and
/// lossless frames.
pub fn synthetic_webm(seed: u64, w: u32, h: u32, log2_cols: u32, log2_rows: u32) -> Vec<u8> {
    let mut hidden = FrameSpec::inter(90);
    hidden.show = false;
    hidden.refresh = 4;
    let mut comp = FrameSpec::inter(70);
    comp.compound = true;
    comp.allow_hp = false;
    let mut plain = FrameSpec::inter(120);
    plain.switchable = false;
    plain.tx_mode = 3;
    let specs = [
        Some(FrameSpec::key(60)),
        Some(FrameSpec::inter(80)),
        Some(hidden),
        Some(comp),
        None,
        Some(plain),
        Some(FrameSpec::key(0)),
        Some(FrameSpec::inter(0)),
    ];
    let f = frames(seed, w, h, log2_cols, log2_rows, &specs);
    let blocks = vec![f[0].clone(), f[1].clone(), superframe(&f[2..4]), f[4].clone(), f[5].clone(), f[6].clone(), f[7].clone()];
    crate::mkv::write_mkv_codec(b"V_VP9", None, &blocks, w, h)
}
