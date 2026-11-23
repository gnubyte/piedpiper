//! The recompressed format and the compress/decompress drivers.
//!
//! Layout: the 4-byte magic `PPM3` or `PPM4` (H.264) or `PPH3` (HEVC), then a header stream (varint length + arithmetic-coded
//! data) holding the file length, NAL length-field size, video sample table, segment
//! boundaries and every byte that is not part of a video sample (container boxes, audio).
//! Runs of such bytes of 512 or more that look random (compressed audio) are flagged and
//! coded at one bit per bit, without the model.
//! Then, per segment: one byte giving its model's hash-table size (log2 of the number of
//! slots), a varint length and an arithmetic-coded stream with the in-band SPS/PPS units in
//! effect at its start and, for every NAL unit of its samples, the header byte(s) and either
//! the raw bytes or, for slices that recode exactly, the slice-header bits, the CABAC bins
//! (coded with the context-mixing model), a tail fix-up and the number of trailing
//! `cabac_zero_word` bytes.
//!
//! Segments are runs of about 3 MB of video, chosen from the content alone. Each has its own
//! model, so they are coded and decoded in parallel without the thread count affecting the
//! output.
//!
//! `PPM4` streams (H.264 with CABAC) also rebuild the decoded pictures on both sides and give
//! the model what they show (`h264::recon`); their segments start at key frames so that every
//! segment has its reference pictures.
//!
//! A slice is recoded only after re-encoding its bins with CABAC has reproduced the original
//! payload bit for bit; anything else is stored as raw bytes. Decompression therefore always
//! rebuilds the exact input file.

use crate::bits::{escape, unescape, Bad, BitWriter, R};
use crate::h264::cabac;
use crate::h264::cavlc;
use crate::h264::mb::{BinIo, Hint, Picture, SliceParser};
use crate::h264::{ParamSets, Pps, SliceHeader, Sps};
use crate::hevc;
use crate::model::{
    code_bin, code_byte, code_flag, code_num, is_simple, sym, ArithDec, ArithEnc, ByteCtx, Coder, Model, Slots,
    MAX_TABLE_BITS, MIN_TABLE_BITS,
};
use crate::mp4;

pub const MAGIC: &[u8; 4] = b"PPM3";
/// Magic of H.264 streams whose model also uses the decoded pictures (same layout; see
/// `h264::recon`).
pub const MAGIC4: &[u8; 4] = b"PPM4";
/// Magic of HEVC streams, which share the `PPM3` layout.
pub const MAGIC_HEVC: &[u8; 4] = b"PPH3";

/// The video codec of a recompressed stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    Avc,
    Hevc,
}

impl Codec {
    /// Length of the NAL unit header in bytes.
    fn hdr_len(self) -> usize {
        match self {
            Codec::Avc => 1,
            Codec::Hevc => 2,
        }
    }

    fn nal_type(self, nal: &[u8]) -> u32 {
        match self {
            Codec::Avc => (nal[0] & 31) as u32,
            Codec::Hevc => hevc::nal_type(nal),
        }
    }

    fn is_slice(self, t: u32) -> bool {
        match self {
            Codec::Avc => t == 1 || t == 5,
            Codec::Hevc => hevc::is_slice(t),
        }
    }

    fn magic(self) -> &'static [u8; 4] {
        match self {
            Codec::Avc => MAGIC,
            Codec::Hevc => MAGIC_HEVC,
        }
    }
}

/// The first video track this module recodes: H.264 if the file has one, else HEVC.
fn find_track(input: &[u8]) -> Option<(Codec, mp4::AvcTrack)> {
    match mp4::find_avc_track(input) {
        Some(t) => Some((Codec::Avc, t)),
        None => mp4::find_hevc_track(input).map(|t| (Codec::Hevc, t)),
    }
}

fn find_codec_track(input: &[u8], codec: Codec) -> Option<mp4::AvcTrack> {
    match codec {
        Codec::Avc => mp4::find_avc_track(input),
        Codec::Hevc => mp4::find_hevc_track(input),
    }
}

/// Per-segment parser state of either codec.
enum SegState {
    Avc { ps: ParamSets, pic: Picture, cpic: Box<cavlc::Picture> },
    Hevc(Box<hevc::recode::State>),
}

impl SegState {
    fn new(codec: Codec, rec: bool) -> Self {
        match codec {
            Codec::Avc => {
                let mut pic = Picture::default();
                pic.rec = rec.then(Box::<crate::h264::recon::Recon>::default);
                SegState::Avc { ps: ParamSets::default(), pic, cpic: Box::default() }
            }
            Codec::Hevc => SegState::Hevc(Box::default()),
        }
    }

    /// Picture management for a slice NAL unit stored raw (decoder side; the compressor did
    /// the same in [`verify_slice`] before it knew the slice would be stored raw).
    fn rec_raw_slice(&mut self, nal: &[u8]) {
        if let SegState::Avc { ps, pic, .. } = self {
            rec_begin(nal, ps, pic);
        }
    }

    /// Records a parameter-set NAL unit; other NAL units are ignored.
    fn add_param(&mut self, nal: &[u8]) {
        match self {
            SegState::Avc { ps, .. } => ps.add_nal(nal),
            SegState::Hevc(s) => s.add_param_nal(nal),
        }
    }

    /// Verifies a slice NAL unit, leaving its bins in `recs`.
    fn verify(&mut self, nal: &[u8], recs: &mut Vec<Rec>) -> Result<Verified, usize> {
        match self {
            SegState::Avc { ps, pic, cpic } => verify_slice(nal, ps, pic, cpic, recs),
            SegState::Hevc(s) => hevc::recode::verify_slice(nal, s, recs),
        }
    }

    /// Bookkeeping after a slice NAL unit has been coded (HEVC dependent slice segments).
    fn after_slice(&mut self, nal: &[u8], recoded: bool) {
        if let SegState::Hevc(s) = self {
            s.after_slice(nal, recoded);
        }
    }
}

const N_FILELEN: u32 = 1;
const N_NLS: u32 = 2;
const N_NSAMP: u32 = 3;
const N_GAP: u32 = 4;
const N_SIZE: u32 = 5;
const N_NALLEN: u32 = 6;
const N_HBITS: u32 = 7;
const N_TZ: u32 = 8;
const N_DROP: u32 = 9;
const N_FIX: u32 = 10;
const F_MODE: u32 = 1;
const F_RAW: u32 = 2;

/// Why a slice was stored raw instead of recoded.
pub const FAIL_NAMES: [&str; 7] =
    ["escaping", "header/unsupported", "alignment", "syntax/PCM", "re-encode mismatch", "trailing data", "other"];

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub samples: usize,
    pub nals: usize,
    pub slices: usize,
    pub recoded: usize,
    /// Original size of the recoded slice NAL units.
    pub slice_bytes: usize,
    pub raw_nal_bytes: usize,
    pub other_bytes: usize,
    pub bins: u64,
    pub fails: [usize; 7],
    pub out_header: usize,
    pub out_other: usize,
    pub out_slices: usize,
    pub out_raw: usize,
    /// Per bin kind: (bins, bits under CABAC's own probabilities, bits under our model).
    pub kind_cost: Vec<(u64, f64, f64)>,
    /// Independently coded runs of samples.
    pub segments: usize,
    /// The video track is HEVC (`kind_cost` then indexes `hevc::ctu::kind`).
    pub hevc: bool,
}

impl Stats {
    fn merge(&mut self, o: &Stats) {
        self.nals += o.nals;
        self.slices += o.slices;
        self.recoded += o.recoded;
        self.slice_bytes += o.slice_bytes;
        self.raw_nal_bytes += o.raw_nal_bytes;
        self.bins += o.bins;
        for (a, b) in self.fails.iter_mut().zip(o.fails) {
            *a += b;
        }
        self.out_slices += o.out_slices;
        self.out_raw += o.out_raw;
        if self.kind_cost.len() < o.kind_cost.len() {
            self.kind_cost.resize(o.kind_cost.len(), (0, 0.0, 0.0));
        }
        for (a, b) in self.kind_cost.iter_mut().zip(&o.kind_cost) {
            a.0 += b.0;
            a.1 += b.1;
            a.2 += b.2;
        }
    }

    fn add_cost(&mut self, r: &Rec, state: Option<u8>, p: u32) {
        if self.kind_cost.is_empty() {
            self.kind_cost = vec![(0, 0.0, 0.0); 32];
        }
        let kc = &mut self.kind_cost[(r.h.kind as usize).min(31)];
        let pm = if r.bin == 1 { p as f64 / 65536.0 } else { 1.0 - p as f64 / 65536.0 };
        let pc = match state {
            Some(s) => {
                let p1 = crate::model::cabac_p1(s) as f64 / 65536.0;
                if r.bin == 1 {
                    p1
                } else {
                    1.0 - p1
                }
            }
            None if r.typ == 1 => 0.5,
            None => 0.995,
        };
        kc.0 += 1;
        kc.1 -= pc.log2();
        kc.2 -= pm.log2();
    }
}

pub(crate) struct Rec {
    h: Hint,
    state: u8,
    typ: u8,
    bin: u8,
}

/// Decodes the original CABAC payload, re-encodes it, and records every bin.
pub(crate) struct VerifyIo<'a, 'r> {
    dec: cabac::Decoder<'a>,
    enc: cabac::Encoder,
    recs: &'r mut Vec<Rec>,
}

impl<'a, 'r> VerifyIo<'a, 'r> {
    /// Starts decoding `rbsp` at byte `start`, recording bins into `recs` (cleared first).
    pub(crate) fn new(rbsp: &'a [u8], start: usize, recs: &'r mut Vec<Rec>) -> Self {
        recs.clear();
        VerifyIo { dec: cabac::Decoder::new(rbsp, start), enc: cabac::Encoder::new(), recs }
    }

    /// The re-encoded payload.
    pub(crate) fn finish(self) -> Vec<u8> {
        self.enc.w.finish()
    }
}

impl BinIo for VerifyIo<'_, '_> {
    fn decision(&mut self, state: u8, h: &Hint) -> R<u32> {
        if self.dec.overrun() {
            return Err(Bad);
        }
        let b = self.dec.decision(state);
        self.enc.decision(state, b);
        self.recs.push(Rec { h: *h, state, typ: 0, bin: b as u8 });
        Ok(b)
    }

    fn bypass(&mut self, h: &Hint) -> R<u32> {
        if self.dec.overrun() {
            return Err(Bad);
        }
        let b = self.dec.bypass();
        self.enc.bypass(b);
        self.recs.push(Rec { h: *h, state: 0, typ: 1, bin: b as u8 });
        Ok(b)
    }

    fn terminate(&mut self, h: &Hint) -> R<u32> {
        if self.dec.overrun() {
            return Err(Bad);
        }
        let b = self.dec.terminate();
        self.enc.terminate(b);
        self.recs.push(Rec { h: *h, state: 0, typ: 2, bin: b as u8 });
        Ok(b)
    }

    fn end_substream(&mut self) -> R<()> {
        if self.dec.overrun() || self.dec.terminate() != 1 {
            return Err(Bad);
        }
        self.enc.terminate(1);
        self.dec.restart();
        self.enc.restart();
        Ok(())
    }
}

/// Decodes bins from the model and re-encodes them with CABAC.
pub(crate) struct ModelIo<'a, 'd> {
    m: &'a mut Model,
    c: &'a mut ArithDec<'d>,
    pub(crate) enc: cabac::Encoder,
    /// Recently announced bins and their table positions (already prefetched).
    soon: [(Hint, Slots); 4],
    soon_i: usize,
}

impl<'a, 'd> ModelIo<'a, 'd> {
    pub(crate) fn new(m: &'a mut Model, c: &'a mut ArithDec<'d>) -> Self {
        let none = Hint { kind: u8::MAX, ..Default::default() };
        ModelIo { m, c, enc: cabac::Encoder::new(), soon: [(none, Slots::default()); 4], soon_i: 0 }
    }

    #[inline]
    fn bin(&mut self, h: &Hint, state: Option<u8>) -> u32 {
        let p = if is_simple(h) {
            self.m.predict(h, state)
        } else {
            match self.soon.iter().find(|e| e.0 == *h) {
                Some(e) => {
                    let s = e.1;
                    self.m.predict_at(&s, h, state)
                }
                None => self.m.predict(h, state),
            }
        };
        let b = self.c.decode(p);
        self.m.update(b);
        b
    }
}

impl BinIo for ModelIo<'_, '_> {
    fn decision(&mut self, state: u8, h: &Hint) -> R<u32> {
        if self.c.overrun() {
            return Err(Bad);
        }
        let b = self.bin(h, Some(state));
        self.enc.decision(state, b);
        Ok(b)
    }

    fn bypass(&mut self, h: &Hint) -> R<u32> {
        if self.c.overrun() {
            return Err(Bad);
        }
        let b = self.bin(h, None);
        self.enc.bypass(b);
        Ok(b)
    }

    fn terminate(&mut self, h: &Hint) -> R<u32> {
        if self.c.overrun() {
            return Err(Bad);
        }
        let b = self.bin(h, None);
        self.enc.terminate(b);
        Ok(b)
    }

    fn end_substream(&mut self) -> R<()> {
        self.enc.terminate(1);
        self.enc.restart();
        Ok(())
    }

    #[inline]
    fn prefetch(&mut self, h: &Hint) {
        let s = self.m.slots(h);
        self.m.prefetch(&s);
        self.soon[self.soon_i & 3] = (*h, s);
        self.soon_i = self.soon_i.wrapping_add(1);
    }
}

#[derive(Default)]
pub(crate) struct Verified {
    pub(crate) hdr_bits: usize,
    /// Bytes to remove from the end of the re-encoded payload ...
    pub(crate) drop: usize,
    /// ... and the original bytes that replace them. Encoders may put arbitrary bits after
    /// the final CABAC stop bit (x264 mixes a pseudo-random bit into the last byte), so the
    /// last byte or two can differ even though every bin is identical.
    pub(crate) fix: Vec<u8>,
    /// The re-encoded bytes that were dropped (model context for `fix`).
    pub(crate) dropped: Vec<u8>,
    pub(crate) trailing: usize,
    pub(crate) rbsp: Vec<u8>,
    /// A CAVLC slice: its bits are coded directly, with no tail fix-up.
    pub(crate) cavlc: bool,
}

const MAX_FIX: usize = 8;

/// Codes the replacement tail bytes, using the bytes CABAC re-encoding produced as context.
fn code_fix<C: Coder>(m: &mut Model, c: &mut C, dropped: &[u8], fix: &mut Vec<u8>, n: usize) {
    for j in 0..n {
        let ours = dropped.get(j).copied().unwrap_or(0) as u32;
        let want = fix.get(j).copied().unwrap_or(0);
        let mut c0 = 1u32;
        for i in (0..8).rev() {
            let bit = ((want >> i) & 1) as u32;
            let h = Hint { ctx: 0, kind: sym::FIX, a: c0 | ours << 8, b: j.min(3) as u32, c: 0, d: 0 };
            let b = code_bin(m, c, &h, None, bit);
            c0 = (c0 << 1) | b;
        }
        if j < fix.len() {
            fix[j] = c0 as u8;
        } else {
            fix.push(c0 as u8);
        }
    }
}

fn bit_at(b: &[u8], i: usize) -> u32 {
    ((b[i >> 3] >> (7 - (i & 7))) & 1) as u32
}

/// Checks that a slice NAL unit can be rebuilt exactly from its bins. On success the
/// slice becomes part of the model's history.
/// The slice's bins are left in `recs` (a buffer reused across slices).
fn verify_slice(
    nal: &[u8],
    ps: &ParamSets,
    pic: &mut Picture,
    cpic: &mut cavlc::Picture,
    recs: &mut Vec<Rec>,
) -> Result<Verified, usize> {
    rec_begin(nal, ps, pic);
    let t = (nal[0] & 31) as u32;
    let rbsp = unescape(&nal[1..]);
    let mut chk = Vec::with_capacity(nal.len());
    escape(&rbsp, &mut chk);
    if chk != nal[1..] {
        return Err(0);
    }
    let (sh, sps, pps) = ps.slice_header(&rbsp, t, ((nal[0] >> 5) & 3) as u32).map_err(|_| 1usize)?;
    if !pps.cabac {
        return verify_cavlc(rbsp, &sh, sps, pps, cpic, recs);
    }
    let start = sh.data_bit.div_ceil(8);
    if start > rbsp.len() || (sh.data_bit..start * 8).any(|i| bit_at(&rbsp, i) != 1) {
        return Err(2);
    }
    let mut io = VerifyIo::new(&rbsp, start, recs);
    let mut p = SliceParser::new(sps, pps, &sh, pic).map_err(|_| 1usize)?;
    p.run(&mut io).map_err(|_| 3usize)?;
    let data = io.finish();
    let fixup = tail_fixup(&rbsp[start..], &data)?;
    p.commit();
    Ok(Verified { hdr_bits: sh.data_bit, rbsp, ..fixup })
}

/// Starts the picture and reference lists of a slice in the reconstruction, if the stream
/// has one. Both sides call it for every slice NAL unit whose header parses, in stream order.
fn rec_begin(nal: &[u8], ps: &ParamSets, pic: &mut Picture) {
    if pic.rec.is_none() || nal.is_empty() {
        return;
    }
    let rbsp = unescape(&nal[1..]);
    let mut chk = Vec::with_capacity(nal.len());
    escape(&rbsp, &mut chk);
    if chk != nal[1..] {
        return;
    }
    if let Ok((sh, sps, _)) = ps.slice_header(&rbsp, (nal[0] & 31) as u32, ((nal[0] >> 5) & 3) as u32) {
        rec_begin_header(&sh, sps, pic);
    }
}

fn rec_begin_header(sh: &SliceHeader, sps: &Sps, pic: &mut Picture) {
    if let Some(rec) = pic.rec.as_mut() {
        rec.begin_slice(&crate::h264::recon::SliceInfo {
            first_mb: sh.first_mb,
            slice_type: sh.slice_type,
            frame_num: sh.frame_num,
            idr: sh.idr,
            nal_ref_idc: sh.nal_ref_idc,
            num_ref: [sh.num_ref_l0, sh.num_ref_l1],
            log2_max_frame_num: sps.log2_max_frame_num,
            poc_type: sps.poc_type,
            log2_max_poc_lsb: sps.log2_max_poc_lsb,
            max_num_ref_frames: sps.max_num_ref_frames,
            wmb: sps.width_mbs as usize,
            hmb: sps.height_mbs() as usize,
            ri: &sh.ri,
        });
    }
}

/// Whether the parameter sets of a track (container and in-band) select CABAC, so the
/// pixel-context model applies.
fn track_uses_cabac(input: &[u8], samples: &[(u64, u32)], nls: usize, params: &[Vec<u8>]) -> bool {
    let is_cabac = |nal: &[u8]| -> bool {
        nal.len() > 1 && nal[0] & 31 == 8 && crate::h264::parse_pps(&unescape(&nal[1..])).is_ok_and(|p| p.cabac)
    };
    if params.iter().any(|n| is_cabac(n)) {
        return true;
    }
    samples.iter().take(64).any(|&(o, s)| {
        let smp = &input[o as usize..(o + s as u64) as usize];
        split_nals(smp, nls).is_some_and(|ns| ns.iter().any(|&(a, l)| is_cabac(&smp[a..a + l])))
    })
}

/// Whether a sample starts a run that needs no earlier picture: an IDR access unit, or one
/// whose slices are all I slices.
fn is_keyframe(smp: &[u8], nals: &[(usize, usize)]) -> bool {
    let mut any = false;
    for &(a, l) in nals {
        let nal = &smp[a..a + l];
        match nal[0] & 31 {
            5 => return true,
            1 => {
                let rbsp = unescape(&nal[1..nal.len().min(16)]);
                let mut r = crate::bits::BitReader::new(&rbsp);
                let (Ok(_), Ok(st)) = (r.ue(), r.ue()) else { return false };
                if st % 5 != 2 {
                    return false;
                }
                any = true;
            }
            _ => {}
        }
    }
    any
}

/// Position of the rbsp_stop_one_bit: the last set bit of the RBSP.
fn stop_bit(rbsp: &[u8]) -> Option<usize> {
    let i = rbsp.iter().rposition(|&b| b != 0)?;
    Some(i * 8 + 7 - rbsp[i].trailing_zeros() as usize)
}

/// Records the bits of a CAVLC slice, read from the original RBSP, for [`code_recs`].
struct CavlcVerify<'a, 'r> {
    rbsp: &'a [u8],
    pos: usize,
    stop: usize,
    recs: &'r mut Vec<Rec>,
}

impl CavlcVerify<'_, '_> {
    #[inline]
    fn record(&mut self, h: &Hint, b: u32) {
        self.recs.push(Rec { h: *h, state: 0, typ: 0, bin: b as u8 });
    }
}

impl cavlc::BitIo for CavlcVerify<'_, '_> {
    #[inline]
    fn bit(&mut self, h: &Hint) -> R<u32> {
        // slice data never reaches the stop bit
        if self.pos >= self.stop {
            return Err(Bad);
        }
        let b = bit_at(self.rbsp, self.pos);
        self.pos += 1;
        self.record(h, b);
        Ok(b)
    }

    fn more(&mut self, h: &Hint, _at_end: bool) -> R<bool> {
        let more = self.pos < self.stop;
        self.record(h, more as u32);
        Ok(more)
    }

    fn peek_ue(&mut self) -> R<Option<u32>> {
        let mut zeros = 0usize;
        loop {
            if self.pos >= self.stop {
                return Err(Bad);
            }
            let b = bit_at(self.rbsp, self.pos);
            self.pos += 1;
            if b == 1 {
                break;
            }
            zeros += 1;
            if zeros > 31 {
                return Err(Bad);
            }
        }
        if self.pos + zeros > self.stop {
            return Err(Bad);
        }
        let mut v = 0u64;
        for _ in 0..zeros {
            v = v << 1 | bit_at(self.rbsp, self.pos) as u64;
            self.pos += 1;
        }
        u32::try_from((1u64 << zeros) - 1 + v).map(Some).map_err(|_| Bad)
    }

    #[inline]
    fn decision(&mut self, h: &Hint, v: u32) -> R<u32> {
        self.record(h, v);
        Ok(v)
    }

    fn put_ue(&mut self, _v: u32) {}

    fn pos(&self) -> usize {
        self.pos
    }
}

/// Decodes CAVLC slice data from the model and writes the bits.
struct CavlcDecode<'a, 'd> {
    m: &'a mut Model,
    c: &'a mut ArithDec<'d>,
    w: BitWriter,
    /// Recently announced bits and their table positions (already prefetched).
    soon: [(Hint, Slots); 4],
    soon_i: usize,
}

impl<'a, 'd> CavlcDecode<'a, 'd> {
    fn new(m: &'a mut Model, c: &'a mut ArithDec<'d>, w: BitWriter) -> Self {
        let none = Hint { kind: u8::MAX, ..Default::default() };
        CavlcDecode { m, c, w, soon: [(none, Slots::default()); 4], soon_i: 0 }
    }

    #[inline]
    fn dec(&mut self, h: &Hint) -> R<u32> {
        if self.c.overrun() {
            return Err(Bad);
        }
        let p = match self.soon.iter().find(|e| e.0 == *h) {
            Some(e) => {
                let s = e.1;
                self.m.predict_at(&s, h, Some(0))
            }
            None => self.m.predict(h, Some(0)),
        };
        let b = self.c.decode(p);
        self.m.update(b);
        Ok(b)
    }
}

impl cavlc::BitIo for CavlcDecode<'_, '_> {
    #[inline]
    fn bit(&mut self, h: &Hint) -> R<u32> {
        let b = self.dec(h)?;
        self.w.put(b);
        Ok(b)
    }

    fn more(&mut self, h: &Hint, _at_end: bool) -> R<bool> {
        Ok(self.dec(h)? == 1)
    }

    fn peek_ue(&mut self) -> R<Option<u32>> {
        Ok(None)
    }

    #[inline]
    fn decision(&mut self, h: &Hint, _v: u32) -> R<u32> {
        self.dec(h)
    }

    fn put_ue(&mut self, v: u32) {
        let x = v as u64 + 1;
        let n = 64 - x.leading_zeros();
        for _ in 1..n {
            self.w.put(0);
        }
        for i in (0..n).rev() {
            self.w.put(((x >> i) & 1) as u32);
        }
    }

    fn pos(&self) -> usize {
        self.w.bit_len()
    }

    #[inline]
    fn prefetch(&mut self, h: &Hint) {
        let s = self.m.slots(h);
        self.m.prefetch(&s);
        self.soon[self.soon_i & 3] = (*h, s);
        self.soon_i = self.soon_i.wrapping_add(1);
    }
}

/// Checks that a CAVLC slice parses exactly up to its stop bit, leaving its bits (and the
/// decisions coded instead of some of them) in `recs`. On success the slice becomes part of
/// the parser's cross-slice state.
fn verify_cavlc(
    rbsp: Vec<u8>,
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    cpic: &mut cavlc::Picture,
    recs: &mut Vec<Rec>,
) -> Result<Verified, usize> {
    let stop = stop_bit(&rbsp).ok_or(5usize)?;
    if stop < sh.data_bit {
        return Err(5);
    }
    recs.clear();
    let mut io = CavlcVerify { rbsp: &rbsp, pos: sh.data_bit, stop, recs };
    let mut p = cavlc::SliceParser::new(sps, pps, sh, cpic).map_err(|_| 1usize)?;
    p.run(&mut io).map_err(|_| 3usize)?;
    if io.pos != stop {
        return Err(5);
    }
    p.commit();
    let trailing = rbsp.len() - (stop / 8 + 1);
    Ok(Verified { hdr_bits: sh.data_bit, trailing, rbsp, cavlc: true, ..Default::default() })
}

/// Rebuilds a CAVLC slice RBSP (without trailing zero bytes) from its header bits and the
/// model.
fn decode_cavlc(
    hdr_bits: &[u8],
    sh: &SliceHeader,
    sps: &Sps,
    pps: &Pps,
    cpic: &mut cavlc::Picture,
    m: &mut Model,
    c: &mut ArithDec,
) -> R<Vec<u8>> {
    let mut w = BitWriter::new();
    for &b in hdr_bits {
        w.put(b as u32);
    }
    let mut io = CavlcDecode::new(m, c, w);
    let mut p = cavlc::SliceParser::new(sps, pps, sh, cpic)?;
    p.run(&mut io)?;
    p.commit();
    let mut w = io.w;
    w.put(1);
    while w.bit_len() & 7 != 0 {
        w.put(0);
    }
    Ok(w.finish())
}

/// Compares a re-encoded payload with the original slice data (`tail`: everything after the
/// slice header) and returns the tail fix-up fields of a [`Verified`] (the others are empty),
/// or failure reason 4 if they differ by more than the fix-up covers.
pub(crate) fn tail_fixup(tail: &[u8], data: &[u8]) -> Result<Verified, usize> {
    // The original slice data ends at its last non-zero byte (it holds the stop bit); any
    // zero bytes after it are cabac_zero_words.
    let orig_len = tail.iter().rposition(|&b| b != 0).map_or(0, |p| p + 1);
    let orig = &tail[..orig_len];
    let common = data.iter().zip(orig).take_while(|(a, b)| a == b).count();
    let drop = data.len() - common;
    let fix = orig[common..].to_vec();
    if drop > MAX_FIX || fix.len() > MAX_FIX {
        return Err(4);
    }
    Ok(Verified { drop, fix, dropped: data[common..].to_vec(), trailing: tail.len() - orig_len, ..Default::default() })
}

/// How many bins ahead the encoder computes table positions and prefetches them.
const AHEAD: usize = 6;

/// Codes recorded bins. All of them are known in advance, so the model's table memory for
/// later bins is prefetched while earlier ones are coded.
fn code_recs(m: &mut Model, enc: &mut ArithEnc, recs: &[Rec], mut stats: Option<&mut Stats>) {
    let mut ring = [Slots::default(); 8];
    for (j, r) in recs.iter().enumerate().take(AHEAD) {
        ring[j & 7] = m.slots(&r.h);
        m.prefetch(&ring[j & 7]);
    }
    for (j, r) in recs.iter().enumerate() {
        if let Some(f) = recs.get(j + AHEAD) {
            let s = m.slots(&f.h);
            m.prefetch(&s);
            ring[(j + AHEAD) & 7] = s;
        }
        let state = (r.typ == 0).then_some(r.state);
        let p = m.predict_at(&ring[j & 7], &r.h, state);
        enc.encode(r.bin as u32, p);
        m.update(r.bin as u32);
        if let Some(st) = stats.as_deref_mut() {
            st.add_cost(r, state, p);
        }
    }
}

/// Context for slice-header bits: the same bit positions of the previous header.
#[derive(Default)]
struct HdrCtx {
    prev: Vec<u8>,
}

impl HdrCtx {
    fn hint(&self, idx: usize, hist: u32, nal_hdr: u8) -> Hint {
        let p = |i: usize| self.prev.get(i).copied().unwrap_or(2) as u32;
        let around = p(idx) | p(idx + 1) << 2 | p(idx.wrapping_sub(1)) << 4;
        Hint { ctx: 0, kind: sym::HDR, a: (hist & 0xFF) | (idx.min(127) as u32) << 8, b: around | (nal_hdr as u32) << 8, c: 0, d: 0 }
    }
}

fn code_nal_hdr<C: Coder>(m: &mut Model, c: &mut C, prev: &mut u8, v: u8) -> u8 {
    let mut c0 = 1u32;
    for i in (0..8).rev() {
        let bit = ((v >> i) & 1) as u32;
        let b = code_bin(m, c, &Hint { ctx: 0, kind: sym::NALHDR, a: c0, b: *prev as u32, c: 0, d: 0 }, None, bit);
        c0 = (c0 << 1) | b;
    }
    *prev = c0 as u8;
    c0 as u8
}

/// Codes the second byte of an HEVC NAL unit header (layer and temporal id), in the context of
/// the first byte (the NAL unit type).
fn code_nal_hdr2<C: Coder>(m: &mut Model, c: &mut C, prev: &mut u8, first: u8, v: u8) -> u8 {
    let mut c0 = 1u32;
    for i in (0..8).rev() {
        let bit = ((v >> i) & 1) as u32;
        let h = Hint { ctx: 0, kind: sym::NALHDR, a: c0 | 0x100, b: *prev as u32 | (first as u32) << 8, c: 0, d: 0 };
        let b = code_bin(m, c, &h, None, bit);
        c0 = (c0 << 1) | b;
    }
    *prev = c0 as u8;
    c0 as u8
}

/// Container bytes between video samples ("gaps") at least this long may be stored raw.
const RAW_GAP_MIN: usize = 512;

/// True if `b` looks like entropy-coded data (compressed audio, mostly): its byte
/// histogram is so close to uniform that coding it with the model would save under about
/// 1% (chi-square against uniform: random data averages 255). Integer-only.
fn looks_random(b: &[u8]) -> bool {
    let mut counts = [0u32; 256];
    for &x in b {
        counts[x as usize] += 1;
    }
    let n = b.len() as u64;
    let sum_sq: u64 = counts.iter().map(|&c| c as u64 * c as u64).sum();
    let chi2 = (256 * sum_sq / n).saturating_sub(n);
    chi2 < 255 + n / 9
}

/// Codes a gap: with the model, or, if it is long and looks random, at one bit per bit
/// without it. Compressed audio interleaved with the video would otherwise cost most of the
/// header stream's coding time (which runs before the parallel segments) for ~1% gain.
fn put_gap(m: &mut Model, enc: &mut ArithEnc, bc: &mut ByteCtx, gap: &[u8]) {
    if gap.len() >= RAW_GAP_MIN {
        let raw = looks_random(gap);
        code_flag(m, enc, F_RAW, 0, raw as u32);
        if raw {
            for &b in gap {
                for i in (0..8).rev() {
                    enc.encode(((b >> i) & 1) as u32, 1 << 15);
                }
            }
            return;
        }
    }
    for &b in gap {
        code_byte(m, enc, sym::RAW_OTHER, bc, b);
    }
}

/// Decodes a gap written by [`put_gap`].
fn get_gap(m: &mut Model, c: &mut ArithDec, bc: &mut ByteCtx, gap: &mut [u8]) {
    if gap.len() >= RAW_GAP_MIN && code_flag(m, c, F_RAW, 0, 0) == 1 {
        for b in gap.iter_mut() {
            let mut v = 0u32;
            for _ in 0..8 {
                v = (v << 1) | c.decode(1 << 15);
            }
            *b = v as u8;
        }
        return;
    }
    for b in gap.iter_mut() {
        *b = code_byte(m, c, sym::RAW_OTHER, bc, 0);
    }
}

/// Compression settings.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// log2 of each segment model's hash-table size (4 bytes per slot). `None` picks one
    /// from the segment size: larger tables predict better but cost memory on both sides.
    pub table_bits: Option<u32>,
    /// Collect per-syntax-element costs in [`Stats::kind_cost`] (slower).
    pub stats: bool,
    /// Worker threads (0 = all cores, capped at 8). Never changes the output.
    pub threads: usize,
    /// Do not model H.264 with the decoded pictures (writes the older `PPM3` stream: faster,
    /// a little larger).
    pub no_pixels: bool,
}

/// Progress callback: `(processed_bytes, total_bytes) -> keep_going`.
pub type Progress<'a> = dyn FnMut(u64, u64) -> bool + 'a;

/// The operation was cancelled by the progress callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

/// True if `input` looks like an MP4/MOV (or Matroska) file with an H.264 or HEVC track
/// (cheap: parses only the container boxes).
pub fn probe(input: &[u8]) -> bool {
    find_track(input).is_some()
}

/// Video data per segment. Segments may start at any sample: the parser never needs
/// reference pictures, so a cut only resets the model's statistics and its memory of the
/// previous picture.
const SEG_TARGET: usize = 3 << 20;
/// Longest segment of a reconstructed (`PPM4`) stream without a key frame to cut at.
const REC_SEG_MAX: usize = 48 << 20;
/// Hash-table size of the model that codes the container bytes.
const HEADER_BITS: u32 = 20;
const MAX_WORKERS: usize = 8;
const N_NSEG: u32 = 11;
const N_SEGSTART: u32 = 12;
const N_PRE: u32 = 13;

fn auto_table_bits(len: usize) -> u32 {
    if len <= 2_000_000 {
        22
    } else {
        23
    }
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_varint(b: &[u8], pos: &mut usize) -> R<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let x = *b.get(*pos).ok_or(Bad)?;
        *pos += 1;
        v |= ((x & 0x7F) as u64) << shift;
        if x & 0x80 == 0 {
            return Ok(v);
        }
    }
    Err(Bad)
}

/// (start, length) of each NAL unit in a length-prefixed sample, or `None` if the length
/// fields do not tile the sample exactly.
fn split_nals(smp: &[u8], nls: usize) -> Option<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < smp.len() {
        if i + nls > smp.len() {
            return None;
        }
        let len = smp[i..i + nls].iter().fold(0usize, |a, &b| a << 8 | b as usize);
        i += nls;
        if len == 0 || i + len > smp.len() {
            return None;
        }
        out.push((i, len));
        i += len;
    }
    Some(out)
}

/// (nal_unit_type, id) of an SPS or PPS NAL unit.
fn param_key(nal: &[u8], codec: Codec) -> Option<(u8, u32)> {
    if codec == Codec::Hevc {
        if nal.len() < 3 || hevc::layer_id(nal) != 0 {
            return None;
        }
        let rbsp = unescape(&nal[2..]);
        return match hevc::nal_type(nal) {
            hevc::NAL_SPS => hevc::sps_id(&rbsp).ok().map(|id| (33, id)),
            hevc::NAL_PPS => crate::bits::BitReader::new(&rbsp).ue().ok().filter(|&id| id < 64).map(|id| (34, id)),
            _ => None,
        };
    }
    let t = nal.first()? & 31;
    let rbsp = unescape(&nal[1..]);
    match t {
        7 => crate::h264::parse_sps(&rbsp).ok().map(|s| (7, s.id)),
        8 => crate::h264::parse_pps(&rbsp).ok().map(|p| (8, p.id)),
        _ => None,
    }
}

/// A run of samples coded with its own model, so runs can be coded in parallel. Runs are
/// cut by content (size), never by thread count, so the output does not depend on how many
/// threads are used.
struct SegPlan {
    first: usize,
    end: usize,
    /// In-band SPS/PPS NAL units in effect where the run starts.
    preamble: Vec<Vec<u8>>,
}

fn plan_segments(input: &[u8], samples: &[(u64, u32)], nls: usize, codec: Codec, rec: bool) -> Option<Vec<SegPlan>> {
    let mut plans = Vec::new();
    let mut start = 0;
    let mut acc = 0usize;
    let mut latest: Vec<((u8, u32), Vec<u8>)> = Vec::new();
    let mut pre = Vec::new();
    for (i, &(o, s)) in samples.iter().enumerate() {
        let smp = &input[o as usize..(o + s as u64) as usize];
        let nals = split_nals(smp, nls)?;
        if nals.iter().any(|&(_, l)| l < codec.hdr_len()) {
            return None;
        }
        // With picture reconstruction a segment starts at a key frame, so its decoder has
        // every reference picture; a run without key frames is cut anyway at REC_SEG_MAX
        // (its pictures then predict from grey until intra blocks refresh them).
        let key_ok = !rec || acc >= REC_SEG_MAX || is_keyframe(smp, &nals);
        if i > start && acc >= SEG_TARGET && key_ok {
            plans.push(SegPlan { first: start, end: i, preamble: std::mem::take(&mut pre) });
            start = i;
            acc = 0;
            pre = latest.iter().map(|(_, n)| n.clone()).collect();
        }
        for &(a, l) in &nals {
            let nal = &smp[a..a + l];
            if let Some(k) = param_key(nal, codec) {
                match latest.iter_mut().find(|(key, _)| *key == k) {
                    Some(e) => e.1 = nal.to_vec(),
                    None => latest.push((k, nal.to_vec())),
                }
            }
        }
        acc += s as usize;
    }
    plans.push(SegPlan { first: start, end: samples.len(), preamble: pre });
    Some(plans)
}

fn worker_count(threads: usize, jobs: usize) -> usize {
    if cfg!(target_family = "wasm") {
        return 1;
    }
    let t = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()).min(MAX_WORKERS) } else { threads };
    t.min(jobs).max(1)
}

/// A job: index and a progress tick (processed bytes -> keep going).
type Job<'a, T> = dyn Fn(usize, &mut dyn FnMut(u64) -> bool) -> T + Sync + 'a;

/// Runs `job(i, tick)` for `i` in `0..n` on up to `threads` threads and returns the results
/// in order. Jobs report processed bytes through `tick`, which returns `false` once the
/// progress callback has asked to stop. Progress is reported from the calling thread.
fn run_jobs<T: Send>(n: usize, threads: usize, total: u64, base: u64, progress: &mut Progress, job: &Job<T>) -> Vec<T> {
    let workers = worker_count(threads, n);
    if workers <= 1 {
        let mut done = base;
        let mut keep = true;
        return (0..n)
            .map(|i| {
                job(i, &mut |b| {
                    done += b;
                    keep = keep && progress(done.min(total), total);
                    keep
                })
            })
            .collect();
    }
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::Mutex;
    let next = AtomicUsize::new(0);
    let done = AtomicU64::new(base);
    let cancel = AtomicBool::new(false);
    let slots: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                sc.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= n {
                        break;
                    }
                    let r = job(i, &mut |b| {
                        done.fetch_add(b, Ordering::Relaxed);
                        !cancel.load(Ordering::Relaxed)
                    });
                    *slots[i].lock().unwrap() = Some(r);
                })
            })
            .collect();
        while !handles.iter().all(|h| h.is_finished()) {
            if !progress(done.load(Ordering::Relaxed).min(total), total) {
                cancel.store(true, Ordering::Relaxed);
            }
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
    });
    slots.into_iter().map(|m| m.into_inner().unwrap().expect("worker finished its job")).collect()
}

/// Recompresses an MP4/MOV file whose video track is H.264. Returns `None` when the file
/// is not one this module handles.
pub fn compress(input: &[u8]) -> Option<(Vec<u8>, Stats)> {
    compress_with(input, &Options::default(), &mut |_, _| true).ok().flatten()
}

/// Like [`compress`], with options and progress reporting.
pub fn compress_with(input: &[u8], opts: &Options, progress: &mut Progress) -> Result<Option<(Vec<u8>, Stats)>, Cancelled> {
    let Some((codec, track)) = find_track(input) else { return Ok(None) };
    if codec == Codec::Hevc && !hevc::recode::quick_check(input, &track) {
        return Ok(None); // nothing recodable: let the caller store the file quickly
    }
    compress_track(input, codec, &track, true, opts, progress)
}

/// [`compress_with`] for a track already found. `in_container`: the parameter sets come
/// from the file's container bytes (MP4/Matroska), which the decoder reads back before it
/// has any samples; otherwise the decoder is given them (see [`decompress_track`]).
pub(crate) fn compress_track(
    input: &[u8],
    codec: Codec,
    track: &mp4::AvcTrack,
    in_container: bool,
    opts: &Options,
    progress: &mut Progress,
) -> Result<Option<(Vec<u8>, Stats)>, Cancelled> {
    let total = input.len() as u64;
    let nls = track.nal_len_size;
    let samples = track.samples.clone();
    if samples.is_empty() {
        return Ok(None);
    }
    let mut end = 0u64;
    for &(o, s) in &samples {
        if o < end || o + s as u64 > input.len() as u64 {
            return Ok(None);
        }
        end = o + s as u64;
    }
    // The decoder reads the parameter sets from the container bytes before it has any
    // samples; make sure it will see the same ones.
    if in_container {
        let mut holes = input.to_vec();
        for &(o, s) in &samples {
            holes[o as usize..(o + s as u64) as usize].fill(0);
        }
        let Some(t2) = find_codec_track(&holes, codec) else { return Ok(None) };
        if t2.param_nals != track.param_nals || t2.nal_len_size != nls || t2.samples != samples {
            return Ok(None);
        }
    }
    let rec = codec == Codec::Avc && !opts.no_pixels && track_uses_cabac(input, &samples, nls, &track.param_nals);
    let Some(plans) = plan_segments(input, &samples, nls, codec, rec) else { return Ok(None) };

    // Header stream: sample table, segment boundaries, container bytes.
    let mut st = Stats { samples: samples.len(), segments: plans.len(), hevc: codec == Codec::Hevc, ..Default::default() };
    let mut m = Model::new(HEADER_BITS);
    let mut enc = ArithEnc::new();
    code_num(&mut m, &mut enc, N_FILELEN, input.len() as u64);
    code_num(&mut m, &mut enc, N_NLS, nls as u64);
    code_num(&mut m, &mut enc, N_NSAMP, samples.len() as u64);
    let mut prev = 0u64;
    for &(o, s) in &samples {
        code_num(&mut m, &mut enc, N_GAP, o - prev);
        code_num(&mut m, &mut enc, N_SIZE, s as u64);
        prev = o + s as u64;
    }
    code_num(&mut m, &mut enc, N_NSEG, plans.len() as u64);
    for w in plans.windows(2) {
        code_num(&mut m, &mut enc, N_SEGSTART, (w[1].first - w[0].first) as u64);
    }
    st.out_header = enc.out.len();
    let mut bc = ByteCtx::default();
    let mut pos = 0usize;
    for &(o, s) in &samples {
        put_gap(&mut m, &mut enc, &mut bc, &input[pos..o as usize]);
        st.other_bytes += o as usize - pos;
        pos = (o + s as u64) as usize;
    }
    put_gap(&mut m, &mut enc, &mut bc, &input[pos..]);
    st.other_bytes += input.len() - pos;
    let header = enc.finish();
    st.out_other = header.len() - st.out_header;
    if !progress(st.other_bytes as u64, total) {
        return Err(Cancelled);
    }

    let base = st.other_bytes as u64;
    let results = run_jobs(plans.len(), opts.threads, total, base, progress, &|i, tick| {
        let plan = &plans[i];
        let size: usize = samples[plan.first..plan.end].iter().map(|&(_, s)| s as usize).sum();
        let bits = opts.table_bits.unwrap_or_else(|| auto_table_bits(size)).clamp(MIN_TABLE_BITS, MAX_TABLE_BITS);
        encode_segment(input, &samples, nls, codec, rec, &track.param_nals, plan, bits, opts.stats, tick)
            .map(|(d, s)| (bits, d, s))
    });
    let mut out = if rec { MAGIC4.to_vec() } else { codec.magic().to_vec() };
    put_varint(&mut out, header.len() as u64);
    out.extend_from_slice(&header);
    for r in results {
        let (bits, data, s) = r?;
        out.push(bits as u8);
        put_varint(&mut out, data.len() as u64);
        out.extend_from_slice(&data);
        st.merge(&s);
    }
    progress(total, total);
    Ok(Some((out, st)))
}

#[allow(clippy::too_many_arguments)]
fn encode_segment(
    input: &[u8],
    samples: &[(u64, u32)],
    nls: usize,
    codec: Codec,
    rec: bool,
    base_params: &[Vec<u8>],
    plan: &SegPlan,
    bits: u32,
    stats: bool,
    tick: &mut dyn FnMut(u64) -> bool,
) -> Result<(Vec<u8>, Stats), Cancelled> {
    let mut st = Stats::default();
    let mut cs = SegState::new(codec, rec);
    for n in base_params {
        cs.add_param(n);
    }
    let mut m = Model::new(bits);
    let mut enc = ArithEnc::new();
    let mut rb = ByteCtx::default();
    code_num(&mut m, &mut enc, N_PRE, plan.preamble.len() as u64);
    for nal in &plan.preamble {
        code_num(&mut m, &mut enc, N_NALLEN, nal.len() as u64);
        for &b in nal {
            code_byte(&mut m, &mut enc, sym::RAW_NAL, &mut rb, b);
        }
        cs.add_param(nal);
    }
    let hl = codec.hdr_len();
    let mut hc = HdrCtx::default();
    let mut prev_hdr = 0u8;
    let mut prev_hdr2 = 0u8;
    let mut recs = Vec::new();
    for &(o, s) in &samples[plan.first..plan.end] {
        if !tick(s as u64) {
            return Err(Cancelled);
        }
        let smp = &input[o as usize..(o + s as u64) as usize];
        let nals = split_nals(smp, nls).expect("validated by plan_segments");
        for (a, len) in nals {
            let nal = &smp[a..a + len];
            st.nals += 1;
            let t = codec.nal_type(nal);
            let before = enc.out.len();
            code_nal_hdr(&mut m, &mut enc, &mut prev_hdr, nal[0]);
            if codec == Codec::Hevc {
                code_nal_hdr2(&mut m, &mut enc, &mut prev_hdr2, nal[0], nal[1]);
            }
            let slice = codec.is_slice(t);
            let res = if slice {
                st.slices += 1;
                cs.verify(nal, &mut recs)
            } else {
                Err(usize::MAX)
            };
            match res {
                Ok(v) => {
                    code_flag(&mut m, &mut enc, F_MODE, t, 1);
                    code_num(&mut m, &mut enc, N_HBITS, v.hdr_bits as u64);
                    let mut hist = 1u32;
                    for k in 0..v.hdr_bits {
                        let h = hc.hint(k, hist, nal[0]);
                        let b = code_bin(&mut m, &mut enc, &h, None, bit_at(&v.rbsp, k));
                        hist = (hist << 1) | b;
                    }
                    hc.prev = (0..v.hdr_bits).map(|k| bit_at(&v.rbsp, k) as u8).collect();
                    code_recs(&mut m, &mut enc, &recs, if stats { Some(&mut st) } else { None });
                    if !v.cavlc {
                        code_num(&mut m, &mut enc, N_DROP, v.drop as u64);
                        code_num(&mut m, &mut enc, N_FIX, v.fix.len() as u64);
                        let mut fix = v.fix.clone();
                        code_fix(&mut m, &mut enc, &v.dropped, &mut fix, v.fix.len());
                    }
                    code_num(&mut m, &mut enc, N_TZ, v.trailing as u64);
                    cs.after_slice(nal, true);
                    st.recoded += 1;
                    st.slice_bytes += len;
                    st.bins += recs.len() as u64;
                    st.out_slices += enc.out.len() - before;
                }
                Err(reason) => {
                    if reason != usize::MAX {
                        st.fails[reason.min(6)] += 1;
                    }
                    code_flag(&mut m, &mut enc, F_MODE, t, 0);
                    code_num(&mut m, &mut enc, N_NALLEN, len as u64);
                    for &b in &nal[hl..] {
                        code_byte(&mut m, &mut enc, sym::RAW_NAL, &mut rb, b);
                    }
                    cs.add_param(nal);
                    if slice {
                        cs.after_slice(nal, false);
                    }
                    st.raw_nal_bytes += len;
                    st.out_raw += enc.out.len() - before;
                }
            }
        }
    }
    if let SegState::Avc { pic, .. } = &mut cs {
        if let Some(rec) = pic.rec.as_mut() {
            rec.finish_picture();
        }
    }
    Ok((enc.finish(), st))
}

/// Decodes a number and checks it against `max` before narrowing it (32-bit targets).
fn num(m: &mut Model, c: &mut ArithDec, id: u32, max: usize) -> R<usize> {
    let v = code_num(m, c, id, 0);
    if v > max as u64 {
        return Err(Bad);
    }
    Ok(v as usize)
}

/// Rebuilds the original file from [`compress`] output.
pub fn decompress(data: &[u8]) -> R<Vec<u8>> {
    decompress_with(data, 0, &mut |_, _| true)
}

/// A segment's output range and samples, handed to one decoding job.
type Region<'a> = std::sync::Mutex<Option<(&'a mut [u8], usize, usize, usize)>>;

/// Like [`decompress`], with a thread count (0 = all cores; never changes the result) and
/// progress reporting (returning `false` aborts with an error).
pub fn decompress_with(data: &[u8], threads: usize, progress: &mut Progress) -> R<Vec<u8>> {
    decompress_track(data, None, threads, progress)
}

/// [`decompress_with`], with the stream's parameter-set NAL units given instead of read from
/// the rebuilt container bytes (streams made by [`compress_track`] without a container).
pub(crate) fn decompress_track(data: &[u8], params: Option<&[Vec<u8>]>, threads: usize, progress: &mut Progress) -> R<Vec<u8>> {
    if data.len() < 5 {
        return Err(Bad);
    }
    let (codec, rec) = match &data[..4] {
        m if m == MAGIC => (Codec::Avc, false),
        m if m == MAGIC4 => (Codec::Avc, true),
        m if m == MAGIC_HEVC => (Codec::Hevc, false),
        _ => return Err(Bad),
    };
    let mut pos = 4;
    let hlen = usize::try_from(get_varint(data, &mut pos)?).map_err(|_| Bad)?;
    let header = data.get(pos..pos.checked_add(hlen).ok_or(Bad)?).ok_or(Bad)?;
    pos += hlen;
    let mut c = ArithDec::new(header);
    let mut m = Model::new(HEADER_BITS);
    let file_len = code_num(&mut m, &mut c, N_FILELEN, 0);
    if file_len > 1 << 40 {
        return Err(Bad);
    }
    let file_len = usize::try_from(file_len).map_err(|_| Bad)?;
    let nls = num(&mut m, &mut c, N_NLS, 4)?;
    if !(1..=4).contains(&nls) {
        return Err(Bad);
    }
    let n = num(&mut m, &mut c, N_NSAMP, file_len)?;
    let mut samples = Vec::with_capacity(n);
    let mut prev = 0u64;
    for _ in 0..n {
        let gap = code_num(&mut m, &mut c, N_GAP, 0);
        let size = code_num(&mut m, &mut c, N_SIZE, 0);
        let o = prev.checked_add(gap).ok_or(Bad)?;
        prev = o.checked_add(size).ok_or(Bad)?;
        if prev > file_len as u64 || size == 0 {
            return Err(Bad);
        }
        samples.push((o as usize, size as usize));
    }
    if n == 0 {
        return Err(Bad);
    }
    let nseg = num(&mut m, &mut c, N_NSEG, n)?;
    if nseg == 0 {
        return Err(Bad);
    }
    let mut starts = vec![0usize];
    for _ in 1..nseg {
        let d = num(&mut m, &mut c, N_SEGSTART, n)?;
        let s = starts.last().copied().unwrap_or(0) + d;
        if d == 0 || s >= n {
            return Err(Bad);
        }
        starts.push(s);
    }
    let mut out = vec![0u8; file_len];
    let mut bc = ByteCtx::default();
    let mut p = 0usize;
    for &(o, s) in &samples {
        get_gap(&mut m, &mut c, &mut bc, &mut out[p..o]);
        p = o + s;
    }
    get_gap(&mut m, &mut c, &mut bc, &mut out[p..]);
    if c.overrun() {
        return Err(Bad);
    }
    let param_nals = match params {
        Some(p) => p.to_vec(),
        None => find_codec_track(&out, codec).ok_or(Bad)?.param_nals,
    };
    let mut segs = Vec::with_capacity(nseg);
    for _ in 0..nseg {
        let bits = *data.get(pos).ok_or(Bad)? as u32;
        pos += 1;
        if !(MIN_TABLE_BITS..=MAX_TABLE_BITS).contains(&bits) {
            return Err(Bad);
        }
        let len = usize::try_from(get_varint(data, &mut pos)?).map_err(|_| Bad)?;
        segs.push((bits, data.get(pos..pos.checked_add(len).ok_or(Bad)?).ok_or(Bad)?));
        pos += len;
    }
    let other = (file_len - samples.iter().map(|&(_, s)| s).sum::<usize>()) as u64;
    // Each segment writes only inside its own byte range of the output.
    let mut regions: Vec<Region> = Vec::with_capacity(nseg);
    let mut rest: &mut [u8] = &mut out;
    let mut consumed = 0usize;
    for k in 0..nseg {
        let (first, end) = (starts[k], if k + 1 < nseg { starts[k + 1] } else { n });
        let lo = samples[first].0;
        let hi = samples[end - 1].0 + samples[end - 1].1;
        let (_, tail) = std::mem::take(&mut rest).split_at_mut(lo - consumed);
        let (region, tail) = tail.split_at_mut(hi - lo);
        rest = tail;
        consumed = hi;
        regions.push(std::sync::Mutex::new(Some((region, lo, first, end))));
    }
    let results = run_jobs(nseg, threads, file_len as u64, other, progress, &|k, tick| {
        let taken = regions[k].lock().unwrap().take();
        let (region, lo, first, end) = taken.expect("each region is decoded once");
        let (bits, seg) = segs[k];
        decode_segment(seg, bits, region, lo, &samples[first..end], nls, codec, rec, &param_nals, tick)
    });
    for r in results {
        r?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn decode_segment(
    data: &[u8],
    bits: u32,
    out: &mut [u8],
    base: usize,
    samples: &[(usize, usize)],
    nls: usize,
    codec: Codec,
    rec: bool,
    base_params: &[Vec<u8>],
    tick: &mut dyn FnMut(u64) -> bool,
) -> R<()> {
    let mut c = ArithDec::new(data);
    let mut m = Model::new(bits);
    let mut cs = SegState::new(codec, rec);
    for nal in base_params {
        cs.add_param(nal);
    }
    let mut rb = ByteCtx::default();
    let npre = num(&mut m, &mut c, N_PRE, 1 << 12)?;
    for _ in 0..npre {
        let len = num(&mut m, &mut c, N_NALLEN, 1 << 16)?;
        let nal: Vec<u8> = (0..len).map(|_| code_byte(&mut m, &mut c, sym::RAW_NAL, &mut rb, 0)).collect();
        cs.add_param(&nal);
    }
    let hl = codec.hdr_len();
    let mut hc = HdrCtx::default();
    let mut prev_hdr = 0u8;
    let mut prev_hdr2 = 0u8;
    for &(o, s) in samples {
        if !tick(s as u64) {
            return Err(Bad);
        }
        let mut w = o - base;
        let end = w + s;
        while w < end {
            let hdr = code_nal_hdr(&mut m, &mut c, &mut prev_hdr, 0);
            let mut head = vec![hdr];
            if codec == Codec::Hevc {
                head.push(code_nal_hdr2(&mut m, &mut c, &mut prev_hdr2, hdr, 0));
            }
            let t = codec.nal_type(&head);
            let nal: Vec<u8> = if code_flag(&mut m, &mut c, F_MODE, t, 0) == 1 {
                if !codec.is_slice(t) {
                    return Err(Bad);
                }
                let hbits = num(&mut m, &mut c, N_HBITS, 1 << 16)?;
                let mut bw = BitWriter::new();
                let mut hist = 1u32;
                let mut hbits_v = Vec::with_capacity(hbits);
                for k in 0..hbits {
                    let h = hc.hint(k, hist, hdr);
                    let b = code_bin(&mut m, &mut c, &h, None, 0);
                    bw.put(b);
                    hbits_v.push(b as u8);
                    hist = (hist << 1) | b;
                }
                hc.prev = hbits_v;
                while bw.bit_len() & 7 != 0 {
                    bw.put(1);
                }
                let mut rbsp = bw.finish();
                // The re-encoded payload, or None when `rbsp` already is the whole CAVLC slice.
                let payload = match &mut cs {
                    SegState::Avc { ps, pic, cpic } => {
                        // The header parser needs a little slack after the header bits.
                        let mut probe = rbsp.clone();
                        probe.extend_from_slice(&[0xFF; 8]);
                        let (sh, sps, pps) = ps.slice_header(&probe, t, ((hdr >> 5) & 3) as u32)?;
                        if sh.data_bit != hbits {
                            return Err(Bad);
                        }
                        rec_begin_header(&sh, sps, pic);
                        if pps.cabac {
                            let mut io = ModelIo::new(&mut m, &mut c);
                            let mut p = SliceParser::new(sps, pps, &sh, pic)?;
                            p.run(&mut io)?;
                            p.commit();
                            Some(io.enc.w.finish())
                        } else {
                            rbsp = decode_cavlc(&hc.prev, &sh, sps, pps, cpic, &mut m, &mut c)?;
                            None
                        }
                    }
                    SegState::Hevc(s) => Some(hevc::recode::rebuild_payload(s, &head, &rbsp, hbits, &mut m, &mut c)?),
                };
                if let Some(mut payload) = payload {
                    let drop = num(&mut m, &mut c, N_DROP, MAX_FIX)?;
                    let nfix = num(&mut m, &mut c, N_FIX, MAX_FIX)?;
                    if drop > payload.len() {
                        return Err(Bad);
                    }
                    let dropped = payload.split_off(payload.len() - drop);
                    let mut fix = Vec::new();
                    code_fix(&mut m, &mut c, &dropped, &mut fix, nfix);
                    payload.extend_from_slice(&fix);
                    rbsp.extend_from_slice(&payload);
                }
                let tz = num(&mut m, &mut c, N_TZ, end - w)?;
                rbsp.resize(rbsp.len() + tz, 0);
                let mut nal = head;
                escape(&rbsp, &mut nal);
                cs.after_slice(&nal, true);
                nal
            } else {
                let len = num(&mut m, &mut c, N_NALLEN, end - w)?;
                if len < hl {
                    return Err(Bad);
                }
                let mut nal = head;
                nal.reserve(len - hl);
                for _ in hl..len {
                    nal.push(code_byte(&mut m, &mut c, sym::RAW_NAL, &mut rb, 0));
                }
                cs.add_param(&nal);
                if codec.is_slice(t) {
                    cs.rec_raw_slice(&nal);
                    cs.after_slice(&nal, false);
                }
                nal
            };
            if c.overrun() || w + nls + nal.len() > end || (nls < 4 && nal.len() >> (8 * nls) != 0) {
                return Err(Bad);
            }
            for k in 0..nls {
                out[w + k] = (nal.len() >> (8 * (nls - 1 - k))) as u8;
            }
            w += nls;
            out[w..w + nal.len()].copy_from_slice(&nal);
            w += nal.len();
        }
    }
    Ok(())
}

/// Development aid: describes every slice that fails to recode.
pub fn diagnose(input: &[u8], limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    let Some(track) = mp4::find_avc_track(input) else { return vec!["no H.264 track".into()] };
    let mut ps = ParamSets::default();
    for n in &track.param_nals {
        ps.add_nal(n);
    }
    if let Some(Some(p)) = ps.pps.iter().find(|p| p.is_some()) {
        out.push(format!("PPS: transform_8x8={} weighted_pred={} bipred={} constrained_intra={}", p.transform_8x8_mode, p.weighted_pred, p.weighted_bipred_idc, p.constrained_intra_pred));
    }
    let mut pic = Picture::default();
    let mut cpic = cavlc::Picture::default();
    let nls = track.nal_len_size;
    let mut idx = 0;
    let mut by_type = [[0usize; 2]; 3];
    for &(o, s) in &track.samples {
        let smp = &input[o as usize..(o + s as u64) as usize];
        let mut i = 0;
        while i + nls <= smp.len() {
            let len = smp[i..i + nls].iter().fold(0usize, |a, &b| a << 8 | b as usize);
            i += nls;
            if i + len > smp.len() || len == 0 {
                break;
            }
            let nal = &smp[i..i + len];
            i += len;
            let t = (nal[0] & 31) as u32;
            if t == 1 || t == 5 {
                let rbsp = unescape(&nal[1..]);
                let hdr = ps.slice_header(&rbsp, t, ((nal[0] >> 5) & 3) as u32);
                let stype = hdr.as_ref().map(|h| h.0.slice_type as usize).unwrap_or(0);
                match verify_slice(nal, &ps, &mut pic, &mut cpic, &mut Vec::new()) {
                    Ok(_) => by_type[stype.min(2)][0] += 1,
                    Err(reason) => {
                        by_type[stype.min(2)][1] += 1;
                        if out.len() < limit {
                            let mut msg = format!("slice {idx} type {} len {len}: {}", ["P", "B", "I"][stype.min(2)], FAIL_NAMES[reason.min(6)]);
                            if let Ok((sh, sps, pps)) = hdr {
                                if !pps.cabac {
                                    // re-run the CAVLC parse on scratch state to see where it stops
                                    let stop = stop_bit(&rbsp).unwrap_or(0);
                                    let mut recs = Vec::new();
                                    let mut io = CavlcVerify { rbsp: &rbsp, pos: sh.data_bit, stop, recs: &mut recs };
                                    let mut scratch = cavlc::Picture::default();
                                    if let Ok(mut p) = cavlc::SliceParser::new(sps, pps, &sh, &mut scratch) {
                                        let r = p.run(&mut io);
                                        msg += &format!(
                                            " | CAVLC parse {:?} first_mb {} | stopped at bit {} of {} (data from {})",
                                            r.is_ok(),
                                            sh.first_mb,
                                            io.pos,
                                            stop,
                                            sh.data_bit
                                        );
                                    }
                                    out.push(msg);
                                    idx += 1;
                                    continue;
                                }
                                let start = sh.data_bit.div_ceil(8);
                                let mut recs = Vec::new();
                                let mut io = VerifyIo { dec: cabac::Decoder::new(&rbsp, start), enc: cabac::Encoder::new(), recs: &mut recs };
                                let mut scratch = Picture::default();
                                if let Ok(mut p) = SliceParser::new(sps, pps, &sh, &mut scratch) {
                                    let r = p.run(&mut io);
                                    let mbs = p.count;
                                    let data = io.enc.w.finish();
                                    let tail = &rbsp[start..];
                                    let diff = data.iter().zip(tail).position(|(a, b)| a != b);
                                    msg += &format!(
                                        " | parse {:?} first_mb {} mbs {} bins {} | reenc {} B vs payload {} B, first diff at {:?}",
                                        r.is_ok(), sh.first_mb, mbs, io.recs.len(), data.len(), tail.len(), diff
                                    );
                                }
                            }
                            out.push(msg);
                        }
                    }
                }
                idx += 1;
            } else {
                ps.add_nal(nal);
            }
        }
    }
    out.push(format!("ok/failed by type: P {:?} B {:?} I {:?}", by_type[0], by_type[1], by_type[2]));
    out
}

/// Decodes the original CABAC payload only (profiling).
struct DecodeOnlyIo<'a> {
    dec: cabac::Decoder<'a>,
    n: u64,
}

impl BinIo for DecodeOnlyIo<'_> {
    fn decision(&mut self, state: u8, _h: &Hint) -> R<u32> {
        if self.dec.overrun() {
            return Err(Bad);
        }
        self.n += 1;
        Ok(self.dec.decision(state))
    }
    fn bypass(&mut self, _h: &Hint) -> R<u32> {
        if self.dec.overrun() {
            return Err(Bad);
        }
        self.n += 1;
        Ok(self.dec.bypass())
    }
    fn terminate(&mut self, _h: &Hint) -> R<u32> {
        if self.dec.overrun() {
            return Err(Bad);
        }
        self.n += 1;
        Ok(self.dec.terminate())
    }
}

/// Development aid: times the stages of compression on one thread, one model.
pub fn profile(input: &[u8], table_bits: u32) -> Vec<String> {
    use std::time::Instant;
    let mut out = Vec::new();
    let Some(track) = mp4::find_avc_track(input) else { return vec!["no H.264 track".into()] };
    let nls = track.nal_len_size;
    // Collect slice NAL units and parameter-set order once.
    let mut units: Vec<&[u8]> = Vec::new();
    for &(o, s) in &track.samples {
        let smp = &input[o as usize..(o + s as u64) as usize];
        let Some(nals) = split_nals(smp, nls) else { return vec!["bad sample".into()] };
        for (a, len) in nals {
            units.push(&smp[a..a + len]);
        }
    }
    let fresh = || {
        let mut ps = ParamSets::default();
        for n in &track.param_nals {
            ps.add_nal(n);
        }
        ps
    };
    // Pass 1: unescape + header + parse + CABAC decode.
    let mut ps = fresh();
    let mut pic = Picture::default();
    let mut bins = 0u64;
    let t0 = Instant::now();
    for nal in &units {
        let t = (nal[0] & 31) as u32;
        if t != 1 && t != 5 {
            ps.add_nal(nal);
            continue;
        }
        let rbsp = unescape(&nal[1..]);
        let Ok((sh, sps, pps)) = ps.slice_header(&rbsp, t, ((nal[0] >> 5) & 3) as u32) else { continue };
        if !pps.cabac {
            continue;
        }
        let start = sh.data_bit.div_ceil(8);
        let mut io = DecodeOnlyIo { dec: cabac::Decoder::new(&rbsp, start), n: 0 };
        let Ok(mut p) = SliceParser::new(sps, pps, &sh, &mut pic) else { continue };
        if p.run(&mut io).is_ok() {
            p.commit();
        }
        bins += io.n;
    }
    let t_dec = t0.elapsed().as_secs_f64();
    out.push(format!("bins {bins}"));
    out.push(format!("parse+decode   {:7.3}s {:6.1} ns/bin", t_dec, t_dec * 1e9 / bins as f64));
    // Pass 2: full verify_slice (decode, re-encode, record bins), then the model.
    let mut ps = fresh();
    let mut pic = Picture::default();
    let mut cpic = cavlc::Picture::default();
    let mut m = Model::new(table_bits);
    let mut enc = ArithEnc::new();
    let mut t_ver = 0f64;
    let mut t_mod = 0f64;
    let mut recs = Vec::new();
    let mut kinds = [0u64; 40];
    for nal in &units {
        let t = (nal[0] & 31) as u32;
        if t != 1 && t != 5 {
            ps.add_nal(nal);
            continue;
        }
        let tv = Instant::now();
        let r = verify_slice(nal, &ps, &mut pic, &mut cpic, &mut recs);
        t_ver += tv.elapsed().as_secs_f64();
        if r.is_ok() {
            let tm = Instant::now();
            code_recs(&mut m, &mut enc, &recs, None);
            t_mod += tm.elapsed().as_secs_f64();
            for r in &recs {
                kinds[(r.h.kind as usize).min(39)] += 1;
            }
        }
    }
    out.push(format!("verify_slice   {:7.3}s {:6.1} ns/bin", t_ver, t_ver * 1e9 / bins as f64));
    out.push(format!("model+arith    {:7.3}s {:6.1} ns/bin  -> {} bytes", t_mod, t_mod * 1e9 / bins as f64, enc.out.len()));
    const NAMES: [&str; 21] = [
        "skip", "mb_type", "sub_type", "mvd", "ref_idx", "dqp", "ipred", "cpred", "cbp_luma", "cbp_chroma", "t8x8", "cbf",
        "sig", "last", "abs", "sign", "abs_suffix", "mvd_suffix", "mvd_sign", "end", "pcm",
    ];
    let mut line = String::from("bins by kind:");
    for (i, &n) in kinds.iter().enumerate() {
        if n > 0 {
            line += &format!(" {}={:.1}%", NAMES.get(i).unwrap_or(&"?"), 100.0 * n as f64 / bins as f64);
        }
    }
    out.push(line);
    out
}
