//! Development tool: remuxes the H.264 or HEVC track of an MP4 into an MPEG transport stream the
//! way common muxers do (video only): access unit delimiters, parameter sets before each IRAP
//! picture, one PES packet per access unit with PTS and DTS, a PCR on each PES start, PAT and
//! PMT (with CRC) every 10 access units, stuffing in the last packet of each PES; 192-byte BDAV
//! packets with arrival time stamps when the output name ends in `.m2ts`.
//! `cargo run --release --example tsmux -- IN.mp4 OUT.ts|OUT.m2ts [MAX_SAMPLES]`
use piedpiper_media::mp4;

const PKT: usize = 188;

fn crc32(d: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in d {
        c ^= (b as u32) << 24;
        for _ in 0..8 {
            c = if c & 0x8000_0000 != 0 { (c << 1) ^ 0x04C1_1DB7 } else { c << 1 };
        }
    }
    c
}

struct Mux {
    out: Vec<u8>,
    cc: [u8; 3],
    bdav: bool,
    ats: u32,
}

impl Mux {
    /// One PES packet or section as transport packets of `pid` (`k` picks the continuity counter).
    fn put(&mut self, pid: u16, k: usize, payload: &[u8], pcr: Option<u64>) {
        let mut used = 0;
        let mut first = true;
        while first || used < payload.len() {
            // adaptation field bytes after its length byte
            let mut af: Option<Vec<u8>> = None;
            if let (true, Some(p)) = (first, pcr) {
                let (base, ext) = (p / 300, p % 300);
                af = Some(vec![
                    0x10,
                    (base >> 25) as u8,
                    (base >> 17) as u8,
                    (base >> 9) as u8,
                    (base >> 1) as u8,
                    ((base & 1) << 7) as u8 | 0x7E | (ext >> 8) as u8,
                    ext as u8,
                ]);
            }
            let left = payload.len() - used;
            let mut room = PKT - 4 - af.as_ref().map_or(0, |v| 1 + v.len());
            if left < room {
                let need = PKT - 4 - left; // adaptation field bytes with its length byte
                let mut v = af.take().unwrap_or_default();
                if v.is_empty() && need >= 2 {
                    v.push(0);
                }
                v.resize(need - 1, 0xFF);
                af = Some(v);
                room = left;
            }
            if self.bdav {
                self.out.extend_from_slice(&(self.ats & 0x3FFF_FFFF).to_be_bytes());
                self.ats = self.ats.wrapping_add(13_500);
            }
            self.out.push(0x47);
            self.out.push(((first as u8) << 6) | (pid >> 8) as u8);
            self.out.push(pid as u8);
            self.out.push(if af.is_some() { 0x30 } else { 0x10 } | (self.cc[k] & 15));
            if let Some(v) = &af {
                self.out.push(v.len() as u8);
                self.out.extend_from_slice(v);
            }
            self.out.extend_from_slice(&payload[used..used + room]);
            used += room;
            self.cc[k] = self.cc[k].wrapping_add(1);
            first = false;
        }
    }

    fn section(&mut self, pid: u16, k: usize, body: &[u8]) {
        let mut s = vec![0u8];
        s.extend_from_slice(body);
        s.extend_from_slice(&crc32(body).to_be_bytes());
        s.resize(PKT - 4, 0xFF);
        self.put(pid, k, &s, None);
    }
}

fn pts_bytes(marker: u8, t: u64) -> [u8; 5] {
    [marker << 4 | ((t >> 29) as u8 & 0x0E) | 1, (t >> 22) as u8, ((t >> 14) as u8) | 1, (t >> 7) as u8, ((t << 1) as u8) | 1]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).expect("read");
    let max: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let (track, hevc) = match mp4::find_hevc_track(&data) {
        Some(t) if mp4::find_avc_track(&data).is_none() => (t, true),
        _ => (mp4::find_avc_track(&data).expect("H.264 or HEVC track"), false),
    };
    let nls = track.nal_len_size;
    let mut mux = Mux { out: Vec::new(), cc: [0; 3], bdav: a[2].ends_with(".m2ts"), ats: 0 };
    let stream_type = if hevc { 0x24 } else { 0x1B };
    let pat = [0x00u8, 0xB0, 0x0D, 0x00, 0x01, 0xC1, 0x00, 0x00, 0x00, 0x01, 0xE1, 0x00];
    let pmt = [0x02u8, 0xB0, 0x12, 0x00, 0x01, 0xC1, 0x00, 0x00, 0xE1, 0x01, 0xF0, 0x00, stream_type, 0xE1, 0x01, 0xF0, 0x00];
    let aud: &[u8] = if hevc { &[0, 0, 0, 1, 0x46, 0x01, 0x50] } else { &[0, 0, 0, 1, 0x09, 0xF0] };
    let n = track.samples.len().min(max);
    for (k, &(o, s)) in track.samples.iter().take(n).enumerate() {
        if k % 10 == 0 {
            mux.section(0, 0, &pat);
            mux.section(0x100, 1, &pmt);
        }
        let smp = &data[o as usize..o as usize + s as usize];
        let mut nals = Vec::new();
        let mut i = 0;
        while i + nls <= smp.len() {
            let len = smp[i..i + nls].iter().fold(0usize, |x, &b| x << 8 | b as usize);
            nals.push(&smp[i + nls..i + nls + len]);
            i += nls + len;
        }
        let irap = nals.iter().any(|n| if hevc { (16..=23).contains(&((n[0] >> 1) & 63)) } else { n[0] & 31 == 5 });
        let mut au = aud.to_vec();
        if irap {
            for p in &track.param_nals {
                au.extend_from_slice(&[0, 0, 0, 1]);
                au.extend_from_slice(p);
            }
        }
        for nal in nals {
            au.extend_from_slice(&[0, 0, 1]);
            au.extend_from_slice(nal);
        }
        let dts = 90_000 + k as u64 * 3003;
        let pts = dts + 3003;
        let mut pes = vec![0, 0, 1, 0xE0, 0, 0, 0x84, 0xC0, 10];
        pes.extend_from_slice(&pts_bytes(3, pts));
        pes.extend_from_slice(&pts_bytes(1, dts));
        pes.extend_from_slice(&au);
        mux.put(0x101, 2, &pes, Some(dts * 300));
    }
    std::fs::write(&a[2], &mux.out).expect("write");
    println!("{n} samples, {} bytes ({})", mux.out.len(), if hevc { "HEVC" } else { "H.264" });
}
