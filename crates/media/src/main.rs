//! Development tool: `ppmedia rt FILE...` compresses, decompresses and verifies files and
//! prints where the bytes went. `ppmedia c IN OUT` / `ppmedia d IN OUT` convert files.
//! `ppmedia synth OUT [SEED FRAMES W H]` writes a synthetic H.264 MP4 (test input);
//! `ppmedia synth-cavlc OUT [SEED FRAMES W H [high]]` one with CAVLC slices (Baseline, or High);
//! `ppmedia synth-hevc OUT [SEED FRAMES CONFIG]` a synthetic HEVC one (see `hevc::testgen`);
//! `ppmedia synth-vp9 OUT [SEED W H]` a synthetic VP9 WebM (see `vp9::synth`);
//! `ppmedia synth-vp8 OUT [SEED W H FRAMES]` a synthetic VP8 one (see `vp8::stream::synth`);
//! `ppmedia synth-mpeg2 OUT [SEED MBW MBH FRAMES]` a synthetic MPEG-2 MKV (see `mpeg2::synth`).
//! `ppmedia prof FILE [TABLE_BITS]` times the compression stages on one thread.
//! `PPM_BITS=n` overrides the model's table size for `rt`; `PPM_KINDS=1` adds per-syntax-element
//! costs; `PPM_THREADS=n` sets the worker threads; `PPM_NODEC=1` skips the decompression check.

use std::time::Instant;

fn human(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2} MB", n as f64 / 1e6)
    } else if n >= 1000 {
        format!("{:.1} KB", n as f64 / 1e3)
    } else {
        format!("{n} B")
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        eprintln!("usage: ppmedia rt FILE... | c IN OUT | d IN OUT | diag FILE [N] | synth OUT ... | synth-cavlc OUT ... | synth-hevc OUT ... | synth-vp9 OUT ... | synth-vp8 OUT ... | synth-mpeg2 OUT ...");
        std::process::exit(2);
    };
    match cmd.as_str() {
        "rt" => {
            let mut ok = true;
            for path in &args[1..] {
                let data = std::fs::read(path).expect("read");
                let t0 = Instant::now();
                let opts = piedpiper_media::Options {
                    stats: std::env::var_os("PPM_KINDS").is_some(),
                    table_bits: std::env::var("PPM_BITS").ok().and_then(|v| v.parse().ok()),
                    threads: std::env::var("PPM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
                    no_pixels: std::env::var_os("PPM_NOPIX").is_some(),
                };
                let Some((packed, st)) = piedpiper_media::compress_with(&data, &opts, &mut |_, _| true).ok().flatten() else {
                    println!("{path}: not an H.264/HEVC MP4 this tool handles");
                    continue;
                };
                let tc = t0.elapsed().as_secs_f64();
                let t1 = Instant::now();
                // PPM_NODEC skips the decompression check (faster model experiments).
                let back = if std::env::var_os("PPM_NODEC").is_some() { Ok(data.clone()) } else { piedpiper_media::decompress(&packed) };
                let td = t1.elapsed().as_secs_f64();
                let good = matches!(&back, Ok(b) if *b == data);
                ok &= good;
                let video = st.slice_bytes + st.raw_nal_bytes;
                println!(
                    "{path}: {} -> {} ({:.2}% smaller)  comp {:.2}s ({:.1} MB/s)  decomp {:.2}s  {}",
                    human(data.len()),
                    human(packed.len()),
                    100.0 * (1.0 - packed.len() as f64 / data.len() as f64),
                    tc,
                    data.len() as f64 / 1e6 / tc,
                    td,
                    if good { "OK" } else { "MISMATCH" }
                );
                println!(
                    "    slices recoded {}/{} ({} of {} video bytes), bins {}; slices {} -> {} ({:.2}% smaller)",
                    st.recoded,
                    st.slices,
                    human(st.slice_bytes),
                    human(video),
                    st.bins,
                    human(st.slice_bytes),
                    human(st.out_slices),
                    100.0 * (1.0 - st.out_slices as f64 / st.slice_bytes.max(1) as f64)
                );
                println!(
                    "    other bytes {} -> {}; raw NALs {} -> {}; header {}",
                    human(st.other_bytes),
                    human(st.out_other),
                    human(st.raw_nal_bytes),
                    human(st.out_raw),
                    human(st.out_header)
                );
                if std::env::var_os("PPM_KINDS").is_some() {
                    // kinds 21.. are CAVLC syntax (h264::cavlc::kind)
                    const NAMES: [&str; 29] = [
                        "skip", "mb_type", "sub_type", "mvd", "ref_idx", "dqp", "ipred", "cpred", "cbp_luma",
                        "cbp_chroma", "t8x8", "cbf", "sig", "last", "abs", "sign", "abs_suffix", "mvd_suffix",
                        "mvd_sign", "end", "pcm", "more_data", "coeff_token", "trailing_sign", "level_prefix",
                        "level_suffix", "total_zeros", "run_before", "pcm_bits",
                    ];
                    let names: &[&str] = if st.hevc { &piedpiper_media::hevc::ctu::kind::NAMES } else { &NAMES };
                    let tot: f64 = st.kind_cost.iter().map(|k| k.1).sum();
                    for (i, &(n, c, mm)) in st.kind_cost.iter().enumerate() {
                        if n > 0 {
                            println!(
                                "      {:<11} bins {:>9}  cabac {:>9.0} B ({:>4.1}%)  model {:>9.0} B  saved {:>5.1}%",
                                names.get(i).copied().unwrap_or("?"),
                                n,
                                c / 8.0,
                                100.0 * c / tot,
                                mm / 8.0,
                                100.0 * (1.0 - mm / c)
                            );
                        }
                    }
                }
                let fails: Vec<String> = st
                    .fails
                    .iter()
                    .zip(piedpiper_media::FAIL_NAMES)
                    .filter(|(n, _)| **n > 0)
                    .map(|(n, name)| format!("{name}: {n}"))
                    .collect();
                if !fails.is_empty() {
                    println!("    stored raw: {}", fails.join(", "));
                }

            }
            if !ok {
                std::process::exit(1);
            }
        }
        "diag" if args.len() >= 2 => {
            let data = std::fs::read(&args[1]).expect("read");
            let limit = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12);
            let lines = if piedpiper_media::mp4::find_hevc_track(&data).is_some() && piedpiper_media::mp4::find_avc_track(&data).is_none() {
                piedpiper_media::hevc::recode::diagnose(&data, limit)
            } else {
                piedpiper_media::container::diagnose(&data, limit)
            };
            for line in lines {
                println!("{line}");
            }
        }
        "aus" if args.len() >= 3 => {
            // Development aid: the H.264 track's samples as Annex B access units, each
            // preceded by its length (u32 LE); parameter sets go in front of the first.
            let data = std::fs::read(&args[1]).expect("read");
            let limit: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
            let t = piedpiper_media::mp4::find_avc_track(&data).expect("H.264 track");
            let mut out = Vec::new();
            for (i, &(o, sz)) in t.samples.iter().take(limit).enumerate() {
                let smp = &data[o as usize..o as usize + sz as usize];
                let mut au = Vec::new();
                if i == 0 {
                    for n in &t.param_nals {
                        au.extend_from_slice(&[0, 0, 0, 1]);
                        au.extend_from_slice(n);
                    }
                }
                let mut p = 0;
                while p + t.nal_len_size <= smp.len() {
                    let mut l = 0usize;
                    for k in 0..t.nal_len_size {
                        l = l << 8 | smp[p + k] as usize;
                    }
                    p += t.nal_len_size;
                    au.extend_from_slice(&[0, 0, 0, 1]);
                    au.extend_from_slice(&smp[p..p + l]);
                    p += l;
                }
                out.extend_from_slice(&(au.len() as u32).to_le_bytes());
                out.extend_from_slice(&au);
            }
            std::fs::write(&args[2], out).expect("write");
        }
        "mkv" if args.len() == 3 => {
            // Remux an H.264 MP4 into Matroska (test input).
            let data = std::fs::read(&args[1]).expect("read");
            let t = piedpiper_media::mp4::find_avc_track_iso(&data).expect("H.264 MP4");
            let sps: Vec<&Vec<u8>> = t.param_nals.iter().filter(|n| n[0] & 31 == 7).collect();
            let pps: Vec<&Vec<u8>> = t.param_nals.iter().filter(|n| n[0] & 31 == 8).collect();
            let s0 = sps[0];
            let mut avcc = vec![1, s0[1], s0[2], s0[3], 0xFC | (t.nal_len_size as u8 - 1), 0xE0 | sps.len() as u8];
            for n in &sps {
                avcc.extend_from_slice(&(n.len() as u16).to_be_bytes());
                avcc.extend_from_slice(n);
            }
            avcc.push(pps.len() as u8);
            for n in &pps {
                avcc.extend_from_slice(&(n.len() as u16).to_be_bytes());
                avcc.extend_from_slice(n);
            }
            let frames: Vec<Vec<u8>> = t.samples.iter().map(|&(o, s)| data[o as usize..o as usize + s as usize].to_vec()).collect();
            std::fs::write(&args[2], piedpiper_media::mkv::write_mkv(&avcc, &frames, 0, 0)).expect("write");
        }
        "prof" if args.len() >= 2 => {
            let data = std::fs::read(&args[1]).expect("read");
            let bits = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(22);
            for line in piedpiper_media::container::profile(&data, bits) {
                println!("{line}");
            }
        }
        "synth" if args.len() >= 2 => {
            let n = |i: usize, d: u64| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
            let file = piedpiper_media::testgen::synthetic_mp4(n(2, 2026), n(3, 12) as usize, n(4, 20) as u32, n(5, 12) as u32);
            std::fs::write(&args[1], file).expect("write");
        }
        "synth-cavlc" if args.len() >= 2 => {
            let n = |i: usize, d: u64| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
            let high = args.get(6).is_some_and(|s| s == "high");
            let file =
                piedpiper_media::testgen::synthetic_mp4_cavlc(n(2, 2026), n(3, 12) as usize, n(4, 20) as u32, n(5, 12) as u32, high);
            std::fs::write(&args[1], file).expect("write");
        }
        "synth-hevc" if args.len() >= 2 => {
            let seed = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2026);
            let frames = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
            let name = args.get(4).map_or("main", String::as_str);
            let Some(cfg) = piedpiper_media::hevc::testgen::config(name) else {
                eprintln!("unknown config {name}; one of {:?}", piedpiper_media::hevc::testgen::CONFIGS);
                std::process::exit(2);
            };
            std::fs::write(&args[1], piedpiper_media::hevc::testgen::synthetic_mp4(seed, frames, &cfg)).expect("write");
        }
        "synth-vp9" if args.len() >= 2 => {
            let n = |i: usize, d: u64| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
            let file = piedpiper_media::vp9::synth::synthetic_webm(n(2, 5), n(3, 96) as u32, n(4, 64) as u32, 0, 0);
            std::fs::write(&args[1], file).expect("write");
        }
        "synth-vp8" if args.len() >= 2 => {
            let n = |i: usize, d: u64| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
            let file = piedpiper_media::vp8::stream::synth::synthetic_webm(n(2, 8), n(3, 96) as u32, n(4, 64) as u32, n(5, 10) as usize);
            std::fs::write(&args[1], file).expect("write");
        }
        "synth-mpeg2" if args.len() >= 2 => {
            let n = |i: usize, d: u64| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
            let file = piedpiper_media::mpeg2::synth::synthetic_mkv(n(2, 7), n(3, 6) as u32, n(4, 4) as u32, n(5, 8) as usize);
            std::fs::write(&args[1], file).expect("write");
        }
        "c" | "d" if args.len() == 3 => {
            let data = std::fs::read(&args[1]).expect("read");
            let out = if cmd == "c" {
                piedpiper_media::compress(&data).expect("unsupported input").0
            } else {
                piedpiper_media::decompress(&data).expect("corrupt input")
            };
            std::fs::write(&args[2], out).expect("write");
        }
        _ => {
            eprintln!("usage: ppmedia rt FILE... | c IN OUT | d IN OUT | diag FILE [N] | synth OUT ... | synth-cavlc OUT ... | synth-hevc OUT ... | synth-vp9 OUT ... | synth-vp8 OUT ... | synth-mpeg2 OUT ...");
            std::process::exit(2);
        }
    }
}
