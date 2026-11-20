//! Development aid: `m2rt [--diag|--nodec] FILE...` recompresses MPEG-2 MKV files, checks the
//! round trip and prints where the bytes went (like vp9rt).
use piedpiper_media::mpeg2::stream;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--diag") {
        for path in &args[1..] {
            let data = std::fs::read(path).expect("read");
            for line in stream::diagnose(&data, 40) {
                println!("{line}");
            }
        }
        return;
    }
    let nodec = args.iter().any(|a| a == "--nodec");
    for path in args.iter().filter(|a| !a.starts_with("--")) {
        let data = std::fs::read(path).expect("read");
        let opts = piedpiper_media::Options { stats: true, ..Default::default() };
        let t0 = Instant::now();
        let Some((packed, st)) = stream::compress_with(&data, &opts, &mut |_, _| true).ok().flatten() else {
            println!("{path}: not recodable MPEG-2");
            continue;
        };
        let tc = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let back = if nodec { Ok(data.clone()) } else { stream::decompress(&packed) };
        let td = t1.elapsed().as_secs_f64();
        let good = matches!(&back, Ok(b) if *b == data);
        println!(
            "{path}: {} -> {} B ({:.2}% smaller) comp {tc:.2}s decomp {td:.2}s {}",
            data.len(),
            packed.len(),
            100.0 * (1.0 - packed.len() as f64 / data.len() as f64),
            if good { "OK" } else { "MISMATCH" }
        );
        println!(
            "  frames {} slices recoded {} ({} B -> {} B, {:.2}% smaller); raw {} B -> {} B; copies {}; other {} B -> {} B",
            st.frames,
            st.recoded,
            st.frame_bytes,
            st.out_frames,
            100.0 * (1.0 - st.out_frames as f64 / st.frame_bytes.max(1) as f64),
            st.raw_bytes,
            st.out_raw,
            st.copies,
            st.other_bytes,
            st.out_other + st.out_header
        );
        if st.audio_frames > 0 {
            println!(
                "  AC-3 frames {} ({} B -> {} B, {:.2}% smaller)",
                st.audio_frames,
                st.audio_bytes,
                st.out_audio,
                100.0 * (1.0 - st.out_audio as f64 / st.audio_bytes.max(1) as f64)
            );
        }
        println!("  fails {:?}; bits {} original {:.0} B model {:.0} B ({:.2}% smaller)", st.fails, st.bools, st.vp9_bits / 8.0, st.model_bits / 8.0, 100.0 * (1.0 - st.model_bits / st.vp9_bits.max(1.0)));
        const NAMES: [&str; 24] = [
            "slice_hdr", "mba", "mb_type", "modes", "quant", "mv_code", "mv_sign", "residual", "dmv", "cbp", "dc_size",
            "dc_diff", "coef", "coef_sign", "escape", "more", "?", "?", "tail", "hdr", "raw", "start", "?", "?",
        ];
        for (i, &(n, v, mm)) in st.kind_cost.iter().enumerate() {
            if n > 0 {
                println!("    {:<10} {:>9} bools  libvpx {:>9.0} B ({:>4.1}%)  model {:>9.0} B  saved {:>5.1}%", NAMES.get(i).copied().unwrap_or("?"), n, v / 8.0, 100.0 * v / st.vp9_bits.max(1.0), mm / 8.0, 100.0 * (1.0 - mm / v.max(1e-9)));
            }
        }
    }
}
