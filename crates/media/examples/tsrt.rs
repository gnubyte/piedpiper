//! Development tool: recompresses H.264 transport streams and checks the round trip.
//! `cargo run --release --example tsrt -- FILE...` (`PPM_THREADS=n`, `PPM_NODEC=1`)
use piedpiper_media::{ts, Options};
use std::time::Instant;

fn main() {
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).expect("read");
        let opts = Options { threads: std::env::var("PPM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(0), ..Default::default() };
        let t0 = Instant::now();
        let Some((packed, st)) = ts::compress_with(&data, &opts, &mut |_, _| true).ok().flatten() else {
            println!("{path}: not a transport stream this handles");
            continue;
        };
        let tc = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let good = std::env::var_os("PPM_NODEC").is_some() || ts::decompress_with(&packed, opts.threads, &mut |_, _| true).is_ok_and(|b| b == data);
        let td = t1.elapsed().as_secs_f64();
        println!(
            "{path}: {} -> {} B ({:.2}% smaller) comp {tc:.1}s decomp {td:.1}s {}; slices recoded {}/{} ({} B -> {} B), fails {:?}",
            data.len(),
            packed.len(),
            100.0 * (1.0 - packed.len() as f64 / data.len() as f64),
            if good { "OK" } else { "MISMATCH" },
            st.recoded,
            st.slices,
            st.slice_bytes,
            st.out_slices,
            st.fails
        );
    }
}
